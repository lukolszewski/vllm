// Turing (sm_75) paged attention for small query counts (decode, MTP verify).
//
// Why: FlashInfer's fa2 kernel on sm_75 (no cp.async, 64 KiB smem, head_dim 256 -> 1 CTA/SM) is latency-bound
// and reads KV at ~230 GB/s of the 577 GB/s the card does. This kernel keeps K/V out of shared memory:
//   * each warp owns 64 of the 256 head dims, loads its K/V slice straight into registers (16 B / 8 B vectors),
//   * S = Q K^T and O^T = V^T P^T run on mma.sync.m16n8k8 (f16 in, f32 accumulate), which sm_75 has,
//   * the 4 per-warp partial S tiles are summed through a small shared buffer, softmax is online (FA2 style),
//   * the KV range is split into `nch` chunks per (request, kv head); a merge kernel combines them by LSE.
// Semantics = flashinfer BatchPrefillWithPagedKVCacheWrapper(layout NHD), causal bottom-right aligned.
// Layout: kv_cache [num_pages, 2, page, HK, 256] (any strides, last dim contiguous), fp16 or fp8 e4m3.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <type_traits>

namespace {
constexpr int D = 256;
constexpr int WARPS = 4;
constexpr int TILE = 32;  // tokens per iteration

__device__ __forceinline__ void mma1688(float* c, uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
}

// x holds two e4m3 bytes at byte 0 and byte 2 (bytes 1/3 ignored) -> half2 bits (low = byte 0).
__device__ __forceinline__ uint32_t f8pair(uint32_t x) {
  uint32_t bits = ((x & 0x00800080u) << 8) | ((x & 0x007f007fu) << 7);
  half2 h = *reinterpret_cast<half2*>(&bits);
  const half2 k256 = __halves2half2(__ushort_as_half(0x5C00), __ushort_as_half(0x5C00));
  h = __hmul2(h, k256);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

template <bool FP8> struct Raw;
template <> struct Raw<true> { using K = uint4; using V = uint2; using E = uint8_t; };   // 16 dims / 8 dims of e4m3
template <> struct Raw<false> { struct K2 { uint4 a, b; }; using K = K2; using V = uint4; using E = half; };

template <int MT, bool FP8>
__global__ void __launch_bounds__(128, 2)
ta_partial(const half* __restrict__ q, const void* __restrict__ kv_raw,
           int64_t s_page, int64_t s_kv, int64_t s_tok, int64_t s_head,
           const int* __restrict__ kv_indptr, const int* __restrict__ kv_indices,
           const int* __restrict__ kv_last, const int* __restrict__ qo_indptr,
           float* __restrict__ o_part, float* __restrict__ lse_part,
           int page, int HQ, int HK, float scale_log2, int nch, int min_tper) {
  using E = typename Raw<FP8>::E;
  using RK = typename Raw<FP8>::K;
  using RV = typename Raw<FP8>::V;
  constexpr int ROWS = MT * 16;
  const E* kvp = reinterpret_cast<const E*>(kv_raw);

  const int chunk = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int G = HQ / HK;
  const int qs = qo_indptr[b], ql = qo_indptr[b + 1] - qs;
  const int pbase = kv_indptr[b], np = kv_indptr[b + 1] - pbase;
  const int kvlen = np > 0 ? (np - 1) * page + kv_last[b] : 0;
  const int R = ql * G;
  const int ntiles = (kvlen + TILE - 1) / TILE;
  const int tper = max((ntiles + nch - 1) / nch, min_tper);
  const int t0 = chunk * tper, t1 = min(ntiles, t0 + tper);
  constexpr int ROWS_ = MT * 16;
  if (t0 >= t1) {  // empty chunk: only the lse marker is read by the merge
    if (warp == 0 && tig == 0) {
      const int64_t cb = (((int64_t)b * HK + kvh) * nch + chunk) * ROWS_;
      for (int r = gid; r < ROWS_; r += 8) lse_part[cb + r] = -INFINITY;
    }
    return;
  }

  // Q as the A operand: rows gid / gid+8 of each 16-row tile, dims warp*64 + tig*16 + kk*2 (+1).
  uint32_t qa[MT][8][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = mt * 16 + gid + h * 8;
      if (r < R) {
        const half* qrow = q + ((int64_t)(qs + r / G) * HQ + kvh * G + r % G) * D + warp * 64 + tig * 16;
        const uint4 w0 = *reinterpret_cast<const uint4*>(qrow);
        const uint4 w1 = *reinterpret_cast<const uint4*>(qrow + 8);
        qa[mt][0][h] = w0.x; qa[mt][1][h] = w0.y; qa[mt][2][h] = w0.z; qa[mt][3][h] = w0.w;
        qa[mt][4][h] = w1.x; qa[mt][5][h] = w1.y; qa[mt][6][h] = w1.z; qa[mt][7][h] = w1.w;
      } else {
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) qa[mt][kk][h] = 0u;
      }
    }

  float m[MT][2], l[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) { m[mt][0] = m[mt][1] = -1e30f; l[mt][0] = l[mt][1] = 0.f; }
  float o[4][2 * MT][4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int n = 0; n < 2 * MT; ++n)
#pragma unroll
      for (int c = 0; c < 4; ++c) o[j][n][c] = 0.f;

  __shared__ float red[MT * 16][WARPS][32];

  RK kr[4];
  RV vr[4][2];
  auto load_k = [&](int tok0) {
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int tok = tok0 + nt * 8 + gid;
      if (tok < kvlen) {
        const int pg = __ldg(kv_indices + pbase + tok / page), off = tok % page;
        const E* kp = kvp + (int64_t)pg * s_page + (int64_t)off * s_tok + (int64_t)kvh * s_head + warp * 64 + tig * 16;
        if constexpr (FP8) kr[nt] = __ldg(reinterpret_cast<const uint4*>(kp));
        else { kr[nt].a = __ldg(reinterpret_cast<const uint4*>(kp)); kr[nt].b = __ldg(reinterpret_cast<const uint4*>(kp) + 1); }
      } else {
        if constexpr (FP8) kr[nt] = make_uint4(0, 0, 0, 0);
        else { kr[nt].a = make_uint4(0, 0, 0, 0); kr[nt].b = kr[nt].a; }
      }
    }
  };
  auto load_v = [&](int tok0) {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int tok = tok0 + ks * 8 + tig * 2 + e;
        if (tok < kvlen) {
          const int pg = __ldg(kv_indices + pbase + tok / page), off = tok % page;
          const E* vp = kvp + (int64_t)pg * s_page + s_kv + (int64_t)off * s_tok + (int64_t)kvh * s_head + warp * 64 + gid * 8;
          vr[ks][e] = __ldg(reinterpret_cast<const RV*>(vp));
        } else {
          if constexpr (FP8) vr[ks][e] = make_uint2(0, 0);
          else vr[ks][e] = make_uint4(0, 0, 0, 0);
        }
      }
  };
  constexpr bool PIPE = (MT == 1);  // MT>=2 is at the register cap; prefetch would spill/occupy
  if constexpr (PIPE) { load_k(t0 * TILE); load_v(t0 * TILE); }
  for (int t = t0; t < t1; ++t) {
    const int tok0 = t * TILE;
    const bool more = PIPE && (t + 1 < t1);
    if constexpr (!PIPE) { load_k(tok0); load_v(tok0); }
    // ---- partial S over this warp's 64 dims ----
    float s[MT][4][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) s[mt][nt][c] = 0.f;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      uint32_t kb[8];
      if constexpr (FP8) {
        const uint32_t w[4] = {kr[nt].x, kr[nt].y, kr[nt].z, kr[nt].w};
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) kb[kk] = f8pair(__byte_perm(w[kk >> 1], 0u, (kk & 1) ? 0x4342 : 0x4140));
      } else {
        kb[0] = kr[nt].a.x; kb[1] = kr[nt].a.y; kb[2] = kr[nt].a.z; kb[3] = kr[nt].a.w;
        kb[4] = kr[nt].b.x; kb[5] = kr[nt].b.y; kb[6] = kr[nt].b.z; kb[7] = kr[nt].b.w;
      }
#pragma unroll
      for (int kk = 0; kk < 8; ++kk)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma1688(s[mt][nt], qa[mt][kk][0], qa[mt][kk][1], kb[kk]);
    }

    if (more) load_k(tok0 + TILE);  // K registers are free now
    // ---- sum the 4 warps' partial S (each warp ends with the full S in the same fragment layout) ----
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) red[(mt * 4 + nt) * 4 + c][warp][lane] = s[mt][nt][c];
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int i = (mt * 4 + nt) * 4 + c;
          s[mt][nt][c] = red[i][0][lane] + red[i][1][lane] + red[i][2][lane] + red[i][3][lane];
        }
    __syncthreads();

    // ---- scale, causal mask, online softmax ----
    float alpha[MT][2];
    uint32_t pb[2 * MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = mt * 16 + gid + h * 8;
        const int lim = (row < R) ? (kvlen - ql + row / G) : -1;  // last allowed token
        float mx = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int tok = tok0 + nt * 8 + tig * 2 + e;
            float v = s[mt][nt][h * 2 + e] * scale_log2;
            v = (tok <= lim) ? v : -INFINITY;
            s[mt][nt][h * 2 + e] = v;
            mx = fmaxf(mx, v);
          }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float mnew = fmaxf(m[mt][h], mx);
        alpha[mt][h] = exp2f(m[mt][h] - mnew);
        m[mt][h] = mnew;
        float sum = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const float p0 = exp2f(s[mt][nt][h * 2] - mnew), p1 = exp2f(s[mt][nt][h * 2 + 1] - mnew);
          sum += p0 + p1;
          pb[mt * 2 + h][nt] = pack_h2(p0, p1);
        }
        l[mt][h] = l[mt][h] * alpha[mt][h] + sum;
      }

    // ---- rescale O (rows of the O^T fragment are tig*2+e, softmax stats live on lane (row%8)*4) ----
#pragma unroll
    for (int n = 0; n < 2 * MT; ++n) {
      const float a0 = __shfl_sync(0xffffffffu, alpha[n >> 1][n & 1], (tig * 2) * 4);
      const float a1 = __shfl_sync(0xffffffffu, alpha[n >> 1][n & 1], (tig * 2 + 1) * 4);
#pragma unroll
      for (int j = 0; j < 4; ++j) { o[j][n][0] *= a0; o[j][n][1] *= a1; o[j][n][2] *= a0; o[j][n][3] *= a1; }
    }

    // ---- O^T[dim, row] += V^T[dim, tok] * P^T[tok, row] ----
#pragma unroll
    for (int ks = 0; ks < 4; ++ks)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t a0, a1;
        if constexpr (FP8) {
          // dims 2j (a0) and 2j+1 (a1) of this thread's 8-dim slice; byte i of an 8-byte uint2
          const int i0 = 2 * j, i1 = 2 * j + 1;
          const uint32_t A0 = (i0 < 4) ? vr[ks][0].x : vr[ks][0].y, B0 = (i0 < 4) ? vr[ks][1].x : vr[ks][1].y;
          const uint32_t b0 = i0 & 3, b1 = i1 & 3;
          a0 = f8pair(__byte_perm(A0, B0, b0 | (b0 << 4) | ((4 + b0) << 8) | ((4 + b0) << 12)));
          a1 = f8pair(__byte_perm(A0, B0, b1 | (b1 << 4) | ((4 + b1) << 8) | ((4 + b1) << 12)));
        } else {
          const uint32_t wa[4] = {vr[ks][0].x, vr[ks][0].y, vr[ks][0].z, vr[ks][0].w};
          const uint32_t wb[4] = {vr[ks][1].x, vr[ks][1].y, vr[ks][1].z, vr[ks][1].w};
          a0 = __byte_perm(wa[j], wb[j], 0x5410);
          a1 = __byte_perm(wa[j], wb[j], 0x7632);
        }
#pragma unroll
        for (int n = 0; n < 2 * MT; ++n) mma1688(o[j][n], a0, a1, pb[n][ks]);
      }
    if (more) load_v(tok0 + TILE);  // V registers are free now
  }

  // ---- finalize this chunk: O/l and lse (log2 domain) ----
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      l[mt][h] += __shfl_xor_sync(0xffffffffu, l[mt][h], 1);
      l[mt][h] += __shfl_xor_sync(0xffffffffu, l[mt][h], 2);
    }
  const int64_t cbase = (((int64_t)b * HK + kvh) * nch + chunk) * ROWS;
  float* op = o_part + cbase * D;
#pragma unroll
  for (int n = 0; n < 2 * MT; ++n) {
    const float l0 = __shfl_sync(0xffffffffu, l[n >> 1][n & 1], (tig * 2) * 4);
    const float l1 = __shfl_sync(0xffffffffu, l[n >> 1][n & 1], (tig * 2 + 1) * 4);
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int r0 = n * 8 + tig * 2, r1 = r0 + 1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int d0 = warp * 64 + gid * 8 + 2 * j;
      if (r0 < R) { op[(int64_t)r0 * D + d0] = o[j][n][0] * i0; op[(int64_t)r0 * D + d0 + 1] = o[j][n][2] * i0; }
      if (r1 < R) { op[(int64_t)r1 * D + d0] = o[j][n][1] * i1; op[(int64_t)r1 * D + d0 + 1] = o[j][n][3] * i1; }
    }
  }
  if (warp == 0 && tig == 0) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = mt * 16 + gid + h * 8;
        if (row < R) lse_part[cbase + row] = l[mt][h] > 0.f ? m[mt][h] + log2f(l[mt][h]) : -INFINITY;
      }
  }
}

__global__ void ta_merge(const float* __restrict__ o_part, const float* __restrict__ lse_part, half* __restrict__ out,
                         const int* __restrict__ qo_indptr, int HQ, int HK, int nch, int ROWS, float v_scale,
                         int64_t o_s_tok, int64_t o_s_head) {
  const int row = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z, d = threadIdx.x;
  const int G = HQ / HK;
  const int qs = qo_indptr[b], ql = qo_indptr[b + 1] - qs;
  if (row >= ql * G) return;
  const int64_t cbase = ((int64_t)b * HK + kvh) * nch * ROWS;
  float M = -INFINITY;
  for (int c = 0; c < nch; ++c) M = fmaxf(M, lse_part[cbase + (int64_t)c * ROWS + row]);
  float acc = 0.f, ws = 0.f;
  for (int c = 0; c < nch; ++c) {
    const float ls = lse_part[cbase + (int64_t)c * ROWS + row];
    if (ls == -INFINITY) continue;
    const float w = exp2f(ls - M);
    acc += w * o_part[(cbase + (int64_t)c * ROWS + row) * D + d];
    ws += w;
  }
  const float r = ws > 0.f ? acc / ws * v_scale : 0.f;
  out[(int64_t)(qs + row / G) * o_s_tok + (int64_t)(kvh * G + row % G) * o_s_head + d] = __float2half_rn(r);
}

// lse_part must hold -inf for chunks a CTA never writes (rows >= R are never read; empty chunks write -inf
// themselves because l == 0), so no memset is needed.
template <int MT, bool FP8>
void launch(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& kv_indptr, const at::Tensor& kv_indices,
            const at::Tensor& kv_last, const at::Tensor& qo_indptr, at::Tensor& o_part, at::Tensor& lse_part,
            at::Tensor& out, int B, int HQ, int HK, int page, float scale_log2, float v_scale, int nch,
            int min_tper, cudaStream_t st) {
  if (nch <= 0) {
    static int slots = 0;
    if (!slots) {
      int per_sm = 0, sms = 0, dev = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ta_partial<MT, FP8>, 128, 0);
      slots = per_sm * sms;
    }
    nch = std::max(1, slots / std::max(1, B * HK));
  }
  dim3 g1(nch, HK, B);
  ta_partial<MT, FP8><<<g1, 128, 0, st>>>(
      reinterpret_cast<const half*>(q.data_ptr()), kv.data_ptr(), kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
      kv_indptr.data_ptr<int>(), kv_indices.data_ptr<int>(), kv_last.data_ptr<int>(), qo_indptr.data_ptr<int>(),
      o_part.data_ptr<float>(), lse_part.data_ptr<float>(), page, HQ, HK, scale_log2, nch, min_tper);
  dim3 g2(MT * 16, HK, B);
  ta_merge<<<g2, D, 0, st>>>(o_part.data_ptr<float>(), lse_part.data_ptr<float>(),
                              reinterpret_cast<half*>(out.data_ptr()), qo_indptr.data_ptr<int>(), HQ, HK, nch, MT * 16,
                              v_scale, out.stride(0), out.stride(1));
}
}  // namespace

at::Tensor run(at::Tensor q, at::Tensor kv, at::Tensor kv_indptr, at::Tensor kv_indices, at::Tensor kv_last,
               at::Tensor qo_indptr, double sm_scale, double k_scale, double v_scale, int64_t max_q_len, int64_t nch,
               c10::optional<at::Tensor> out_opt, c10::optional<at::Tensor> ws_opt, int64_t min_tper) {
  TORCH_CHECK(q.scalar_type() == at::kHalf && q.dim() == 3 && q.size(2) == D && q.stride(2) == 1 && q.stride(1) == D,
              "q must be fp16 [T, HQ, 256] with contiguous heads");
  TORCH_CHECK(kv.dim() == 5 && kv.size(1) == 2 && kv.size(4) == D && kv.stride(4) == 1, "kv_cache must be [P,2,page,HK,256]");
  const bool fp8 = kv.scalar_type() == at::ScalarType::Float8_e4m3fn;
  TORCH_CHECK(fp8 || kv.scalar_type() == at::kHalf, "kv_cache must be fp16 or float8_e4m3fn");
  const int HQ = q.size(1), HK = kv.size(3), page = kv.size(2);
  TORCH_CHECK(HQ % HK == 0, "HQ % HK");
  const int B = qo_indptr.size(0) - 1;
  const int G = HQ / HK;
  const int MT = (int)((max_q_len * G + 15) / 16);
  TORCH_CHECK(MT >= 1 && MT <= 3, "max_q_len * G must be <= 48");
  TORCH_CHECK(nch >= 1, "nch");
  const c10::cuda::CUDAGuard guard(q.device());
  at::Tensor out = out_opt.has_value() ? *out_opt : at::empty({q.size(0), HQ, D}, q.options());
  const int64_t rows = MT * 16;
  const int64_t need = (int64_t)B * HK * nch * rows * (D + 1);
  at::Tensor ws;
  if (ws_opt.has_value() && ws_opt->numel() * ws_opt->element_size() >= need * 4) ws = *ws_opt;
  else ws = at::empty({need}, q.options().dtype(at::kFloat));
  float* base = reinterpret_cast<float*>(ws.data_ptr());
  at::Tensor o_part = at::from_blob(base, {(int64_t)B * HK * nch * rows * D}, q.options().dtype(at::kFloat));
  at::Tensor lse_part = at::from_blob(base + (int64_t)B * HK * nch * rows * D, {(int64_t)B * HK * nch * rows},
                                      q.options().dtype(at::kFloat));
  const float scale_log2 = (float)(sm_scale * k_scale * 1.4426950408889634);
  cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
#define TA_L(MT_, F_) launch<MT_, F_>(q, kv, kv_indptr, kv_indices, kv_last, qo_indptr, o_part, lse_part, out, B, HQ, HK, page, scale_log2, (float)v_scale, (int)nch, (int)std::max<int64_t>(1, min_tper), st)
  if (fp8) { if (MT == 1) TA_L(1, true); else if (MT == 2) TA_L(2, true); else TA_L(3, true); }
  else { if (MT == 1) TA_L(1, false); else if (MT == 2) TA_L(2, false); else TA_L(3, false); }
#undef TA_L
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "Turing paged attention (small q_len)");
}
