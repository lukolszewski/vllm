# SPDX-License-Identifier: Apache-2.0
"""int8 per-token-per-head KV cache kernels (Turing sm_75 arm), pure torch + triton.

Cache layout (one transformer layer).  The KV cache is a raw byte buffer of ``B``
blocks; block ``b`` starts at byte ``b * Sb`` (``Sb`` = block stride in bytes,
``Sb >= N*H*520``, ``Sb % 16 == 0``, may be padded by a hybrid allocator).
``N`` = tokens per block, ``H`` = kv heads, ``D`` = head dim (256).

Inside one block:

* data region ``[0, N*H*512)``: row ``r = n*H + h`` at byte ``r*512`` holds
  ``K[0..255]`` int8 then ``V[0..255]`` int8  (16-byte aligned rows).
* scale region at byte ``N*H*512 + r*8``: ``k_scale`` float32 then ``v_scale`` float32.

Zero-copy torch views over a raw ``torch.uint8`` tensor of ``B*Sb`` bytes
(see :func:`make_views`):

    kv_int8   = as_strided(raw.view(torch.int8),    (B,2,N,H,D), (Sb, D, H*2*D, 2*D, 1))
    kv_scales = as_strided(raw.view(torch.float32), (B,N,H,2),   (Sb//4, 2*H, 2, 1),
                           storage_offset=N*H*2*D//4)

``kv_int8[b,0,n,h,:]`` is K of token ``(b,n)`` head ``h``, ``kv_int8[b,1,n,h,:]`` is V,
``kv_scales[b,n,h,0/1]`` is the k/v scale.  All kernels take these two views and use
their *strides* (passed to Triton) - nothing is assumed about contiguity or padding.

Quantisation (matches :func:`oracle_write` exactly), per token ``t``, per kv head
``h``, separately for K and V (``x`` = ``D`` fp16 values, computed in fp32)::

    absmax = max(|x|)
    scale  = absmax / 127                       (0 if absmax == 0)
    q      = clamp(rint(x * (127 / absmax)), -127, 127).to(int8)   (0 if absmax == 0)
    x_hat  = float(q) * scale                   (dequant)

Both kernels are CUDA-graph capturable: no host sync, no allocation, no ``.item()``.
"""

from __future__ import annotations

import triton
import triton.language as tl

# torch is only needed by the host wrappers / the oracles.  Importing it lazily keeps
# the Triton kernels usable under ``TRITON_INTERPRET=1`` with plain numpy buffers
# (CPU-only sanity checks of the index arithmetic), which is handy off the GPU.
try:  # pragma: no cover - trivial
    import torch
except Exception:  # pragma: no cover - trivial
    torch = None

QUANT_MAX = 127.0
"""int8 symmetric quantisation max (absmax maps to +-127)."""

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_FLOOR = getattr(tl, "floor", None) or tl.math.floor

try:  # triton >= 3.0
    from triton.language.extra.libdevice import rint as _LIBDEVICE_RINT
except Exception:  # pragma: no cover - version dependent
    try:  # triton 2.2/2.1
        from triton.language.extra.cuda.libdevice import rint as _LIBDEVICE_RINT
    except Exception:
        _LIBDEVICE_RINT = None


if _LIBDEVICE_RINT is not None:

    @triton.jit
    def _rint(x):
        """Round to nearest, ties to even (IEEE rint)."""
        return _LIBDEVICE_RINT(x)

else:  # pragma: no cover - only on old triton

    @triton.jit
    def _rint(x):
        # floor(x + 0.5) with the .5 ties pushed back to even.
        y = x + 0.5
        f = _FLOOR(y)
        tie = y == f
        odd = (f - 2.0 * _FLOOR(f * 0.5)) == 1.0
        return tl.where(tie & odd, f - 1.0, f)


@triton.jit
def _quant(x, BLOCK_D: tl.constexpr):
    """x: [BLOCK_H, BLOCK_D] fp32 (padding lanes == 0) -> (q fp32 integral, scale fp32)."""
    absmax = tl.max(tl.abs(x), axis=1)                       # [BLOCK_H]
    nonzero = absmax > 0
    inv = 127.0 / tl.where(nonzero, absmax, 1.0)             # fp32, like the oracle
    q = _rint(x * inv[:, None])
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    q = tl.where(nonzero[:, None], q, 0.0)
    scale = tl.where(nonzero, absmax / 127.0, 0.0)
    return q, scale


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

@triton.jit
def _kv_write_kernel(
    key_ptr,       # [T, H, D] fp16
    value_ptr,     # [T, H, D] fp16
    kv_ptr,        # kv_int8 view, int8
    scale_ptr,     # kv_scales view, fp32
    slot_ptr,      # [T] int64 slot = block*N + n, < 0 -> skip
    stride_kt, stride_kh, stride_kd,        # key strides (elements)
    stride_vt, stride_vh, stride_vd,        # value strides (elements)
    stride_cb, stride_ckv, stride_cn, stride_ch,   # kv_int8 strides 0,1,2,3 (dim 4 == 1)
    stride_sb, stride_sn, stride_sh,               # kv_scales strides 0,1,2 (dim 3 == 1)
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    slot = tl.load(slot_ptr + tok).to(tl.int64)
    if slot < 0:
        return
    blk = slot // N
    n = slot % N

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    hmask = offs_h < H
    dmask = offs_d < D
    m = hmask[:, None] & dmask[None, :]

    kv_row = kv_ptr + blk * stride_cb + n * stride_cn + offs_h[:, None] * stride_ch \
        + offs_d[None, :]
    sc_row = scale_ptr + blk * stride_sb + n * stride_sn + offs_h * stride_sh

    # ---- K ------------------------------------------------------------------
    k = tl.load(key_ptr + tok * stride_kt + offs_h[:, None] * stride_kh
                + offs_d[None, :] * stride_kd, mask=m, other=0.0).to(tl.float32)
    q, scale = _quant(k, BLOCK_D)
    tl.store(kv_row, q.to(tl.int32).to(tl.int8), mask=m)
    tl.store(sc_row, scale, mask=hmask)

    # ---- V ------------------------------------------------------------------
    v = tl.load(value_ptr + tok * stride_vt + offs_h[:, None] * stride_vh
                + offs_d[None, :] * stride_vd, mask=m, other=0.0).to(tl.float32)
    q, scale = _quant(v, BLOCK_D)
    tl.store(kv_row + stride_ckv, q.to(tl.int32).to(tl.int8), mask=m)
    tl.store(sc_row + 1, scale, mask=hmask)


@triton.jit
def _dequant_gather_kernel(
    kv_ptr,        # kv_int8 view, int8
    scale_ptr,     # kv_scales view, fp32
    page_ptr,      # [P] int32/int64 block ids, any order, may repeat
    out_ptr,       # [P_max, 2, N, H, D] fp16, contiguous-ish (strides passed)
    stride_cb, stride_ckv, stride_cn, stride_ch,   # kv_int8 strides 0..3
    stride_sb, stride_sn, stride_sh,               # kv_scales strides 0..2
    stride_ob, stride_os, stride_on, stride_oh,    # out strides 0..3 (dim 4 == 1)
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)          # index into page_ids / out
    nb = tl.program_id(1)         # token tile inside the block
    s = tl.program_id(2)          # 0 = K, 1 = V

    page = tl.load(page_ptr + i).to(tl.int64)

    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    nmask = offs_n < N
    m = nmask[:, None, None] & (offs_h[None, :, None] < H) & (offs_d[None, None, :] < D)

    q = tl.load(kv_ptr + page * stride_cb + s * stride_ckv
                + offs_n[:, None, None] * stride_cn + offs_h[None, :, None] * stride_ch
                + offs_d[None, None, :], mask=m, other=0).to(tl.float32)
    sc = tl.load(scale_ptr + page * stride_sb + offs_n[:, None] * stride_sn
                 + offs_h[None, :] * stride_sh + s,
                 mask=nmask[:, None] & (offs_h[None, :] < H), other=0.0)
    y = (q * sc[:, :, None]).to(out_ptr.dtype.element_ty)

    tl.store(out_ptr + i.to(tl.int64) * stride_ob + s * stride_os
             + offs_n[:, None, None] * stride_on + offs_h[None, :, None] * stride_oh
             + offs_d[None, None, :], y, mask=m)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

def _check_views(kv_int8, kv_scales):
    """Validate the two cache views, return (B, N, H, D, Sb_bytes)."""
    if kv_int8.ndim != 5 or kv_int8.shape[1] != 2:
        raise ValueError(f"kv_int8 must be [B,2,N,H,D], got {tuple(kv_int8.shape)}")
    if kv_scales.ndim != 4:
        raise ValueError(f"kv_scales must be [B,N,H,2], got {tuple(kv_scales.shape)}")
    B, _, N, H, D = kv_int8.shape
    if tuple(kv_scales.shape) != (B, N, H, 2):
        raise ValueError(f"kv_scales {tuple(kv_scales.shape)} does not match kv_int8 "
                         f"{tuple(kv_int8.shape)}")
    if kv_int8.stride(4) != 1:
        raise ValueError("kv_int8 must be contiguous in the head dim (stride 4 == 1)")
    if kv_scales.stride(3) != 1:
        raise ValueError("kv_scales must be contiguous in the k/v dim (stride 3 == 1)")
    if D & (D - 1):
        raise ValueError(f"head dim D must be a power of two for now, got {D}")
    sb = kv_int8.stride(0)
    if sb < N * H * 2 * D:
        raise ValueError(f"block stride {sb} smaller than N*H*2*D = {N * H * 2 * D}")
    if torch is not None:
        if kv_int8.dtype != torch.int8:
            raise ValueError(f"kv_int8 must be int8, got {kv_int8.dtype}")
        if kv_scales.dtype != torch.float32:
            raise ValueError(f"kv_scales must be float32, got {kv_scales.dtype}")
    return B, N, H, D, sb


def _cache_strides(kv_int8, kv_scales):
    return (kv_int8.stride(0), kv_int8.stride(1), kv_int8.stride(2), kv_int8.stride(3),
            kv_scales.stride(0), kv_scales.stride(1), kv_scales.stride(2))


def make_views(raw, B, N, H, D, Sb):
    """Zero-copy (kv_int8, kv_scales) views over a raw uint8 buffer of ``B*Sb`` bytes.

    ``raw`` may be a ``torch.uint8`` tensor (device or host).  Provided for tests and
    for the backend that owns the storage; it is *not* required by the kernels.
    """
    if torch is None:
        raise RuntimeError("make_views needs torch")
    if raw.dtype != torch.uint8:
        raise ValueError("raw must be a torch.uint8 buffer")
    if raw.numel() < B * Sb:
        raise ValueError(f"raw holds {raw.numel()} bytes, need {B * Sb}")
    if Sb % 16:
        raise ValueError(f"block stride {Sb} must be a multiple of 16 bytes")
    if Sb < N * H * (2 * D + 8):
        raise ValueError(f"block stride {Sb} smaller than N*H*520 = {N * H * (2 * D + 8)}")
    if not raw.is_contiguous() or raw.storage_offset() != 0:
        raw = raw.contiguous()
    kv_int8 = torch.as_strided(raw.view(torch.int8), (B, 2, N, H, D),
                               (Sb, D, H * 2 * D, 2 * D, 1))
    kv_scales = torch.as_strided(raw.view(torch.float32), (B, N, H, 2),
                                 (Sb // 4, 2 * H, 2, 1), N * H * 2 * D // 4)
    return kv_int8, kv_scales


def state_bytes(N, H, D=256):
    """Bytes per token-head of the cache (2*D int8 + 2 fp32 scales) -> 520 for D=256."""
    return 2 * D + 8


def write(key, value, kv_int8, kv_scales, slot_mapping, num_warps=None):
    """Quantise ``key``/``value`` per (token, head) and store them at ``slot_mapping``.

    key, value   [T, H, D] fp16 (any stride on dim 0/1, last dim contiguous)
    kv_int8      [B, 2, N, H, D] int8 view of the cache
    kv_scales    [B, N, H, 2] float32 view of the cache
    slot_mapping [T] int64 cuda, slot = block*N + n; slot < 0 -> token skipped

    CUDA-graph safe: no allocation, no host sync.
    """
    B, N, H, D, _sb = _check_views(kv_int8, kv_scales)
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    T, kH, kD = key.shape
    if kH != H or kD != D:
        raise ValueError(f"key {tuple(key.shape)} incompatible with cache H={H} D={D}")
    if slot_mapping.numel() != T:
        raise ValueError(f"slot_mapping has {slot_mapping.numel()} entries, expected {T}")
    if torch is not None:
        if key.dtype != torch.float16 or value.dtype != torch.float16:
            raise ValueError("key/value must be float16")
        if key.stride(2) != 1 or value.stride(2) != 1:
            raise ValueError("key/value must be contiguous in the head dim")
    if T == 0:
        return

    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_D = triton.next_power_of_2(D)
    if num_warps is None:
        num_warps = min(8, max(1, (BLOCK_H * BLOCK_D) // 256))
    (scb, sckv, scn, sch, ssb, ssn, ssh) = _cache_strides(kv_int8, kv_scales)

    _kv_write_kernel[(T,)](
        key, value, kv_int8, kv_scales, slot_mapping,
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        scb, sckv, scn, sch, ssb, ssn, ssh,
        N,
        H=H, D=D, BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )


def dequant_gather(kv_int8, kv_scales, page_ids, out, BLOCK_N=None, num_warps=None):
    """Dequantise whole blocks into ``out``.

    page_ids [P] int32/int64 cuda block ids (any order, may repeat)
    out      [P_max, 2, N, H, D] fp16 (P_max >= P); rows ``i >= P`` are left untouched

    ``out[i, s, n, h, :] = fp16(kv_int8[page_ids[i], s, n, h, :] *
                                kv_scales[page_ids[i], n, h, s])``
    """
    B, N, H, D, _sb = _check_views(kv_int8, kv_scales)
    if out.ndim != 5 or out.shape[1] != 2 or out.shape[2:] != (N, H, D):
        raise ValueError(f"out must be [P_max,2,{N},{H},{D}], got {tuple(out.shape)}")
    if out.stride(4) != 1:
        raise ValueError("out must be contiguous in the head dim")
    P = page_ids.numel()
    if P == 0:
        return
    if out.shape[0] < P:
        raise ValueError(f"out has {out.shape[0]} rows, need {P}")
    if torch is not None and out.dtype != torch.float16:
        raise ValueError("out must be float16")

    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_D = triton.next_power_of_2(D)
    if BLOCK_N is None:
        BLOCK_N = max(1, min(triton.next_power_of_2(N), 4096 // (BLOCK_H * BLOCK_D)))
    BLOCK_N = triton.next_power_of_2(BLOCK_N)
    if num_warps is None:
        num_warps = min(8, max(1, (BLOCK_N * BLOCK_H * BLOCK_D) // 1024))
    (scb, sckv, scn, sch, ssb, ssn, ssh) = _cache_strides(kv_int8, kv_scales)

    _dequant_gather_kernel[(P, triton.cdiv(N, BLOCK_N), 2)](
        kv_int8, kv_scales, page_ids, out,
        scb, sckv, scn, sch, ssb, ssn, ssh,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N,
        H=H, D=D, BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# torch oracles (same signatures, used by the tests and by other harnesses)
# ---------------------------------------------------------------------------

def oracle_write(key, value, kv_int8, kv_scales, slot_mapping):
    """Pure-torch reference for :func:`write` (in-place on the cache views)."""
    if torch is None:
        raise RuntimeError("the oracles need torch")
    _B, N, H, D, _sb = _check_views(kv_int8, kv_scales)
    slots = slot_mapping.long()
    idx = torch.nonzero(slots >= 0, as_tuple=True)[0]
    if idx.numel() == 0:
        return
    blk = slots.index_select(0, idx) // N
    n = slots.index_select(0, idx) % N
    for src, s in ((key, 0), (value, 1)):
        x = src.index_select(0, idx).to(torch.float32)          # [t, H, D]
        absmax = x.abs().amax(dim=-1)                           # [t, H]
        nz = absmax > 0
        inv = 127.0 / torch.where(nz, absmax, torch.ones_like(absmax))
        q = torch.round(x * inv.unsqueeze(-1))
        q = torch.clamp(q, -127.0, 127.0)
        q = torch.where(nz.unsqueeze(-1), q, torch.zeros_like(q))
        kv_int8[blk, s, n, :, :] = q.to(torch.int8)
        kv_scales[blk, n, :, s] = torch.where(nz, absmax / 127.0,
                                              torch.zeros_like(absmax))


def oracle_dequant_gather(kv_int8, kv_scales, page_ids, out):
    """Pure-torch reference for :func:`dequant_gather`."""
    if torch is None:
        raise RuntimeError("the oracles need torch")
    _B, N, H, D, _sb = _check_views(kv_int8, kv_scales)
    P = page_ids.numel()
    if P == 0:
        return
    pages = page_ids.long()
    q = kv_int8.index_select(0, pages).to(torch.float32)         # [P,2,N,H,D]
    sc = kv_scales.index_select(0, pages)                        # [P,N,H,2]
    y = q * sc.permute(0, 3, 1, 2).unsqueeze(-1)                 # [P,2,N,H,D]
    out[:P].copy_(y.to(out.dtype))
