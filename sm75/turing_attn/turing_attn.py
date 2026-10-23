"""Python entry for the Turing paged-attention op (see turing_attn.cu).

run(q, kv_cache, kv_indptr, kv_indices, kv_last_page_len, qo_indptr, sm_scale, k_scale, v_scale,
    out=None, max_q_len=None, nch=None, workspace=None)
Same semantics as flashinfer BatchPrefillWithPagedKVCacheWrapper("NHD") planned causal=True.
max_q_len: pass it (static in vLLM: 1 for decode, num_spec+1 for MTP) to avoid a host sync.
nch: KV chunks per (request, kv head); default env TURING_ATTN_NCH or 136 // (B*HK).
"""
import os
import torch

_mod = None
_ws = {}  # device index -> persistent fp32 split-KV workspace (graph-capture safe once allocated)
_WS_FLOATS = int(os.environ.get("TURING_ATTN_WS_MB", "32")) * (1 << 20) // 4


def _load():
    global _mod
    if _mod is None:
        try:
            import turing_attn_C as m  # prebuilt (image)
        except ImportError:
            from torch.utils.cpp_extension import load
            here = os.path.dirname(os.path.abspath(__file__))
            m = load(
                name="turing_attn_C",
                sources=[os.path.join(here, "turing_attn.cu")],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_75,code=sm_75", "-lineinfo"],
                build_directory=os.environ.get("TURING_ATTN_BUILD") or None,
                verbose=bool(os.environ.get("TURING_ATTN_VERBOSE")),
            )
        _mod = m
    return _mod


def default_nch(batch: int, hk: int) -> int:
    env = os.environ.get("TURING_ATTN_NCH")
    if env:
        return max(1, int(env))
    return max(1, 136 // max(1, batch * hk))


def run(q, kv_cache, kv_indptr, kv_indices, kv_last_page_len, qo_indptr, sm_scale, k_scale=1.0, v_scale=1.0,
        out=None, max_q_len=None, nch=None, workspace=None, min_tper=None):
    m = _load()
    if min_tper is None:
        min_tper = int(os.environ.get("TURING_ATTN_MIN_TILES", "1"))
    if max_q_len is None:
        max_q_len = int((qo_indptr[1:] - qo_indptr[:-1]).max().item())
    if nch is None:
        nch = default_nch(qo_indptr.numel() - 1, kv_cache.shape[3])
    if workspace is None:
        di = q.device.index
        workspace = _ws.get(di)
        if workspace is None:
            workspace = _ws[di] = torch.empty(_WS_FLOATS, dtype=torch.float32, device=q.device)
    if torch.is_tensor(k_scale):
        k_scale = float(k_scale)
    if torch.is_tensor(v_scale):
        v_scale = float(v_scale)
    return m.run(q, kv_cache, kv_indptr, kv_indices, kv_last_page_len, qo_indptr, float(sm_scale), float(k_scale),
                 float(v_scale), int(max_q_len), int(nch), out, workspace, int(min_tper))
