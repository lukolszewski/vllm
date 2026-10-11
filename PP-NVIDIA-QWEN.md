# pp-nvidia-qwen: pipeline-parallel Qwen3.8-Flash-Next (Qwen4Exp) on NVIDIA GPUs

This branch is upstream vLLM at commit `7d0b4e57a` (nightly 0.31.1rc1.dev253, 2026-10-09)
plus commits that make Qwen4Exp models (Qwen3.8-Flash-Next) run with
`--pipeline-parallel-size > 1`, load GPTQ checkpoints that quantize only the
Linear layers, use `--kv-cache-dtype fp8` on Ampere GPUs, and serve the PLE
n-gram table from SSD instead of pinned host RAM:

1. **Allow PP when the PLE layer stays on stage 0** — the model-config verifier and
   both `model_state` modules refused PP outright. They now require
   `VLLM_PP_LAYER_PARTITION` and check that its first entry keeps the PLE n-gram
   layer on rank 0 (the only rank that receives raw `input_ids`).
2. **GPTQ loading** — unquantized PLE table shards and indexer projection under a
   GPTQ-family quant config (`ngram_embedding.py`, `qsa.py`, `nvidia/model.py`).
   Needed at PP=1 too.
3. **KV-cache allocation under PP** — `get_kv_cache_config_from_groups` emitted
   tensors for layers living on other ranks, which killed the non-first workers
   with `StopIteration` (`vllm/v1/core/kv_cache_utils.py`).
4. **fp8 (e4m3) KV cache on sm < 89** — Triton 3.7 has no `fp8e4nv` conversion on
   Ampere, so `--kv-cache-dtype fp8` crashed every QSA kernel on the RTX 3090 with
   `ValueError("type fp8e4nv not supported in this architecture")`. New
   `nvidia/ops/fp8_ampere.py` converts e4m3fn <-> f32/f16 in software (RNE, saturate
   to 448, NaN preserved); on sm < 89 (or `VLLM_QSA_EMULATE_E4M3=1`) the caches are
   handed to Triton as uint8 views and the pointer element type picks the branch at
   compile time. The byte format is unchanged, sm >= 89 is untouched. Proven bitwise
   against the native cast on an RTX 5060 Ti (helpers, `qsa_prepare` cache bytes,
   `qsa_sparse_paged_attention` outputs) and end-to-end on the six 3090s. It is not
   free: the software conversion costs about 3.3x on prefill (2.3k vs 7.8k tok/s at
   50k tokens, 250k TTFT 104 s vs 33 s) while decode is unchanged; what it buys is KV
   capacity, 7.44x vs 4.03x of 262k on the six 3090s (table below).

5. **PLE table on SSD (`host_file_gather`)** — port of upstream PR #58815. With
   `--engram-config '{"host_file_gather": true}'` the 102 GB BF16 PLE n-gram table is
   no longer pinned in host RAM: the rows each step needs are read from the
   safetensors shards with `preadv` (page cache does the caching) and staged into a
   fixed device buffer. Under PP only the rank holding the PLE module gets a
   non-empty `file_gather_modules` list; the staging hook is a no-op elsewhere.
   Host RAM "available" with the server idle went from 15 GiB to 141 GiB on the
   test box, load time from ~400 s to 240 s, decode and prefill unchanged (see
   "PLE on SSD / fp8 KV" below).

Only Python files change; the branch runs on the stock
`vllm/vllm-openai:cu129-nightly-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe` image by
bind-mounting the changed files over `/usr/local/lib/python3.12/dist-packages/vllm/`
(or by installing this tree). That nightly image's torchcodec links a missing
`libnvrtc.so.13` and breaks `vllm serve` at import; `pip uninstall -y torchcodec`
in a derived image fixes it.

## Tested configuration

Six RTX 3090 (24 GB), AMD 7950X, 187 GB RAM, checkpoint
[btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit](https://huggingface.co/btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit)
(GPTQ 4-bit g32, Linear layers only; PLE table in BF16, host-offloaded):

```
VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8 \
vllm serve /path/to/Qwen3.8-Flash-Next-GPTQ-4bit \
  --served-model-name qwen3.8-flash-next --dtype bfloat16 \
  --pipeline-parallel-size 6 --tensor-parallel-size 1 \
  --distributed-executor-backend mp \
  --max-model-len 262144 --max-num-seqs 4 --gpu-memory-utilization 0.92 \
  --engram-config '{"host_file_gather": true}' \
  --trust-remote-code
```

The partition must sum to the layer count (48) and its first entry must be >= 2 so
that the PLE layer (index 1) is on stage 0. With `host_file_gather` the 102 GB BF16
PLE table stays on disk (container RSS ~28 GB); with `{"cpu_offload": true}` instead
it is pinned in host RAM (RSS ~143 GB). The KV cache at this setting holds about
1.06M tokens in bf16 across the six GPUs (4.03 x 262k). Adding `--kv-cache-dtype fp8
--max-num-seqs 5` raises that to 1,950,576 tokens (7.44 x 262k) with decode speed
unchanged (~55 tok/s), needle passing at 50k/200k and five concurrent 250k-token
sessions resident, at the price of ~3.3x slower prefill on these Ampere cards (see
"PLE on SSD / fp8 KV"). bf16 is the default here; fp8 is the capacity option.

Measured on that box (1 session, 250k-token prompt, 128 greedy tokens): ~9,000 tok/s
prefill and ~58 tok/s decode; needle-in-haystack passes at 50k and 200k. Full grid,
server logs, raw JSON and the comparison with llama.cpp are in
https://github.com/lukolszewski/llama.cpp-multigpu/pull/6.

Known limits: five concurrent greedy requests are not byte-identical to their solo
runs (batch-composition nondeterminism, outputs stay coherent); model load to
healthy takes 6-8 minutes; the AMD `model_state` change is untested.

## Relation to alanoo81/flashnext-v620-pp3

That fork ran the model at PP=3 on Radeon V620s on a September 2026 vLLM. Of its six
PP patches only the model_state guard removal is still relevant here (this branch keeps
a partition check instead of deleting the guard). The gpu_worker / gpu_model_runner
guards, the `ple_offload` worker and its import fix are gone from upstream (PLE host
offload is now `EngramConfig.cpu_offload`), and the HC-mixer weight skipping on
non-last ranks is upstreamed. The config-verifier guard, the GPTQ loading fixes and
the KV-cache projection fix in this branch did not exist as problems in that tree.

## PLE on SSD / fp8 KV

Measured 2026-10-11 on the box above, one window each, same image and checkpoint,
`--kv-cache-dtype fp8 --max-num-seqs 5 --max-model-len 262144`, prefix caching on.
"Pinned" = `{"cpu_offload": true}`, "SSD" = `{"host_file_gather": true}` (PLE
shards on a PCIe 4 NVMe, 57k random 4 KiB reads/s).

| | pinned PLE | PLE on SSD |
|---|---|---|
| load to `/health` | 406 s | 240 s |
| host `free -g` idle: used / shared / buff-cache / available (GiB) | 171 / 128 / 145 / 15 | 45 / 0 / 111 / 141 |
| container RSS idle | ~143 GB | 28 GB |
| swap used during load | ~20 GB | none added (20 GB pre-existing) |
| GPU KV cache (fp8) | 1,835,008 tokens | 1,950,576 tokens |
| 1-session prefill 5k / 50k tokens | 1,314 / 2,227 tok/s | 1,329 / 2,276 tok/s |
| 1-session decode at 5k / 50k / 250k | 54.7 / 55.3 / 55.6 tok/s | 55.5 / 55.3 / 56.6 tok/s |
| 5 x 250k concurrent | all resident | all resident |
| needle 50k / 200k | PASS / PASS | PASS / PASS |

The 102 GB table becomes page cache instead of anonymous memory, so it is evicted
under pressure and re-read on demand. In practice the page cache held most of the
table across restarts: a second window measured prefill and decode at 50k and 250k
tokens on a "cold" start (first start after hours of other serving) and again after
`cat ple-*.safetensors > /dev/null` (68 s, i.e. mostly cached already), and the two
differed by at most 1.5 % in either direction, so no pre-warm step is needed.

The same window ran the fp8-vs-bf16 KV comparison, both with the PLE on SSD, one
session per row, prefix caching on, distinct prompts per run (no cache hits):

| | fp8 KV (`--max-num-seqs 5`) | bf16 KV (`--max-num-seqs 4`) |
|---|---|---|
| load to `/health` | 270 s | 240 s |
| GPU KV cache | 1,950,576 tokens (7.44 x 262k) | 1,057,462 tokens (4.03 x 262k) |
| prefill 50k / 250k tokens | 2,260-2,295 / 2,377-2,394 tok/s | 7,831 / 7,550 tok/s |
| TTFT 50k / 250k | 22 / 104 s | 6.3 / 33 s |
| decode 50k / 250k (1 session) | 54.9-56.2 / 56.5-57.0 tok/s | 55.1 / 56.7 tok/s |
| 250k x 128 tokens, all seats busy | 5 sessions: 25.7 tok/s each, 78.6 aggregate | 4 sessions: 38.6 tok/s each, 83.0 aggregate |
| host `free -g` available idle | 141 GiB | 142 GiB |

bf16 with the PLE on SSD prefills as fast as the original pinned-PLE bf16 runs
(7,630 tok/s at 50k), so `host_file_gather` itself costs nothing measurable; the
3.3x prefill gap is the software e4m3 path of item 4 (the only difference between the
two columns). A native-speed e4m3 load path in the QSA kernels would close it. Raw
logs and JSON for both windows are in the `vllm-pp6` run directories referenced from
https://github.com/lukolszewski/llama.cpp-multigpu/pull/7.

## Amending

The code commits are independent; `git diff 7d0b4e57a --name-only -- vllm/` lists the
overlay files (thirteen, incl. the new `nvidia/ops/fp8_ampere.py` and
`vllm/config/engram.py`) that can be bind-mounted
over the stock nightly image; a bind mount creates the mount point for a file the image
does not have.
