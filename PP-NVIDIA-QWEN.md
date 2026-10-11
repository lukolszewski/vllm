# pp-nvidia-qwen: pipeline-parallel Qwen3.8-Flash-Next (Qwen4Exp) on NVIDIA GPUs

This branch is upstream vLLM at commit `7d0b4e57a` (nightly 0.31.1rc1.dev253, 2026-10-09)
plus three commits that make Qwen4Exp models (Qwen3.8-Flash-Next) run with
`--pipeline-parallel-size > 1`, and load GPTQ checkpoints that quantize only the
Linear layers:

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

Only Python files change; the branch runs on the stock
`vllm/vllm-openai:cu129-nightly-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe` image by
bind-mounting the seven files over `/usr/local/lib/python3.12/dist-packages/vllm/`
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
  --max-model-len 262144 --max-num-seqs 5 --gpu-memory-utilization 0.92 \
  --engram-config '{"cpu_offload": true}' \
  --no-enable-prefix-caching --trust-remote-code
```

The partition must sum to the layer count (48) and its first entry must be >= 2 so
that the PLE layer (index 1) is on stage 0. With `cpu_offload` the 102 GB BF16 PLE
table lives in host RAM (container RSS ~143 GB). The KV cache at this setting holds
about 1.0M tokens in bf16 across the six GPUs.

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

## Amending

The three code commits are independent; `git diff 7d0b4e57a -- vllm/` reproduces the
seven-file overlay that can be bind-mounted over the stock nightly image.
