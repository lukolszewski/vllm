# Benchmarks: Qwen3.8-Flash-Next on 6 x RTX 3090 with the ampere-multigpu branch

All measurements on machine-01 (6 x RTX 3090 24 GB, AMD 7950X, 187 GB DDR5-3600, PCIe 4 NVMe; GPU0/1 on PCIe 4 x8 with
NVLink, GPU2-5 behind Thunderbolt), 2026-10-11 and 2026-10-12, image `vllm/vllm-openai:cu129-nightly-7d0b4e57a…` with
this branch's files mounted over it, checkpoint btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit, `VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8`,
`--max-model-len 262144 --gpu-memory-utilization 0.92`, prefix caching on, distinct prompts per cell (no cache hits
unless stated). Raw JSON and the client scripts are in [benches/machine-01-7950x-6x3090/](benches/machine-01-7950x-6x3090/);
file names carry the date and the configuration.

Metric definitions (client side, OpenAI API, streaming): prefill tok/s = prompt tokens / time to first token (TTFT);
generation tok/s per session = 128 generated tokens / (wall - TTFT); "sum" = sum of the per-session generation rates.
"N sessions decoding together" = round 2 of `decode_round2.py`: each session re-sends its cached 250k prefix, so all N
generate at once with no prefill in flight. Needle = a code word placed in the prompt must be returned verbatim.

## 1. KV-cache modes (the configuration choice), PLE on SSD

| | bf16 KV, `--max-num-seqs 4` | **int8 KV, `--max-num-seqs 7`** | fp8 KV (software e4m3), `--max-num-seqs 5` |
|---|---|---|---|
| load to `/health` | 240 s | 225 s | 270 s |
| GPU KV cache | 1,057,462 tokens (4.03 x 262k) | **1,922,389 (7.33 x)** | 1,950,576 (7.44 x) |
| prefill 50k / 250k, 1 session | 7,831 / 7,550 t/s | **6,506 / 6,574** | 2,260-2,295 / 2,377-2,394 |
| TTFT 50k / 250k | 6.3 / 33 s | **7.6 / 37.8 s** | 22 / 104 s |
| generation 50k / 250k, 1 session | 55.1 / 56.7 t/s | **56.5 / 57.6** | 54.9-56.2 / 56.5-57.0 |
| 250k, N sessions decoding together (per session / sum) | 4: 38.6 / 154; 3: 41.9 / 126; 2: 50.3 / 101 | **7: 29.4 / 206; 5: 37.9 / 190; 3: 44.2 / 133** | 5: 25.7-29.1 / 128-145; 4: 32.5 / 130; 3: 38.9 / 117; 2: 46.6 / 93 |
| wall for 128 tokens, N sessions at 250k (max over sessions, incl. 1-5 s prefix re-read) | 4: 6.2 s; 3: 5.2 s | **7: 8.0 s; 5: 6.1 s; 3: 4.7 s** | 5: 7.9-8.1 s; 4: 7.2 s; 3: 5.6 s |
| needle 50k / 200k | pass / pass | pass / pass | pass / pass |
| host RAM available, idle | 142 GiB | 141 GiB | 141 GiB |

int8 (`--kv-cache-dtype int8_per_token_head`, int8 values plus one fp32 absmax scale per token and kv head, 264 bytes
per token-head versus 512 bf16 / 256 fp8) keeps 83-87 % of the bf16 prefill with 1.8x its capacity, generates as fast
solo and faster than fp8 under load. The software e4m3 path is bit-exact with the native cast (verified on an RTX 5060
Ti) but costs 3.3x on prefill and ~25 % on multi-session generation, so it is the capacity-only option.

Per-session generation declines with the session count under PP (57.6 -> 44.2 -> 37.9 -> 29.4 at 1/3/5/7) while the sum
rises (58 -> 133 -> 190 -> 206); the spread between sessions inside one run (22-40 t/s at 7) is the pipeline stagger,
not noise.

## 2. Against llama.cpp-multigpu on the same machine

Same model (llama.cpp: UD-Q4_K_XL GGUF with q8_0 KV; vLLM: GPTQ-INT4 with int8 KV), same six cards. llama.cpp numbers
from [llama.cpp-multigpu](https://github.com/lukolszewski/llama.cpp-multigpu) (2026-10-05 grid, server timings) and its
production launcher.

| | llama.cpp-multigpu | vLLM, this branch (int8 KV) |
|---|---|---|
| PLE table | 26.8 GiB IQ4_NL GGUF, mmap from SSD | 102 GB bf16 safetensors, `host_file_gather` from SSD |
| start to healthy | 94-101 s warm, 2-3 min cold | 225 s |
| 262k contexts resident | 5 | 7.33 |
| prefill 50k / 250k, 1 session | 2,127 / 2,111 t/s | 6,506 / 6,574 t/s |
| TTFT 250k | ~118 s | 37.8 s |
| generation 50k / 250k, 1 session | 41.9 / 33.7 t/s | 56.5 / 57.6 t/s |
| 5 sessions decoding together (per session / sum) | ~30 / ~150 (at 200k) | 37.9 / 190 (at 250k) |
| 7 sessions | n/a | 29.4 / 206 |
| 2 sessions | 41.7 / 83 (at 4k) | 50.3 / 101 (bf16 KV run) |
| host RAM available, idle | 154 GiB | 141 GiB |

vLLM wins prefill 3x and single-session generation 1.7x and holds two more full contexts; llama.cpp's per-user rate
barely moves with concurrency (34 -> 30 at 5) while vLLM's halves (57 -> 29 at 7), so at five users the sums are close.
Prefill on both engines is first-come-first-served per request: on vLLM seven 230k prompts sent together got their
first token after 42 / 70 / 101 / 129 / 161 / 196 / 225 s (7.1k t/s aggregate, i.e. no throughput lost, only waiting).

## 3. PLE table on SSD versus pinned in host RAM (fp8 KV, 5 seats, 2026-10-11)

| | pinned (`{"cpu_offload": true}`) | on SSD (`{"host_file_gather": true}`) |
|---|---|---|
| load to `/health` | 406 s | 240 s |
| host `free -g` idle: used / available | 171 / 15 GiB (+20 GB swap) | 45 / 141 GiB |
| container RSS idle | ~143 GB | 28 GB |
| GPU KV cache (fp8) | 1,835,008 tokens | 1,950,576 tokens |
| prefill 5k / 50k, 1 session | 1,314 / 2,227 t/s | 1,329 / 2,276 t/s |
| generation 5k / 50k / 250k | 54.7 / 55.3 / 55.6 t/s | 55.5 / 55.3 / 56.6 t/s |
| needle 50k / 200k | pass / pass | pass / pass |

bf16 KV with the table on SSD prefills at 7,831 t/s (50k) against 8,365 pinned (separate run, prefix cache off): the
placement costs nothing measurable. A cold start (page cache empty after hours of other serving) and a warm one differed
by at most 1.5 % in prefill and generation, so no pre-warming is needed; the table simply becomes page cache.

## 4. Host-RAM KV tier (`--kv-offloading-size 36 --kv-offloading-backend native`, int8 KV, 7 seats, 2026-10-12)

Nine distinct 230k prompts (2.07M tokens > the 1.92M-token GPU KV cache) were sent, seven concurrently and two more one
at a time, so the two oldest contexts had to leave the GPUs; then the oldest two were re-sent, plus a still-resident
one and a fresh one:

| request | TTFT |
|---|---|
| oldest context, evicted from GPU, re-sent | 3.15 s |
| second-oldest, evicted, re-sent | 2.79 s |
| context still on the GPU, re-sent | 1.89 s |
| fresh 230k prompt (full prefill) | 42.8 s |

Server metrics during the returns: `kv_offload_load_bytes` 6.15 GB in 12 loads (2 returns x 6 pipeline ranks), 0.8 s
per rank-load. Note that vLLM's "GPU KV cache usage" counts only blocks of running requests and never shows eviction;
the load metrics do. Size the tier at ~4.5 GiB per 262k context in int8.

## 5. NVFP4 checkpoint (RedHatAI/Qwen3.8-Flash-Next-NVFP4, experts-only W4A4, 2026-10-12)

Loads with no code change (vLLM picks the Marlin NvFp4 MoE backend and warns that FP4 is emulated as weight-only on
this GPU). Same grid as the int8 column of §1, int8 KV, 7 seats:

| | GPTQ-INT4 | NVFP4 |
|---|---|---|
| load | 225 s | 241 s |
| GPU KV cache | 1,922,389 tokens | 1,987,220 tokens |
| prefill 50k / 250k | 6,506 / 6,574 t/s | 5,801 / 5,750 t/s |
| TTFT 250k | 37.8 s | 43.2 s |
| generation, 1 session | 56.5-57.6 t/s | 50.6-51.3 t/s |
| 7 / 5 / 3 sessions, sum | 206 / 190 / 133 t/s | 200 / 149 / 118 t/s |
| needle | pass / pass | pass / pass |

Both formats run weight-only W4A16 through Marlin on Ampere, and NVFP4 moves slightly more bytes per weight, so the
claimed FP4 speed-ups do not apply without FP4 tensor cores. GPTQ-INT4 stays the recommendation.

## 6. Things that did not work

- `--max-num-batched-tokens 16384` (to let a short prompt in beside a long prefill): the larger profile step costs ~1 GiB
  of activation memory per card and the GPU KV cache fell to 969,650 tokens (3.7 x 262k). `8192` kept the KV (1,840,645)
  but the QSA indexer's prefill selection ran out of memory on the first 230k prompt (`qsa_select_paged_prefill`,
  512 MiB allocation with 433 MiB free). Keep the default 2048 on 24 GB cards.
- fp8 KV on Ampere via the unfused `reshape_and_cache` path: not implemented; the fused prepare kernel is the default
  and is the only store path for int8 as well.

## 7. Reproducing

`readme_grid_openai.py URL LABEL OUT.json SESSIONS SIZES` (prefill/generation grid), `openai_check.py URL --conc 0`
(needle), `decode_round2.py URL LABEL OUT.json N 250000 128` (N sessions decoding together), `tier_test.py URL OUT.json
230000 7` (RAM tier), all in [benches/machine-01-7950x-6x3090/](benches/machine-01-7950x-6x3090/). Prompts are
synthetic word streams with a per-session seed, so every run of a script hits no prefix cache from another run.
