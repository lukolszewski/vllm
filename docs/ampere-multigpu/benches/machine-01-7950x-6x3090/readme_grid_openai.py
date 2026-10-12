#!/usr/bin/env python3
"""OpenAI-API benchmark grid: for each size, N sessions send ONE streaming /v1/completions request at once
(max_tokens 128, greedy, cache disabled) and both phases are timed from that single stream: the gap
send -> first token chunk is the prefill, first -> last token chunk is the generation.
usage: readme_grid_openai.py URL LABEL OUT.json [slots=1,5] [sizes=5000,50000,150000,200000,250000]"""
import json, sys, time, threading, urllib.request, urllib.error, subprocess

GEN = 128
WORDS = ("system latency throughput kernel tensor gradient manifold entropy quantize scheduler pipeline embedding attention routing expert cache vector matrix decode prefill context window inference bandwidth compute memory buffer sparse dense token logits softmax").split()

URL = LABEL = OUT = MODEL = None
SLOTS, SIZES = [], []


def prompt(i, size, n_words):
    body = " ".join(WORDS[(i*7 + k*13) % len(WORDS)] for k in range(n_words))
    return f"[grid {LABEL} size {size} stream {i}] Deep context priming block. {body}. Now continue at length:"


def gpu():
    try: return subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10).stdout.strip().replace("\n",";")
    except Exception as e: return str(e)


def model_name():
    d = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/models"), timeout=60).read())
    return d["data"][0]["id"]


def open_stream(p):
    """POST one streaming completion; returns (response, t_send). timeout 7200 s as in the reference client."""
    body = {"model": MODEL, "prompt": p, "max_tokens": GEN, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True, "cache_prompt": False}
    r = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    t_send = time.time()
    try: return urllib.request.urlopen(r, timeout=7200), t_send
    except urllib.error.HTTPError as e:
        try: detail = e.read().decode("utf-8", "replace")[:200]
        except Exception: detail = ""
        raise RuntimeError(f"HTTP {e.code} {e.reason} {detail}")


def stream_req(p, n_words):
    """One streaming request per cell/session. Records t_send, t_first_token (first chunk with non-empty
    choices[0].text), t_last_token, the concatenated text and usage.prompt_tokens/completion_tokens."""
    resp, t_send = open_stream(p)
    text, usage, chunks, t_first, t_last = [], None, 0, None, None
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue          # SSE data line; ignore comments/blank
            data = line[5:].strip()
            if data == "[DONE]": break
            try: c = json.loads(data)
            except Exception: continue
            if c.get("usage"): usage = c["usage"]              # final usage chunk (stream_options.include_usage)
            ch = (c.get("choices") or [{}])[0]
            txt = ch.get("text") or ""
            if txt:
                now = time.time()
                if t_first is None: t_first = now
                t_last = now; chunks += 1
                text.append(txt)
    finally:
        try: resp.close()
        except Exception: pass
    if t_first is None: raise RuntimeError("no tokens streamed")
    d = {"t_send": t_send, "t_first": t_first, "t_last": t_last, "text": "".join(text)}
    if usage is None:                                          # fallback: count chunks / reuse the 1.07 rule
        d["usage_missing"] = True
        d["prompt_tokens"] = int(round(n_words * 1.07))
        d["completion_tokens"] = max(chunks, 1)
    else:
        d["prompt_tokens"] = int(usage.get("prompt_tokens") or 0)
        d["completion_tokens"] = int(usage.get("completion_tokens") or 0)
    if not d["prompt_tokens"] or not d["completion_tokens"]: raise RuntimeError(f"empty usage {usage}")
    return d


def phase(prompts, n_words):
    """Run len(prompts) sessions concurrently in N threads, released together so that min(t_send) is aligned."""
    res = [None]*len(prompts)
    gate = threading.Event()
    def run(i):
        try:
            gate.wait(); res[i] = (stream_req(prompts[i], n_words), 0.0)
        except Exception as e: res[i] = ({"error": str(e)[:200]}, 0.0)
    th = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    t0 = time.time(); [t.start() for t in th]; gate.set(); [t.join() for t in th]; return res, time.time() - t0


rows = []


def save(): json.dump({"client": "readme_grid_openai.py", "label": LABEL, "url": URL, "gen_tokens": GEN, "rows": rows}, open(OUT, "w"), indent=1)


def main():
    global URL, LABEL, OUT, SLOTS, SIZES, MODEL
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help") or len(argv) < 3: print(__doc__.strip()); sys.exit(0)
    URL, LABEL, OUT = argv[0], argv[1], argv[2]
    SLOTS = [int(x) for x in (argv[3] if len(argv) > 3 else "1,5").split(",")]
    SIZES = [int(x) for x in (argv[4] if len(argv) > 4 else "5000,50000,150000,200000,250000").split(",")]
    try: MODEL = model_name()
    except Exception as e: print(f"{LABEL} cannot get model name from {URL}/v1/models: {str(e)[:200]}", flush=True); sys.exit(1)
    print(f"{time.strftime('%H:%M:%S')} {LABEL} model={MODEL} slots={SLOTS} sizes={SIZES}", flush=True)
    for n in SLOTS:
        for size in SIZES:
            n_words = int(size / 1.07)   # measured 1.07 tokens per word for this generator
            prompts = [prompt(10 + i, size, n_words) for i in range(n)]
            row = {"slots": n, "size_tokens": size, "gpu_before": gpu(), "t_start": time.strftime("%H:%M:%S")}
            res, _ = phase(prompts, n_words)
            errs = [r[0]["error"] for r in res if "error" in r[0]]
            if errs:
                row["pp_error"] = errs; row["tg_error"] = errs
                rows.append(row); save(); print(f"{time.strftime('%H:%M:%S')} {LABEL} slots={n} size={size} PREFILL ERROR {errs[0]}", flush=True); continue
            pp = [r[0] for r in res]
            pn = [r["prompt_tokens"] for r in pp]; gen_n = [r["completion_tokens"] for r in pp]
            ttft = [r["t_first"] - r["t_send"] for r in pp]
            pps = [r["prompt_tokens"] / max(r["t_first"] - r["t_send"], 1e-6) for r in pp]
            tgs = [(r["completion_tokens"] - 1) / max(r["t_last"] - r["t_first"], 1e-6) for r in pp]
            wall_pp = max(r["t_first"] for r in pp) - min(r["t_send"] for r in pp)
            wall_tg = max(r["t_last"] for r in pp) - min(r["t_first"] for r in pp)
            row.update(pp_agg=round(sum(pn)/max(wall_pp,1e-6), 1), pp_slot_min=round(min(pps),1), pp_slot_mean=round(sum(pps)/n,1), pp_slot_max=round(max(pps),1), prompt_n=pn, ttft_min=round(min(ttft),1), ttft_max=round(max(ttft),1), pp_wall=round(wall_pp,1))
            row.update(tg_agg=round(sum(gen_n)/max(wall_tg,1e-6), 2), tg_slot_min=round(min(tgs),2), tg_slot_max=round(max(tgs),2), tg_slot_mean=round(sum(tgs)/n,2), prompt_n_round2=pn, tg_wall=round(wall_tg,1), texts=[r.get("text","")[:80] for r in pp])
            if any(r.get("usage_missing") for r in pp): row["usage_missing"] = True
            row["gpu_after"] = gpu(); rows.append(row); save()
            print(f"{time.strftime('%H:%M:%S')} {LABEL} slots={n} size={size} pn={pn[0]} pp_agg={row.get('pp_agg')} t/s (slot {row.get('pp_slot_mean')}) ttft={row.get('ttft_max')}s | tg_agg={row.get('tg_agg')} t/s (slot {row.get('tg_slot_mean')}, min {row.get('tg_slot_min')}) pn2={row.get('prompt_n_round2')} {row.get('tg_error','')} | text={row.get('texts',[''])[0][:50]!r}", flush=True)
    print("GRID DONE", LABEL, flush=True)


if __name__ == "__main__": main()
