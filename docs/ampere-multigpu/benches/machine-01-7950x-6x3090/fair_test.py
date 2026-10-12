#!/usr/bin/env python3
"""fair_test.py URL OUT.json [long_tokens=230000] [short_tokens=3000] [stagger_s=10]
Prefill fairness probe (vLLM --max-num-batched-tokens 16384 --long-prefill-token-threshold 8192 vs the default 2048 budget).
Every prompt uses a fresh seed (no prefix-cache hits). TTFT = time to the first streamed chunk. max_tokens 8, temperature 0.
 phase 0  short solo               -> short baseline
 phase 1  long solo                -> long baseline (prefill speed under the new budget; W6 cold 230k = 42.8 s)
 phase 2  long A, short B at +stagger -> the case that matters: short must NOT wait for the whole long prefill
 phase 3  long C, long D at +stagger  -> long-behind-long (both share the budget; no max-long knob in vLLM V1)
"""
import json, sys, time, threading, urllib.request
URL, OUT = sys.argv[1], sys.argv[2]
LT = int(sys.argv[3]) if len(sys.argv) > 3 else 230000
ST = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
STAG = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
TPW = 1.07
WORDS = ("system latency throughput kernel tensor gradient manifold entropy quantize scheduler pipeline embedding attention routing expert cache vector matrix decode prefill context window inference bandwidth compute memory buffer sparse dense token logits softmax").split()
def prompt(seed, tokens):
    n = int(tokens / TPW); rng = (seed * 7919 + 17)
    words = [WORDS[(rng * (k + 1) + k * k) % len(WORDS)] for k in range(n)]
    return f"[fair-test session {seed}] " + " ".join(words) + f"\n\nSession {seed}: summarise the text above in one sentence."
MODEL = json.load(urllib.request.urlopen(URL + "/v1/models"))["data"][0]["id"]
rows = []; lock = threading.Lock(); T0 = time.time()
def req(seed, tokens, tag):
    body = {"model": MODEL, "prompt": prompt(seed, tokens), "max_tokens": 8, "temperature": 0, "stream": True}
    r = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for line in resp:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                if first is None: first = time.time()
                n += 1
    return {"tag": tag, "seed": seed, "tokens": tokens, "sent_at": round(t0 - T0, 1), "ttft": round((first or time.time()) - t0, 2), "wall": round(time.time() - t0, 2), "chunks": n}
def run(seed, tokens, tag):
    try: r = req(seed, tokens, tag)
    except Exception as e: r = {"tag": tag, "seed": seed, "tokens": tokens, "error": str(e)[:200]}
    with lock: rows.append(r); print(json.dumps(r), flush=True)
def pair(seed_a, tok_a, tag_a, seed_b, tok_b, tag_b):
    ta = threading.Thread(target=run, args=(seed_a, tok_a, tag_a)); ta.start()
    time.sleep(STAG)
    tb = threading.Thread(target=run, args=(seed_b, tok_b, tag_b)); tb.start()
    ta.join(); tb.join()
print(f"phase 0: short solo ({ST} tokens)", flush=True); run(100, ST, "short-solo")
print(f"phase 1: long solo ({LT} tokens)", flush=True); run(101, LT, "long-solo")
time.sleep(3)
print(f"phase 2: long A ({LT}) then short B ({ST}) at +{STAG}s  <- the case that matters", flush=True)
pair(102, LT, "long-A-with-short-behind", 103, ST, "short-B-behind-long")
time.sleep(3)
print(f"phase 3: long C ({LT}) then long D ({LT}) at +{STAG}s", flush=True)
pair(104, LT, "long-C-with-long-behind", 105, LT, "long-D-behind-long")
json.dump({"url": URL, "long_tokens": LT, "short_tokens": ST, "stagger_s": STAG, "rows": rows}, open(OUT, "w"), indent=1)
print("FAIR TEST DONE", flush=True)
