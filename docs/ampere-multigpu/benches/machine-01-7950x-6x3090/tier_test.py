#!/usr/bin/env python3
"""W5: does the vLLM host-RAM KV tier (--kv-offloading-size) give a returning user its 250k context back without a re-prefill?
usage: tier_test.py URL OUT.json [prompt_tokens=250000] [seats=7]
Phase 1 (cold): prompts 0..seats-1 concurrently (fills the GPU KV), then prompt `seats` alone (LRU must evict one finished
context from the GPU; with the tier on it goes to host RAM). Phase 2 (returns): re-send prompt 0 (expected: evicted from GPU ->
served from the RAM tier), prompt `seats` (still on GPU), prompt 1 (GPU or RAM), and a fresh prompt (cold reference).
TTFT of each is the measurement: RAM-tier hit = seconds, re-prefill = ~38 s at 250k. temperature 0, max_tokens 8, streaming."""
import json, sys, time, threading, urllib.request
URL, OUT = sys.argv[1], sys.argv[2]
PT = int(sys.argv[3]) if len(sys.argv) > 3 else 230000   # 250k overflowed max_model_len for one seed in W5 (HTTP 400)
SEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 7
TPW = 1.07
WORDS = ("system latency throughput kernel tensor gradient manifold entropy quantize scheduler pipeline embedding attention routing expert cache vector matrix decode prefill context window inference bandwidth compute memory buffer sparse dense token logits softmax").split()
def prompt(i):
    n = int(PT / TPW); rng = (i * 7919 + 17)
    words = [WORDS[(rng * (k + 1) + k * k) % len(WORDS)] for k in range(n)]
    return f"[tier-test session {i}] " + " ".join(words) + f"\n\nSession {i}: summarise the text above in one sentence."
MODEL = json.load(urllib.request.urlopen(URL + "/v1/models"))["data"][0]["id"]
def req(i, tag):
    body = {"model": MODEL, "prompt": prompt(i), "max_tokens": 8, "temperature": 0, "stream": True}
    r = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for line in resp:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                if first is None: first = time.time()
                n += 1
    return {"tag": tag, "prompt": i, "ttft": round((first or time.time()) - t0, 2), "wall": round(time.time() - t0, 2), "chunks": n}
rows = []; lock = threading.Lock()
def run(i, tag):
    try: r = req(i, tag)
    except Exception as e: r = {"tag": tag, "prompt": i, "error": str(e)[:200]}
    with lock: rows.append(r); print(json.dumps(r), flush=True)
print(f"phase 1: {SEATS} cold prompts x {PT} tokens concurrently", flush=True)
ths = [threading.Thread(target=run, args=(i, "cold-concurrent")) for i in range(SEATS)]
[t.start() for t in ths]; [t.join() for t in ths]
print(f"phase 1b: prompts {SEATS} and {SEATS+1} cold, one at a time (9 x {PT} > GPU KV 1.92M tokens -> the LRU contexts must leave the GPU)", flush=True)
run(SEATS, "cold-evictor-1"); run(SEATS + 1, "cold-evictor-2")
time.sleep(5)
print("phase 2: returns (prompt 0 = oldest, must come from the RAM tier if the tier works)", flush=True)
run(0, "return-prompt0-expected-evicted"); run(SEATS + 1, "return-evictor2-on-gpu"); run(1, "return-prompt1-expected-evicted"); run(SEATS + 2, "cold-reference")
json.dump({"url": URL, "prompt_tokens": PT, "seats": SEATS, "rows": rows}, open(OUT, "w"), indent=1)
print("TIER TEST DONE", flush=True)
