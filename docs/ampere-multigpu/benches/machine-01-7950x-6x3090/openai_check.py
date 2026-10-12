#!/usr/bin/env python3
"""Correctness checks against an OpenAI-compatible server (non-streaming, temperature 0, max_tokens 64 / 512 for the needle).
usage: openai_check.py URL [--sizes 50000,200000] [--conc 5] [--conc-size 20000]   (--conc 0 = needle checks only)
(a) needle: for each size build a filler of ~size tokens with the same WORDS generator used by the grid
    clients, hide a random 8-letter code word at ~40% depth and restart the sentence at the end -> PASS if the word is completed.
(b) concurrency equality: run CONC distinct prompts of ~CONC_SIZE tokens solo (sequentially), then all of
    them concurrently; PASS only if every concurrent completion is byte-equal to its solo completion.
Prints PASS/FAIL per check, a JSON summary line at the end and writes check-<ts>.json in cwd. Exit 0 only if all pass."""
import json, random, string, sys, threading, time, urllib.request, urllib.error

GEN = 64            # concurrency check
NEEDLE_GEN = 32     # needle: the prompt repeats the sentence up to the opening quote, so the model copies the word; a bare "Answer:" question made it emit EOS at once on llama.cpp
WORDS = ("system latency throughput kernel tensor gradient manifold entropy quantize scheduler pipeline embedding attention routing expert cache vector matrix decode prefill context window inference bandwidth compute memory buffer sparse dense token logits softmax").split()

URL, MODEL = None, None
SIZES, CONC, CONC_SIZE = [50000, 200000], 5, 20000


def filler(sid, n_words):
    """Same word-sequence generator as readme_grid.py / readme_grid_openai.py; sid shifts the phase."""
    return [WORDS[(sid*7 + k*13) % len(WORDS)] for k in range(n_words)]


def rand_word():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(8))


def needle_prompt(size, sid, code):
    """Filler of ~size tokens with the code sentence inserted at ~40% depth."""
    words = filler(sid, int(size / 1.07))
    pos = int(len(words) * 0.4)
    sent = f'The secret code word for this document is "{code}".'
    body = " ".join(words[:pos]) + " " + sent + " " + " ".join(words[pos:])
    return f"[check size {size} stream {sid}] Deep context priming block. {body}\n\nAs stated above, the secret code word for this document is \""


def conc_prompt(size, sid):
    """~size tokens, distinct per sid (different stream id -> different word sequence) + short instruction."""
    body = " ".join(filler(sid, int(size / 1.07)))
    return f"[check conc size {size} stream {sid}] Deep context priming block. {body}. List five words from the text above:"


def complete(prompt, max_tokens=GEN):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    r = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(r, timeout=7200).read())
    except urllib.error.HTTPError as e:
        try: detail = e.read().decode("utf-8", "replace")[:200]
        except Exception: detail = ""
        raise RuntimeError(f"HTTP {e.code} {e.reason} {detail}")
    return resp["choices"][0].get("text", "")


def model_name():
    d = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/models"), timeout=60).read())
    return d["data"][0]["id"]


def log(msg): print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def check_needle(size, sid):
    code = rand_word()
    p = needle_prompt(size, sid, code)
    try:
        txt = complete(p, NEEDLE_GEN)
    except Exception as e:
        log(f"FAIL needle size={size} request error: {str(e)[:200]}")
        return {"pass": False, "code": code, "error": str(e)[:200]}
    ok = code.lower() in txt.lower()
    if ok: log(f'PASS needle size={size} code="{code}" -> completion={txt.strip()[:60]!r}')
    else: log(f'FAIL needle size={size} code="{code}" not in completion: {txt.strip()[:200]!r}')
    return {"pass": ok, "code": code, "completion": txt[:200]}


def check_concurrency():
    prompts = [conc_prompt(CONC_SIZE, i) for i in range(CONC)]
    solo, err = [], None
    for i, p in enumerate(prompts):                       # solo pass: one request at a time, in order
        try: solo.append(complete(p))
        except Exception as e:
            err = f"request {i} failed in solo pass: {str(e)[:200]}"
            log(f"FAIL concurrency {err}")
            return {"pass": False, "error": err}
    conc = [None]*CONC
    gate = threading.Event()
    def run(i):
        try:
            gate.wait(); conc[i] = complete(prompts[i])
        except Exception as e: conc[i] = f"ERROR: {str(e)[:200]}"
    th = [threading.Thread(target=run, args=(i,)) for i in range(CONC)]
    [t.start() for t in th]; gate.set(); [t.join() for t in th]
    bad = [{"index": i, "solo": solo[i], "concurrent": conc[i]} for i in range(CONC) if conc[i] != solo[i]]
    if bad:
        log(f"FAIL concurrency conc={CONC} size={CONC_SIZE} {len(bad)}/{CONC} completions differ")
        for m in bad:
            log(f"  [{m['index']}] solo:       {m['solo']!r}")
            log(f"  [{m['index']}] concurrent: {m['concurrent']!r}")
        return {"pass": False, "conc": CONC, "size": CONC_SIZE, "mismatches": bad}
    log(f"PASS concurrency conc={CONC} size={CONC_SIZE} all {CONC} concurrent completions byte-equal to solo")
    return {"pass": True, "conc": CONC, "size": CONC_SIZE, "mismatches": []}


def main():
    global URL, MODEL, SIZES, CONC, CONC_SIZE
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"): print(__doc__.strip()); sys.exit(0)
    if argv[0].startswith("-"): print("openai_check.py: URL is required\n" + __doc__.strip()); sys.exit(0)
    URL = argv[0]
    opt = {k: v for k, v in [(a[2:], b) for a, b in zip(argv[1::2], argv[2::2]) if a.startswith("--")]}
    if "sizes" in opt: SIZES = [int(x) for x in opt["sizes"].split(",")]
    if "conc" in opt: CONC = int(opt["conc"])
    if "conc-size" in opt: CONC_SIZE = int(opt["conc-size"])
    unknown = [a for a in argv[1:] if a.startswith("--") and a[2:] not in ("sizes", "conc", "conc-size")]
    if unknown: print(f"openai_check.py: unknown option(s) {unknown}\n" + __doc__.strip()); sys.exit(0)
    try: MODEL = model_name()
    except Exception as e: log(f"FAIL cannot get model name from {URL}/v1/models: {str(e)[:200]}"); sys.exit(1)
    log(f"url={URL} model={MODEL} sizes={SIZES} conc={CONC} conc-size={CONC_SIZE} max_tokens={GEN}")
    needles = {}
    for k, size in enumerate(SIZES): needles[str(size)] = check_needle(size, 900 + k)
    concurrency = check_concurrency() if CONC > 0 else {"pass": True, "conc": 0, "size": CONC_SIZE, "mismatches": [], "skipped": True}
    if CONC <= 0: log("SKIP concurrency (--conc 0)")
    summary = {"needle": needles, "concurrency": concurrency}
    ok = all(v["pass"] for v in needles.values()) and concurrency["pass"]
    name = f"check-{time.strftime('%Y%m%d-%H%M%S')}.json"
    json.dump({"client": "openai_check.py", "url": URL, "model": MODEL, "sizes": SIZES, "conc": CONC,
               "conc_size": CONC_SIZE, "all_pass": ok, **summary}, open(name, "w"), indent=1)
    log(f"wrote {name}")
    print(json.dumps(summary), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__": main()
