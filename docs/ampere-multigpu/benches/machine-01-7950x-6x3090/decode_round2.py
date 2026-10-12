#!/usr/bin/env python3
"""Like-for-like DECODE-ONLY measurement against an OpenAI-compatible server (stdlib only).

usage: decode_round2.py URL LABEL OUT.json [sessions=5] [prompt_tokens=250000] [gen=128]

Round 1: all `sessions` sessions concurrently POST /v1/completions (max_tokens=8, temperature=0,
stream=true) and are awaited in full -- this prefills every session and leaves its prefix cached.
Round 2: the SAME prompt per session, again concurrently, with max_tokens=gen, temperature=0,
ignore_eos=true, stream=true and stream_options.include_usage=true. Because the prompt is identical
to round 1, round 2 hits the prefix cache, so ttft2 is (mostly) cache-hit prefill and
tokens/(wall2-ttft2) is the decode rate of a slot under load.

Prompts are built exactly as in/readme_grid_openai.py does (WORDS and prompt() copied verbatim,
same 1.07 tokens-per-word factor, same 10+i per-session index so prompts differ per session).

Timings: t_send = just before urlopen, t_first/t_last = first/last chunk carrying non-empty text.
ttftN = t_first-t_send, wallN = t_last-t_send. tg_slot = tokens/(wall2-ttft2);
tg_total = sum(tokens) / (max(t_last) - min(t_send)) over the sessions that succeeded in round 2.

Prints one summary line on stdout; per-session errors go to stderr AND into that row as
"error": "..." (the run continues, exit code 1 if any row errored). 1800 s timeout per request."""

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

TIMEOUT = 1800            # per request: long prefills
TPW = 1.07                # measured tokens per word for this generator (as in the reference client)

# ---- copied verbatim from in/readme_grid_openai.py -------------------------------
WORDS = ("system latency throughput kernel tensor gradient manifold entropy quantize scheduler pipeline embedding attention routing expert cache vector matrix decode prefill context window inference bandwidth compute memory buffer sparse dense token logits softmax").split()


def prompt(i, size, n_words):
    body = " ".join(WORDS[(i*7 + k*13) % len(WORDS)] for k in range(n_words))
    return f"[grid {LABEL} size {size} stream {i}] Deep context priming block. {body}. Now continue at length:"
# ----------------------------------------------------------------------------------

URL = LABEL = OUT = MODEL = None
SESSIONS, PROMPT_TOKENS, GEN = 5, 250000, 128


def gpu():
    """nvidia-smi snapshot the way the reference client does; absent nvidia-smi -> None."""
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=10).stdout.strip().replace("\n", ";")
    except Exception:
        return None


def model_name():
    d = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/models"), timeout=60).read())
    return d["data"][0]["id"]


def open_stream(p, max_tokens, usage_and_eos):
    """POST one streaming completion; returns (response, t_send)."""
    body = {"model": MODEL, "prompt": p, "max_tokens": max_tokens, "temperature": 0, "stream": True}
    if usage_and_eos:
        body["ignore_eos"] = True
        body["stream_options"] = {"include_usage": True}
    r = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    t_send = time.time()
    try:
        return urllib.request.urlopen(r, timeout=TIMEOUT), t_send
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            detail = ""
        raise RuntimeError(f"HTTP {e.code} {e.reason} {detail}")


def stream_req(p, max_tokens, usage_and_eos):
    """One streaming request; records t_send, t_first/t_last (first/last chunk with non-empty text),
    the number of text chunks, the final usage dict (or None) and the concatenated text."""
    resp, t_send = open_stream(p, max_tokens, usage_and_eos)
    text, usage, chunks, t_first, t_last = [], None, 0, None, None
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue                                   # SSE data line; ignore comments/blank
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                c = json.loads(data)
            except Exception:
                continue
            if c.get("usage"):
                usage = c["usage"]                         # final usage chunk (stream_options.include_usage)
            choices = c.get("choices") or []
            if not choices:
                continue                                   # usage-only chunk
            ch = choices[0]
            txt = ch.get("text") or ""
            # Any choice chunk is a generated token (an immediate EOS arrives as text ""
            # with finish_reason "stop"), so time it even when the text is empty.
            now = time.time()
            if t_first is None:
                t_first = now
            t_last = now
            chunks += 1
            if txt:
                text.append(txt)
    finally:
        try:
            resp.close()
        except Exception:
            pass
    if t_first is None:
        raise RuntimeError("no tokens streamed")
    return {"t_send": t_send, "t_first": t_first, "t_last": t_last, "chunks": chunks,
            "usage": usage, "text": "".join(text)}


def phase(prompts, max_tokens, usage_and_eos):
    """Run len(prompts) requests concurrently in N threads released together, so min(t_send) aligns.
    Returns the per-session result dicts ({"error": ...} on failure) -- never raises."""
    res = [None] * len(prompts)
    gate = threading.Event()

    def run(i):
        try:
            gate.wait()
            res[i] = {"ok": stream_req(prompts[i], max_tokens, usage_and_eos)}
        except Exception as e:
            res[i] = {"error": str(e)[:200]}

    th = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    [t.start() for t in th]
    gate.set()
    [t.join() for t in th]
    return res


DOC = {}


def save():
    json.dump(DOC, open(OUT, "w"), indent=1)


def fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def main():
    global URL, LABEL, OUT, SESSIONS, PROMPT_TOKENS, GEN, MODEL
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help") or len(argv) < 3:
        print(__doc__.strip())
        sys.exit(0)
    URL, LABEL, OUT = argv[0].rstrip("/"), argv[1], argv[2]
    SESSIONS = int(argv[3]) if len(argv) > 3 else 5
    PROMPT_TOKENS = int(argv[4]) if len(argv) > 4 else 250000
    GEN = int(argv[5]) if len(argv) > 5 else 128

    DOC.update({"label": LABEL, "url": URL, "sessions": SESSIONS, "prompt_tokens": PROMPT_TOKENS,
                "gen": GEN, "rows": [], "tg_slot_mean": None, "tg_total": None,
                "gpu_before": None, "gpu_after": None,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished": None})

    try:
        MODEL = model_name()
    except Exception as e:
        print("decode_round2 %s: cannot get model name from %s/v1/models: %s"
              % (LABEL, URL, str(e)[:200]), file=sys.stderr)
        DOC["error"] = str(e)[:200]
        save()
        sys.exit(1)

    n_words = int(PROMPT_TOKENS / TPW)                       # same 1.07 rule as the reference client
    prompts = [prompt(10 + i, PROMPT_TOKENS, n_words) for i in range(SESSIONS)]
    DOC["gpu_before"] = gpu()

    # ---- round 1: concurrent prefill of every session, awaited in full ------------
    r1 = phase(prompts, 8, False)
    rows = []
    for i, r in enumerate(r1):
        row = {"session": i}
        if "error" in r:
            row["error"] = "round1: " + r["error"]
        else:
            d = r["ok"]
            row["ttft1"] = round(d["t_first"] - d["t_send"], 3)
            row["wall1"] = round(d["t_last"] - d["t_send"], 3)
        rows.append(row)
    DOC["rows"] = rows
    save()

    # ---- round 2: the same prompts, decode with the prefix already cached ---------
    live = [i for i in range(SESSIONS) if "error" not in rows[i]]
    r2 = phase([prompts[i] for i in live], GEN, True) if live else []
    starts, ends = [], []
    for k, i in enumerate(live):
        row, r = rows[i], r2[k]
        if "error" in r:
            row["error"] = "round2: " + r["error"]
            continue
        d = r["ok"]
        usage = d["usage"] or {}
        tokens = int(usage.get("completion_tokens") or 0) or max(d["chunks"], 1)   # usage else text chunks
        ttft2 = d["t_first"] - d["t_send"]
        wall2 = d["t_last"] - d["t_send"]
        row["ttft2"] = round(ttft2, 3)
        row["tokens"] = tokens
        row["wall2"] = round(wall2, 3)
        row["tg_slot"] = round(tokens / max(wall2 - ttft2, 1e-6), 2)
        starts.append(d["t_send"])
        ends.append(d["t_last"])

    ok = [r for r in rows if "error" not in r and r.get("tokens")]
    if ok:
        DOC["tg_slot_mean"] = round(sum(r["tg_slot"] for r in ok) / len(ok), 2)
        DOC["tg_total"] = round(sum(r["tokens"] for r in ok) / max(max(ends) - min(starts), 1e-6), 2)
    for r in rows:
        if "error" in r:
            print("decode_round2 %s session %d ERROR %s" % (LABEL, r["session"], r["error"]), file=sys.stderr)

    DOC["gpu_after"] = gpu()
    DOC["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save()

    mean_ttft = round(sum(r["ttft2"] for r in ok) / len(ok), 3) if ok else None
    print(f"decode_round2 {LABEL}: sessions={SESSIONS} prompt={PROMPT_TOKENS} gen={GEN} "
          f"ttft2 mean {fmt(mean_ttft, '.2f')}s tg_slot mean {fmt(DOC['tg_slot_mean'], '.1f')} t/s "
          f"aggregate {fmt(DOC['tg_total'], '.1f')} t/s", flush=True)
    sys.exit(1 if any("error" in r for r in rows) else 0)


if __name__ == "__main__":
    main()
