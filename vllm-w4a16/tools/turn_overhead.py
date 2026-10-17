#!/usr/bin/env python3
# Where a turn's time goes outside prefill. For a conversation of --depth tokens, one cold request fills the prefix
# cache; then the same prompt is sent warm three ways, --reps times each:
#   chat        /v1/chat/completions with the messages (what an agent client sends)
#   text        /v1/completions with the rendered prompt text (no chat template, still tokenized by the server)
#   ids         /v1/completions with the prompt as token ids (no template, no tokenization)
# For each: client TTFT, the server's TTFT / queue / prefill histogram deltas, and the two gaps they leave:
#   client - server TTFT   (HTTP, request parsing, chat template, tokenization: whatever runs before arrival_time)
#   server TTFT - prefill  (arrival in the frontend to scheduling in the engine core, plus the first token's way back)
# --probe N repeats a short prompt with max_tokens=1 and logprobs N times (the temperature-0 repeatability probe of
# journal/01-baseline.md). Idle asserts as in scoreboard.py.
# Usage: URL=http://localhost:8090 python3 vllm-w4a16/tools/turn_overhead.py --depth 128000 --reps 2 --probe 10
import argparse, json, os, random, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib as B
from benchlib import out

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8090"))
ap.add_argument("--depth", type=int, default=128000)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--probe", type=int, default=0)
ap.add_argument("--label", default="run")
ap.add_argument("--out-dir", default=B.RESULTS)
a = ap.parse_args()
B.set_base(a.url)
stamp = time.strftime("%Y%m%d-%H%M%S")
B.open_log(os.path.join(a.out_dir, f"turn-overhead-{a.label}-{stamp}.log"))
R = {"label": a.label, "depth": a.depth, "runs": [], "probe": []}


def post(path, body, stream):
    req = urllib.request.Request(B.BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    m0 = B.assert_idle("before request")
    t0 = time.time(); first = None; usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        if not stream: raw = json.loads(r.read()); t1 = time.time()
        else:
            raw = None
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d = json.loads(line[5:])
                ch = (d.get("choices") or [{}])[0]
                if first is None and (ch.get("text") or (ch.get("delta") or {}).get("content")): first = time.time()
                if d.get("usage"): usage = d["usage"]
            t1 = time.time()
    m1 = B.assert_idle("after request")
    return raw, (first or t1) - t0, usage, B.mdelta(m0, m1)


def row(kind, ttft, usage, d):
    s_ttft = d["time_to_first_token_seconds_sum"]; pre = d["request_prefill_time_seconds_sum"]
    r = {"kind": kind, "pt": usage.get("prompt_tokens"), "hits": int(d["prefix_cache_hits"]), "client_ttft": ttft,
         "server_ttft": s_ttft, "queue": d["request_queue_time_seconds_sum"], "prefill": pre,
         "client_minus_server": ttft - s_ttft, "server_minus_prefill": s_ttft - pre,
         "preemptions": int(d["num_preemptions"])}
    out(f"  {kind:5} pt {r['pt']:>7} hits {r['hits']:>7}  client TTFT {ttft:6.3f}  server {s_ttft:6.3f}  prefill {pre:6.3f}"
        f"  client-server {1e3*r['client_minus_server']:6.0f} ms  server-prefill {1e3*r['server_minus_prefill']:6.0f} ms")
    R["runs"].append(r)


B.wait_ready(); B.assert_idle("start")
if a.probe:
    prompt = "Write a Python function that returns the n-th Fibonacci number."
    out(f"repeatability probe: {a.probe} x max_tokens=1 with logprobs: {prompt!r}")
    for i in range(a.probe):
        raw, _, usage, d = post("/v1/chat/completions", {"model": B.MODEL, "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5,
                                "chat_template_kwargs": {"enable_thinking": False}}, False)
        lp = raw["choices"][0]["logprobs"]["content"][0]
        top = [(t["token"], round(t["logprob"], 4)) for t in lp["top_logprobs"]]
        R["probe"].append({"token": lp["token"], "logprob": lp["logprob"], "top": top})
        out(f"  {i}: {lp['token']!r} {lp['logprob']:.4f}  top {top[:3]}")
    lps = [p["logprob"] for p in R["probe"]]
    out(f"  distinct top-1 logprobs: {len(set(round(x, 6) for x in lps))} of {len(lps)}, range {min(lps):.4f} .. {max(lps):.4f}")

if a.depth:
    rng = random.Random(B.salt())
    msgs = [{"role": "user", "content": f"Session {B.salt()}. " + B.filler_needle(a.depth, rng) +
             "\n\nReply with one short sentence about the text above."}]
    tok = json.loads(urllib.request.urlopen(urllib.request.Request(
        B.BASE + "/tokenize", data=json.dumps({"model": B.MODEL, "messages": msgs, "add_generation_prompt": True,
                                               "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"}), timeout=600).read())
    ids = tok["tokens"]
    text = json.loads(urllib.request.urlopen(urllib.request.Request(
        B.BASE + "/detokenize", data=json.dumps({"model": B.MODEL, "tokens": ids}).encode(),
        headers={"Content-Type": "application/json"}), timeout=600).read())["prompt"]
    out(f"depth {a.depth}: {len(ids)} prompt tokens; cold request to fill the cache")
    chat_body = {"model": B.MODEL, "messages": msgs, "max_tokens": 8, "temperature": 0, "stream": True,
                 "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    _, ttft, usage, d = post("/v1/chat/completions", chat_body, True); row("cold", ttft, usage, d)
    for rep in range(a.reps):
        _, ttft, usage, d = post("/v1/chat/completions", chat_body, True); row("chat", ttft, usage, d)
        body = {"model": B.MODEL, "prompt": text, "max_tokens": 8, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        _, ttft, usage, d = post("/v1/completions", body, True); row("text", ttft, usage, d)
        body["prompt"] = ids
        _, ttft, usage, d = post("/v1/completions", body, True); row("ids", ttft, usage, d)
path = os.path.join(a.out_dir, f"turn-overhead-{a.label}-{stamp}.json")
json.dump(R, open(path, "w"), indent=1); out(f"wrote {path}")
