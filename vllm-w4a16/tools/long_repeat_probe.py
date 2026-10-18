#!/usr/bin/env python3
# Temperature-0 repeatability of a long cold prompt (T14, journal/16-nondeterminism.md). repeat_probe.py uses
# prompt_logprobs, whose logits for every position do not fit at depth, so this sends one fixed-seed prompt of about
# --tokens tokens --reps times with a fresh cache_salt each time (no prefix-cache hit, so every repeat runs the full
# prefill, asserted from the hit counter), generates --max-tokens tokens with top-5 logprobs, and reports how many
# distinct outputs came back, the first token where each repeat differs from repeat 0 and the largest logprob spread
# over the common prefix. Covers the QSA top-k path, which selects among more than 2,048 positions only at depth.
# Usage: URL=http://localhost:8090 python3 vllm-w4a16/tools/long_repeat_probe.py --label R30 [--tokens 16000] [--reps 4]
import argparse, json, os, random, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib as B
from benchlib import out

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8090"))
ap.add_argument("--tokens", type=int, default=16000)
ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=16)
ap.add_argument("--label", default="run")
ap.add_argument("--out-dir", default=B.RESULTS)
a = ap.parse_args()
B.set_base(a.url)
stamp = time.strftime("%Y%m%d-%H%M%S")
B.open_log(os.path.join(a.out_dir, f"long-repeat-probe-{a.label}-{a.tokens}-{stamp}.log"))
prompt = B.filler_needle(a.tokens, random.Random(1234)) + "\n\nSummarize the text above in one sentence."
B.wait_ready(); B.assert_idle("start")
runs = []
for i in range(a.reps):
    B.assert_idle("before request")
    m0 = B.metrics()
    body = {"model": B.MODEL, "prompt": prompt, "max_tokens": a.max_tokens, "temperature": 0, "logprobs": 5,
            "return_tokens_as_token_ids": True, "cache_salt": f"t14-{stamp}-{i}"}
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(B.BASE + "/v1/completions", data=json.dumps(body).encode(),
                   headers={"Content-Type": "application/json"}), timeout=3600).read())
    B.assert_idle("after request")
    d = B.mdelta(m0, B.metrics())
    hits = int(d.get("prefix_cache_hits", 0))
    assert hits == 0, f"repeat {i} hit the prefix cache ({hits} tokens)"
    lp = r["choices"][0]["logprobs"]
    ids = [int(t.split(":")[1]) for t in lp["tokens"]]
    runs.append({"ids": ids, "logprobs": lp["token_logprobs"], "prompt_tokens": r["usage"]["prompt_tokens"],
                 "seconds": round(time.time() - t0, 2)})
    out(f"run {i}: {r['usage']['prompt_tokens']} prompt tokens, {runs[-1]['seconds']} s, ids {ids[:8]}..., "
        f"first logprobs {[round(x, 4) for x in lp['token_logprobs'][:4]]}")
ref = runs[0]
firsts, spread = [], 0.0
for r in runs[1:]:
    j = next((k for k, (x, y) in enumerate(zip(ref["ids"], r["ids"])) if x != y), None)
    firsts.append(j)
    n = len(ref["ids"]) if j is None else j
    spread = max([spread] + [abs(x - y) for x, y in zip(ref["logprobs"][:n], r["logprobs"][:n])])
distinct = len({json.dumps([r["ids"], r["logprobs"]]) for r in runs})
out(f"distinct outputs (ids and logprobs) {distinct} of {a.reps}; first differing token vs run 0: {firsts}; "
    f"max |dlogprob| over common prefixes {spread:.5f}")
path = os.path.join(a.out_dir, f"long-repeat-probe-{a.label}-{a.tokens}-{stamp}.json")
json.dump({"label": a.label, "tokens": a.tokens, "runs": runs, "distinct": distinct, "first_diff": firsts,
           "max_spread": spread}, open(path, "w"), indent=1)
out(f"wrote {path}")
