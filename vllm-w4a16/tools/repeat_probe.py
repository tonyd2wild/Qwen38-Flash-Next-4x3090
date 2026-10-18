#!/usr/bin/env python3
# Temperature-0 repeatability of one forward pass: the same prompt sent --reps times, one request at a time on an idle
# server, with prompt_logprobs (the logprob of every prompt token given the ones before it, computed in the prefill)
# and max_tokens=1. Prints the first position where each run differs from run 0 and the spread per position. A
# bit-reproducible lane shows 1 distinct value everywhere; floating-point reordering shows small spreads; a discrete
# switch (an expert-routing flip, stale state) shows large spreads that start at a specific position.
# Usage: URL=http://localhost:8090 python3 vllm-w4a16/tools/repeat_probe.py --label R17 [--reps 6] [--copies 3]
import argparse, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib as B
from benchlib import out

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8090"))
ap.add_argument("--reps", type=int, default=6)
ap.add_argument("--copies", type=int, default=3, help="the sentence repeated this many times (prompt length)")
ap.add_argument("--label", default="run")
ap.add_argument("--out-dir", default=B.RESULTS)
a = ap.parse_args()
B.set_base(a.url)
stamp = time.strftime("%Y%m%d-%H%M%S")
B.open_log(os.path.join(a.out_dir, f"repeat-probe-{a.label}-{stamp}.log"))
prompt = "Write a Python function that returns the n-th Fibonacci number." * a.copies
B.wait_ready(); B.assert_idle("start")
runs, nexts = [], []
for i in range(a.reps):
    B.assert_idle("before request")
    body = {"model": B.MODEL, "prompt": prompt, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0, "logprobs": 5}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(B.BASE + "/v1/completions", data=json.dumps(body).encode(),
                   headers={"Content-Type": "application/json"}), timeout=120).read())
    B.assert_idle("after request")
    ch = r["choices"][0]
    vals = [min(p.values(), key=lambda x: x.get("rank", 99))["logprob"] for p in ch["prompt_logprobs"][1:]]
    runs.append(vals); nexts.append(ch["logprobs"]["top_logprobs"][0])
    out(f"run {i}: {len(vals)} positions, first 8 {[round(x, 3) for x in vals[:8]]}, next-token top {list(nexts[-1].items())[:2]}")
n = len(runs[0])
spread = [max(r[j] for r in runs) - min(r[j] for r in runs) for j in range(n)]
first = [next((j for j in range(n) if r[j] != runs[0][j]), None) for r in runs[1:]]
distinct = len({tuple(r) for r in runs})
out(f"distinct runs {distinct} of {a.reps}; first differing position vs run 0: {first}")
out(f"per-position spread: max {max(spread):.4f}, positions with any spread {sum(1 for s in spread if s > 0)} of {n}")
out(f"spread by position: {[round(s, 3) for s in spread]}")
path = os.path.join(a.out_dir, f"repeat-probe-{a.label}-{stamp}.json")
json.dump({"label": a.label, "prompt": prompt, "runs": runs, "next": nexts, "spread": spread, "first_diff": first,
           "distinct": distinct}, open(path, "w"), indent=1)
out(f"wrote {path}")
