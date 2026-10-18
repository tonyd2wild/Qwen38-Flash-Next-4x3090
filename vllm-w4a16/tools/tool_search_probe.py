#!/usr/bin/env python3
# The quality gate's tool_search request repeated --reps times on an idle server (T11 R21, T4 R25; the first probe for
# T14, journal/INDEX.md). Same messages, tools and sampling as quality_gate.py's tool_search prompt: non-streamed,
# temperature 0, top-5 logprobs, token ids. Token 9 is a near-tie that flips between 79114 (the baseline, margin 3.75)
# and 28. R21 and R25 picked 28 at -0.38687 in the gate, while repeats right after the gate gave 79114 at margins of
# 1.25-5.25 (R21) and 5.3-7.8 nat (R25). Prints the hit count, token --pos with its logprob and the top 3 per run, and
# how many distinct outputs came back. Idle asserts as in scoreboard.py (benchlib.chat checks before and after).
# Usage: URL=http://localhost:8090 python3 vllm-w4a16/tools/tool_search_probe.py [--reps 12] [--pos 9] [--out FILE]
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib as B

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8090"))
ap.add_argument("--reps", type=int, default=12)
ap.add_argument("--pos", type=int, default=9)
ap.add_argument("--out", default="")
a = ap.parse_args()
B.set_base(a.url)
msgs = [{"role": "system", "content": B.SYSTEM},
        {"role": "user", "content": "Search the repository for the regex 'def create_order' and tell me where it is defined."}]
seen = collections.Counter(); rows = []
for i in range(a.reps):
    r = B.chat(msgs, 256, stream=False, logprobs=True, top_logprobs=5, return_tokens_as_token_ids=True, tools=B.TOOLS,
               tool_choice="auto")
    lp = r["raw"]["choices"][0]["logprobs"]["content"]
    ids = tuple(int(t["token"].split(":")[1]) for t in lp)
    t = lp[a.pos]; top = [(x["token"], round(x["logprob"], 3)) for x in t["top_logprobs"][:3]]
    seen[ids] += 1
    rows.append({"hits": r["hits"], "token": ids[a.pos], "logprob": t["logprob"], "top3": top})
    print(i, "hits", r["hits"], f"tok{a.pos}", t["token"], round(t["logprob"], 3), top)
print("distinct outputs:", len(seen), list(seen.values()))
if a.out:
    json.dump({"pos": a.pos, "runs": rows, "distinct": len(seen)}, open(a.out, "w"), indent=1)
    print("wrote", a.out)
