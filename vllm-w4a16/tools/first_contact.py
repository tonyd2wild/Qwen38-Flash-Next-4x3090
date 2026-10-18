#!/usr/bin/env python3
# What the first requests after a boot pay for Triton JIT (T11, journal/14-p9-fixed-overheads.md). Run it right after
# "Application startup complete", before anything else talks to the server (the gate and the scoreboard warm up, so they
# come after). Two passes of the same request shapes, each prompt freshly salted (cold, no prefix-cache hits): pass 1
# compiles whatever the Triton cache lacks, pass 2 finds every kernel compiled. pass1 - pass2 per row is the JIT tax.
# Shapes: a short prompt, prompts of 33 to 30K tokens (new prefill-chunk sizes specialize kernels again), count to 100.
# Idle asserts as in scoreboard.py.
# Usage: URL=http://localhost:8090 python3 vllm-w4a16/tools/first_contact.py --label R20
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib as B
from benchlib import out

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8090"))
ap.add_argument("--label", default="run")
ap.add_argument("--out-dir", default=B.RESULTS)
a = ap.parse_args()
B.set_base(a.url)
stamp = time.strftime("%Y%m%d-%H%M%S")
B.open_log(os.path.join(a.out_dir, f"first-contact-{a.label}-{stamp}.log"))
SIZES = [33, 65, 200, 1000, 4000, 8000, 30000]
COUNT = "Count from 1 to 100, separated by spaces. Output only the numbers."
R = {"label": a.label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "passes": []}

B.wait_ready(); B.assert_idle("start")
for p in (1, 2):
    rows = []
    out(f"pass {p}")
    r = B.chat("Say OK.", 8); rows.append({"kind": "short", "pt": r["pt"], "ttft": r["ttft"], "wall": r["wall"]})
    out(f"  short        pt {r['pt']:>6}  TTFT {r['ttft']:7.3f} s")
    for n in SIZES:
        rng = random.Random(B.salt())
        body = B.filler_needle(max(n - 20, 1), rng)
        r = B.chat(f"Run {B.salt()}. {body}\n\nReply with one word.", 4)
        rows.append({"kind": f"cold{n}", "pt": r["pt"], "ttft": r["ttft"], "wall": r["wall"], "hits": r["hits"]})
        out(f"  cold {n:>6}  pt {r['pt']:>6}  TTFT {r['ttft']:7.3f} s  hits {r['hits']}")
    r = B.chat(COUNT, 512)
    rows.append({"kind": "count", "pt": r["pt"], "ct": r["ct"], "ttft": r["ttft"], "dec": r["dec"], "wall": r["wall"]})
    out(f"  count        TTFT {r['ttft']:7.3f} s  decode {r['dec']:6.1f} tok/s  wall {r['wall']:6.2f} s")
    R["passes"].append(rows)
p1, p2 = R["passes"]
R["jit_tax_s"] = sum(x["wall"] for x in p1) - sum(x["wall"] for x in p2)
out(f"pass 1 - pass 2 wall: {R['jit_tax_s']:.2f} s  (" + ", ".join(f"{x['kind']} {x['wall'] - y['wall']:+.2f}" for x, y in zip(p1, p2)) + ")")
path = os.path.join(a.out_dir, f"first-contact-{a.label}-{stamp}.json")
json.dump(R, open(path, "w"), indent=1)
out(f"wrote {path}")
