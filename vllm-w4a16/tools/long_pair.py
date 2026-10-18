#!/usr/bin/env python3
# Concurrent long cold prompts (T4, journal/04-p3-mamba-leak.md): N salted cold prompts of about --tokens tokens each,
# sent at the same time. Each prefills alongside the others, so the pool has to hold N long requests at once; the
# Mamba page leak turns that into preemptions and recomputation. Reports each request's prompt tokens and TTFT, the
# wall time, and the server-wide preemption delta. Idle asserts as in scoreboard.py, before and after.
# This lane's 2,048-token step budget fits only one 832-aligned 1,664-token chunk, so concurrent cold prompts prefill one
# after the other. --stagger is the case that collides: request 0 (--tokens, decoding --decode tokens) holds its blocks
# while it decodes, and the others (--tokens2 each) arrive as soon as it emits its first token.
# Usage: URL=http://localhost:8090 python3 vllm-w4a16/tools/long_pair.py --label R24 [--n 2] [--tokens 100000]
#        ... --stagger --decode 1500 --tokens2 40000
import argparse, json, os, random, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib as B
from benchlib import out

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8090"))
ap.add_argument("--label", default="run")
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--tokens", type=int, default=100000)
ap.add_argument("--stagger", action="store_true")
ap.add_argument("--decode", type=int, default=30, help="max_tokens of request 0")
ap.add_argument("--tokens2", type=int, default=40000, help="prompt size of the later requests with --stagger")
ap.add_argument("--out-dir", default=B.RESULTS)
a = ap.parse_args()
B.set_base(a.url)
stamp = time.strftime("%Y%m%d-%H%M%S")
kind = "stagger" if a.stagger else "pair"
B.open_log(os.path.join(a.out_dir, f"long-{kind}-{a.label}-{stamp}.log"))

B.wait_ready(); B.assert_idle("start")
prompts = []
for i in range(a.n):
    rng = random.Random(B.salt())
    n = a.tokens2 if a.stagger and i else a.tokens
    ask = "Retell the text above in full detail, as long as you can." if a.stagger and i == 0 else "Summarize the text above in one sentence."
    prompts.append(f"Run {B.salt()}. " + B.filler_bench(n, rng) + "\n\n" + ask)
res = [None] * a.n
def go(i):
    res[i] = B.chat(prompts[i], a.decode if i == 0 else 30, check_idle=False, ignore_eos=True)
m0 = B.metrics(); t0 = time.perf_counter()
th = [threading.Thread(target=go, args=(i,)) for i in range(a.n)]
th[0].start()
if a.stagger:  # the others start when request 0 has produced its first output token
    while B.mget(B.metrics(), "generation_tokens_total") <= B.mget(m0, "generation_tokens_total"):
        time.sleep(0.2)
    t_start2 = time.perf_counter() - t0
for t in th[1:]: t.start()
for t in th: t.join()
wall = time.perf_counter() - t0; d = B.mdelta(m0, B.metrics())
B.assert_idle("end")
pt = sum(r["pt"] for r in res)
R = {"label": a.label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "n": a.n, "tokens": a.tokens, "wall_s": wall,
     "prompt_tokens": pt, "preemptions": int(d["num_preemptions"]), "aggregate_prefill_tps": pt / max(r["ttft"] for r in res),
     "stagger": a.stagger, "decode": a.decode, "tokens2": a.tokens2 if a.stagger else None,
     "later_start_s": t_start2 if a.stagger else 0.0,
     "requests": [{"pt": r["pt"], "ct": r["ct"], "ttft": r["ttft"], "dec": r["dec"], "wall": r["wall"], "finish": r["finish"]} for r in res]}
for i, r in enumerate(res):
    out(f"  request {i}: {r['pt']:,} prompt tokens, TTFT {r['ttft']:.1f} s, {r['ct']} tokens at {r['dec']:.1f} tok/s, wall {r['wall']:.1f} s, finish {r['finish']}")
if a.stagger:
    out(f"later requests started {R['later_start_s']:.1f} s in")
out(f"{a.n} x ~{a.tokens:,} cold: {pt:,} prompt tokens, last first token after {max(r['ttft'] for r in res):.1f} s "
    f"({R['aggregate_prefill_tps']:,.0f} tok/s aggregate prefill), wall {wall:.1f} s, preemptions {R['preemptions']}")
path = os.path.join(a.out_dir, f"long-{kind}-{a.label}-{stamp}.json")
json.dump(R, open(path, "w"), indent=1)
out(f"wrote {path}")
