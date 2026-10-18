#!/usr/bin/env python3
# The scoreboard of results/perf-2026-10-notes.md, one section per group of rows. bench-host.py's metric definitions
# verbatim (decode = completion tokens after the first / time after the first token; cold = fresh random prefix;
# warm = the same prompt again; TTFT from the HTTP request to the first streamed content token), plus incremental turns
# at depth, a code decode run, a ~200K needle, a ~155K cold prefill with free GPU memory, and 2/4/6 concurrent streams.
# Every request asserts num_requests_running and num_requests_waiting are 0 before and after (abort otherwise): nothing
# else may talk to the server during a run. Order on a candidate boot: quality_gate.py --compare first, then this.
# Usage (stdlib only; nvidia-smi optional, for the free-memory column):
#   URL=http://localhost:8090 python3 vllm-w4a16/tools/scoreboard.py --label R13
#   ... --sections decode,ttft           # decode ttft incremental needle200k cold155k concurrency (default: all)
#   ... --quick --label smoke            # smoke test: small sizes, one run, no long context
# Writes results/scoreboard-<label>-<time>.json and .log (raw per-request numbers in the JSON).
import argparse, json, os, random, statistics, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib as B
from benchlib import out

SECTIONS = ["decode", "ttft", "incremental", "needle200k", "cold155k", "concurrency"]
ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8090"))
ap.add_argument("--label", default="run")
ap.add_argument("--sections", default=",".join(SECTIONS))
ap.add_argument("--quick", action="store_true", help="smoke test: small sizes, 1 run, no needle/155K, concurrency 2 only")
ap.add_argument("--no-warmup", action="store_true")
ap.add_argument("--out-dir", default=B.RESULTS)
a = ap.parse_args()
B.set_base(a.url)
secs = [s.strip() for s in a.sections.split(",") if s.strip()]
for s in secs:
    if s not in SECTIONS: sys.exit(f"unknown section {s}; choose from {','.join(SECTIONS)}")
if a.quick: secs = [s for s in secs if s not in ("needle200k", "cold155k")]
stamp = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(a.out_dir, exist_ok=True)
log_path = B.open_log(os.path.join(a.out_dir, f"scoreboard-{a.label}-{stamp}.log"))
R = {"label": a.label, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "quick": a.quick, "sections": secs}

B.wait_ready()
R["server"] = B.server_info()
out(f"scoreboard {R['started']}  {B.BASE}  model {B.MODEL}  label {a.label}  vllm {R['server'].get('version')}  sections {','.join(secs)}")
B.assert_idle("start")
if not a.no_warmup: B.warmup()
m_start = B.metrics()

COUNT = "Count from 1 to 100, separated by commas, nothing else."
PROSE = "Write a detailed technical explanation of how a hybrid attention and state-space language model serves long contexts."
CODE = ("Write a Python module that implements a thread-safe LRU cache class with get, put and delete methods, a max size, "
        "optional per-entry TTL, type hints and Google-style docstrings, followed by a set of pytest tests for it. Output only the code.")
NEEDLE = "The secret code for the 3090 lane is 4471-ROSEWOOD."

def decode_runs(tag, prompt, runs, gen):
    rs = [B.chat(prompt, gen) for _ in range(runs)]
    d = {}
    for r in rs: d = B.add_deltas(d, r["d"])
    acc = B.acceptance(d)
    for k, r in enumerate(rs):
        out(f"  {tag} run {k+1}: {r['ct']} tokens, TTFT {r['ttft']*1000:.0f} ms, decode {r['dec']:.1f} tok/s")
    med = statistics.median(r["dec"] for r in rs)
    out(f"  {tag.upper()} median {med:.1f} tok/s, TTFT median {statistics.median(r['ttft'] for r in rs)*1000:.0f} ms, "
        f"acceptance mean length {acc.get('mean_len', 'n/a')} per position {acc.get('per_pos', 'n/a')}")
    return {"median_tps": med, "acceptance": acc, "runs": [{k: v for k, v in r.items() if k != "text"} | {"head": r["text"][:80]} for r in rs]}

def slim(r):
    return {k: v for k, v in r.items() if k not in ("text", "reasoning")} | {"head": r["text"][:80]}

S = {}  # summary rows, transcribed into the notes file
if "decode" in secs:
    runs = 1 if a.quick else 5
    out(f"\n== Decode, single stream (count median of {runs}, prose 1 run, code median of {1 if a.quick else 3})")
    R["decode"] = {"count": decode_runs("count", COUNT, runs, 512),
                   "prose": decode_runs("prose", PROSE, 1, 512),
                   "code": decode_runs("code", CODE, 1 if a.quick else 3, 512)}
    for k in ("count", "prose", "code"):
        S[f"decode_{k}"] = f"{R['decode'][k]['median_tps']:.1f} (accept {R['decode'][k]['acceptance'].get('mean_len', 'n/a')})"

if "ttft" in secs:
    sizes = [1000, 4000] if a.quick else [1000, 16000, 64000, 128000]
    out("\n== TTFT by prompt size (cold = fresh prefix, warm = same prompt again)")
    out(f"  {'target':>8} {'prompt':>8}  {'cold TTFT':>10} {'prefill':>12}  {'warm TTFT':>10} {'warm hits':>10}  {'decode@ctx':>10} {'preempt':>8}")
    R["ttft"] = []
    for size in sizes:
        rng = random.Random(f"{time.time()}-{size}")
        prompt = f"Run {rng.getrandbits(48):012x}. " + B.filler_bench(size, rng) + "\n\nSummarize the text above in three sentences."
        c = B.chat(prompt, 128); w = B.chat(prompt, 128)
        pre = c["pt"] / c["ttft"] if c["ttft"] > 0 else 0
        pre_n = int(c["d"]["num_preemptions"] + w["d"]["num_preemptions"])
        out(f"  {size:>8} {c['pt']:>8,}  {c['ttft']:>9.2f}s {pre:>8,.0f} t/s  {w['ttft']:>9.2f}s {w['hits']:>10,}  {c['dec']:>6.1f} t/s {pre_n:>8}")
        R["ttft"].append({"target": size, "cold": slim(c), "warm": slim(w), "prefill_tps": pre, "preemptions": pre_n})
        S[f"cold_{size}"] = f"{c['ttft']:.2f} s / {pre:,.0f}" + (f" ({pre_n} preemptions)" if pre_n else "")
        S[f"warm_{size}"] = f"{w['ttft']:.2f}"; S[f"decode_ctx_{size}"] = f"{c['dec']:.1f}"

if "incremental" in secs:
    plan = [(4000, [1000])] if a.quick else [(16000, [1000, 4000]), (64000, [4000, 16000]), (128000, [4000, 16000])]
    out("\n== Incremental turns (cached conversation + N new tokens; each N appends to the same cached base)")
    out(f"  {'cached':>8} {'new':>6} {'prompt':>8}  {'TTFT':>8} {'hits':>8}  {'srv queue':>9} {'srv prefill':>11} {'srv TTFT':>8}  {'base TTFT':>9}")
    R["incremental"] = []
    for depth, news in plan:
        rng = random.Random(B.salt())
        base = [{"role": "user", "content": f"Run {B.salt()} notes. " + B.filler_needle(depth, rng) + "\n\nSummarize the text above in one sentence."}]
        b = B.chat(base, 64)
        conv = base + [{"role": "assistant", "content": b["text"]}]
        for n in news:
            msgs = conv + [{"role": "user", "content": "More log: " + B.filler_needle(n, rng) + "\n\nSummarize the new log in one sentence."}]
            r = B.chat(msgs, 64); d = r["d"]
            out(f"  {depth:>8} {n:>6} {r['pt']:>8,}  {r['ttft']:>7.2f}s {r['hits']:>8,}  {d['request_queue_time_seconds_sum']:>8.3f}s "
                f"{d['request_prefill_time_seconds_sum']:>10.3f}s {d['time_to_first_token_seconds_sum']:>7.3f}s  {b['ttft']:>8.2f}s")
            R["incremental"].append({"depth": depth, "new": n, "base": slim(b), "turn": slim(r)})
            S[f"incr_{depth}_{n}"] = f"{r['ttft']:.2f}"

def long_prefill(tag, prompt, check=None):
    with B.GpuMemPoller() as poll:
        r = B.chat(prompt, 30)
    pre = r["pt"] / r["ttft"] if r["ttft"] > 0 else 0
    ok = None if check is None else (check in r["text"])
    out(f"  {tag}: {r['pt']:,} prompt tokens, TTFT {r['ttft']:.1f} s ({pre:,.0f} tok/s prefill), preemptions {int(r['d']['num_preemptions'])}, "
        f"min free per card {poll.summary()}" + ("" if ok is None else f", {'CORRECT' if ok else 'WRONG: ' + r['text'].strip()[:80]}"))
    return {"result": slim(r), "prefill_tps": pre, "correct": ok, "preemptions": int(r["d"]["num_preemptions"]), "min_free_gib": poll.min_free}

if "needle200k" in secs:
    out("\n== Needle ~200K")
    rng = random.Random(B.salt()); body = B.filler_needle(200000, rng); cut = int(len(body) * 0.4)
    prompt = f"Run {B.salt()}. " + body[:cut] + " " + NEEDLE + " " + body[cut:] + "\n\nQuestion: What is the secret code for the 3090 lane? Answer with the code only."
    R["needle200k"] = n = long_prefill("needle", prompt, "4471-ROSEWOOD")
    S["needle200k"] = f"{'correct' if n['correct'] else 'WRONG'} / {n['result']['ttft']:.0f} s / {n['prefill_tps']:,.0f}"

if "cold155k" in secs:
    out("\n== ~155K cold prefill")
    rng = random.Random(B.salt())
    prompt = f"Run {B.salt()}. " + B.filler_bench(155000, rng) + "\n\nSummarize the text above in one sentence."
    R["cold155k"] = c = long_prefill("155K cold", prompt)
    mf = c["min_free_gib"]
    S["cold155k"] = f"{c['prefill_tps']:,.0f}, {c['preemptions']}, " + (f"{min(mf):.2f} (per card {', '.join(f'{x:.2f}' for x in mf)})" if mf else "n/a")

if "concurrency" in secs:
    levels = [2] if a.quick else [2, 4, 6]; rounds = 1 if a.quick else 2
    out(f"\n== Concurrency (N streams x {rounds} rounds, agent system prompt + 8 tools, tool_choice none, max 512, temperature 0)")
    R["concurrency"] = []
    for n in levels:
        res = []; lock = threading.Lock(); errs = []
        def one(i):
            for rnd in range(rounds):
                msgs = [{"role": "system", "content": B.SYSTEM}, {"role": "user", "content": B.TASKS[(i + rnd) % len(B.TASKS)] +
                        " Before touching any tool, write the complete implementation plan as prose: every file you will read and change, "
                        "the exact code you intend to write for the core function, the tests you will add, and the risks. At least 600 words. "
                        "Do not call tools in this message."}]
                try: r = B.chat(msgs, 512, check_idle=False, tools=B.TOOLS, tool_choice="none")
                except Exception as e:
                    with lock: errs.append(str(e)[:150]); continue
                with lock: res.append(r)
        m0 = B.assert_idle(f"concurrency {n} start"); t0 = time.time()
        th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        [t.start() for t in th]; [t.join() for t in th]
        wall = time.time() - t0; m1 = B.assert_idle(f"concurrency {n} end"); d = B.mdelta(m0, m1)
        tot = sum(r["ct"] for r in res); tt = sorted(r["ttft"] for r in res); decs = [r["dec"] for r in res]
        pq = lambda q: tt[min(len(tt) - 1, int(round(q * (len(tt) - 1))))] if tt else float("nan")
        row = {"streams": n, "rounds": rounds, "requests": len(res), "errors": errs, "wall_s": wall, "completion_tokens": tot,
               "aggregate_tps": tot / wall if wall else 0, "per_stream_decode_median": statistics.median(decs) if decs else 0,
               "per_stream_decode_min": min(decs) if decs else 0, "ttft_p50": pq(0.5), "ttft_p99": pq(0.99),
               "hits": int(d["prefix_cache_hits"]), "preemptions": int(d["num_preemptions"]), "acceptance": B.acceptance(d)}
        R["concurrency"].append(row | {"requests_detail": [slim(r) for r in res]})
        out(f"  {n} streams: {len(res)} requests, {tot} tokens in {wall:.1f} s = {row['aggregate_tps']:.1f} tok/s aggregate; per-stream decode "
            f"median {row['per_stream_decode_median']:.1f} (min {row['per_stream_decode_min']:.1f}); TTFT p50 {row['ttft_p50']:.2f} s p99 {row['ttft_p99']:.2f} s"
            + (f"; ERRORS {len(errs)}: {errs[0]}" if errs else ""))
    S["conc_agg"] = " / ".join(f"{r['aggregate_tps']:.1f}" for r in R["concurrency"])
    S["conc_dec"] = " / ".join(f"{r['per_stream_decode_median']:.1f}" for r in R["concurrency"])
    S["conc_ttft"] = " / ".join(f"{r['ttft_p50']:.2f}-{r['ttft_p99']:.2f}" for r in R["concurrency"])

m_end = B.assert_idle("end")
R["totals"] = B.mdelta(m_start, m_end); R["summary"] = S; R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
json_path = os.path.join(a.out_dir, f"scoreboard-{a.label}-{stamp}.json")
json.dump(R, open(json_path, "w"), indent=1)

ROWS = [("Decode, count to 100, median of 5 (tok/s)", ["decode_count"]), ("Decode, prose 512 tokens (tok/s)", ["decode_prose"]),
        ("Decode, code prompt (tok/s)", ["decode_code"]),
        ("Cold TTFT 1K / prefill tok/s", ["cold_1000"]), ("Cold TTFT 4K / prefill tok/s (quick)", ["cold_4000"]),
        ("Cold TTFT 16K / prefill tok/s", ["cold_16000"]), ("Cold TTFT 64K / prefill tok/s", ["cold_64000"]),
        ("Cold TTFT 128K / prefill tok/s", ["cold_128000"]),
        ("Warm TTFT 1K / 16K / 64K / 128K (s)", ["warm_1000", "warm_16000", "warm_64000", "warm_128000"]),
        ("Decode at 64K / 128K context (tok/s)", ["decode_ctx_64000", "decode_ctx_128000"]),
        ("Incremental TTFT, 16K cached + 1K / 4K new (s)", ["incr_16000_1000", "incr_16000_4000"]),
        ("Incremental TTFT, 64K cached + 4K / 16K new (s)", ["incr_64000_4000", "incr_64000_16000"]),
        ("Incremental TTFT, 128K cached + 4K / 16K new (s)", ["incr_128000_4000", "incr_128000_16000"]),
        ("Incremental TTFT, 4K cached + 1K new (quick)", ["incr_4000_1000"]),
        ("Needle ~200K: correct / TTFT / prefill tok/s", ["needle200k"]),
        ("155K cold prefill tok/s, preemptions, min free GiB per card", ["cold155k"]),
        ("Concurrency 2 / 4 / 6: aggregate tok/s", ["conc_agg"]), ("Concurrency 2 / 4 / 6: per-stream decode median (tok/s)", ["conc_dec"]),
        ("Concurrency 2 / 4 / 6: TTFT p50-p99 (s)", ["conc_ttft"])]
out(f"\n== Scoreboard rows ({a.label}), notes-file order")
for name, keys in ROWS:
    if any(k in S for k in keys): out(f"| {name} | {' / '.join(S.get(k, 'n/a') for k in keys)} |")
tot = R["totals"]
out(f"\n  preemptions during run: {int(tot['num_preemptions'])}; whole-run acceptance {B.acceptance(tot)}")
out(f"json: {json_path}\nlog: {log_path}")
