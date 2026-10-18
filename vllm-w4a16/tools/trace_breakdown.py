#!/usr/bin/env python3
# Sums a torch-profiler trace (the vLLM /start_profile output, *.pt.trace.json.gz) by kernel category for each kind of
# engine step, so every component can be set against its floor (results/perf-2026-10-notes.md, "Floor table").
# Steps come from the GPU-side execute_context_<n>(<tok>)_generation_<m>(<tok>) annotations: prefill chunks
# (context only, >= --min-chunk tokens), partial prefill, decode (generation only) and mixed. The annotation ends before
# the MTP draft and the sampler run, so a step is accounted over its period, from its start to the next step's start
# (the last step of a burst, with no successor within --max-gap-ms, is left out of the averages). Per step kind it
# reports the period, the annotated span, busy time (union of kernels, memcpys and memsets on all streams), idle, and
# per category the kernel time (and the part after the annotation: draft, sampler, input prep), calls, time per token. Idle gaps are attributed to the CPU work
# that ran on the launching thread between the end of the previous GPU activity and the launch of the next kernel
# (the innermost op at each instant), which separates "GPU waited for the host" from "GPU waited for a sync".
# Usage (stdlib only):
#   python3 vllm-w4a16/tools/trace_breakdown.py .cache/ple/prof-pc1                   # rank 0 of a profile directory
#   ... .cache/ple/prof-pc1 --rank all             # one summary line per rank (NCCL time includes waiting for peers)
#   ... path/to/trace.json.gz --json out.json      # also write the numbers as JSON
#   ... --kernels 25                               # list the top kernels per step kind with their category
import argparse, bisect, collections, glob, gzip, json, os, re, sys

CATS = [  # (category, regex on the kernel name); first match wins
    ("nccl_allreduce", r"ncclDevKernel_AllReduce|ncclKernel_AllReduce"),
    ("nccl_allgather", r"ncclDevKernel_AllGather|ncclKernel_AllGather"),
    ("nccl_other", r"nccl"),
    ("moe_marlin", r"[Mm]arlin"),
    ("moe_glue", r"moe_|topkGating|topk_softmax|act_and_mul|silu_and_mul|count_and_sort|moe_align"),
    ("qsa_topk", r"persistent_topk"),
    ("qsa_attn", r"_qsa_|reshape_and_cache|qsa"),
    ("gdn", r"chunk_|_causal_conv1d|recompute_w_u|merge_16x16|_fused_post_conv|solve_tril|fused_recurrent|gated_delta"
            r"|layer_norm_fwd|fused_gdn|causal_conv"),
    ("hc", r"_hc_"),
    ("gdn_decode", r"gdn_|recurrent"),
    ("ple", r"ple|ngram"),
    ("spec_sample", r"rejection|_resample|logits|sampl|argmax|gumbel|_topk_topp|softmax_kernel|cunn_SoftMax|"
                    r"log_softmax|draft|eagle|mtp"),
    ("gemm", r"gemm|cutlass|cublas|gemv|splitKreduce|s16816|s1688|sm80_xmma|ampere_"),
    ("norm_rope", r"rms_norm|rmsnorm|rope|rotary|layernorm|layer_norm"),
    ("elementwise", r"elementwise|vectorized|unrolled|copy|fill|index|reduce_kernel|cat|scatter|gather|where|"
                    r"arange|cumsum|sort|topk|radix|masked"),
]
CATS = [(c, re.compile(r)) for c, r in CATS]
ORDER = [c for c, _ in CATS] + ["other", "memcpy", "memset"]

ap = argparse.ArgumentParser()
ap.add_argument("path", help="a profile directory (picks rank files) or one *.pt.trace.json(.gz)")
ap.add_argument("--rank", default="0", help="rank number or 'all'")
ap.add_argument("--min-chunk", type=int, default=1024, help="context tokens at or above which a step is a full chunk")
ap.add_argument("--max-gap-ms", type=float, default=100.0, help="a longer gap to the next step ends a burst")
ap.add_argument("--min-gap-us", type=float, default=5.0, help="idle intervals shorter than this are not attributed")
ap.add_argument("--kernels", type=int, default=0, help="list the top N kernels per step kind")
ap.add_argument("--gaps", type=int, default=5, help="list the top N idle-gap patterns (previous -> next GPU op) per kind")
ap.add_argument("--json", help="write the summary as JSON here")
a = ap.parse_args()


def cat_of(name):
    for c, r in CATS:
        if r.search(name): return c
    return "other"


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f: return json.load(f)


STEP_RE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def step_kind(name):
    m = STEP_RE.search(name)
    if not m: return None, 0
    nctx, ctok, ngen, gtok = map(int, m.groups())
    if ctok and not gtok: return ("prefill_chunk" if ctok >= a.min_chunk else "prefill_partial"), ctok
    if gtok and not ctok: return "decode", gtok
    if ctok and gtok: return "mixed", ctok + gtok
    return "empty", 0


def union_len(iv):
    """Total length of the union of (start, end) intervals, plus the merged list."""
    tot, merged = 0.0, []
    for s, e in sorted(iv):
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]: merged[-1][1] = e
        else: merged.append([s, e])
    for s, e in merged: tot += e - s
    return tot, merged


class Innermost:
    """Innermost CPU event (cpu_op / cuda_runtime / cuda_driver) at any instant on one thread."""
    def __init__(self, evs):
        evs = sorted(evs, key=lambda e: (e["ts"], -e.get("dur", 0)))
        # sweep into flat segments of (start, end, innermost name)
        self.seg = []
        stack, t = [], None
        pts = []
        for e in evs:
            pts.append((e["ts"], 1, e)); pts.append((e["ts"] + e.get("dur", 0), 0, e))
        pts.sort(key=lambda p: (p[0], p[1]))  # ends before starts at the same instant
        for ts, kind, e in pts:
            if t is not None and ts > t:
                self.seg.append((t, ts, stack[-1]["name"] if stack else "(no cpu op: python / scheduler)"))
            if kind: stack.append(e)
            else:
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i] is e: del stack[i]; break
            t = ts
        self.starts = [s[0] for s in self.seg]

    def credit(self, t0, t1, acc):
        i = max(0, bisect.bisect_right(self.starts, t0) - 1)
        covered = 0.0
        while i < len(self.seg) and self.seg[i][0] < t1:
            s, e, n = self.seg[i]
            ov = min(e, t1) - max(s, t0)
            if ov > 0: acc[n] += ov; covered += ov
            i += 1
        if t1 - t0 - covered > 0: acc["(no cpu op: python / scheduler)"] += t1 - t0 - covered


def analyse(path):
    d = load(path)
    ev = d["traceEvents"]
    gpu_ann = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation" and STEP_RE.search(e.get("name", ""))),
                     key=lambda e: e["ts"])
    # cluster per-stream annotations of one step into one window
    steps = []
    for e in gpu_ann:
        s, t = e["ts"], e["ts"] + e["dur"]
        if steps and steps[-1]["name"] == e["name"] and s < steps[-1]["t1"]:
            steps[-1]["t1"] = max(steps[-1]["t1"], t); continue
        kind, tok = step_kind(e["name"])
        steps.append({"name": e["name"], "kind": kind, "tok": tok, "t0": s, "t1": t})
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    gpu.sort(key=lambda e: e["ts"])
    runtime = {}
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = e.get("args", {}).get("correlation")
            if c is not None: runtime[c] = e
    cpu_by_tid = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver") and "dur" in e:
            cpu_by_tid[e["tid"]].append(e)
    inner = {}
    starts = [g["ts"] for g in gpu]
    for i, st in enumerate(steps):
        nxt = steps[i + 1]["t0"] if i + 1 < len(steps) else None
        st["complete"] = nxt is not None and nxt - st["t1"] < a.max_gap_ms * 1e3
        st["p1"] = nxt if st["complete"] else st["t1"]
    for st in steps:
        lo = bisect.bisect_left(starts, st["t0"]); hi = bisect.bisect_left(starts, st["p1"])
        ks = gpu[lo:hi]
        st["n_kernels"] = sum(1 for k in ks if k["cat"] == "kernel")
        busy, merged = union_len([(k["ts"], k["ts"] + k["dur"]) for k in ks])
        st["span"] = st["p1"] - st["t0"]; st["ann"] = st["t1"] - st["t0"]; st["busy"] = busy
        cat = collections.Counter(); cnt = collections.Counter(); kern = collections.Counter(); kcnt = collections.Counter()
        post = collections.Counter()
        for k in ks:
            c = {"gpu_memcpy": "memcpy", "gpu_memset": "memset"}.get(k["cat"]) or cat_of(k["name"])
            cat[c] += k["dur"]; cnt[c] += 1; kern[(c, k["name"][:100])] += k["dur"]; kcnt[(c, k["name"][:100])] += 1
            if k["ts"] >= st["t1"]: post[c] += k["dur"]
        st["cat"], st["cnt"], st["kern"], st["kcnt"], st["post"] = cat, cnt, kern, kcnt, post
        # NCCL time not overlapped by any other GPU work (all-reduce is on the main stream here; this checks it)
        nc = [(k["ts"], k["ts"] + k["dur"]) for k in ks if k["cat"] == "kernel" and "nccl" in k["name"]]
        other = [(k["ts"], k["ts"] + k["dur"]) for k in ks if not (k["cat"] == "kernel" and "nccl" in k["name"])]
        nu, _ = union_len(nc); ou, _ = union_len(other); allu, _ = union_len(nc + other)
        st["nccl_exposed"] = allu - ou
        # idle gaps inside the span, attributed to the CPU work before the next kernel's launch
        gaps = collections.Counter(); idle_sync = 0.0; idle_host = 0.0; pat = collections.Counter()
        edges = [(st["t0"], st["t0"])] + [tuple(m) for m in merged] + [(st["p1"], st["p1"])]
        nxt_by_start = {k["ts"]: k for k in ks}
        end_of = {}
        for k in ks: end_of[k["ts"] + k["dur"]] = k
        for (s0, e0), (s1, e1) in zip(edges, edges[1:]):
            g = s1 - e0
            if g < a.min_gap_us: continue
            k = nxt_by_start.get(s1)
            pv = end_of.get(e0)
            pat[(pv["name"][:48] if pv else "(step start)", k["name"][:48] if k else "(step end)")] += g
            r = runtime.get(k.get("args", {}).get("correlation")) if k else None
            if not r:
                gaps["(unattributed)"] += g; continue
            launch = r["ts"]
            if launch <= e0:  # launched before the GPU went idle: the GPU waited on something else (a sync / event)
                idle_sync += g; gaps["(queued before idle: launch latency / event wait)"] += g; continue
            idle_host += g
            tid = r["tid"]
            if tid not in inner: inner[tid] = Innermost(cpu_by_tid[tid])
            tmp = collections.Counter(); inner[tid].credit(e0, min(launch, s1), tmp)
            if s1 > launch: tmp["(launch to start)"] += s1 - launch
            for n, v in tmp.items(): gaps[n] += v
        st["gaps"] = gaps; st["idle_host"] = idle_host; st["idle_sync"] = idle_sync; st["pat"] = pat
    return d, steps


def summarise(steps):
    by = collections.OrderedDict()
    for st in steps:
        if st["kind"] in (None, "empty"): continue
        by.setdefault(st["kind"], []).append(st)
    for kind in list(by):  # averages over complete periods when there are any
        if any(s["complete"] for s in by[kind]): by[kind] = [s for s in by[kind] if s["complete"]]
    out = {}
    for kind, ss in by.items():
        n = len(ss); tok = sum(s["tok"] for s in ss) / n
        mean = lambda f: sum(f(s) for s in ss) / n
        r = {"steps": n, "complete": all(s["complete"] for s in ss), "tokens_per_step": tok,
             "period_ms": mean(lambda s: s["span"]) / 1e3, "annotated_ms": mean(lambda s: s["ann"]) / 1e3,
             "busy_ms": mean(lambda s: s["busy"]) / 1e3, "kernels_per_step": mean(lambda s: s["n_kernels"]),
             "idle_ms": mean(lambda s: s["span"] - s["busy"]) / 1e3,
             "idle_host_ms": mean(lambda s: s["idle_host"]) / 1e3, "idle_sync_ms": mean(lambda s: s["idle_sync"]) / 1e3,
             "nccl_exposed_ms": mean(lambda s: s["nccl_exposed"]) / 1e3,
             "tok_per_s": tok / (mean(lambda s: s["span"]) / 1e6), "cat": {}, "gaps": {}, "kernels": []}
        for c in ORDER:
            t = sum(s["cat"][c] for s in ss) / n
            if t: r["cat"][c] = {"ms": t / 1e3, "calls": sum(s["cnt"][c] for s in ss) / n,
                                 "after_ann_ms": sum(s["post"][c] for s in ss) / n / 1e3,
                                 "us_per_tok": t / tok if tok else 0}
        g = collections.Counter()
        for s in ss: g.update(s["gaps"])
        r["gaps"] = {k: v / n / 1e3 for k, v in g.most_common(12)}
        pt = collections.Counter()
        for s in ss: pt.update(s["pat"])
        r["gap_patterns"] = [{"prev": p, "next": q, "ms": v / n / 1e3} for (p, q), v in pt.most_common(a.gaps)]
        kk = collections.Counter(); kc = collections.Counter()
        for s in ss: kk.update(s["kern"]); kc.update(s["kcnt"])
        r["kernels"] = [{"cat": c, "name": nm, "ms": v / n / 1e3, "calls": kc[(c, nm)] / n}
                        for (c, nm), v in kk.most_common(a.kernels or 40)]
        out[kind] = r
    return out


def report(path, steps, S):
    print(f"== {path}")
    for kind, r in S.items():
        print(f"\n[{kind}] {r['steps']} steps x {r['tokens_per_step']:.0f} tok   period {r['period_ms']:.2f} ms "
              f"({r['tok_per_s']:.0f} tok/s{'' if r['complete'] else ', INCOMPLETE: no next step'})   "
              f"annotated {r['annotated_ms']:.2f}   busy {r['busy_ms']:.2f}   idle {r['idle_ms']:.2f} "
              f"(host-starved {r['idle_host_ms']:.2f}, queued {r['idle_sync_ms']:.2f})   "
              f"{r['kernels_per_step']:.0f} kernels/step   NCCL not overlapped {r['nccl_exposed_ms']:.2f} ms")
        print(f"  {'category':<16}{'ms/step':>10}{'after ann':>10}{'calls':>9}{'us/tok':>9}{'% busy':>8}")
        for c, v in r["cat"].items():
            print(f"  {c:<16}{v['ms']:>10.2f}{v['after_ann_ms']:>10.2f}{v['calls']:>9.0f}{v['us_per_tok']:>9.2f}"
                  f"{100*v['ms']/r['busy_ms']:>7.1f}%")
        if r["gaps"]:
            print("  idle attributed to (ms/step):")
            for k, v in r["gaps"].items():
                if v >= 0.005: print(f"    {v:8.2f}  {k}")
        if r["gap_patterns"]:
            print("  largest idle patterns (ms/step): previous GPU op -> next GPU op")
            for g in r["gap_patterns"]:
                if g["ms"] >= 0.005: print(f"    {g['ms']:8.2f}  {g['prev']}  ->  {g['next']}")
        if a.kernels:
            print("  top kernels (ms/step, calls/step):")
            for k in r["kernels"][:a.kernels]:
                print(f"    {k['ms']:8.2f} {k['calls']:6.0f}  {k['cat']:<14} {k['name']}")


paths = []
if os.path.isdir(a.path):
    files = sorted(glob.glob(os.path.join(a.path, "*rank*.pt.trace.json*")))
    if a.rank == "all": paths = files
    else:
        paths = [f for f in files if re.search(rf"_rank{a.rank}\.", f)]
    if not paths: sys.exit(f"no rank trace in {a.path}")
else: paths = [a.path]

allout = {}
for p in paths:
    d, steps = analyse(p)
    S = summarise(steps)
    allout[os.path.basename(p)] = S
    if a.rank == "all" and len(paths) > 1:
        line = "  ".join(f"{k}: period {r['period_ms']:.1f} busy {r['busy_ms']:.1f} allreduce "
                         f"{r['cat'].get('nccl_allreduce', {}).get('ms', 0):.1f}" for k, r in S.items())
        print(f"{os.path.basename(p)[:30]}  {line}")
    else:
        report(p, steps, S)
if a.json:
    with open(a.json, "w") as f: json.dump(allout, f, indent=1)
