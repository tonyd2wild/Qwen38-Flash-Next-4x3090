#!/usr/bin/env python3
# Compares the records of patch/det_trace.py (T14, journal/16-nondeterminism.md): groups them by rank, root module,
# token count and the fingerprint of the root's inputs (the same request sent again), and against the first record of
# each group prints the first module call whose fingerprint differs, whether it is an input (the divergence happened in
# code between module calls) or an output (inside the module, if its inputs matched), which of the first 64 rows
# (tokens) differ, and how many entries differ in all. Ends with the count of first divergences per module name with
# layer numbers folded, over all groups and ranks. --outputs looks only at "out" and "post" entries for the first
# divergence: an output buffer passed in as an argument (wrapped custom ops, the QSA indexer) holds stale bytes before
# the call, so its "in" entry differs without meaning anything; the inputs of that call are then listed with their
# status. Stdlib only.
# Usage: python3 vllm-w4a16/tools/det_trace_diff.py [DIR=.cache/ple/det-trace] [--root Qwen4ExpForConditionalGeneration]
#        [--after UNIX_TIME] [--before UNIX_TIME] [--show 8] [--rank R] [--outputs]
import argparse, collections, glob, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("dir", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.cache/ple/det-trace"))
ap.add_argument("--root", default="")
ap.add_argument("--after", type=float, default=0.0)
ap.add_argument("--before", type=float, default=1e12)
ap.add_argument("--rank", type=int, default=None)
ap.add_argument("--show", type=int, default=8, help="differing entries listed per record after the first")
ap.add_argument("--outputs", action="store_true", help="first divergence among out/post entries only")
a = ap.parse_args()

recs = []
for f in sorted(glob.glob(os.path.join(a.dir, "rank*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if a.after <= r["t0"] <= a.before and (not a.root or r["root"] == a.root) and (a.rank is None or r["rank"] == a.rank):
            recs.append(r)
recs.sort(key=lambda r: (r["rank"], r["t0"]))


def sig(r):
    e = r["entries"]
    head = [x for x in e[:8] if x[0] == e[0][0] and x[1].startswith("in")] if e else []
    return (r["rank"], r["root"], r["ntok"], json.dumps([x[1:5] for x in head]))


def fold(name):
    return re.sub(r"\.\d+\.", ".N.", "." + name + ".").strip(".")


groups = collections.OrderedDict()
for r in recs:
    groups.setdefault(sig(r), []).append(r)
firsts = collections.Counter()
print(f"{len(recs)} records in {len(groups)} groups from {a.dir}")
for (rank, root, ntok, _), rs in groups.items():
    if len(rs) < 2:
        continue
    ref = rs[0]
    distinct = len({json.dumps([x[4:] for x in r["entries"]]) for r in rs})
    print(f"\n== rank {rank} {root} ntok {ntok}: {len(rs)} records, {distinct} distinct, {len(ref['entries'])} entries each")
    for k, r in enumerate(rs[1:], 1):
        ea, eb = ref["entries"], r["entries"]
        if len(ea) != len(eb):
            print(f"  rec {k}: entry count {len(eb)} != {len(ea)}")
        diffs = [i for i, (x, y) in enumerate(zip(ea, eb)) if (x[4:] != y[4:] or x[:4] != y[:4])
                 and (not a.outputs or not x[1].startswith("in"))]
        if not diffs:
            print(f"  rec {k}: identical")
            continue
        i = diffs[0]; x, y = ea[i], eb[i]
        rows = [j for j, (p, q) in enumerate(zip(x[5], y[5])) if p != q]
        prev = ea[i - 1] if i else None
        print(f"  rec {k}: {len(diffs)} of {len(ea)} entries differ; first #{i} {x[0]} {x[1]} {x[2]} {x[3]}"
              f" rows {rows[:12]}{'...' if len(rows) > 12 else ''} ({len(rows)} of {len(x[5])})"
              f"; previous equal entry: {prev[0] + ' ' + prev[1] if prev else '-'}")
        if a.outputs:
            j = i - 1
            while j >= 0 and ea[j][0] == x[0] and not ea[j][1].startswith("in"):
                j -= 1
            while j >= 0 and ea[j][0] == x[0] and ea[j][1].startswith("in"):
                st = "same" if ea[j][4:] == eb[j][4:] else "DIFF"
                print(f"      input of that call: #{j} {ea[j][1]} {ea[j][2]} {ea[j][3]} {st}")
                j -= 1
        for j in diffs[1:1 + a.show]:
            z = ea[j]; zr = [t for t, (p, q) in enumerate(zip(z[5], eb[j][5])) if p != q]
            print(f"      #{j} {z[0]} {z[1]} {z[2]} rows {zr[:8]}{'...' if len(zr) > 8 else ''} ({len(zr)})")
        firsts[(fold(x[0]), x[1])] += 1
print("\nfirst divergence by module (layer numbers folded), all groups and ranks:")
for (n, key), c in firsts.most_common(20):
    print(f"  {c:4d}  {n} {key}")
