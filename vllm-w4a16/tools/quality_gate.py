#!/usr/bin/env python3
# Quality gate for candidate lanes: lossless means temperature-0 outputs match the frozen baseline except for
# floating-point reordering. 20 frozen prompts (code, prose, tool calls, long-context QA, vision, thinking on/off) are
# sent non-streamed at temperature 0 with logprobs; --record freezes token ids, top-5 logprobs, text and tool calls;
# --compare reports per prompt IDENTICAL, or the first divergent token with the baseline's top1-top2 margin there and
# the logprob drift over the common prefix. Both modes also run the drop-in checks: every tool call parses, the needle
# at ~30K/~100K/~200K, a ~155K cold prefill (no error, preemptions, min free GiB per card), and the boot log's pool
# (>= 262,144 tokens), max_num_seqs 6, max_model_len 262,144, qwen3_xml, qwen3 reasoning, thinking off by default.
# Run it FIRST after a boot (it does its own warm-up): the prefix cache is part of the compute path, so the record and
# every compare should see the same cold cache. The per-prompt cache hits are stored and a mismatch is flagged.
# Usage (stdlib only):
#   URL=http://localhost:8090 python3 vllm-w4a16/tools/quality_gate.py --record [results/quality-baseline-2026-10-17.json]
#   URL=http://localhost:8090 python3 vllm-w4a16/tools/quality_gate.py --compare [baseline.json] --label R15
#   URL=... python3 vllm-w4a16/tools/quality_gate.py --rescore results/quality-compare-R15-....json   # new thresholds, no requests
#   ... --skip-long (no needles, no 155K)   ... --boot-log FILE | --container NAME (default vllm_qwen38fn_3090)
# Exit 0 = PASS, 1 = FAIL (a check failed), 2 = REVIEW (a divergence looks larger than floating-point reordering).
import argparse, base64, json, os, random, re, struct, subprocess, sys, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib as B
from benchlib import out

DEFAULT_BASELINE = os.path.join(B.RESULTS, "quality-baseline-2026-10-17.json")
# Calibrated on R13/R14 (journal/01-baseline.md): the prefix lane is not bit-reproducible at temperature 0 even on one
# boot. Over 3 same-lane passes (60 prompt-runs, 2,503 compared positions) first divergences sat at baseline margins of
# 0-0.5 nat with one at 1.25, the worst prefix drift was 0.451 nat, and the per-position |dlogprob| p99 was 0.122.
MARGIN_NAT = 2.0   # a first divergence where the baseline's top1-top2 margin exceeds this is SUSPECT
DRIFT_NAT = 1.0    # a chosen-token logprob drift above this over one prompt's common prefix is SUSPECT
P99_NAT = 0.25     # p99 of |dlogprob| over every common-prefix position of the run; about 2x the measured noise
POOL_MIN = 262144

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--record", nargs="?", const=DEFAULT_BASELINE, metavar="PATH")
g.add_argument("--compare", nargs="?", const=DEFAULT_BASELINE, metavar="PATH")
g.add_argument("--rescore", metavar="COMPARE_JSON", help="re-judge a saved compare report with the current thresholds (no requests)")
ap.add_argument("--baseline", default=DEFAULT_BASELINE, help="baseline for --rescore")
ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8090"))
ap.add_argument("--label", default="run")
ap.add_argument("--skip-long", action="store_true", help="skip the needles and the 155K prefill")
ap.add_argument("--no-warmup", action="store_true")
ap.add_argument("--boot-log", help="boot log file (default: docker logs of --container)")
ap.add_argument("--container", default=os.environ.get("NAME", "vllm_qwen38fn_3090"))
ap.add_argument("--out-dir", default=B.RESULTS)
a = ap.parse_args()
B.set_base(a.url)
mode = "record" if a.record else ("rescore" if a.rescore else "compare")
stamp = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(a.out_dir, exist_ok=True)
log_path = B.open_log(os.path.join(a.out_dir, f"quality-{mode}-{a.label}-{stamp}.log"))

# ---- frozen prompt set (never edit after the baseline is recorded; add new prompts under new ids)
def png_halves(w=64, h=32, left=(220, 30, 30), right=(30, 60, 220)):
    raw = b"".join(b"\x00" + bytes(left) * (w // 2) + bytes(right) * (w - w // 2) for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")

def doc(n_tokens, seed, facts):
    """Fixed-seed filler with facts planted at fixed fractions."""
    rng = random.Random(seed); words = B.filler_needle(n_tokens, rng).split(); n = len(words)
    for frac, fact in sorted(facts, reverse=True):
        words.insert(int(n * frac), fact)
    return " ".join(words)

BUGGY = '''def binary_search(xs: list[int], target: int) -> int:
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < target:
            lo = mid
        elif xs[mid] > target:
            hi = mid
        else:
            return mid
    return -1'''
NESTED = '''def ship_cost(order):
    if order is not None:
        if order["country"] == "US":
            if order["weight"] > 10:
                return 25
            else:
                if order["express"]:
                    return 15
                else:
                    return 8
        else:
            if order["weight"] > 10:
                return 60
            else:
                return 35
    else:
        return 0'''
SUMMARY_TEXT = ("The migration of the billing service from the old monolith started in March. The team first copied the invoice tables into "
                "a separate database and ran both systems side by side for six weeks, comparing every generated invoice. Two mismatches were "
                "found: rounding of tax on multi-currency orders, and a missing late fee for accounts suspended mid-cycle. Both were fixed in "
                "the new service before the cut-over. The cut-over itself happened on a Sunday night and took forty minutes, most of it spent "
                "replaying queued payment events. Since then, invoice generation time dropped from eleven minutes to under two, and on-call "
                "pages for billing fell by half. The remaining work is retiring the old cron jobs and moving the dunning emails to the new "
                "notification pipeline, planned for the next quarter.")
def sys_tools(user):
    return [{"role": "system", "content": B.SYSTEM}, {"role": "user", "content": user}]
FACTS30 = [(0.15, "Note: the backup relay is called Marigold."), (0.55, "Note: the maintenance window opens at 03:40."),
           (0.85, "Note: the audit lead's badge number is 7731.")]
FACTS16 = [(0.1, "Incident: disk full on node kestrel."), (0.4, "Incident: certificate expired on node heron."),
           (0.7, "Incident: fan failure on node osprey.")]

P = []  # (id, kind, messages, max_tokens, extra, expect)
P += [("code_func", "code", "Write a Python function `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]` that merges overlapping "
       "closed intervals, with a docstring and three doctest examples. Output only the code.", 384, {}, "def merge_intervals"),
      ("code_bugfix", "code", "This binary search never terminates for some inputs. Find the bug, then give the corrected function and a one-line explanation.\n\n"
       "```python\n" + BUGGY + "\n```", 384, {}, "mid + 1"),
      ("code_refactor", "code", "Refactor this function to use early returns and no nested if/else, keeping the behaviour identical. Output only the code.\n\n"
       "```python\n" + NESTED + "\n```", 384, {}, "def ship_cost"),
      ("code_sql", "code", "Write a PostgreSQL query that returns the top 5 customers by total order value over the last 30 days, given tables "
       "customers(id, name), orders(id, customer_id, created_at) and order_items(order_id, price, qty). Then explain it in two sentences.", 384, {}, "LIMIT 5"),
      ("code_bash", "code", "Write a bash script that prints the 10 largest files under the directory given as $1, ignoring any .git directory, "
       "with human-readable sizes. Use set -euo pipefail. Output only the script.", 384, {}, "set -euo pipefail")]
P += [("prose_explain", "prose", "Explain how a hash map handles collisions, in about 200 words.", 256, {}, None),
      ("prose_summary", "prose", SUMMARY_TEXT + "\n\nSummarize the text above in exactly three bullet points.", 256, {}, None),
      ("prose_email", "prose", "Write a short, polite email to a colleague asking to move Tuesday's design review to Thursday afternoon.", 256, {}, "Thursday"),
      ("prose_compare", "prose", "Compare TCP and UDP in a short markdown table, then add two sentences on when to use each.", 256, {}, "UDP")]
TOOL_EXTRA = {"tools": B.TOOLS, "tool_choice": "auto"}
P += [("tool_read", "tool", sys_tools("Read the file app/main.py so we can look at the routes."), 256, TOOL_EXTRA, "read_file"),
      ("tool_tests", "tool", sys_tools("Run the test suite, filtered to tests/test_orders.py."), 256, TOOL_EXTRA, "run_tests"),
      ("tool_search", "tool", sys_tools("Search the repository for the regex 'def create_order' and tell me where it is defined."), 256, TOOL_EXTRA, "search"),
      ("tool_diff", "tool", sys_tools("Show me the current git diff for app/services/orders.py."), 256, TOOL_EXTRA, "git_diff"),
      ("tool_write", "tool", sys_tools("Create the file scripts/hello.py containing a script that prints hello world. The file does not exist yet, so just write it."),
       256, TOOL_EXTRA, "write_file")]
P += [("long_qa_8k", "long", doc(8000, "qa8k", [(0.3, "Note: the spare lane token is 9182.")]) +
       "\n\nWhat is the spare lane token? Answer with the number only.", 64, {}, "9182"),
      ("long_list_16k", "long", doc(16000, "list16k", FACTS16) + "\n\nList every node that had an incident, in order of appearance, comma separated.", 64, {}, "kestrel"),
      ("long_qa_30k", "long", doc(30000, "qa30k", FACTS30) + "\n\nWhat is the backup relay called, and when does the maintenance window open? One sentence.",
       64, {}, "Marigold")]
IMG = "data:image/png;base64," + base64.b64encode(png_halves()).decode()
P += [("vision_halves", "vision", [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG}},
       {"type": "text", "text": "This image has two halves. What colour is the left half and what colour is the right half? Answer in five words or fewer."}]}],
       32, {}, "red"),
      ("think_off", "think", "What is 17 * 23? Answer with the number only.", 16, {}, "391"),
      ("think_on", "think", "What is 17 * 23? Answer with the number only.", 1024, {"thinking": True}, "391")]

# ---- helpers
def parse_lp(raw):
    lc = (raw["choices"][0].get("logprobs") or {}).get("content") or []
    tid = lambda t: int(t.split(":", 1)[1]) if isinstance(t, str) and t.startswith("token_id:") else t
    ids = [tid(e["token"]) for e in lc]; lps = [e["logprob"] for e in lc]
    top = [sorted(([tid(t["token"]), t["logprob"]] for t in e.get("top_logprobs") or []), key=lambda x: -x[1]) for e in lc]
    return ids, lps, top

def run_prompt(pid, kind, msgs, max_tokens, extra, expect):
    extra = dict(extra); thinking = extra.pop("thinking", False)
    r = B.chat(msgs, max_tokens, stream=False, thinking=thinking, logprobs=True, top_logprobs=5, return_tokens_as_token_ids=True, **extra)
    ids, lps, top = parse_lp(r["raw"])
    rec = {"kind": kind, "ids": ids, "lps": lps, "top": top, "text": r["text"], "reasoning": r["reasoning"], "finish": r["finish"],
           "tool_calls": r["tool_calls"], "pt": r["pt"], "ct": r["ct"], "hits": r["hits"], "checks": {}}
    ck = rec["checks"]
    if kind == "tool":
        names = [tc["function"]["name"] for tc in r["tool_calls"]]
        try: parsed = all(isinstance(json.loads(tc["function"]["arguments"]), dict) for tc in r["tool_calls"]); err = None
        except Exception as e: parsed = False; err = str(e)[:80]
        known = {t["function"]["name"] for t in B.TOOLS}
        ck["tool_parse"] = bool(r["tool_calls"]) and parsed and all(n in known for n in names)
        ck["tool_expected"] = expect in names
        rec["tool_names"] = names; rec["tool_err"] = err
    elif expect:
        ck["expected"] = expect.lower() in (r["text"] or "").lower()
    if pid == "think_off": ck["no_reasoning_by_default"] = not r["reasoning"].strip()
    if pid == "think_on": ck["reasoning_split"] = bool(r["reasoning"].strip())
    return rec

def compare(b, c):
    bt, ct_ = b["ids"], c["ids"]; n = min(len(bt), len(ct_))
    i = next((k for k in range(n) if bt[k] != ct_[k]), None)
    upto = n if i is None else i
    drift = max((abs(b["lps"][k] - c["lps"][k]) for k in range(upto)), default=0.0)
    res = {"len_base": len(bt), "len_cand": len(ct_), "prefix_drift": round(drift, 4)}
    if i is None and len(bt) == len(ct_):
        res["status"] = "IDENTICAL" if drift <= DRIFT_NAT else "SUSPECT"; return res
    if i is None: i = n  # one output is a prefix of the other (finish or length differs)
    res["first_divergence"] = i
    if i < len(bt):
        bt_top = b["top"][i]; res["base_token"] = bt[i]
        res["base_margin"] = round(bt_top[0][1] - bt_top[1][1], 4) if len(bt_top) > 1 else None
    if i < len(ct_):
        res["cand_token"] = ct_[i]
        cand_top = dict((t, lp) for t, lp in c["top"][i])
        res["cand_lp_of_base_token"] = round(cand_top[bt[i]], 4) if i < len(bt) and bt[i] in cand_top else None
        res["cand_margin"] = round(c["top"][i][0][1] - c["top"][i][1][1], 4) if len(c["top"][i]) > 1 else None
    m = res.get("base_margin")
    res["status"] = "SUSPECT" if (m is None or m > MARGIN_NAT or drift > DRIFT_NAT) else "FP-SIZED"
    return res

def prefix_dlps(b, c):
    n = min(len(b["ids"]), len(c["ids"]))
    i = next((k for k in range(n) if b["ids"][k] != c["ids"][k]), n)
    return [abs(b["lps"][k] - c["lps"][k]) for k in range(i)]
def p99(vals):
    v = sorted(vals); return v[min(len(v) - 1, int(0.99 * len(v)))] if v else 0.0

def boot_facts():
    try:
        if a.boot_log: txt = open(a.boot_log).read()
        else: p = subprocess.run(["docker", "logs", a.container], capture_output=True, text=True, timeout=60); txt = p.stdout + p.stderr
    except Exception as e:
        return {"error": f"no boot log: {e}"}
    f = {}
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", txt); f["pool_tokens"] = int(m[-1].replace(",", "")) if m else None
    for k, pat in [("max_num_seqs", r"'max_num_seqs': (\d+)"), ("max_model_len", r"'max_model_len': (\d+)"),
                   ("tool_call_parser", r"'tool_call_parser': '(\w+)'"), ("reasoning_parser", r"'reasoning_parser': '(\w+)'")]:
        m = re.search(pat, txt); f[k] = (int(m.group(1)) if m.group(1).isdigit() else m.group(1)) if m else None
    f["thinking_off_default"] = "'default_chat_template_kwargs': {'enable_thinking': False}" in txt
    f["checks"] = {"pool_ge_262144": (f["pool_tokens"] or 0) >= POOL_MIN, "max_num_seqs_6": f["max_num_seqs"] == 6,
                   "max_model_len_262144": f["max_model_len"] == 262144, "qwen3_xml": f["tool_call_parser"] == "qwen3_xml",
                   "qwen3_reasoning": f["reasoning_parser"] == "qwen3", "thinking_off_default": f["thinking_off_default"]}
    return f

NEEDLE = "The secret code for the 3090 lane is 4471-ROSEWOOD."
def long_checks():
    L = {}
    for tk in (30000, 100000, 200000):
        body = B.filler_needle(tk, random.Random(f"needle-{tk}")); cut = int(len(body) * 0.4)
        prompt = body[:cut] + " " + NEEDLE + " " + body[cut:] + "\n\nQuestion: What is the secret code for the 3090 lane? Answer with the code only."
        try:
            with B.GpuMemPoller() as poll: r = B.chat(prompt, 30)
            ok = "4471-ROSEWOOD" in r["text"]
            L[f"needle_{tk}"] = {"pt": r["pt"], "ttft": r["ttft"], "hits": r["hits"], "preemptions": int(r["d"]["num_preemptions"]),
                                 "min_free_gib": poll.min_free, "text": r["text"], "checks": {"correct": ok}}
            out(f"  needle {tk:>6}: {r['pt']:,} tok, TTFT {r['ttft']:.1f} s, hits {r['hits']:,}, preemptions {int(r['d']['num_preemptions'])}, "
                f"min free {poll.summary()}: {'CORRECT' if ok else 'WRONG: ' + r['text'].strip()[:80]}")
        except Exception as e:
            L[f"needle_{tk}"] = {"error": str(e)[:200], "checks": {"correct": False}}; out(f"  needle {tk}: ERROR {str(e)[:200]}")
    rng = random.Random(B.salt())
    prompt = f"Run {B.salt()}. " + B.filler_bench(155000, rng) + "\n\nSummarize the text above in one sentence."
    try:
        with B.GpuMemPoller() as poll: r = B.chat(prompt, 30)
        L["cold_155k"] = {"pt": r["pt"], "ttft": r["ttft"], "prefill_tps": r["pt"] / r["ttft"], "preemptions": int(r["d"]["num_preemptions"]),
                          "min_free_gib": poll.min_free, "checks": {"completed": r["ct"] > 0}}
        out(f"  155K cold: {r['pt']:,} tok, TTFT {r['ttft']:.1f} s ({r['pt']/r['ttft']:,.0f} tok/s), preemptions {int(r['d']['num_preemptions'])}, "
            f"min free {poll.summary()}")
    except Exception as e:
        L["cold_155k"] = {"error": str(e)[:200], "checks": {"completed": False}}; out(f"  155K cold: ERROR {str(e)[:200]}")
    return L

if mode == "rescore":
    base = json.load(open(a.baseline)); cand = json.load(open(a.rescore)); dl = []; sus = []; counts = {}
    out(f"rescore {a.rescore} against {a.baseline}; thresholds margin {MARGIN_NAT}, drift {DRIFT_NAT}, p99 {P99_NAT} nat")
    for pid, rec in cand["prompts"].items():
        b = base["prompts"].get(pid)
        if not b or "ids" not in b or "lps" not in rec: continue
        cmp = compare(b, rec); dl += prefix_dlps(b, rec); counts[cmp["status"]] = counts.get(cmp["status"], 0) + 1
        if cmp["status"] == "SUSPECT": sus.append(pid)
        out(f"  {pid:<15} {cmp['status']:<10} {str(cmp.get('first_divergence', '')):>5} {cmp.get('base_margin') or '':>7} {cmp['prefix_drift']:>7.3f}")
    if p99(dl) > P99_NAT: sus.append("aggregate_p99")
    v = "FAIL" if cand.get("fails") else ("REVIEW" if sus else "PASS")
    out(f"VERDICT {v} ({', '.join(f'{k} {n}' for k, n in sorted(counts.items()))}; p99 |dlogprob| {p99(dl):.3f} over {len(dl)} positions)"
        + (f"; SUSPECT {', '.join(sus)}" if sus else "") + (f"; FAIL {', '.join(cand['fails'])}" if cand.get("fails") else ""))
    sys.exit({"PASS": 0, "FAIL": 1, "REVIEW": 2}[v])

# ---- run
B.wait_ready()
info = B.server_info()
out(f"quality gate {mode} {time.strftime('%Y-%m-%d %H:%M:%S')}  {B.BASE}  label {a.label}  vllm {info.get('version')}")
base = json.load(open(a.compare)) if mode == "compare" else None
B.assert_idle("start")
if not a.no_warmup: B.warmup()
models = info.get("models") if isinstance(info.get("models"), list) else []
model_ok = any(m.get("id") == B.MODEL and m.get("max_model_len") == 262144 for m in models)
G = {"mode": mode, "label": a.label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "server": info, "boot": boot_facts(),
     "endpoint_checks": {"model_id_and_len": model_ok}, "prompts": {}, "thresholds": {"margin_nat": MARGIN_NAT, "drift_nat": DRIFT_NAT, "p99_nat": P99_NAT}}
out(f"  boot facts: { {k: v for k, v in G['boot'].items() if k != 'checks'} }")

fails, warns, suspects = [], [], []
def check(scope, checks, base_checks=None):
    for k, v in checks.items():
        if v: continue
        # content expectations are a regression only if the baseline met them; parse/drop-in checks always fail
        if k in ("expected", "tool_expected") and (mode == "record" or not (base_checks or {}).get(k, False)):
            warns.append(f"{scope}:{k}")
        else: fails.append(f"{scope}:{k}")
for k, v in G["boot"].get("checks", {"boot_log": False}).items():
    if not v: fails.append(f"boot:{k}")
check("endpoint", G["endpoint_checks"])

out(f"\n== Frozen prompts ({len(P)}), temperature 0, non-streamed, top-5 logprobs")
out(f"  {'id':<15} {'pt':>6} {'ct':>4} {'hits':>6}  {'status':<10} {'first div':>9} {'margin':>7} {'drift':>7}  checks")
for pid, kind, msgs, mt, extra, expect in P:
    try: rec = run_prompt(pid, kind, msgs, mt, extra, expect)
    except Exception as e:
        out(f"  {pid:<15} ERROR {str(e)[:150]}"); fails.append(f"{pid}:error"); G["prompts"][pid] = {"error": str(e)[:300]}; continue
    G["prompts"][pid] = rec
    st, fd, mg, dr = "recorded", "", "", ""
    if base is not None:
        b = base["prompts"].get(pid)
        if not b or "ids" not in b: st = "NO-BASE"; warns.append(f"{pid}:no_baseline")
        else:
            cmp = compare(b, rec); rec["compare"] = cmp; st = cmp["status"]
            fd = str(cmp.get("first_divergence", "")); mg = "" if cmp.get("base_margin") is None else f"{cmp['base_margin']:.3f}"
            dr = f"{cmp['prefix_drift']:.3f}"
            if st == "SUSPECT": suspects.append(pid)
            if b.get("hits") != rec["hits"]: rec["cache_state_differs"] = [b.get("hits"), rec["hits"]]
            if kind == "tool" and b.get("tool_names") != rec.get("tool_names"): warns.append(f"{pid}:tool_names_differ")
    check(pid, rec["checks"], (base or {}).get("prompts", {}).get(pid, {}).get("checks"))
    cs = " ".join(f"{k}={'ok' if v else 'NO'}" for k, v in rec["checks"].items())
    extra_note = f" cache {rec['cache_state_differs'][0]}->{rec['cache_state_differs'][1]}" if rec.get("cache_state_differs") else ""
    out(f"  {pid:<15} {rec['pt']:>6} {rec['ct']:>4} {rec['hits']:>6}  {st:<10} {fd:>9} {mg:>7} {dr:>7}  {cs}{extra_note}")

if not a.skip_long:
    out("\n== Long-context checks")
    G["long"] = long_checks()
    for k, v in G["long"].items(): check(k, v["checks"])
B.assert_idle("end")

dl = [x for pid, rec in G["prompts"].items() if "compare" in rec for x in prefix_dlps(base["prompts"][pid], rec)] if base else []
G["p99_dlogprob"] = p99(dl); G["compared_positions"] = len(dl)
if base and G["p99_dlogprob"] > P99_NAT: suspects.append("aggregate_p99")
verdict = "FAIL" if fails else ("REVIEW" if suspects else "PASS")
G["verdict"] = verdict; G["fails"] = fails; G["warns"] = warns; G["suspects"] = suspects
if mode == "record":
    path = a.record
else:
    path = os.path.join(a.out_dir, f"quality-compare-{a.label}-{stamp}.json")  # keeps the logprobs: noise analysis needs them
json.dump(G, open(path, "w"), indent=None, separators=(",", ":"))
counts = {}
for rec in G["prompts"].values():
    s = rec.get("compare", {}).get("status", "recorded" if "ids" in rec else "error"); counts[s] = counts.get(s, 0) + 1
out(f"\nVERDICT {verdict}  ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}; thresholds margin {MARGIN_NAT}, drift {DRIFT_NAT}, p99 {P99_NAT} nat)"
    + (f"; p99 |dlogprob| {G['p99_dlogprob']:.3f} over {len(dl)} positions" if base else ""))
if fails: out(f"  FAIL: {', '.join(fails)}")
if warns: out(f"  warn: {', '.join(warns)}")
if suspects: out(f"  SUSPECT (review the first divergent token): {', '.join(suspects)}")
out(f"json: {path}\nlog: {log_path}")
sys.exit({"PASS": 0, "FAIL": 1, "REVIEW": 2}[verdict])
