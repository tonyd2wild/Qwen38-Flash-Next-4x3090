#!/usr/bin/env python3
# Shared helpers for scoreboard.py and quality_gate.py (stdlib only), so both measure the way bench-host.py does:
# temperature 0, thinking off unless asked, TTFT at the client from the HTTP request to the first streamed content token,
# decode = completion tokens after the first / time after the first token, cache hits from /metrics around each request.
# Every measured request asserts the server is idle (num_requests_running and num_requests_waiting are 0) before and
# after, because the /metrics counters are server-wide: another client during a run would contaminate the numbers.
import json, os, random, re, shutil, subprocess, sys, threading, time, urllib.request

BASE = os.environ.get("URL", "http://localhost:8090").rstrip("/")
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

def set_base(url):
    global BASE
    BASE = url.rstrip("/")

# ---- logging: print and append to a log file
_logf = None
def open_log(path):
    global _logf
    _logf = open(path, "w")
    return path
def out(s=""):
    print(s, flush=True)
    if _logf: _logf.write(s + "\n"); _logf.flush()

# ---- prompt text
WORDS = ("the fleet ran a quiet audit of every relay and lane while the sparks idled under the aurora of a late summer sky "
         "and the ledger filled with rows about pools tokens seats and the slow patient work of measuring instead of guessing").split()
def filler_bench(n_tokens, rng):
    """bench-host.py's filler: about 1.05 tokens per word (used for the cold/warm TTFT rows)."""
    return " ".join(rng.choice(WORDS) for _ in range(int(n_tokens / 1.05)))
def filler_needle(n_tokens, rng):
    """needle_3090.py / prefix_cache_3090.py's filler (shuffled passes over WORDS). Those scripts assumed 0.75 tokens per
    word; this text measures about 1.05 (their "50K" prompt was 69,834 tokens), so the size here is in real tokens."""
    n = int(n_tokens / 1.05); words = []
    while len(words) < n: words.extend(rng.sample(WORDS, len(WORDS)))
    return " ".join(words[:n])
def salt():
    return "%012x" % random.SystemRandom().getrandbits(48)

# ---- /metrics
_LINE = re.compile(r'^(vllm:[A-Za-z0-9_:]+)(?:\{([^}]*)\})? ([-+0-9.eE]+|NaN|[+-]Inf)$', re.M)
def metrics():
    """Counters and gauges summed over label sets; per-position spec counters keyed 'name[pos]'; buckets skipped."""
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    m = {}
    for name, labels, val in _LINE.findall(txt):
        if name.endswith("_bucket") or name.endswith("_created"): continue
        try: v = float(val)
        except ValueError: continue
        pos = re.search(r'position="(\d+)"', labels or "")
        key = f"{name}[{pos.group(1)}]" if pos else name
        m[key] = m.get(key, 0.0) + v
    return m
def mget(m, name, default=0.0):
    return m.get("vllm:" + name, default)
def mdelta(m0, m1):
    """Counter deltas worth keeping per request: cache, preemptions, spec decode, latency histogram sums."""
    keys = ["prefix_cache_queries_total", "prefix_cache_hits_total", "num_preemptions_total",
            "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total",
            "time_to_first_token_seconds_sum", "request_queue_time_seconds_sum", "request_prefill_time_seconds_sum",
            "request_decode_time_seconds_sum", "request_inference_time_seconds_sum", "e2e_request_latency_seconds_sum"]
    d = {k.replace("_total", ""): mget(m1, k) - mget(m0, k) for k in keys}
    for k in m1:
        if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total["):
            pos = k[k.index("[") + 1:-1]
            d[f"accepted_pos{pos}"] = m1[k] - m0.get(k, 0.0)
    return d
def acceptance(d):
    """Mean acceptance length (1 + accepted/drafts) and the per-position rates from an mdelta() dict."""
    drafts = d.get("spec_decode_num_drafts", 0)
    if not drafts: return {"drafts": 0}
    pos = sorted((int(k[len("accepted_pos"):]), v) for k, v in d.items() if k.startswith("accepted_pos"))
    return {"drafts": int(drafts), "mean_len": round(1 + d.get("spec_decode_num_accepted_tokens", 0) / drafts, 3),
            "per_pos": [round(v / drafts, 3) for _, v in pos]}
def add_deltas(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}

def assert_idle(tag, wait_s=10.0):
    """Abort unless running and waiting are both 0. Polls for up to wait_s so a just-finished request can drain."""
    t_end = time.time() + wait_s
    while True:
        m = metrics(); run, wait = mget(m, "num_requests_running"), mget(m, "num_requests_waiting")
        if run == 0 and wait == 0: return m
        if time.time() > t_end:
            out(f"ABORT [{tag}]: server not idle (num_requests_running={run:g}, num_requests_waiting={wait:g}); "
                "another client is talking to the server, measurements would be contaminated")
            sys.exit(3)
        time.sleep(0.5)

# ---- requests
def wait_ready(tries=360):
    for i in range(tries):
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=5); return True
        except Exception:
            if i == 0: out(f"waiting for {BASE} ...")
            time.sleep(5)
    out("endpoint never came up"); sys.exit(1)

def server_info():
    info = {"url": BASE}
    try: info["models"] = json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=10).read())["data"]
    except Exception as e: info["models"] = str(e)
    try: info["version"] = json.loads(urllib.request.urlopen(BASE + "/version", timeout=10).read()).get("version")
    except Exception: info["version"] = None
    return info

def chat(messages, max_tokens, stream=True, thinking=False, check_idle=True, timeout=3600, **extra):
    """One chat request. Streamed: TTFT and decode as bench-host.py. Not streamed: the full JSON is returned in 'raw'.
    With check_idle the server must be idle before and after, and /metrics deltas are attached as 'd'."""
    if isinstance(messages, str): messages = [{"role": "user", "content": messages}]
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if stream: body.update(stream=True, stream_options={"include_usage": True})
    body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    m0 = assert_idle("before request") if check_idle else None
    t0 = time.time(); first = last = None; usage = {}; text = ""; reasoning = ""; tcs = {}; finish = None; raw = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not stream:
            raw = json.loads(r.read()); msg = raw["choices"][0]["message"]; usage = raw.get("usage") or {}
            text = msg.get("content") or ""; reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            finish = raw["choices"][0].get("finish_reason")
            tcs = {i: tc for i, tc in enumerate(msg.get("tool_calls") or [])}
        else:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d = json.loads(line[5:])
                if d.get("choices"):
                    ch = d["choices"][0]; delta = ch.get("delta", {})
                    if delta.get("content"):
                        now = time.time(); first = first or now; last = now; text += delta["content"]
                    reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                    for tc in delta.get("tool_calls") or []:
                        t = tcs.setdefault(tc.get("index", 0), {"function": {"name": "", "arguments": ""}})
                        f = tc.get("function") or {}
                        t["function"]["name"] += f.get("name") or ""; t["function"]["arguments"] += f.get("arguments") or ""
                    finish = ch.get("finish_reason") or finish
                if d.get("usage"): usage = d["usage"]
    t1 = time.time()
    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    res = {"pt": pt, "ct": ct, "wall": t1 - t0, "text": text, "reasoning": reasoning, "finish": finish,
           "tool_calls": [tcs[i] for i in sorted(tcs)]}
    if stream:
        res["ttft"] = (first or t1) - t0
        res["dec"] = (ct - 1) / (last - first) if first and last and last > first and ct > 1 else 0.0
    if raw is not None: res["raw"] = raw
    if check_idle:
        m1 = assert_idle("after request"); res["d"] = mdelta(m0, m1)
        res["hits"] = int(res["d"]["prefix_cache_hits"])
    return res

def warmup():
    """First requests after a boot JIT-compile Triton kernels (count-to-100 reads 82-86 tok/s instead of ~189). Not measured."""
    out("warm-up: short prompt, count to 100, one ~8K prompt")
    chat("Say OK.", 8)
    chat("Count from 1 to 100, separated by commas, nothing else.", 400)
    rng = random.Random(salt())
    chat(f"Warm-up {salt()}. " + filler_bench(8000, rng) + "\n\nSummarize the text above in one sentence.", 32)

# ---- GPU memory poller (nvidia-smi in the sandbox or on the host; optional)
class GpuMemPoller:
    def __init__(self, interval=0.5):
        self.interval = interval; self.min_free = None; self._stop = threading.Event(); self._t = None
        self.available = shutil.which("nvidia-smi") is not None
    def _poll(self):
        while not self._stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10).stdout
                free = [(float(t) - float(u)) / 1024 for u, t in (l.split(",") for l in o.strip().splitlines())]
                self.min_free = free if self.min_free is None else [min(a, b) for a, b in zip(self.min_free, free)]
            except Exception: pass
            self._stop.wait(self.interval)
    def __enter__(self):
        if self.available: self._t = threading.Thread(target=self._poll, daemon=True); self._t.start()
        return self
    def __exit__(self, *a):
        self._stop.set()
        if self._t: self._t.join(timeout=15)
    def summary(self):
        if not self.available: return "n/a (no nvidia-smi)"
        if not self.min_free: return "n/a"
        return " / ".join(f"{x:.2f}" for x in self.min_free) + " GiB"

# ---- agent-style prompt material (copied from sixagent_load.py, which runs on import and cannot be imported)
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": {
    "path": {"type": "string", "description": "absolute path"}, "content": {"type": "string", "description": "file content or command text"},
    "pattern": {"type": "string", "description": "regex"}, "line": {"type": "integer"}}, "required": ["path"]}}} for n, d in [
    ("read_file", "Read a file from the repository and return its contents with line numbers."),
    ("write_file", "Write the given content to a file, creating parent directories as needed."),
    ("edit_file", "Replace an exact string in a file with new content; fails if the string is not unique."),
    ("run_command", "Run a shell command in the project root and return stdout, stderr and the exit code."),
    ("search", "Search the repository for a regex pattern and return matching file paths and lines."),
    ("list_dir", "List files and directories under a path, one level deep."),
    ("git_diff", "Return the current git diff for the working tree or a given path."),
    ("run_tests", "Run the project's test suite, optionally filtered by a pattern, and return the summary.")]]
SYSTEM = ("You are a coding agent working inside a Python service repository (FastAPI + SQLAlchemy + pytest). Use the tools to inspect and modify files. "
          "Always read a file before editing it. Keep edits minimal. After changes, run the tests. Explain each step briefly, then call the next tool. "
          "Repository layout: app/main.py (routes), app/models.py (ORM), app/services/orders.py (business logic), app/db.py, tests/ (pytest), scripts/, README.md. "
          "Coding standards: type hints everywhere, no bare excepts, log with structlog, keep functions under 40 lines, docstrings in Google style. "
          "Never delete files. Never run destructive git commands. If a tool fails, read the error and try a different approach once before asking.")
TASKS = ["Add an idempotency key to POST /orders so repeated requests with the same key return the original order. Plan the change, then start with the models.",
         "The nightly job in scripts/reconcile.py double counts refunds when an order has partial shipments. Find the bug and propose a fix with a test.",
         "Add rate limiting to the public /search endpoint using a token bucket per API key, configurable via settings. Start by reading app/main.py.",
         "Migrate app/services/orders.py from synchronous SQLAlchemy sessions to async sessions without changing the public function signatures.",
         "Write a CLI in scripts/export_orders.py that streams orders as CSV for a date range, with pagination and a progress bar. Begin with the data model.",
         "Our tests take 9 minutes. Profile tests/ and propose three concrete changes to cut that in half, then implement the first one."]
