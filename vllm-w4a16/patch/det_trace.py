# Bit-exact activation tracer for T14 (journal/16-nondeterminism.md). A vLLM general plugin: the entry point in
# patch/qwen4exp_det_trace-0.1.dist-info names register(), which vLLM calls in every process (worker_base.py:271 for
# the workers). Off unless QWEN4EXP_DET_TRACE=1. It wraps nn.Module.__call__ (global module hooks do not see keyword
# arguments, and vLLM passes most inputs by keyword) and, while /root/.cache/det-trace/on exists, opens a record for
# every top-level module call (the model forward, compute_logits, the MTP drafter) that runs eagerly, outside CUDA-graph
# capture, with at most QWEN4EXP_DET_TRACE_MAX_TOKENS tokens. Inside a record it fingerprints the tensor inputs and
# outputs of every module call, in call order: the bytes as int32 widened to int64, wrapped sums of x, x*x and x times a
# 1..1024 ramp, and per row (first dimension = tokens) for the first 64 rows. The fingerprints are computed on the
# current stream right after the module, so they see what the next kernel reads. One torch.cuda.synchronize() per
# record, then a JSON line in <dir>/rank<R>-pid<P>.jsonl. QWEN4EXP_DET_TRACE_FUNCS="mod:attr,mod:Class.attr,..." also
# wraps those functions (the custom ops inside a module, e.g. vllm._custom_ops:moe_sum), patched when the first record
# opens: their tensor arguments before the call ("in"), the same arguments after it ("post", for ops that write into an
# output argument) and the return value ("out"). QWEN4EXP_DET_TRACE_QSA=1 adds two views to a wrapped
# qsa_indexer._prefill_logits, whose buffers otherwise always differ through stale regions: "view.k" (the first
# request's compressed-key pages gathered through its page table, in logical order, up to the last visible block) and
# "view.logits" (the scores of each row's visible columns only; the rest set to 0). Reads tensors only; changes no
# values. QWEN4EXP_DET_TRACE_DUMP="<label suffix>:<min visible>:<calls>" (e.g. "_topk:8192:0,1,2") saves, on rank 0, the
# wrapped qsa_indexer._topk's valid scores, visible counts and returned selection to <dir>/dump/ for the listed calls
# of a record (in call order) whose largest visible count exceeds the minimum. Diff with
# tools/det_trace_diff.py. Mounted by DET_TRACE=1 in launch/qwen38fn-w4a16-3090-tp4.sh.
import json
import os
import time

_DIR = os.environ.get("QWEN4EXP_DET_TRACE_DIR", "/root/.cache/det-trace")
_MAX_TOKENS = int(os.environ.get("QWEN4EXP_DET_TRACE_MAX_TOKENS", "512"))
_FUNCS = [f for f in os.environ.get("QWEN4EXP_DET_TRACE_FUNCS", "").split(",") if f]
_QSA_VIEWS = os.environ.get("QWEN4EXP_DET_TRACE_QSA") == "1"
_DUMP = os.environ.get("QWEN4EXP_DET_TRACE_DUMP", "")
_ROWS = 64
_CHUNK = 1 << 20  # int32 elements per fingerprint chunk (4 MiB), so the int64 temporaries stay small next to a prefill

_installed = False
_depth = 0
_rec = None  # open record: dict(root, ntok, entries=[(name, key, shape, fp_tensor | list | str)])
_names: dict[int, str] = {}
_named_roots: set[int] = set()
_seq = 0
_on_cache = [0.0, False]
_ramp: dict = {}
_funcs_patched = False


def _sentinel_on() -> bool:
    now = time.monotonic()
    if now - _on_cache[0] > 0.5:
        _on_cache[0] = now
        _on_cache[1] = os.path.exists(os.path.join(_DIR, "on"))
    return _on_cache[1]


def _ramp_for(device):
    import torch
    r = _ramp.get(device)
    if r is None:
        r = _ramp[device] = torch.arange(1, 1025, dtype=torch.int64, device=device)
    return r


def _fp(t, ntok):
    """[s1, s2, s3, row_0, ..., row_k] as one int64 tensor on t's device."""
    import torch
    x = t.detach()
    if not x.is_contiguous():
        x = x.contiguous()
    u = x.reshape(-1).view(torch.uint8)
    pad = (-u.numel()) % 4
    if pad:
        u = torch.cat([u, u.new_zeros(pad)])
    w = u.view(torch.int32)
    ramp = _ramp_for(w.device)
    zero = torch.zeros((), dtype=torch.int64, device=w.device)
    s1, s2, s3 = zero, zero, zero
    for i in range(0, w.numel(), _CHUNK):
        c = w[i:i + _CHUNK].to(torch.int64)
        s1 = s1 + c.sum()
        s2 = s2 + (c * c).sum()
        m = c.numel() // 1024 * 1024
        s3 = s3 + (c[:m].view(-1, 1024) * ramp).sum() + (c[m:] * ramp[:c.numel() - m]).sum()
    parts = [torch.stack([s1, s2, s3])]
    if ntok and x.dim() >= 1 and x.shape[0] == ntok:
        rows = x.reshape(ntok, -1)[:_ROWS]
        rb = rows.contiguous().view(torch.uint8)
        if rb.shape[1] % 4 == 0 and rb.shape[1] > 0:
            ri = rb.view(torch.int32)
            for j in range(0, ri.shape[0], 8):
                c = ri[j:j + 8].to(torch.int64)
                parts.append((c * c).sum(1) * 3 + c.sum(1))
    return torch.cat(parts)


def _tensors(obj, key, out, depth=0):
    import torch
    if isinstance(obj, torch.Tensor):
        out.append((key, obj))
    elif depth < 2 and isinstance(obj, (tuple, list)):
        for i, o in enumerate(obj):
            _tensors(o, f"{key}.{i}", out, depth + 1)
    elif depth < 2 and isinstance(obj, dict):
        for k, o in obj.items():
            _tensors(o, f"{key}.{k}", out, depth + 1)


def _record(name, kind, obj):
    ts = []
    _tensors(obj, kind, ts)
    for key, t in ts:
        try:
            v = _fp(t, _rec["ntok"]) if t.numel() > 0 else "empty"
        except Exception as e:  # never break the forward
            v = f"error {type(e).__name__}: {e}"
        _rec["entries"].append((name, key, list(t.shape), str(t.dtype).replace("torch.", ""), v))


def _first_tensor(args, kwargs):
    import torch
    for k in ("input_ids", "hidden_states", "x"):
        if isinstance(kwargs.get(k), torch.Tensor):
            return kwargs[k]
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch.Tensor):
            return a
    return None


def _qsa_views(label, args, kwargs, out):
    import torch
    q, k_cache, page_table, _qsl, visible_blocks = args[:5]
    off = kwargs.get("query_offset", args[6] if len(args) > 6 else 0)
    n = kwargs.get("num_queries", args[7] if len(args) > 7 else q.shape[0])
    vis = visible_blocks[off:off + n].long()
    max_vis = int(vis.max()) if vis.numel() else 0
    npages = -(-max_vis // k_cache.shape[1])
    view_k = k_cache[page_table[0, :npages].long()].reshape(-1, *k_cache.shape[2:])[:max_vis]
    cols = torch.arange(max_vis, device=out.device)
    view_l = torch.where(cols[None, :] < vis[:, None], out[:, :max_vis], torch.zeros((), dtype=out.dtype, device=out.device))
    _record(label, "view.k", view_k)
    _record(label, "view.logits", view_l)


def _dump_topk(label, args):
    import torch
    suffix, min_vis, calls = _DUMP.split(":")
    if not label.endswith(suffix) or not torch.distributed.is_initialized() or torch.distributed.get_rank() != 0:
        return
    n = _rec.setdefault("dump_calls", 0)
    _rec["dump_calls"] = n + 1
    logits, vis, block_indices = args[0], args[1], args[4]
    max_vis = int(vis.max()) if vis.numel() else 0
    if max_vis <= int(min_vis) or str(n) not in calls.split(","):
        return
    os.makedirs(os.path.join(_DIR, "dump"), exist_ok=True)
    path = os.path.join(_DIR, "dump", f"pid{os.getpid()}-rec{_seq + 1}-call{n}.pt")
    torch.save({"logits": logits[:, :max_vis].cpu(), "vis": vis.cpu(), "block_indices": block_indices.cpu(),
                "columns": logits.shape[1], "call": n, "t": time.time()}, path)


def _wrap_fn(label, fn):
    def wrapped(*args, **kwargs):
        if _rec is None:
            return fn(*args, **kwargs)
        _record(label, "in", (args, kwargs))
        out = fn(*args, **kwargs)
        _record(label, "post", (args, kwargs))
        _record(label, "out", out)
        if _DUMP:
            try:
                _dump_topk(label, args)
            except Exception as e:
                print(f"[det_trace] dump failed: {type(e).__name__}: {e}", flush=True)
        if _QSA_VIEWS and label.endswith("_prefill_logits"):
            try:
                _qsa_views(label, args, kwargs, out)
            except Exception as e:
                _rec["entries"].append((label, "view.error", [], "", f"{type(e).__name__}: {e}"))
        return out
    return wrapped


def _patch_funcs():
    import importlib
    global _funcs_patched
    _funcs_patched = True
    for spec in _FUNCS:
        mod, _, attr = spec.partition(":")
        try:
            owner = importlib.import_module(mod)
            *path, name = attr.split(".")
            for p in path:
                owner = getattr(owner, p)
            setattr(owner, name, _wrap_fn("fn:" + spec, getattr(owner, name)))
            print(f"[det_trace] pid {os.getpid()}: wrapped {spec}", flush=True)
        except Exception as e:
            print(f"[det_trace] pid {os.getpid()}: cannot wrap {spec}: {type(e).__name__}: {e}", flush=True)


def _open(module, args, kwargs):
    import torch
    global _rec
    if not _sentinel_on() or torch.cuda.is_current_stream_capturing():
        return False
    if _FUNCS and not _funcs_patched:
        _patch_funcs()
    t = _first_tensor(args, kwargs)
    ntok = int(t.shape[0]) if t is not None and t.dim() >= 1 else 0
    if ntok > _MAX_TOKENS:
        return False
    root = type(module).__name__
    if id(module) not in _named_roots:
        _named_roots.add(id(module))
        for n, m in module.named_modules():
            _names.setdefault(id(m), n or root)
    _rec = {"root": root, "ntok": ntok, "t0": time.time(), "entries": []}
    return True


def _close():
    import torch
    global _rec, _seq
    rec, _rec = _rec, None
    torch.cuda.synchronize()
    gpu = [e[4] for e in rec["entries"] if isinstance(e[4], torch.Tensor)]
    flat = torch.cat([g.to("cpu") for g in gpu]).tolist() if gpu else []
    pos, ents = 0, []
    for name, key, shape, dtype, v in rec["entries"]:
        if isinstance(v, torch.Tensor):
            n = v.numel()
            vals, pos = flat[pos:pos + n], pos + n
            ents.append([name, key, shape, dtype, vals[:3], vals[3:]])
        else:
            ents.append([name, key, shape, dtype, v, []])
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    _seq += 1
    os.makedirs(_DIR, exist_ok=True)
    line = {"rank": rank, "pid": os.getpid(), "seq": _seq, "root": rec["root"], "ntok": rec["ntok"],
            "t0": rec["t0"], "t1": time.time(), "entries": ents}
    with open(os.path.join(_DIR, f"rank{rank}-pid{os.getpid()}.jsonl"), "a") as f:
        f.write(json.dumps(line) + "\n")


def register():
    global _installed
    if os.environ.get("QWEN4EXP_DET_TRACE") != "1" or _installed:
        return
    import torch
    orig = torch.nn.Module.__call__

    def _call(self, *args, **kwargs):
        global _depth
        if _rec is None:
            if _depth or not _open(self, args, kwargs):
                _depth += 1
                try:
                    return orig(self, *args, **kwargs)
                finally:
                    _depth -= 1
            name = _names.get(id(self), type(self).__name__)
            _record(name, "in", (args, kwargs))
            _depth += 1
            try:
                out = orig(self, *args, **kwargs)
                _record(name, "out", out)
            finally:
                _depth -= 1
                _close()
            return out
        name = _names.get(id(self), type(self).__name__)
        _record(name, "in", (args, kwargs))
        _depth += 1
        try:
            out = orig(self, *args, **kwargs)
        finally:
            _depth -= 1
        _record(name, "out", out)
        return out

    torch.nn.Module.__call__ = _call
    _installed = True
    print(f"[det_trace] installed in pid {os.getpid()}: dir {_DIR}, max tokens {_MAX_TOKENS}", flush=True)
