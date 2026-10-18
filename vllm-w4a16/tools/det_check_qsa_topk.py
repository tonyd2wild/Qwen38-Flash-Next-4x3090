#!/usr/bin/env python3
# Checks the QSADET overlay (patch/qsa_indexer.py, T14, journal/16-nondeterminism.md) with no server running: on
# synthetic scores (random, bf16-rounded, relu-like with exact-zero ties, stale or NaN columns past the visible ones,
# prefill and decode row counts, 2.1K-260K context) and, if /dump holds R34's dumps (.cache/ple/det-trace-r34/dump),
# on real indexer scores, it compares QSADET's _topk with a torch reference (stable descending sort of the same keys,
# first 512, ascending; rows with at most 512 visible blocks arange then -1), counts distinct outputs over repeats,
# counts rows where the image's persistent_topk returns a set that is not the top 512, and times both. The last part
# runs 32-row slices of the dumps through the decode-size path (persistent_topk hint, checked by counts) and times 12
# calls in a CUDA graph at decode sizes.
# Usage: docker run --rm --device nvidia.com/gpu=0 -e QWEN4EXP_QSA_DET=1 \
#   -v $PWD/vllm-w4a16/patch/qsa_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py:ro \
#   -v $PWD/.cache/ple/det-trace-r34/dump:/dump:ro -v $PWD/vllm-w4a16/tools:/tools:ro --entrypoint python3 \
#   qwen38fn-vllm:8a728663 /tools/det_check_qsa_topk.py
import glob, json, time, torch
import vllm._C_stable_libtorch  # noqa: F401
import vllm.models.qwen4_exp.nvidia.ops.qsa_indexer as qi
K = 512
ws = torch.empty((qi._TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device="cuda")
def keys(x):
    b = x.view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(b >= 0x80000000, 0xFFFFFFFF - b, b + 0x80000000)
def reference(lg, vis):
    rows, cols = lg.shape
    vis_l = vis.long()
    k = keys(lg).masked_fill(torch.arange(cols, device=lg.device)[None, :] >= vis_l[:, None], -1)
    order = torch.sort(k, dim=1, descending=True, stable=True).indices[:, :K]
    ref = torch.sort(order.to(torch.int32), 1).values
    short = vis_l <= K
    ar = torch.arange(K, device=lg.device, dtype=torch.int32)[None, :].expand(rows, K)
    ref[short] = torch.where(ar[short] < vis[short, None], ar[short], torch.full_like(ar[short], -1))
    return ref
def run(lg, vis, det):
    qi._QSA_DET = det
    o = torch.randint(-2**31, 2**31 - 1, (lg.shape[0], K), dtype=torch.int32, device="cuda")
    qi._topk(lg, vis, 2048, 4, o, ws); return o
def case(name, lg, vis, reps=5):
    ref = reference(lg, vis)
    outs = [run(lg, vis, True) for _ in range(reps)]
    img = run(lg, vis, False)
    full = vis.long() > K
    kth = torch.sort(keys(lg).masked_fill(torch.arange(lg.shape[1], device="cuda")[None, :] >= vis.long()[:, None], -1), 1, descending=True).values[:, K - 1]
    img_inexact = int(((keys(lg).gather(1, img.long().clamp(min=0)) < kth[:, None]).any(1) & full).sum())
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): run(lg, vis, False)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    for _ in range(10): run(lg, vis, True)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    print(json.dumps({"case": name, "rows": lg.shape[0], "max_vis": int(vis.max()), "distinct": len({o.cpu().numpy().tobytes() for o in outs}),
                      "equals_reference": bool(torch.equal(outs[0], ref)), "image_inexact_rows": img_inexact,
                      "image_ms": round((t1 - t) / 10 * 1e3, 3), "qsadet_ms": round((t2 - t1) / 10 * 1e3, 3)}), flush=True)
COLS = 65728
g = torch.Generator(device="cuda").manual_seed(0)
for name, rows, ctx0, kind in [("prefill 2.1K", 439, 1664, "randn"), ("prefill 16K", 1664, 14300, "randn"), ("prefill 35K", 1664, 33280, "randn"),
                               ("prefill 64K", 1664, 62000, "randn"), ("prefill 260K", 1664, 260000, "randn"), ("decode 4 rows at 128K", 4, 131000, "randn"),
                               ("decode 24 rows at 250K", 24, 250000, "randn"), ("prefill 35K bf16", 1664, 33280, "bf16"), ("prefill 35K relu 300 positive", 1664, 33280, "zeros"),
                               ("mixed short/full rows", 1664, 1200, "randn")]:
    vis = ((torch.arange(rows, device="cuda") + ctx0) // 4 + 1).clamp(max=COLS).to(torch.int32)
    lg = torch.randn(rows, COLS, device="cuda", generator=g)
    if kind == "bf16": lg = lg.to(torch.bfloat16).float()
    if kind == "zeros":
        lg = torch.where(torch.rand(rows, COLS, device="cuda", generator=g) < 300 / 8500, lg.abs() + 0.01, torch.zeros_like(lg))
    beyond = torch.arange(COLS, device="cuda")[None, :] >= vis.long()[:, None]
    lg[beyond] = float("nan") if kind == "randn" else 1e30
    case(name, lg, vis)
for f in sorted(glob.glob("/dump/*.pt"))[:6]:
    d = torch.load(f)
    full = torch.full((d["logits"].shape[0], d["columns"]), float("nan"), device="cuda"); full[:, :d["logits"].shape[1]] = d["logits"].cuda()
    case("real " + f.split("/")[-1], full, d["vis"].cuda(), reps=5)
# hint path on real deep rows: 32-row slices around rows where the image's kernel was inexact
for f in sorted(glob.glob("/dump/*.pt"))[:8]:
    d = torch.load(f)
    full = torch.full((d["logits"].shape[0], d["columns"]), float("nan"), device="cuda"); full[:, :d["logits"].shape[1]] = d["logits"].cuda()
    vis = d["vis"].cuda()
    img = run(full, vis, False)
    kth = torch.sort(keys(full).masked_fill(torch.arange(full.shape[1], device="cuda")[None, :] >= vis.long()[:, None], -1), 1, descending=True).values[:, K - 1]
    bad = ((keys(full).gather(1, img.long().clamp(min=0)) < kth[:, None]).any(1) & (vis.long() > K)).nonzero().flatten()
    r0 = int(bad[0]) // 32 * 32 if bad.numel() else 1600
    case(f"hint path, rows {r0}-{r0 + 31} of " + f.split("/")[-1], full[r0:r0 + 32].contiguous(), vis[r0:r0 + 32].contiguous(), reps=10)

# CUDA-graph timing at decode sizes (12 calls = 12 QSA layers)
for rows, ctx in ((4, 32000), (4, 128000), (4, 250000), (24, 128000), (24, 250000)):
    vis = ((torch.arange(rows, device="cuda") + ctx) // 4 + 1).to(torch.int32)
    lg = torch.randn(rows, COLS, device="cuda")
    res = {"rows": rows, "ctx": ctx}
    for det in (False, True):
        qi._QSA_DET = det
        outs = [torch.empty((rows, K), dtype=torch.int32, device="cuda") for _ in range(12)]
        ws = torch.empty((qi._TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device="cuda")
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for o in outs: qi._topk(lg, vis, 2048, 4, o, ws)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for o in outs: qi._topk(lg, vis, 2048, 4, o, ws)
        for _ in range(5): g.replay()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(200): g.replay()
        torch.cuda.synchronize(); res["qsadet_ms" if det else "image_ms"] = round((time.perf_counter() - t) / 200 * 1e3, 4)
    res["added_ms_per_step_12_layers"] = round(res["qsadet_ms"] - res["image_ms"], 4)
    print(json.dumps(res), flush=True)
