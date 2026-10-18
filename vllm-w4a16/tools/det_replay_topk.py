#!/usr/bin/env python3
# Replays R34's top-k dumps (T14, journal/16-nondeterminism.md; DET_TRACE_DUMP="_topk:8192:0,1,2" writes them to
# .cache/ple/det-trace/dump/, kept in .cache/ple/det-trace-r34/dump/) with no server running. The dumps of two runs
# of the same prompt are paired in time order: are the visible scores bit-identical, does the selection differ, and
# does the patched _topk on each run's scores reproduce what the server returned? Then the image's persistent_topk is
# run 40 times on one inexact call's scores, idle and with a second stream busy, counting rows whose selection holds a
# score below the true 512th.
# Usage: docker run --rm --device nvidia.com/gpu=0 -e QWEN4EXP_QSA_DET=1 \
#   -v $PWD/vllm-w4a16/patch/qsa_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py:ro \
#   -v $PWD/.cache/ple/det-trace-r34/dump:/dump:ro -v $PWD/vllm-w4a16/tools:/tools:ro --entrypoint python3 \
#   qwen38fn-vllm:8a728663 /tools/det_replay_topk.py
import glob, json, os, re, torch
import vllm._C_stable_libtorch  # noqa: F401
import vllm.models.qwen4_exp.nvidia.ops.qsa_indexer as qi
assert qi._QSA_DET
files = sorted(glob.glob("/dump/*.pt"), key=lambda f: torch.load(f)["t"])
dumps = [torch.load(f) | {"f": os.path.basename(f)} for f in files]
half = len(dumps) // 2
A, B = dumps[:half], dumps[half:]
ws = torch.empty((qi._TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device="cuda")
for a, b in zip(A, B):
    vis = a["vis"]; assert torch.equal(vis, b["vis"])
    cols = torch.arange(a["logits"].shape[1])[None, :] < vis[:, None].long()
    la, lb = a["logits"].masked_fill(~cols, 0), b["logits"].masked_fill(~cols, 0)
    same_logits = bool(torch.equal(la, lb))
    rows_sel_diff = (a["block_indices"] != b["block_indices"]).any(1)
    res = {"A": a["f"], "B": b["f"], "call": a["call"], "max_vis": int(vis.max()), "valid_logits_equal": same_logits,
           "rows_selection_differs": int(rows_sel_diff.sum())}
    # replay: the patched _topk on each run's scores, padded with garbage to the server's width
    for name, d in (("A", a), ("B", b)):
        full = torch.randn(d["logits"].shape[0], d["columns"], device="cuda") * 100
        full[:, :d["logits"].shape[1]] = d["logits"].cuda()
        o = torch.randint(-2**31, 2**31 - 1, d["block_indices"].shape, dtype=torch.int32, device="cuda")
        qi._topk(full, vis.cuda(), 2048, 4, o, ws)
        res[f"replay_{name}_equals_server_{name}"] = bool(torch.equal(o.cpu(), d["block_indices"]))
        bad = (o.cpu() != d["block_indices"]).any(1)
        res[f"rows_server_{name}_differs_from_replay"] = int(bad.sum())
        if bad.any():
            r = int(bad.nonzero()[0]); srv = d["block_indices"][r]; rep = o.cpu()[r]
            lg = d["logits"][r]
            res[f"example_{name}"] = {"row": r, "vis": int(vis[r]), "server_head": srv[:8].tolist(), "replay_head": rep[:8].tolist(),
                                      "server_invalid": int(((srv < 0) | (srv.long() >= vis[r])).sum()),
                                      "server_dups": int(srv.numel() - torch.unique(srv).numel()),
                                      "server_min_score": float(lg[srv.long().clamp(0, lg.numel() - 1)].min()),
                                      "kth_score": float(torch.topk(lg[:int(vis[r])], 512).values[-1])}
    print(json.dumps(res), flush=True)

# persistent_topk alone, repeated on one dumped call
qi._QSA_DET = False  # the image's _topk: persistent_topk only
d = torch.load(sorted(glob.glob("/dump/*call1.pt"))[1])   # a layer-7 call that was inexact in the server
vis = d["vis"].cuda(); rows, K = d["block_indices"].shape
full = torch.randn(rows, d["columns"], device="cuda") * 100; full[:, :d["logits"].shape[1]] = d["logits"].cuda()
masked = full.masked_fill(torch.arange(d["columns"], device="cuda")[None, :] >= vis[:, None].long(), float("-inf"))
kth = torch.topk(masked, K, 1).values[:, -1]
def inexact(o):
    return int((torch.gather(masked, 1, o.long().clamp(min=0)) < kth[:, None]).any(1).sum())
print("server output inexact rows:", inexact(d["block_indices"].cuda()))
side = torch.cuda.Stream(); a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for mode in ("idle", "side-stream matmuls", "side-stream small kernels"):
    tot = 0; worst = 0
    for i in range(40):
        if mode != "idle":
            with torch.cuda.stream(side):
                for _ in range(20):
                    if mode == "side-stream matmuls": a @ a
                    else: a.add_(0.0)
        ws = torch.empty((qi._TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device="cuda")
        o = torch.empty((rows, K), dtype=torch.int32, device="cuda")
        qi._topk(full, vis, 2048, 4, o, ws)
        torch.cuda.synchronize()
        n = inexact(o); tot += n; worst = max(worst, n)
    print(json.dumps({"mode": mode, "calls": 40, "inexact_rows_total": tot, "worst_call": worst}), flush=True)
