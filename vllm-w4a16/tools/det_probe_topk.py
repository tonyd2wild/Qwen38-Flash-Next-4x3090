#!/usr/bin/env python3
# Run-to-run repeatability of the QSA indexer top-k (torch.ops._C.persistent_topk, as called by
# vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py::_topk) on fixed logits: is the output order identical, and is the
# selected set identical? A varying order changes the summation order of the sparse attention that consumes it.
# Usage: docker run --rm --device nvidia.com/gpu=all -v $PWD/vllm-w4a16/tools:/tools:ro --entrypoint python3 \
#          qwen38fn-vllm:8a728663 /tools/det_probe_topk.py
import json, torch
import vllm._C_stable_libtorch  # noqa: F401  registers torch.ops._C (this build ships the stable-ABI library)
import vllm.models.qwen4_exp.nvidia.ops.qsa_indexer as qi

k = 2048 // 4  # indexer_budget / indexer_compress_ratio
ws = torch.empty((qi._TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device="cuda")
res = {}
for name, rows, cols, ctx0 in [("short prompt (29 tokens, all visible)", 29, 8, 0),
                               ("1664-token chunk at 64K (k of 16K visible)", 1664, 16640, 62000)]:
    g = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(rows, cols, device="cuda", generator=g)
    vis = ((torch.arange(rows, device="cuda") + ctx0) // 4 + 1).clamp(max=cols).to(torch.int32)
    outs = []
    for _ in range(20):
        o = torch.full((rows, k), -1, dtype=torch.int32, device="cuda")
        qi._topk(logits, vis, 2048, 4, o, ws); outs.append(o)
    torch.cuda.synchronize()
    exact = 1 + len({o.cpu().numpy().tobytes() for o in outs[1:] if not torch.equal(o, outs[0])})
    sets = 1 + len({torch.sort(o, 1).values.cpu().numpy().tobytes() for o in outs[1:]
                    if not torch.equal(torch.sort(o, 1).values, torch.sort(outs[0], 1).values)})
    res[name] = {"distinct_outputs": exact, "distinct_selected_sets": sets}
print(json.dumps(res))
