#!/usr/bin/env python3
# Checks the MOEDET overlay (patch/moe_align_block_size.py, T14, journal/16-nondeterminism.md) with no server running:
# for 9-2,048 tokens, two ranks, uniform and skewed routing, with and without expert_map, the per-expert kernel's
# sorted_token_ids must equal _canonical_sorted_ids (the torch reference in the same file) and repeat exactly; then it
# times 48 calls (one per MoE layer) in a CUDA graph and eagerly, with and without MOEDET.
# Usage: docker run --rm --device nvidia.com/gpu=0 -e QWEN4EXP_MOE_DET=1 \
#   -v $PWD/vllm-w4a16/patch/moe_align_block_size.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/moe_align_block_size.py:ro \
#   -v $PWD/vllm-w4a16/tools:/tools:ro --entrypoint python3 qwen38fn-vllm:8a728663 /tools/det_check_moe_align.py
import json, time, torch
import vllm.model_executor.layers.fused_moe.moe_align_block_size as M
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
dev = torch.device("cuda", 0); g = torch.Generator(device=dev).manual_seed(3)
EG, E, TOPK = 512, 128, 10
def bs_for(m):
    me = -(-m * E // EG)
    for b in [8, 16, 32, 48, 64]:
        if me * TOPK / E / b < 0.9: return b
    return 64
cases = bad = 0; distinct_max = 1
for rank in (0, 3):
    emap = torch.full((EG,), -1, dtype=torch.int32, device=dev); emap[rank*E:(rank+1)*E] = torch.arange(E, dtype=torch.int32, device=dev)
    for skew in (0.0, 1.5):
        for m in (9, 12, 24, 33, 100, 331, 1028, 1664, 2048):
            for use_map in (True, False):
                x = torch.randn(m, 2560, dtype=torch.bfloat16, device=dev, generator=g)
                lg = torch.randn(m, EG, dtype=torch.bfloat16, device=dev, generator=g); lg[:, rank*E:rank*E+16] += skew
                tw, ti, _ = fused_topk(x, lg, TOPK, renormalize=True); bs = bs_for(m)
                M._MOE_DET = False
                s, e, n = M.moe_align_block_size(ti, bs, EG, emap if use_map else None, ignore_invalid_experts=use_map)
                ref = M._canonical_sorted_ids(s.clone(), e, n, bs, ti.numel())
                M._MOE_DET = True
                outs = [M.moe_align_block_size(ti, bs, EG, emap if use_map else None, ignore_invalid_experts=use_map) for _ in range(6)]
                n0 = int(n)
                ok = all(torch.equal(o[0][:n0], ref[:n0]) and int(o[2]) == n0 for o in outs)
                cases += 1; bad += int(not ok)
                distinct_max = max(distinct_max, len({bytes(o[0][:n0].cpu().numpy()) for o in outs}))
print(json.dumps({"cases": cases, "mismatches_vs_torch_reference": bad, "max_distinct_orders": distinct_max}))
emap = torch.full((EG,), -1, dtype=torch.int32, device=dev); emap[256:384] = torch.arange(E, dtype=torch.int32, device=dev)
for m in (12, 24, 1028, 1664):
    x = torch.randn(m, 2560, dtype=torch.bfloat16, device=dev, generator=g); lg = torch.randn(m, EG, dtype=torch.bfloat16, device=dev, generator=g)
    tw, ti, _ = fused_topk(x, lg, TOPK, renormalize=True); bs = bs_for(m)
    res = {"M": m, "bs": bs}
    for det in (False, True):
        M._MOE_DET = det
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): M.moe_align_block_size(ti, bs, EG, emap, ignore_invalid_experts=True)
        torch.cuda.current_stream().wait_stream(st); gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(48): M.moe_align_block_size(ti, bs, EG, emap, ignore_invalid_experts=True)
        for _ in range(3): gr.replay()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(100): gr.replay()
        torch.cuda.synchronize(); res[f"graph_ms_48_{'det' if det else 'kernel'}"] = round((time.perf_counter() - t) / 100 * 1e3, 4)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20):
            for _ in range(48): M.moe_align_block_size(ti, bs, EG, emap, ignore_invalid_experts=True)
        torch.cuda.synchronize(); res[f"eager_ms_48_{'det' if det else 'kernel'}"] = round((time.perf_counter() - t) / 20 * 1e3, 3)
    print(json.dumps(res))
