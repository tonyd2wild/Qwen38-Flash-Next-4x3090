#!/usr/bin/env python3
# Bitwise repeatability of the Marlin W4A16 MoE path of one rank under expert parallelism, with no server running (T14,
# journal/16-nondeterminism.md): R28's trace found the first run-to-run difference in a rank's moe_sum output while the
# router outputs were identical and only moe_align_block_size's token order varied. Shapes are this lane's per rank:
# 128 local experts of 512 (the rank's contiguous slice through expert_map), top-10, K=2560, N=640, uint4b8 weights with
# group size 128 (random bits; any int32 pattern is a valid packed weight), random bf16 activations and router logits.
#   e2e      fused_marlin_moe as the lane calls it (moe_align_block_size inside), --reps times: distinct outputs, and
#            distinct sorted_token_ids
#   fixed    _fused_marlin_moe with one fixed sorted_token_ids, then moe_sum, --reps times
#   permute  the same, with each expert's token list shuffled inside its blocks (padding kept at the end): does a token's
#            GEMM result depend on its slot? Compares only rows of local experts; output buffers are zeroed per call
#   inblock  the same shuffle, but only inside each block of block_size_m slots: does the slot matter, or only the block?
#   canon    the candidate fix: each expert's token list re-sorted ascending after moe_align_block_size (canonical_ids
#            below), then the e2e path --reps times
#   sum      moe_sum alone on one fixed input
# --skew adds a bias to 16 experts' router logits so that hot experts get more than block_size_m tokens, as real
# routing does (with uniform random logits no expert of 34 tokens fills a block).
# Usage (needs the image and a free GPU):
#   docker run --rm --device nvidia.com/gpu=0 -v $PWD/vllm-w4a16/tools:/tools:ro --entrypoint python3 \
#     qwen38fn-vllm:8a728663 /tools/det_probe_moe.py [--reps 30] [--tokens 4,34,331] [--rank 2]
import argparse, json

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--tokens", default="4,34,331")
ap.add_argument("--rank", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skew", type=float, default=0.0)
a = ap.parse_args()

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import _fused_marlin_moe, fused_marlin_moe
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.scalar_type import scalar_types

dev = torch.device("cuda", 0)
g = torch.Generator(device=dev).manual_seed(a.seed)
E_GLOBAL, E, TOPK, K, N, GS = 512, 128, 10, 2560, 640, 128
w1 = torch.randint(-2**31, 2**31 - 1, (E, K // 16, 2 * N * 2), dtype=torch.int32, device=dev, generator=g)
w2 = torch.randint(-2**31, 2**31 - 1, (E, N // 16, K * 2), dtype=torch.int32, device=dev, generator=g)
s1 = (0.002 + 0.004 * torch.rand(E, K // GS, 2 * N, device=dev, generator=g)).to(torch.bfloat16)
s2 = (0.002 + 0.004 * torch.rand(E, N // GS, K, device=dev, generator=g)).to(torch.bfloat16)
empty = torch.empty((E, 0), dtype=torch.int32, device=dev)
expert_map = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=dev)
expert_map[a.rank * E:(a.rank + 1) * E] = torch.arange(E, dtype=torch.int32, device=dev)
qt = scalar_types.uint4b8
ws = marlin_make_workspace_new(dev, 4)


def bits(t):
    return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def inner(x, tw, ti, sid, eid, npad, bsm):
    """The lane's _fused_marlin_moe + moe_sum with a given token order; returns (gemm1 out, moe out, summed)."""
    M = x.shape[0]
    c13 = torch.zeros(M * TOPK * max(2 * N, K), dtype=x.dtype, device=dev)
    c2 = torch.zeros(M * TOPK, N, dtype=x.dtype, device=dev)
    mo = _fused_marlin_moe(hidden_states=x, w1=w1, w2=w2, bias1=None, bias2=None, w1_scale=s1, w2_scale=s2,
                           topk_weights=tw, num_topk=TOPK, quant_type=qt, apply_router_weight_on_input=False,
                           expert_map=expert_map, block_size_m=bsm, sorted_token_ids=sid, expert_ids=eid,
                           num_tokens_post_padded=npad, topk_ids=ti, g_idx1=empty, g_idx2=empty,
                           sort_indices1=empty, sort_indices2=empty, workspace=ws, intermediate_cache13=c13,
                           intermediate_cache2=c2, is_k_full=True).view(-1, TOPK, K)
    out = torch.empty_like(x)
    ops.moe_sum(mo, out, ti, expert_map)
    return mo, out


def canonical_ids(sid, eid, npad, bsm, numel):
    """Each expert's segment of sorted_token_ids in ascending order (the sentinel numel sorts after every pair)."""
    pos = torch.arange(sid.numel(), device=sid.device)
    seg = torch.where(pos < npad, eid[(pos // bsm).clamp_(max=eid.numel() - 1)].long(), eid.numel() + 1)
    key = seg * (numel + 1) + sid.long()
    return (torch.sort(key).values % (numel + 1)).to(sid.dtype)


def shuffle(sid_c, eid_c, n, bsm, sentinel, gen, per_block):
    s = sid_c.clone(); nblk = n // bsm; b = 0
    while b < nblk:
        e = int(eid_c[b]); b2 = b + 1
        while not per_block and b2 < nblk and int(eid_c[b2]) == e:
            b2 += 1
        seg = s[b * bsm:b2 * bsm]
        v = seg[seg < sentinel]
        seg[:v.numel()] = v[torch.randperm(v.numel(), generator=gen)]
        b = b2
    return s


def block_size_for(M):
    Mest = -(-M * E // E_GLOBAL)
    for b in [8, 16, 32, 48, 64]:
        if Mest * TOPK / E / b < 0.9:
            return b
    return 64


res = {"rank": a.rank, "reps": a.reps, "tests": []}
for M in [int(t) for t in a.tokens.split(",")]:
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev, generator=g)
    logits = torch.randn(M, E_GLOBAL, dtype=torch.bfloat16, device=dev, generator=g)
    if a.skew:
        logits[:, a.rank * E:a.rank * E + 16] += a.skew
    tw, ti, _ = fused_topk(x, logits, TOPK, renormalize=True)
    valid = (expert_map[ti.long()] >= 0)  # [M, TOPK] pairs this rank computes
    bsm = block_size_for(M)
    counts = torch.bincount(expert_map[ti.long()][valid].long(), minlength=E)
    r = {"M": M, "block_size_m": bsm, "local_pairs": int(valid.sum()), "max_tokens_per_expert": int(counts.max()),
         "experts_over_one_block": int((counts > bsm).sum())}
    # e2e
    outs, sids = set(), set()
    for _ in range(a.reps):
        o = fused_marlin_moe(x, w1, w2, None, None, s1, s2, tw, ti, quant_type_id=qt.id, global_num_experts=E_GLOBAL,
                             expert_map=expert_map, g_idx1=empty, g_idx2=empty, sort_indices1=empty,
                             sort_indices2=empty, workspace=ws, is_k_full=True)
        outs.add(bits(o))
        sid, _, _ = moe_align_block_size(ti, bsm, E_GLOBAL, expert_map, ignore_invalid_experts=True)
        sids.add(bits(sid))
    r["e2e_distinct_out"], r["e2e_distinct_sorted_ids"] = len(outs), len(sids)
    # fixed order
    sid, eid, npad = moe_align_block_size(ti, bsm, E_GLOBAL, expert_map, ignore_invalid_experts=True)
    ref_mo, ref_out = inner(x, tw, ti, sid, eid, npad, bsm)
    fo = set()
    for _ in range(a.reps):
        mo, out = inner(x, tw, ti, sid, eid, npad, bsm)
        fo.add(bits(out) + bits(mo[valid]))
    r["fixed_distinct"] = len(fo)
    # permuted within each expert's blocks, then only within each block
    sentinel = ti.numel()
    sid_c, eid_c, n = sid.cpu(), eid.cpu(), int(npad.item())
    gen = torch.Generator().manual_seed(a.seed + 1)
    for name, per_block in (("permute", False), ("inblock", True)):
        pm, po, rows_changed, maxdiff = set(), set(), set(), 0.0
        for _ in range(a.reps):
            s = shuffle(sid_c, eid_c, n, bsm, sentinel, gen, per_block)
            mo, out = inner(x, tw, ti, s.to(dev), eid, npad, bsm)
            pm.add(bits(mo[valid])); po.add(bits(out))
            d = (mo != ref_mo).any(-1) & valid
            rows_changed.update(int(t) for t in d.any(-1).nonzero().flatten())
            maxdiff = max(maxdiff, float((mo.float() - ref_mo.float())[valid].abs().max()))
        r[f"{name}_distinct_moe_out_local_rows"], r[f"{name}_distinct_summed"] = len(pm), len(po)
        r[f"{name}_tokens_changed"], r[f"{name}_max_abs_diff"] = len(rows_changed), maxdiff
    # the candidate fix: canonical order after the atomic align, through the whole path
    co, cs = set(), set()
    for _ in range(a.reps):
        s0, e0, n0 = moe_align_block_size(ti, bsm, E_GLOBAL, expert_map, ignore_invalid_experts=True)
        s1c = canonical_ids(s0, e0, n0, bsm, ti.numel())
        cs.add(bits(s1c[:int(n0.item())]))
        mo, out = inner(x, tw, ti, s1c, e0, n0, bsm)
        co.add(bits(out))
    r["canon_distinct_sorted_ids"], r["canon_distinct_out"] = len(cs), len(co)
    # moe_sum alone
    so = set()
    for _ in range(a.reps):
        out = torch.empty_like(x); ops.moe_sum(ref_mo, out, ti, expert_map); so.add(bits(out))
    r["sum_distinct"] = len(so)
    res["tests"].append(r)
    print(json.dumps(r), flush=True)
print(json.dumps(res))
