#!/usr/bin/env python3
# Offline check of the align-mode Mamba page leak and its fix (T4, journal/04-p3-mamba-leak.md), no GPU and no server.
# Runs inside the lane image with patch/single_type_kv_cache_manager.py mounted over the image's file, and drives the
# image's own KVCacheManager (coordinator, block pool, managers) with this lane's group layout: 4 Mamba groups in
# align mode with 3 speculative blocks, one full-attention group, one QSA ring; 385 blocks of 832 tokens, retention 0.
# The driver plays the scheduler for one request at a time the way async scheduling does (1,664-token block-aligned
# chunks, one chunk in flight, allocate_slots before each chunk, self-preemption on a failed allocation, re-admission
# through get_computed_blocks with the shared-prefix junction stop), then decodes with 3 draft tokens and random
# acceptance, then finishes the request. Per prompt: preemptions, prefill tokens computed, peak blocks per Mamba group,
# and a pool check (every block free at the end). --fix 0 is the image's behaviour, --fix 1 the overlay's MAMBAFIX.
# Usage (repo root):
#   docker run --rm --entrypoint python3 -e QWEN4EXP_MAMBA_FIX=1 -v $PWD/vllm-w4a16:/lane:ro \
#     -v $PWD/vllm-w4a16/patch/single_type_kv_cache_manager.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py:ro \
#     qwen38fn-vllm:8a728663 /lane/tools/mamba_leak_sim.py [--prompts 99726,128000,154495,199404,262000] [--out ...]
import argparse, json, os, random, sys

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default="99726,128000,154495,199404,262000")
ap.add_argument("--decode", type=int, default=3000, help="decode tokens after each prompt")
ap.add_argument("--blocks", type=int, default=385)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="")
a = ap.parse_args()

import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import single_type_kv_cache_manager as stm
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (CircularBufferSpec, FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec,
                                        MambaSpec)
from vllm.v1.request import Request, RequestStatus

FIX = os.environ.get("QWEN4EXP_MAMBA_FIX") == "1"
assert hasattr(stm, "_MAMBA_FIX") and stm._MAMBA_FIX == FIX, "overlay not mounted, or the env flag did not reach it"
try:
    stm.register_all_kvcache_specs(None)
except Exception as e:  # the platform hook needs a VllmConfig; the built-in specs are registered before it
    print(f"(register_all_kvcache_specs: {type(e).__name__} after the built-in specs)")

BS, PAGE, SPEC, CHUNK, LOOKAHEAD = 832, 6_234_000, 3, 2048, 3
u8 = torch.uint8
mamba = MambaSpec(block_size=BS, shapes=((PAGE,),), dtypes=(u8,), mamba_cache_mode="align", num_speculative_blocks=SPEC)
attn = FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=1, dtype=u8, page_size_padded=PAGE)
ring = CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=1, head_size_v=0, dtype=u8, page_size_padded=PAGE)
groups = [KVCacheGroupSpec([f"gdn.{g}"], mamba) for g in range(4)] + [KVCacheGroupSpec(["qsa"], attn),
                                                                       KVCacheGroupSpec(["ring"], ring)]
cfg = KVCacheConfig(num_blocks=a.blocks, kv_cache_tensors=[], kv_cache_groups=groups, prefix_cache_retention_interval=0)
mgr = KVCacheManager(cfg, max_model_len=262144, scheduler_block_size=BS, hash_block_size=BS, enable_caching=True,
                     use_eagle=False)
init_none_hash(sha256)
hasher = get_request_block_hasher(BS, sha256)
mamba_mgrs = [m for m in mgr.coordinator.single_type_managers if isinstance(m, stm.MambaManager)]
pool = mgr.block_pool
rng = random.Random(a.seed)


def split(req, start):
    """scheduler._mamba_block_aligned_split for a cold prefill chunk (no eagle drop, no checkpoints)."""
    end = min(start + CHUNK, req.num_prompt_tokens)
    if end < req.num_prompt_tokens:
        end = end // BS * BS
    last_cache = req.num_tokens - req.num_tokens % BS
    stops = [s for s in (last_cache, start + (req.shared_prefix_boundary - start) // BS * BS
                         if start < req.shared_prefix_boundary < end else 0) if start < s < end]
    return (min(stops) if stops else end) - start


def run(prompt_len, rid):
    req = Request(f"req-{rid}", [rng.randrange(1000, 200000) for _ in range(prompt_len)],
                  SamplingParams(max_tokens=a.decode + 8, temperature=0.0), None, block_hasher=hasher)
    inflight = []  # [num_tokens, stale] per scheduled step, oldest first
    st = {"prompt": prompt_len, "preemptions": 0, "prefill_tokens": 0, "peak_mamba": 0, "peak_used": 0,
          "first_preempt_at": None, "readmit_hits": []}
    decoded, waiting = 0, True

    def complete_oldest():
        nonlocal decoded
        n, stale, kind = inflight.pop(0)
        req.num_in_flight_tokens -= n
        if stale or kind != "decode":
            return
        acc = rng.randint(1, 1 + SPEC)  # accepted tokens of this step; the rest of the 1 + SPEC rows are rejected
        req.num_computed_tokens -= (1 + SPEC) - acc
        req.append_output_token_ids([rng.randrange(1000, 200000) for _ in range(acc)])
        decoded += acc

    while decoded < a.decode:
        if len(inflight) == 2:
            complete_oldest()
        mgr.new_step_starts()  # scheduler.schedule() does this first
        if waiting:
            blocks, hit, spb = mgr.get_computed_blocks(req)
            req.shared_prefix_boundary = spb
            n = split(req, hit)
            new = mgr.allocate_slots(req, n, num_new_computed_tokens=hit, new_computed_blocks=blocks,
                                     num_lookahead_tokens=LOOKAHEAD, has_scheduled_reqs=False)
            if new is None:
                assert inflight, "a lone request cannot be admitted into an idle pool"
                complete_oldest(); continue
            if req.num_preemptions:
                st["readmit_hits"].append(hit)
            req.status, req.num_computed_tokens, waiting = RequestStatus.RUNNING, hit, False
            kind = "prefill"
        else:
            if req.num_computed_tokens < req.num_prompt_tokens:
                n, kind = split(req, req.num_computed_tokens), "prefill"
            elif inflight and inflight[-1][2] == "prefill":
                complete_oldest(); continue  # the first decode step needs the sampled token of the last chunk
            else:
                n, kind = 1 + SPEC, "decode"
            new = mgr.allocate_slots(req, n, num_lookahead_tokens=LOOKAHEAD)
            if new is None:  # scheduler.py:713-723: a lone request preempts itself
                st["preemptions"] += 1
                st["first_preempt_at"] = st["first_preempt_at"] or req.num_computed_tokens
                mgr.free(req)
                req.status, req.num_computed_tokens, waiting = RequestStatus.PREEMPTED, 0, True
                req.num_preemptions += 1
                for f in inflight:
                    f[1] = True
                continue
        req.num_computed_tokens += n
        req.num_in_flight_tokens += n
        inflight.append([n, False, kind])
        if kind == "prefill":
            st["prefill_tokens"] += n
        held = [sum(not b.is_null for b in m.req_to_blocks.get(req.request_id, [])) for m in mamba_mgrs]
        st["peak_mamba"] = max(st["peak_mamba"], max(held))
        st["peak_used"] = max(st["peak_used"], pool.num_gpu_blocks - 1 - pool.get_num_free_blocks())
    while inflight:
        complete_oldest()
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    mgr.free(req)
    st["free_after"] = pool.get_num_free_blocks()
    st["pool_ok"] = st["free_after"] == pool.num_gpu_blocks - 1
    return st


print(f"MAMBAFIX {'on' if FIX else 'off'}: {a.blocks} blocks of {BS} tokens, chunk {CHUNK}, decode {a.decode} tokens")
R = {"fix": FIX, "blocks": a.blocks, "runs": []}
for i, p in enumerate(int(x) for x in a.prompts.split(",")):
    st = run(p, i)
    R["runs"].append(st)
    print(f"  prompt {p:>7}: preemptions {st['preemptions']}  prefill tokens {st['prefill_tokens']:>7} "
          f"({st['prefill_tokens'] / p:.2f}x)  first preempt at {st['first_preempt_at']}  re-admission hits "
          f"{st['readmit_hits']}  peak blocks per Mamba group {st['peak_mamba']}  peak used {st['peak_used']}  "
          f"pool {'ok' if st['pool_ok'] else 'LEAKED ' + str(pool.num_gpu_blocks - 1 - st['free_after'])}")
if a.out:
    json.dump(R, open(a.out, "w"), indent=1)
    print("wrote", a.out)
sys.exit(0 if all(r["pool_ok"] for r in R["runs"]) else 1)
