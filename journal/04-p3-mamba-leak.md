# 04: Align-mode Mamba page leak (T4, portfolio P3)

Status: done · Model: Opus 5.5 · Effort: xhigh · Started: 2026-10-18 01:18 · Wall time: 1:32 (three boots)

## Goal

Long cold prompts on the caching lanes are preemption-bound: 128K and 155K cold get 2 preemptions each, ~200K gets 4 (`01-baseline.md`). The hostpin lane does 1,213-1,310 tok/s there against 2,189 at 64K (R22). Find out why the Mamba groups run the 384-block pool dry, fix it in the KV manager, and validate on the scoreboard's `cold155k`, `needle200k` and 128K cold rows. Success: 0 preemptions for a lone request up to the 262,144 context, with 155K cold at the 64K prefill rate (less the attention cost of depth). Nothing else may move: decode, warm and incremental rows stay at R22 within noise, and the gate passes.

## Hypothesis and kill test

**H1 (primary): under async scheduling the align-mode state-block free in `MambaManager.remove_skipped_blocks` runs one chunk late and never fires during prefill. Each chunk's state block stays held until the request ends.** Code read, all line numbers in the image's `vllm/v1/core/single_type_kv_cache_manager.py`:

- `allocate_new_blocks` (align branch, 1700-1808) records `last_state_block_idx[req] = prev_block_len - 1 - num_speculative_blocks`, the state slot of the previous chunk, every time it allocates a new state block. During prefill that is every chunk (2 blocks of 832 tokens = 1,664 tokens per chunk). The slot is overwritten at the next allocation.
- `remove_skipped_blocks` (1556-1584) frees `blocks[last_state_block_idx]` only if `last_state_block_idx < cdiv(processed, 832) - 1`. `KVCacheManager.allocate_slots` (`kv_cache_manager.py:503`) now passes `processed = total_computed - request.num_in_flight_tokens`: "free on the processed-token basis".
- With async scheduling (on in this lane, `max_concurrent_batches` 2), `EngineCore.step_with_batch_queue` schedules chunk j+1 while chunk j is in flight. So at that call `processed = (j-1)·1664`, the threshold is `2(j-1) - 1 = 2j-3`, and `last_state_block_idx` is chunk j-1's state slot, also `2j-3`. `2j-3 < 2j-3` is false. The allocation that follows overwrites the index with `2j-1`, and slot `2j-3` is never looked at again. In sync scheduling `processed` would be `j·1664`, and the test would pass.
- The base-class sweep (`_remove_blocks_in_range`, 620-645) cannot catch the leftovers. It walks backward from the last skipped block and stops at the first null, and in align mode the table alternates: even slots are nulls (relocated speculative slots), odd slots are states. So it stops at once.
- Decode does not leak: a block boundary comes every 832 tokens, and the processed boundary passes the old state slot within a step or two, long before the index is overwritten.

Arithmetic, with the pool being 385 blocks, 1 of them the null block (the boot log's 299,482 tokens = 385/337 × 262,144; 337 = 316 attention + 1 ring + 4 Mamba groups × 5). A lone request at chunk k holds 2k+1 attention blocks (one for the 3 lookahead tokens), 1 ring block and 4 × (k + 3) Mamba blocks, 6k + 14 in all. Chunk k needs 6 new blocks, so the schedule of chunk 62 fails. That is a preemption at 61 chunks, 101,504 tokens. Checks against what was measured:

| Observation (R13/R14/R22, `01-baseline.md`) | H1 predicts |
|---|---|
| 0 preemptions at 99,726 tokens and below; "the needle just fits" | 60 chunks fit (6·60 + 14 = 374 ≤ 384) |
| 2 preemptions for every cold prompt from 128K to 155K | Attempt 1 dies at 101.5K. Re-admission: attention hits 101.5K, Mamba hits nothing (retention interval 0 keeps no interior states), so the hit is 0, but `find_longest_cache_hit` reports the 101.5K difference as an uncached shared prefix, so `shared_prefix_boundary` ≈ 101.5K (`kv_cache_manager.py:285-290`). Attempt 2 recomputes from 0, its chunk split stops at that junction (`scheduler.py:465-469`) and caches the state there, then dies at about the same place (preemption 2). Attempt 3 resumes from the cached junction with a fresh Mamba footprint and finishes |
| R10: the 155K prompt "ran 155 steps for 256,446 tokens" | 101.5K + 101.5K + (155K - 101.5K) ≈ 256K ✓ |
| 4 preemptions at ~200K; TTFT 170.7 s against 91 s at the 64K rate (1.87x) | Attempt 3 dies again around 168K (6k' > 244 more chunks), and the junction trick repeats: 101.5 + 101.5 + 66.5 + 66.5 + 31.4 = 367K tokens for 199.4K (1.84x) ✓ |
| 128K cold 1,213 tok/s, 155K 1,310 tok/s against 2,189 at 64K (R22) | 229.5K tokens computed for 128K (1.79x) and 256K for 155K (1.66x): ~2,170 and ~2,170 tok/s of real work ✓ |

Alternatives this must rule out (each has a distinct trace signature):

- **H2: cached state blocks keep a reference** (retention or partial-tail CoW). The trace would show leaked blocks with `block_hash` set or `ref_cnt > 1`. Under H1 they have `ref_cnt == 1`, no hash (interior states with retention 0), and the request is the only owner.
- **H3: speculative-block relocation leaks.** The trace would show growth other than exactly +1 non-null block per chunk per group, or growth during decode.
- **H4: the internal-checkpoint path** (`_needs_internal_checkpoint`, `num_prefill_checkpoint_blocks`). The trace would show checkpoint blocks being allocated; H1 expects `num_prefill_checkpoint_blocks == 0` here.
- **H5: the leak is not Mamba at all** (attention lookahead, the ring, CoW). The trace would show the attention group above `cdiv(n + 3, 832)` blocks, or the ring above 1.

**Kill test:** an instrumented boot (fix off) shows Mamba group 0 holding `k + 3` non-null blocks at chunk k, `remove_skipped_blocks` freeing nothing during prefill, the first preemption at about 101.5K, and re-admission hits of 0, then about 101.5K. If any of these fails, H1 is wrong. If the fix-on boot still preempts below 262K for a lone request, the fix is wrong. If the gate fails, or decode or warm rows move beyond noise, the fix is killed.

## Setup

- Branch / commit: `perf-comm-ple` @ `56a69e3`
- Lane and flags: `launch-qwen38fn-w4a16-hostpin.sh` (TRITON_CACHE=1, TOKSPLIT=1 by default), `GPU_ARGS="--device nvidia.com/gpu=all"`, image `qwen38fn-vllm:8a728663`, plus the new knobs below
- Ledger rows: R23 (diagnosis: trace on, fix off), R24 (candidate: fix on, trace on), R25 (the wrapper as shipped: fix on by default, trace off)
- Box state before: GPUs free (15-91 MiB at 01:18), host server down, page cache warm (R22 ran until 01:08)

## Design

**Fix (F1): never forget a state block that is waiting to be freed.** Replace the single `last_state_block_idx[req]` with a per-request list of state slots waiting to be freed. `allocate_new_blocks` appends where it used to overwrite. `remove_skipped_blocks` frees every listed slot that passes the original test (`idx < cdiv(processed, block_size) - 1`, unchanged) and keeps the rest. A block is still freed only when no in-flight step can read it: the safety predicate is the image's, the only change is that a slot that is not yet eligible stays on the list. Under the 1-deep async pipeline a prefilling request then holds 3 state blocks per group: the one the in-flight chunk reads, the one it writes, and the one the next chunk writes. With the 3 speculative blocks that is 6 per group, where the leak gives k + 3. The pool sizer (`MambaSpec.max_memory_usage_bytes`, align: 2 + spec) assumes 5 per group. The extra block per group (4 per request) is inherent to async scheduling and does not change the logged capacity: a 262,144-token request needs 316 + 1 + 24 = 341 of 384 blocks.

Alternatives considered:

- F2, test against `total_computed` instead of `processed` (free one chunk earlier, 5 per group). Rejected: it frees the block the in-flight chunk reads, and safety would then rest on CUDA stream order, the kind of hazard the processed basis was introduced to avoid. It saves 4 blocks per request; not worth the risk.
- F3, free the old slot inside `allocate_new_blocks` when overwriting it. The same hazard as F2.
- F4, `--no-async-scheduling`. H1 predicts it removes the leak with no code, but it gives up the async overlap that decode relies on. It could serve as a zero-code confirmation boot if the trace is ambiguous; not a candidate.
- A larger `--prefix-cache-retention-interval` keeps more junction states, so restarts are cheaper, but the preemptions stay. `--mamba-block-size 4992` "changed nothing" (README). Neither touches the cause.
- The checkpoint author's fix (`runtime/vllm-overlay/v1/core/sched/scheduler.py:382-440`) belongs to another vLLM version, with a different scheduler split and KV manager. Reference only.

Where: an overlay of `vllm/v1/core/single_type_kv_cache_manager.py` (the image's file plus marked changes), `vllm-w4a16/patch/single_type_kv_cache_manager.py`, mounted by a new base-launcher knob. With both env flags unset the overlay runs the image's code paths exactly, so the knob can stay mounted. Knobs: `MAMBA_FIX=0|1` (`QWEN4EXP_MAMBA_FIX=1`: F1) and `MAMBA_TRACE=0|1` (`QWEN4EXP_MAMBA_TRACE=1`: log lines below). Either one mounts the overlay. Defaults 0 leave every existing lane byte-for-byte unchanged.

**Instrumentation** (`MAMBATRACE` log lines from the EngineCore process, only for requests whose prompt is ≥ 32K tokens, plus every free of a request):

1. Once at start: `num_blocks`, every group's spec type, block size, `num_speculative_blocks`, `num_prefill_checkpoint_blocks`, `mamba_cache_mode`, retention interval.
2. Every align-mode allocation in Mamba group 0 (all four groups share one block table shape): request, `num_tokens`, `processed`, table length, non-null blocks held, the slot recorded or listed, pool free blocks, held blocks per group type (attention, ring, Mamba sum).
3. Every `remove_skipped_blocks` in Mamba group 0 that frees or declines a listed slot: slot, threshold, freed or kept, and the freed block's `ref_cnt` and whether it had a hash (H2).
4. Every `pop_blocks_for_free` (preemption or finish): non-null blocks returned per group.
5. Every Mamba `find_longest_cache_hit` with a hit or a long query: hit length (re-admission after preemption).
6. The preemption count comes from `/metrics` (`scoreboard.py` already reports it).

**Test plan:**

1. Offline, no GPU (`tools/mamba_leak_sim.py`, in the image with the overlay mounted). Build a `KVCacheManager` with this lane's group layout (4 Mamba align groups with 3 speculative blocks, full attention, ring; 385 blocks of 832 tokens). Drive one request the way the async scheduler does: 1,664-token chunks with one chunk in flight, self-preemption on a failed allocation, re-admission through `get_computed_blocks` with the junction stop. Then decode with speculative tokens and random rejections. Fix off must reproduce 0 / 2 / 2 / 4 preemptions at 99.7K / 128K / 155K / 199.4K. Fix on must give 0 up to 262,143 tokens, at most 6 Mamba blocks per group, no free of a slot at or above the processed threshold, and every block back in the pool at the end.
2. R23, fix off, trace on: `scoreboard.py --sections cold155k,needle200k`, then save the trace. This is the kill test above.
3. R24, fix on, trace on (it adds a few log lines per chunk to the scheduler process, too few to measure): `quality_gate.py --compare` first, then the full `scoreboard.py`. Compare with R22 on every row; the trace shows the holding stays at 6 per group.
4. Keep or kill on the gate and on rows other than cold/needle staying within R22's noise (decode ±2.5%, warm ±0.1 s). If kept, the hostpin wrapper defaults `MAMBA_FIX=1`.

## What I did

1. Read the KV manager, coordinator, block pool, scheduler and engine core of the image (copied out with `docker run --rm --entrypoint bash qwen38fn-vllm:8a728663 -c 'tar cf ... v1/core config/cache.py ...'`). Wrote the Design above before any code.
2. Overlay `vllm-w4a16/patch/single_type_kv_cache_manager.py`: the image's file (sha256 `128b98a0…`) with the MAMBAFIX and MAMBATRACE changes, all marked. `python3 -m py_compile` passes.
3. Offline simulation `vllm-w4a16/tools/mamba_leak_sim.py` (command in its header; `--decode 3000`, prompts 99,726 / 128,000 / 154,495 / 199,404 / 262,000). Results: `results/mamba-leak-sim-T4-fix{0,1}.json`; output in `.cache/ple/t4-sim-fix{0,1}.out`. The first version forgot `new_step_starts()`, so a re-admitted request saw its own just-cached state as "cached this step" and could not be admitted; fixed in the harness.
4. Launcher knobs `MAMBA_FIX` and `MAMBA_TRACE` in `qwen38fn-w4a16-3090-tp4.sh` (either one mounts the overlay; both 0 mounts nothing). The overlay hash check covers the new mount.
5. R23, 01:46:39: `GPU_ARGS="--device nvidia.com/gpu=all" MAMBA_TRACE=1 bash vllm-w4a16/launch/launch-qwen38fn-w4a16-hostpin.sh`, overlays 10/10, ready 01:51:44.
   ```bash
   URL=http://localhost:8090 python3 vllm-w4a16/tools/scoreboard.py --sections cold155k,needle200k --label R23
   URL=http://localhost:8090 python3 vllm-w4a16/tools/long_pair.py --label R23 --n 2 --tokens 100000
   URL=http://localhost:8090 python3 vllm-w4a16/tools/long_pair.py --label R23 --n 2 --tokens 100000 --stagger --decode 1500 --tokens2 40000
   docker logs vllm_qwen38fn_3090 > .cache/ple/r23-boot.log 2>&1; docker stop -t 60 vllm_qwen38fn_3090
   ```
   `long_pair.py` is new: N cold prompts at once. `--stagger` was added after the first run showed that concurrent cold prefills do not overlap in this lane.
6. R24, 02:01:43: `GPU_ARGS="--device nvidia.com/gpu=all" MAMBA_FIX=1 MAMBA_TRACE=1 bash vllm-w4a16/launch/launch-qwen38fn-w4a16-hostpin.sh`, ready 02:07:46.
   ```bash
   URL=http://localhost:8090 python3 vllm-w4a16/tools/quality_gate.py --compare --label R24-mambafix
   URL=http://localhost:8090 python3 vllm-w4a16/tools/scoreboard.py --label R24
   URL=http://localhost:8090 python3 vllm-w4a16/tools/long_pair.py --label R24 --n 2 --tokens 100000
   URL=http://localhost:8090 python3 vllm-w4a16/tools/long_pair.py --label R24 --n 2 --tokens 100000 --stagger --decode 1500 --tokens2 40000
   URL=http://localhost:8090 python3 vllm-w4a16/tools/scoreboard.py --sections concurrency --no-warmup --label R24-concrepeat
   docker logs vllm_qwen38fn_3090 > .cache/ple/r24-boot.log 2>&1; docker stop -t 60 vllm_qwen38fn_3090
   ```
7. The hostpin wrapper now defaults `MAMBA_FIX=1`. R25, 02:26:02: the wrapper with no knobs set (`QWEN4EXP_MAMBA_FIX=1`, `QWEN4EXP_MAMBA_TRACE=0` in the container); `quality_gate.py --compare --label R25-mambafix-default`, `scoreboard.py --sections decode,cold155k --label R25`.

## Measurements

**Offline simulation, the image's code with the lane's group layout** (`mamba_leak_sim.py`):

| Prompt (tokens) | Fix off: preemptions | prefill tokens computed | re-admission hits | Mamba blocks per group, peak | Fix on: preemptions | Mamba blocks per group, peak | pool blocks used, peak (fix on) |
|---:|---:|---:|---|---:|---:|---:|---:|
| 99,726 | 0 | 99,726 (1.00x) | - | 64 | 0 | 6 | 145 |
| 128,000 | 2 | 229,504 (1.79x) | 0, 101,504 | 64 | 0 | 6 | 179 |
| 154,495 | 2 | 255,999 (1.66x) | 0, 101,504 | 64 | 0 | 6 | 211 |
| 199,404 | 4 | 367,468 (1.84x) | 0, 101,504, 101,504, 168,064 | 64 | 0 | 6 | 265 |
| 262,000 | 8 | 504,944 (1.93x) | 0, 101,504, 101,504, 168,064, 168,064, 212,992, 212,992, 242,944 | 64 | 0 | 6 | 340 |

With the fix off, the first preemption comes at 101,504 tokens in every run, as computed in the hypothesis. The 0 / 2 / 2 / 4 pattern is the measured one, and 255,999 prefill tokens for the 155K prompt matches R10's 256,446 (README). The re-admission hits show the junction mechanism: the first restart hits nothing, and each later restart resumes at the previous failure point. With the fix on, every run ends with all 384 blocks free, after 3,000 decode tokens with random draft rejections.

**R23, fix off, trace on: the kill test** (`.cache/ple/r23-boot.log`, summarized with a scratch parser of the `MAMBATRACE` lines; `results/scoreboard-R23-20261018-015200.json`, `long-pair-R23-*.json`, `long-stagger-R23-*.json`):

- Startup: pool **385 blocks** of 832 tokens (the inferred number). Groups 0-3 `MambaSpec` align with 3 speculative blocks and **0 prefill checkpoint blocks** (H4 out), group 4 the QSA ring (`CircularBufferSpec`, block 8), group 5 attention (`MLAAttentionSpec` under `FullAttentionManager`, block 832).
- Needle (199,407 tokens; correct, 168.3 s, 1,185 tok/s, 4 preemptions): 184 allocations, **0 frees, 184 keeps**. Every keep reads `slot N >= N`: the recorded slot equals the threshold exactly, the off-by-one-chunk of H1. Group 0's held blocks minus the chunk index are constant across each attempt, so +1 per chunk exactly (H3 out). Attempts: 33,280-101,504 tokens (held 23 -> 64, free 9 after group 0's allocation, so 4 after the other groups take theirs); released with 64 held and 4 free; again 33,280-101,504; then 103,168-168,064 twice (held 5 -> 44); then 169,728-199,407. Re-admission hits 0, 101,504, 101,504, 168,064: as simulated.
- 155K cold (154,527 tokens, 116.4 s, 1,328 tok/s, 2 preemptions): the same, attempts to 101,504, 101,504, then 103,168-154,527; hits 0, 101,504.
- Group 5 and the ring (H5): at the first failure the pool has 380 blocks in use = 4 × 64 Mamba + 123 attention (2 × 61 + 1 lookahead) + 1 ring. Nothing else holds blocks.
- H2 does not apply: no interior state is ever freed, so no extra reference keeps one. The blocks are held by the request's own block table, which is H1.
- 2 x 100K cold at once: no collision, because this lane's step budget (2,048) fits only one 832-aligned 1,664-token chunk, so the second prefill waits for the first: TTFT 45.1 and 90.4 s, 0 preemptions.
- Staggered (`--stagger`): a 99,691-token request that decodes 1,500 tokens (170.8 tok/s), and a 39,863-token request sent at its first output token. The second one's first allocation comes at 02:00:40, after the first request released its blocks at 02:00:27: it waited out the whole decode and got its first token 26.5 s after sending. The leak delays a second seat rather than preempting it here. With the fix its prefill should overlap the first request's decode.

**R24, fix on (trace on): the candidate** (`results/scoreboard-R24-20261018-021245.json`, gate `results/quality-compare-R24-mambafix-20261018-020808.json`, `long-pair-R24-*.json`, `long-stagger-R24-*.json`, `.cache/ple/r24-boot.log`):

| Metric | Baseline (R22) | R24 | Delta |
|---|---:|---:|---:|
| Decode, count to 100, median of 5 (tok/s) | 207.5 | 206.5 | -0.5% |
| Decode, prose 512 tokens (tok/s) | 148.0 | 151.9 | +2.6% (1 run) |
| Decode, code prompt, median of 3 (tok/s) | 183.0 | 183.4 | +0.2% |
| Cold TTFT 1K / 16K / 64K: prefill tok/s | 2,077 / 2,198 / 2,189 | 2,065 / 2,176 / 2,160 | -0.6% / -1.0% / -1.3% |
| **Cold TTFT 128K / prefill tok/s** | 105.22 s / 1,213, 2 preemptions | **59.38 s / 2,149, 0** | **1.77x** |
| Warm TTFT 1K / 16K / 64K / 128K (s) | 0.16 / 0.18 / 0.42 / 0.38 | 0.16 / 0.18 / 0.37 / 0.37 | ≤ 0.05 |
| Decode at 64K / 128K context (tok/s) | 131.3 / 144.4 | 150.7 / 136.6 | noisy row |
| Incremental 16K + 1K / 4K (s) | 0.61 / 2.04 | 0.63 / 2.05 | +0.02 |
| Incremental 64K + 4K / 16K (s) | 2.26 / 7.80 | 2.28 / 7.89 | +0.02 / +0.09 |
| Incremental 128K + 4K / 16K (s) | 2.27 / 7.90 | 2.28 / 7.99 | +0.01 / +0.09 |
| **Needle ~200K** | correct / 171 s / 1,168, 4 preemptions | **correct / 94 s / 2,121, 0** | **1.82x** |
| **155K cold prefill tok/s, preemptions** | 1,310, 2 | **2,132, 0** | **1.63x** |
| Concurrency 2 / 4 / 6: aggregate tok/s | 165.0 / 259.5 / 337.1 | 188.3 / 251.4 / 324.9; repeat 189.5 / 283.3 / 328.7 | same-boot spread up to 13% |
| Concurrency 2 / 4 / 6: per-stream decode median | 111.2 / 83.6 / 62.3 | 105.6 / 71.7 / 59.7; repeat 104.2 / 77.2 / 61.3 | |
| Staggered 100K (1,500 decode) + 40K: the 40K's TTFT | 26.5 s (R23) | 18.4 s | -8.1 s |
| Staggered: the 100K's decode while the 40K prefills | 170.8 tok/s (R23, alone) | 55.4 tok/s | shares steps |
| KV pool (tokens) | 299,482 | 299,482 | 0 |

The trace with the fix on: Mamba group 0 holds exactly 6 blocks at every one of 712 traced prefill allocations (never more), 697 state blocks freed, all with `ref_cnt` 1 (4 of them hashed: boundary states). Requests end holding 4 (state + 3 speculative). Warm and incremental turns start at 5 (the hit state + 3 speculative + the new state) and settle at 6.

**R25, the wrapper as shipped** (`MAMBA_FIX=1` by default, trace off; `results/quality-compare-R25-mambafix-default-20261018-023253.json`, `results/scoreboard-R25-*.json`): needles correct with 0 preemptions (200K 94.4 s); 155K 2,124 tok/s in the gate and 2,122 in the scoreboard, 0 preemptions; count 208.0 tok/s; prose 140.6 (acceptance 2.68) and code 178.9 (3.43), which follow the text they generate (R24: 2.89 and 3.49).

128K and 155K cold now run at the 64K rate less the attention cost of depth: 2,149 and 2,132 tok/s against 2,160 at 64K. Those runs had been computing 229.5K and 256K tokens; the real work was always ~2,150 tok/s, as the hypothesis table predicted (~2,170).

## Result

**Kept** (R24; the hostpin wrapper defaults `MAMBA_FIX=1`, confirmed by R25).

H1 was the leak. The simulation with the image's own block manager reproduced the measured 0 / 2 / 2 / 4 preemptions and R10's 256K tokens for a 155K prompt. The first instrumented boot showed the exact off-by-one-chunk in the trace, with no state block freed during prefill and the predicted preemption points and re-admission hits. So the task did not need escalating.

The fix keeps every recorded state slot on a list until the image's own safety test frees it. It holds 6 blocks per Mamba group per prefilling request where the leak held one more per chunk. It removes every preemption of a lone request up to the 200K needle (and, offline, up to 262K). Long cold prompts then prefill at the 64K rate: 128K 1.77x, 155K 1.63x, needle 1.82x. Decode, short cold, warm and incremental rows stay inside R22's noise, and the gate passes.

## Quality gate

R24 (`results/quality-compare-R24-mambafix-20261018-020808.json`): **PASS**. 14 IDENTICAL, 6 FP-SIZED:
- `code_bugfix` at token 201, margin 0.25;
- `prose_explain` at 46, 0.125;
- `prose_summary` at 102, 0.125;
- `prose_email` at 38, 0.125;
- `prose_compare` at 89, 0.000;
- `vision_halves` at token 0, margin 0.875. Earlier gates had this prompt IDENTICAL or FP-SIZED at token 1, margin 0.375 (R14, R22). The fix cannot reach a 113-token request that decodes 8 tokens. One chunk and no block boundary means no state slot is ever recorded, so its code path is the image's, and this is the T14 noise; its answer passed.

p99 |dlogprob| 0.116 over 1,658 positions (R14 noise 0.119-0.124). All 5 tool calls parse to the expected tool. Thinking is off by default, and the reasoning split works when on. Long QA is correct at 8K/16K/30K. Needles: 29,958 tok in 13.8 s, 99,726 in 46.4 s, 199,393 in 94.1 s, all correct with **0 preemptions**. 155K cold: 154,580 tok in 72.7 s (2,126 tok/s), **0 preemptions**, min free 0.66 / 0.66 / 0.61 / 0.66 GiB. Pool 299,482.

R25 (shipping config): **REVIEW**. 11 IDENTICAL, 8 FP-SIZED (among them `tool_write` token 4 at margin 1.25, the known T1 case), p99 0.100 nat, needles and 155K as above, and 1 SUSPECT: `tool_search` at token 9.
- **The value:** the baseline has 79114 at -0.023 with 28 at -3.77; R25 picked 28 at -0.38687 over 79114 at -1.13687.
- **R21 had the same numbers to 5 digits,** with no Mamba fix; R22 and R24 picked 79114.
- **The fix's code path for this request is the image's.** 1,309 prompt tokens with an 832-token prefix hit and 38 output tokens is one chunk and no new block. The only recorded state slot is the hit block, and both versions free it at the same step.
- **12 idle repeats right after the gate on this boot** (`.cache/ple/r21-repeat-tool-search.py`, output in `.cache/ple/r25-repeat-tool-search.out`) all pick 79114, with margins 5.3-7.8 nat (R21's repeats: 1.25-5.25).

Reviewed: T14 noise, not the fix; noted in INDEX as T14's sharpest probe.

The fix changes which pages hold which state, never the computation: the same state is read from and written to the same logical slots. The divergences are the lane's usual near-tie noise.

## Side findings

- **Concurrent cold prefills never overlap in this lane.** The step budget is 2,048 tokens and align mode cuts chunks at 832-token boundaries. After one 1,664-token chunk, 384 tokens are left, less than a block, so a second cold prompt waits until the first has finished prefilling (2 x 100K: TTFT 45 and 90 s). A bigger budget or chunk (T8) would let two prefills share a step.
- **With the fix, a long prefill shares steps with another seat's decode.** Before, the leaked blocks happened to keep a second long prompt out of the pool until the first request finished decoding. Now the second prompt is admitted at once: its TTFT improves (26.5 -> 18.4 s in the staggered test), and the decoding seat drops to ~55 tok/s while each step also carries a 1,664-token chunk. That is vLLM's ordinary chunked-prefill trade-off, which shorter prompts already hit; a policy that protects decode (smaller chunks while decodes run) would be its own task.
- **Why the old pattern was finite (2, 2, 4) instead of an endless preempt-restart loop.** With retention interval 0 no interior Mamba state is cached, so the first restart hits nothing. But `find_longest_cache_hit` reports the attention-only prefix as an uncached shared prefix. The scheduler then stops a chunk exactly at that junction and caches its state, so the next restart resumes there. Each preemption pair cost ~101.5K recomputed tokens at first, then ~66.5K.
- **The pool sizer assumes 5 Mamba blocks per group per request** (`MambaSpec.max_memory_usage_bytes`, align: 2 + speculative). With async scheduling a prefilling request needs 6: the block the in-flight chunk reads, the one it writes, and the one the next chunk writes. That is 4 blocks per seat more than the sizer counts. The logged 299,482 tokens are unchanged, and a 262,144-token request still fits (316 + 1 + 24 = 341 of 384 blocks).
- **The fix also frees the hit-state block of warm turns.** On admission with a prefix hit, the image records the hit state slot. The next allocation overwrites it under the same lag, so the image kept a reference to that cached block until the request ended. The fix frees it a chunk later, which makes the cached block evictable while the request runs.
- **New tools:**
  - `tools/mamba_leak_sim.py` (offline, the image's KV manager with this lane's groups) is a cheap regression test for any later KV-manager or chunk change; T8 should run it with `CHUNK=4096`.
  - `tools/long_pair.py` (`--stagger`) measures a second seat behind a long request.
- **Weight loading varied** between 144 s (R23) and 223 s (R24) on the same lane with nothing changed (page cache).
- The trace overlay costs nothing measurable, but it is a log line per chunk for long prompts. Leave `MAMBA_TRACE` off except for diagnosis.

## Open questions

- Whether vLLM after 8a728663 already fixed this path upstream (the processed-token basis in `allocate_slots` is recent). When the image moves, drop the overlay rather than port it, and rerun `mamba_leak_sim.py` with `QWEN4EXP_MAMBA_FIX` unset against the new image.
- How much the decode-during-prefill slowdown matters for the agent workload (one long seat plus short ones). It is not a T4 regression, but it is now reachable for long prompts too.
- Seats at depth: with the fix, two 128K contexts fit together (2 × (154 attention + 1 ring + 24 Mamba) = 358 of 384 blocks), three 100K contexts do not (3 × (121 + 1 + 24) = 438). Untested on the server.

## Next step

T14 (temperature-0 nondeterminism; Opus 5.5 / xhigh), with the `tool_search` token 9 observation above as its first probe, then T10. The hostpin lane now defaults to `MAMBA_FIX=1`; the prefix and tuned lanes are unchanged (both knobs default to 0 in the base launcher, so nothing is mounted for them). The server was **stopped** after R25 (GPUs at 15-92 MiB at 02:45).
