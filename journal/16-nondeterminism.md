# 16: Temperature-0 nondeterminism, root cause (T14)

Status: done · Model: Opus 5.5 · Effort: xhigh · Started: 2026-10-18 03:11 · Wall time: 3:35 (12 boots, R26-R38; R26 lost to a session crash)

## Goal

Find why the same request on an idle server gives different temperature-0 logprobs from run to run, so that a later candidate that changes numerics can be judged against a known noise floor, or against bit-exact outputs if the source can be removed. Success: the source is named down to the kernel or the line, and one change makes repeats of the probe requests bit-identical, with the noise back when the change is reverted. A lossless fix becomes a knob; whether it ships is decided on its cost.

## Hypothesis and kill test

What is known (T1, T2, T11, T4):
- `repeat_probe.py` (33 tokens, `prompt_logprobs`): 6 of 6 runs distinct on the prefix lane (R15), with MTP off (R17) and on the tuned lane without prefix caching (R18). Every position differs, from position 0, the prediction from one token (spread 0.02-0.1 nat at most positions, up to 0.4-0.7 at a few). R17 and R18 show no recurring values: each run differs from every other at every position.
- Ruled out by microbenchmarks: cuBLAS bf16 GEMMs (also at shifted addresses), the NCCL all-reduce, the QSA top-k on a short context. Marlin atomics are off at this commit.
- Ruled out by boots: speculative decoding (R17), prefix caching and align mode (R18).
- `tool_search` token 9 in the gate: on R21 and R25 it gives the same logprobs to 5 digits (token 28 at -0.38687, 79114 at -1.13687), on R22 and R24 the baseline token. Idle repeats on one boot spread its margin over 1.25-7.8 nat.

Hypotheses, each with the observation that would pin it:

| # | Mechanism | Fits | Pinned by |
|---|---|---|---|
| H1 | **Shared-expert side stream.** `SharedExperts` runs the shared expert on `aux_stream()` when a batch has at most 256 tokens (`VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`; `fused_moe/runner/shared_experts.py:121`). The runner's own comment says the shared input aliases the routed input unless the routed experts quantize their activations (`moe_runner.py:285`); W4A16 does not, and the guard is applied on ROCm only | short prompts and every decode step; timing-dependent | the first divergent entry of a trace is the shared expert or the MoE block with equal inputs; `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` makes repeats identical; prompts above 256 tokens repeat exactly |
| H2 | **Stale memory read.** A kernel reads memory that was not written for this request: an uninitialised scratch buffer (`torch.empty` partials of the QSA split-k), Mamba, conv or PLE state of a fresh request, or KV slots beyond the context, without an exact mask | a deterministic function of the request history, which would explain bit-equal values on different boots after the same gate sequence | a module that reads state diverges with equal inputs, and the divergence follows the physical block or the previous request |
| H3 | **Order-dependent reduction.** Atomics or a split reduction whose order varies: GDN chunk kernels, QSA split-k merge, HC kernels, PLE conv, Marlin MoE | a continuous spread on every run | one module diverges with equal inputs on every run, with no dependence on history or token count |
| H4 | **Async-scheduling input race.** A pinned host buffer for step N+1 overwritten before step N's copy lands (positions, slot mapping, state indices) | timing-dependent; would usually give large errors | the root module's inputs (token ids, positions) or attention metadata differ between repeats |

Kill test for the method: if the tracer itself makes the repeats bit-identical (its syncs and extra kernels change the timing), the trace cannot localise a race. That alone points to H1 or H4, and the next boot tests H1 by its environment switch instead.

## Setup

- Branch / commit: `perf-comm-ple` @ `3bf6d17`
- Lane: the hostpin lane as shipped (`launch-qwen38fn-w4a16-hostpin.sh`: `TRITON_CACHE=1 TOKSPLIT=1 MAMBA_FIX=1`), `GPU_ARGS="--device nvidia.com/gpu=all"`, image `qwen38fn-vllm:8a728663`. R25 is the reference for the shipping config.
- Ledger rows: R26 onward
- Box state before: GPUs free (15-91 MiB at 03:11), host server down, no server in the sandbox

## Design

**A bit-exact activation tracer, one diagnostic boot.** Instead of one boot per suspect, fingerprint the input and output of every `nn.Module` call in every eager forward on every rank, send the same request several times, and find the first module call whose fingerprint differs between repeats. If its inputs are identical, the nondeterminism starts inside that module.

- `vllm-w4a16/patch/det_trace.py`, mounted as `vllm/det_trace.py` and loaded in every vLLM process as a general plugin. A `*.dist-info` directory with one entry point is mounted into `dist-packages`, and `worker_base.py:271` calls `load_general_plugins()` in each worker. This needs no change to any shipped file; the launcher knob `DET_TRACE=1` mounts both and sets `QWEN4EXP_DET_TRACE=1`.
- `register()` installs global forward pre- and post-hooks (`torch.nn.modules.module.register_module_forward_{pre_,}hook`). A call that starts while no record is open opens one (the model's forward, `compute_logits`, the MTP drafter), when `/root/.cache/det-trace/on` exists, the stream is not capturing a CUDA graph and the call has at most `QWEN4EXP_DET_TRACE_MAX_TOKENS` tokens (default 512). CUDA-graph decode replays run no Python modules, so decode is not traced; prefill is, and the probe already diverges in prefill.
- Fingerprint of a tensor: its bytes as int32, widened to int64, then the wrapped sums of x, x² and x times a 1..1024 ramp, plus the same per row (first dimension = tokens) for the first 64 rows. Computed in 4M-element chunks on the current stream, in stream order, so the value is the one the next kernel reads. Side-stream modules get their fingerprints on the side stream before its done-event, so the order still holds. One `torch.cuda.synchronize()` per record, then a JSON line per rank in `/root/.cache/det-trace/rank<N>.jsonl`.
- `vllm-w4a16/tools/det_trace_diff.py` groups records by rank, root module and the root's input fingerprint. Against the first record of each group it reports the first differing entry (module name, input or output, which rows), how many entries differ, and which module is first most often.

Alternatives considered:
- One boot per environment switch (`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`, `--no-async-scheduling`, `PLE_MODE=none`, no expert parallel). That is 6-7 min per boot plus probes and each answers yes or no. A trace answers "where" for all hypotheses at once, and the switches stay available to confirm.
- The torch profiler's kernel list. It shows which kernels run, not which ones vary.
- A sitecustomize shim, or an edit to a shipped overlay to load the tracer. The plugin entry point needs neither.

Probe plan for the trace boot (R26), in this order on an idle server:
1. Tracing off: `repeat_probe.py --reps 6` at 33 tokens and at ~330 tokens (above the 256-token side-stream threshold), and `tool_search_probe.py --reps 6`. This is the untraced baseline for this boot.
2. `touch .cache/ple/det-trace/on`, then the same probes with fewer repeats. Check that they still diverge with the tracer on (the method's kill test), then `rm` the sentinel.
3. `det_trace_diff.py` on the files. Then a confirming boot with the indicated switch or fix; success is 1 distinct output in 6 or more repeats of every probe.

Lossless: the tracer reads tensors and changes no values. A fix for a race or a stale read changes only which of the currently possible outputs comes out; it still goes through `quality_gate.py --compare` like any candidate.

## What I did

```bash
# tracer smoke test on one GPU, no server: a toy model, 3 identical forwards + 1 with a perturbed weight + 1 under graph capture
docker run --rm --device nvidia.com/gpu=0 -v $PWD/vllm-w4a16/patch:/p:ro -v $SP/dt-test:/t -v $SP/dt-test/out:/out --entrypoint python3 qwen38fn-vllm:8a728663 /t/t.py
python3 vllm-w4a16/tools/det_trace_diff.py $SP/dt-test/out   # repeats identical; the perturbed one first differs at l.1.a out; capture not recorded
# R26, launched 03:21:40
DET_TRACE=1 GPU_ARGS="--device nvidia.com/gpu=all" bash vllm-w4a16/launch/launch-qwen38fn-w4a16-hostpin.sh
```

- R26: overlays 11/11 and the plugin check passed; `[det_trace] installed` printed in the API server, the engine core and all four workers. The hostpin setup finished at 03:26:04 (179 s, 0 retries). Then the session running this task ended during weight loading. The sandbox's podman service restarted at 03:30, and the container was left `exited` with no further log lines and nothing measured. At 03:32: GPUs 15-92 MiB, no vLLM process, RAM back to 119 GiB available (the pinned table was released), no server on either port.

- R27 (03:35:53, the same command as R26; ready 03:40:23), then the probe sequence (`$SP/r27-probes.sh`, output in `.cache/ple/r27-probes.out`):
  ```bash
  python3 -c "...; B.wait_ready(); B.assert_idle('start'); B.warmup()"
  python3 vllm-w4a16/tools/repeat_probe.py --label R27-off-33 --reps 6 --copies 3      # tracer off
  python3 vllm-w4a16/tools/repeat_probe.py --label R27-off-330 --reps 6 --copies 30
  python3 vllm-w4a16/tools/tool_search_probe.py --reps 6 --out vllm-w4a16/results/tool-search-probe-R27-off.json
  touch .cache/ple/det-trace/on                                                          # tracer recording
  python3 vllm-w4a16/tools/repeat_probe.py --label R27-on-33 --reps 4 --copies 3
  python3 vllm-w4a16/tools/repeat_probe.py --label R27-on-330 --reps 4 --copies 30
  python3 vllm-w4a16/tools/tool_search_probe.py --reps 4 --out vllm-w4a16/results/tool-search-probe-R27-on.json
  rm .cache/ple/det-trace/on
  python3 vllm-w4a16/tools/det_trace_diff.py .cache/ple/det-trace [--outputs] [--rank 0]
  ```
  The trace (968 records, 4 x 32 MB) is kept in `.cache/ple/det-trace-r27/` (gitignored).

- R28 (03:46:27): R27 plus the stages inside the MoE block, wrapped as functions:
  ```bash
  F="vllm._custom_ops:topk_softmax,vllm._custom_ops:moe_align_block_size,vllm._custom_ops:moe_wna16_marlin_gemm,vllm._custom_ops:moe_sum,vllm.model_executor.layers.fused_moe.modular_kernel:FusedMoEExpertsModular.activation,vllm.model_executor.layers.fused_moe.runner.moe_runner:tensor_model_parallel_all_reduce"
  DET_TRACE=1 DET_TRACE_FUNCS="$F" GPU_ARGS="--device nvidia.com/gpu=all" bash vllm-w4a16/launch/launch-qwen38fn-w4a16-hostpin.sh
  # warm-up; repeat_probe.py --label R28-off-33 --reps 4 --copies 3; touch on; --label R28-on-33 and R28-on-330 --reps 4; rm on
  ```
  Trace kept in `.cache/ple/det-trace-r28/`. Then the server was stopped for microbenchmarks on one GPU:
  ```bash
  docker run --rm --device nvidia.com/gpu=0 -v $PWD/vllm-w4a16/tools:/tools:ro --entrypoint python3 qwen38fn-vllm:8a728663 \
    /tools/det_probe_moe.py --reps 30 --tokens 4,34,331 --rank 2                 # and --tokens 34,96,331,1664 with --skew 0 / 1.0 / 2.5
  # the MOEDET overlay offline: same sets, layout and padded count as the kernel; one order; CUDA-graph timing
  docker run --rm --device nvidia.com/gpu=0 -e QWEN4EXP_MOE_DET=1 \
    -v $PWD/vllm-w4a16/patch/moe_align_block_size.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/moe_align_block_size.py:ro \
    -v $SP:/sp:ro --entrypoint python3 qwen38fn-vllm:8a728663 /sp/moedet_check.py          # then /sp/moedet_triton_check.py
  ```
- R29 (04:03:56): `MOE_DET=1 DET_TRACE=1 GPU_ARGS=... bash vllm-w4a16/launch/launch-qwen38fn-w4a16-hostpin.sh`; the R27 probes, `tool_search_probe.py --reps 12`, and the new `long_repeat_probe.py --label R29-moedet --tokens 1500|2100|4000|16000 --reps 4`, then 2,100 tokens traced (`--reps 3`). Trace kept in `.cache/ple/det-trace-r29/`. Then, with no server, the top-k studies (`$SP/topk_study.py`, `$SP/topk_short.py`: the image's `_topk` repeated on fixed logits) and the first QSADET overlay check (mounted like the MOEDET one at `.../models/qwen4_exp/nvidia/ops/qsa_indexer.py` with `-e QWEN4EXP_QSA_DET=1`). `$SP` scripts were not kept; their final versions are the three `det_check_*` / `det_replay_*` tools below.
- R30 (04:14:48): `MOE_DET=1 QSA_DET=1 GPU_ARGS=... bash vllm-w4a16/launch/launch-qwen38fn-w4a16-hostpin.sh`; `quality_gate.py --compare --label R30-det`, the repeat probes, `long_repeat_probe.py --tokens 2100|4000|16000|24000|32000|40000|48000|64000`.
- R31 (04:33:40): `MOE_DET=1 QSA_DET=1 DET_TRACE=1 DET_TRACE_MAX_TOKENS=2048 DET_TRACE_FUNCS=<qsa_indexer:_topk,qsa_indexer:_prefill_logits> ...`; `long_repeat_probe.py --tokens 40000 --reps 2` untraced, then traced. Trace in `.cache/ple/det-trace-r31/`. Offline: the image's `_topk` with a zeroed, garbage or reused workspace, and with -inf / 0 / 1e30 / NaN / random stale scores past the visible columns (`$SP/topk_ws.py`, `$SP/topk_stale.py`).
- R32 (04:44:32): `MAMBA_FIX=0 MOE_DET=1 QSA_DET=1 ...`; `long_repeat_probe.py --tokens 40000|64000 --reps 3`.
- R33 (04:53:56): R31 plus `DET_TRACE_QSA=1` (logical views), only `_prefill_logits` wrapped; two traced 40K repeats. Trace in `.cache/ple/det-trace-r33/`. Offline: relu-like scores with exact-zero ties through the image's and the patched `_topk` (`$SP/topk_zeros.py`).
- R34 (05:06:01): R33 plus `DET_TRACE_DUMP="_topk:8192:0,1,2"`, `_topk` wrapped too; two traced 36K repeats. Dumps in `.cache/ple/det-trace-r34/dump/` (1.8 GB, gitignored). Offline: `tools/det_replay_topk.py` (the replay and the stress loop), then the exact top-k kernel and its checks: `tools/det_check_qsa_topk.py`.
- R35 (05:21:13): `MOE_DET=1 QSA_DET=1 ...` with the final QSADET; gate, all repeat probes up to 100K, the full `scoreboard.py --label R35`. Offline after it: the per-expert MOEDET kernel, `tools/det_check_moe_align.py`.
- R36 (05:46:21): the final MOEDET and QSADET; gate (compared bitwise with R35's and R30's reports), repeat probes, full scoreboard, two more `scoreboard.py --sections concurrency --no-warmup` runs.
- R37 (06:09:06): the hostpin lane without the two fixes, same session: `scoreboard.py --sections decode,concurrency --label R37` plus two concurrency repeats (the A/B).
- R38 (06:18:00): the hostpin wrapper as shipped (now `MOE_DET=1 QSA_DET=1` by default, no knobs given):
  ```bash
  GPU_ARGS="--device nvidia.com/gpu=all" bash vllm-w4a16/launch/launch-qwen38fn-w4a16-hostpin.sh
  python3 vllm-w4a16/tools/quality_gate.py --compare --label R38-shipped
  python3 vllm-w4a16/tools/quality_gate.py --record vllm-w4a16/results/quality-baseline-det-2026-10-18.json --label R38-record
  python3 vllm-w4a16/tools/quality_gate.py --compare vllm-w4a16/results/quality-baseline-det-2026-10-18.json --label R38-vs-det
  ```
- Mistake: after R35's scoreboard I started the MOEDET microbenchmark with R35 still up. It failed at its first allocation (CUDA OOM); the server answered 200 and logged no error, and every R35 measurement had finished before. Stopped, then reran.

## Measurements

**Source 1: the MoE token order.** R27's trace, every rank and every repeat: the first tensor that differs is a MoE block's output (`mlp.experts out`, layer 0, or layer 1 or 3 at 33 tokens), in 1-2 token rows. Its inputs, the router logits (`mlp.gate out`) and the shared expert's output are bit-identical, and so is everything before it. R28 inside that block, first MoE layer:

| Stage (rank 2, 34 tokens) | Repeats |
|---|---|
| `topk_softmax` outputs (ids, weights) | identical |
| `moe_align_block_size` `sorted_token_ids` | **different on every call** (the order of each expert's tokens; the sets, `expert_ids` and the padded count are identical) |
| `moe_wna16_marlin_gemm` x2, activation | differ (these buffers also hold stale rows for other ranks' experts, so the fingerprint alone does not say more) |
| `moe_sum` output | differs in 2 token rows in 2 of 3 repeats (other ranks: identical, or differ in other repeats) |
| MoE all-reduce output | differs on all ranks |

`det_probe_moe.py`, one GPU, this lane's per-rank shapes (`results/` not written; output in the session):

| Tokens, routing | `fused_marlin_moe` repeated | `sorted_token_ids` distinct | Fixed order | Tokens moved between an expert's blocks | Shuffled inside each block | `moe_sum` alone |
|---|---|---|---|---|---|---|
| 331, uniform (first run) | 3 of 30 distinct | 30 | 1 | 28 distinct, 13 tokens change, max 6.1e-5 | - | 1 |
| 331, uniform (second draw) | 1 | 30 | 1 | 13 tokens change, max 6.1e-5 | 0 change | 1 |
| 34 / 96, uniform | 1 | 8 / 29 | 1 | 0 (no expert over 8 tokens) | 0 | 1 |
| 34-1664, skewed (16 hot experts) | 1 | 5-30 | 1 | 0 | 0 | 1 |

So the Marlin MoE kernel is deterministic for a given token order, and the order inside an 8-row block never matters. Which block of an expert a token lands in can change its result by one bf16 step. Not every layout is sensitive: the skewed ones were not, and the sensitive layouts presumably split K across SMs at different points (not proven). The atomic align kernel decides the block.

**Fix 1, `MOE_DET=1`** (`patch/moe_align_block_size.py`): after the kernel, each expert's token list goes back in ascending order. When the batch has no more tokens than `block_size_m`, this is skipped: no expert can then fill two blocks. That covers decode at 1-2 streams (4-8 tokens). Up to 256 token-expert pairs a single Triton program re-places the tokens: segment start plus the count of earlier pairs with the same expert. Above that a `torch.sort` over (segment, pair index) does it. Checks:
- Both paths give the same layout, the same per-expert sets and the same padded count as the kernel.
- The Triton and torch paths agree in 306 of 306 random cases (9-25 tokens, 3 ranks, 3 skews, with and without `expert_map`).
- 1 distinct order in 30 repeats at every size, in a CUDA graph as well.
- Cost in a CUDA graph, 48 calls (one per MoE layer): +0.24 ms per step at 12 tokens, +0.60 ms at 24 tokens. The torch path alone would cost 2.8 ms, and an untiled Triton version 4.8 ms at 24 tokens (register spills).

R29 (`MOE_DET=1`): **33 tokens 1 of 6 distinct, 330 tokens 1 of 6, `tool_search` 1 distinct in 12 (token 9 = 79114 every time), 1,505 tokens 1 of 4.** Against R27 on the same lane without the fix: 6 of 6, 6 of 6, 2 of 6, and the 1,505-token prompt untested.

**Source 2: the QSA top-k.** R29, still with `MOE_DET=1`: 2,103 / 3,998 / 15,957 tokens give 4 of 4 distinct outputs each (`long_repeat_probe.py`, max spread 0.15-0.33 nat), while 1,505 tokens gives 1 of 4. The trace of the 2,103-token prompt's second chunk (439 tokens): the first difference is `layers.3.self_attn.indexer out` (the first QSA layer's selection, `[439, 2051]` int32) with identical inputs. Its first 64 rows, whose context is below the 2,048-token budget, are identical. The 1,505-token prompt never selects (fewer than 512 compressed blocks visible).

`persistent_topk` on fixed logits (`$SP/topk_study.py`, 30 runs; the same measurement as T2's `det_probe_topk.py`, extended):

| Case | Order distinct | Set distinct | Set = exact top-k by value |
|---|---|---|---|
| 439 rows at 2.1K, fp32 random | 30 | 1 | yes |
| 1,664 rows at 16K / 64K, fp32 random | 30 | 2 | yes; the swapped blocks score exactly the cut |
| 4 rows at 64K | 30 | 1 | yes |
| 1,664 rows at 16K, bf16-rounded scores | 30 | 30 | every swapped block scores exactly the cut |

So the kernel is exact, but it returns the 512 blocks in a new order on every call, and among blocks tied with the cut it takes a different one each time. Rows with at most 512 visible blocks come back as `arange(visible)` then -1, deterministically.

**Fix 2, `QSA_DET=1`** (`patch/qsa_indexer.py`): after `persistent_topk`, each row with more than 512 visible blocks is rewritten by one Triton program per row. The cut is the smallest selected score. The row gets every block above the cut plus the lowest-index blocks equal to it, in ascending order. Checks on six cases (fp32, bf16-rounded and 1/8-step scores; 2.1K to 128K; prefill and decode rows):
- 1 distinct output in 30 calls.
- Equal to a torch reference (stable descending sort, first 512, ascending).
- The same multiset of selected scores as the image's top-k.
- Short rows untouched.
- Cost per QSA layer: a 1,664-row chunk at 64K goes from 0.37 to 0.69 ms (12 layers: +3.9 ms on a ~800 ms step); 24 decode rows at 128K go from 0.04 to 0.09 ms in eager timing (launch-bound).

That first version of the fix (R30) took the cut from `persistent_topk`'s own selection. R30: everything up to 32K tokens became bit-identical (2.1K, 4K, 16K, 24K, 32K: 1 distinct each), but 40K, 48K and 64K still gave 3 of 3 distinct.

**Source 3: `persistent_topk` is not exact on real scores at depth.**
- R31: tracing whole 1,664-token chunks of two cold 40K repeats, chunks 1-20 match in every module output and every top-k selection. Chunk 21 (context 33,280-34,944) first differs at QSA layer 7's selection.
- R32 rules out T4's Mamba fix: with `MAMBA_FIX=0`, 40K and 64K still give 3 of 3.
- R33 adds logical views at `_prefill_logits`: the request's own key pages in logical order, and the visible scores only. In chunk 20, layer 7's views are bit-identical across the two runs, yet its selection differs.
- R34 dumps the `_topk` calls. Replayed offline on the same scores (`tools/det_replay_topk.py`), the image's `persistent_topk` returns a set that is **not the top 512** in about 6 rows per call. It does so idle or under load, and in which rows varies per call. The drafter's QSA layer has up to 175 such rows. The example row selected a block at 9.9525 while the true 512th score is 9.9678.
- Synthetic scores never trigger it: random fp32, bf16-rounded, 1/8-step and relu-like with thousands of exact-zero ties, with -inf/0/1e30/NaN/random stale columns, a zeroed or garbage workspace, 14K-200K visible.
- The real scores cluster tightly near the cut (9.6-10.0 over ~8K blocks), and only that makes it fail. A first-version cut taken from such a selection is itself wrong, so the rewrite inherited the error.
- vLLM's release notes for v0.20.1/0.20.2 list fixes in this kernel ("inter-CTA init race on RadixRowState", #41444; a memset that must run at graph capture, #41665). This image's compiled kernel, at 0.28.1rc1, still misses rows on these inputs; I could not read its source here.

**Fix 2, final form, `QSA_DET=1`** (`patch/qsa_indexer.py`): `_topk` no longer trusts `persistent_topk` for its set.
- 64 rows or more (prefill): `persistent_topk` is not called. One Triton program per row runs a radix select: 4 passes of 8 bits over an order-preserving 64-bit key of the fp32 score. That gives the exact 512th-largest key; a fifth pass writes the blocks above it plus the lowest-index ties, in ascending order.
- Fewer rows (decode): `persistent_topk` runs as a hint. A count kernel (rows x 16 column slices) counts each slice's scores above and equal to the hint's smallest key; split 0 stores that key, so nothing reads the hint while it is being overwritten. The hint is exact iff fewer than 512 lie above and at least 512 at or above. Then each slice writes its part at offsets from the slices before it. Otherwise split 0 runs the radix select for that row.
- Rows with at most 512 visible blocks get `arange(visible)` then -1, as the image writes them.

Checks (`tools/det_check_qsa_topk.py`):
- Equal to a torch reference (stable descending sort of the same keys, first 512, ascending; short rows `arange`) in all 24 cases: synthetic prefill at 2.1K-260K, decode rows at 32K-250K, bf16, tie-heavy, mixed short and full rows, the real dumps, and 32-row hint-path slices containing rows where the image's kernel was inexact (the fallback fires).
- 1 distinct output in 5-10 repeats each.

Cost:

| Call | Image `_topk` | `QSA_DET` |
|---|---|---|
| 1,664-row chunk at 16K / 35K / 64K / 260K (eager) | 0.47 / 0.62 / 0.90 / 1.83 ms | 0.28 / 0.57 / 1.00 / 4.06 ms |
| 12 layers in a CUDA graph, 4 decode rows at 32K / 128K / 250K | | +0.09 / +0.08 / +0.18 ms per step |
| 12 layers in a CUDA graph, 24 decode rows at 128K / 250K | | +0.19 / +0.33 ms per step |

**The lane with both fixes is bit-reproducible at temperature 0** (R35, R36, R38):

| Probe | Without (R27, R29, R30) | With both (R35 / R36) |
|---|---|---|
| `repeat_probe.py`, 33 / 330 tokens, `prompt_logprobs` | 6 of 6 / 6 of 6 distinct | 1 of 6 / 1 of 6 |
| `tool_search_probe.py`, 12 repeats | 2 outputs (token 9 flips between 79114 and 28) | 1 (79114, margin 3.75 every time) |
| `long_repeat_probe.py`, 2.1K / 4K / 16K | 4 of 4 distinct (R29) | 1 of 4 / 1 of 3 / 1 of 4 |
| 40K / 64K / 100K | 3 of 3 / 3 of 3 / - | 1 of 3 / 1 of 3 (1 of 2) / 1 of 2 |
| Gate, 20 prompts, R30 vs R35 vs R36 vs R38 (four boots) | 6-9 of 20 differ on a same-boot repeat (T1) | all 20 bit-identical in ids and logprobs |
| Gate against a record made on the same boot (R38) | p99 \|dlogprob\| 0.12 (T1, R14) | 20 IDENTICAL, drift 0.000, p99 0.000 over 2,019 positions |

Cost (scoreboard, R36 against R37 on the same session; `results/scoreboard-R36-*`, `scoreboard-R37*`):

| Metric | R37 (without) | R36 (with) |
|---|---:|---:|
| Decode, count to 100 (tok/s) | 204.6 | 204.1 |
| Concurrency 2 / 4 / 6, aggregate tok/s, three runs each | 181.0-190.0 / 261.9-269.7 / 302.7-304.9 | 187.9-194.8 / 263.9-265.9 / 301.0-308.9 |
| Cold TTFT 1K / 64K / 128K (R24 for R37's lane) | 0.50 s / 29.52 s / 59.38 s | 0.51 s / 29.36 s / 59.56 s |
| Warm TTFT 1K / 16K / 64K / 128K (R24) | 0.16 / 0.18 / 0.37 / 0.37 s | 0.16 / 0.19 / 0.38 / 0.36 s |
| Needle ~200K, 155K cold (R24) | 94 s, 2,132 tok/s | 95 s, 2,123 tok/s |

The prose and code decode rows changed with their text: the outputs are now fixed and differ from R24's draw (acceptance 2.72 against 2.89 on prose; R25 showed the same drop). Microbenchmarks put the overhead at +0.13-0.15 ms per decode step for MOEDET at 12-24 tokens, 0 at 1-2 streams. QSADET adds +0.08-0.18 ms at 1 stream and +0.19-0.33 ms at 6 streams at 128K-250K depth, and +1-2 ms per deep prefill step at 64K-260K (+0.1-0.2%). All are inside the scoreboard's noise.

## Result

**Root cause found; both fixes kept, on by default in the hostpin wrapper.** There were three sources, all in the order or choice of what gets summed:
1. **The MoE token order.** `moe_align_block_size` fills each expert's blocks in the order its atomics happen to run. Which 8-row (prefill: up to 48-row) block of its expert a token lands in can change its Marlin W4A16 GEMM result by one bf16 step. It hits every prompt longer than `block_size_m` tokens and every multi-stream decode step. This was the whole short-prompt noise of T1/T2.
2. **The QSA top-k order and ties.** `persistent_topk` returns the 512 selected blocks in a new order on every call, and takes a different block among those tied with the cut. The sparse attention sums in that order. This hits every prompt beyond the 2,048-token budget.
3. **The QSA top-k set itself, at depth.** On real indexer scores beyond ~8K visible blocks (~33K tokens), `persistent_topk` returns a set that is not the top 512 in a few rows per call, and which rows varies. This is a selection error, not only reordering: in the example row a block scoring 9.9525 replaces one at 9.9678.

`MOE_DET=1` (`patch/moe_align_block_size.py`) puts each expert's tokens in ascending order. `QSA_DET=1` (`patch/qsa_indexer.py`) computes the exact top-k: lowest index on ties, ascending order. Both fix only which of the computations' valid orders and tie choices is taken; QSADET in addition replaces a slightly wrong selection with the exact one. Gate PASS, no measurable cost in a same-session A/B. With both, a request repeats bit for bit on an idle server, run to run and boot to boot.

T4's Mamba fix was not involved (R32). Async scheduling, prefix caching, speculation and the PLE path were already excluded or never showed in a trace.

## Quality gate

`quality_gate.py --compare` against the frozen R14 baseline on R30, R35, R36 and R38: **PASS** each time, with the same report every time: 13 IDENTICAL and 7 FP-SIZED, p99 0.110 (`results/quality-compare-R38-shipped-*.json`).
- The 7 FP-SIZED prompts (`code_func`, `code_sql`, `code_bash`, `prose_*`) first diverge from R14 at margins 0-0.375 nat. R14 is itself one draw of the old noise, and these are now the same divergences every time.
- `tool_write` (token 4, margin 1.25) and `vision_halves` (token 0, margin 0.875), the two known noisy prompts, are IDENTICAL now. `tool_search` token 9 gives 79114 at margin 3.75 every time.
- All tool calls parse. Needles correct at 29,958 / 99,726 / 199,393 tokens with 0 preemptions (200K 93-95 s). 155K cold 2,104-2,154 tok/s, 0 preemptions, min free 0.60 GiB. Pool 299,482.

New: `results/quality-baseline-det-2026-10-18.json`, recorded on R38 (the shipped wrapper). Comparing against it on the same boot gives 20 IDENTICAL, drift 0.000. A later change that should be numerically neutral must give IDENTICAL against it; anything else shows its exact first divergence. The frozen R14 file is unchanged.

## Side findings

- **`persistent_topk` in this image is not exact on real QSA scores at depth.** This affects every lane that runs the QSA indexer without `QSA_DET`: the prefix and tuned lanes, and the host's daily driver.
  - In a few rows per layer and chunk (up to 175 in the MTP drafter's layer), it selects a block below the true cut.
  - Synthetic scores never trigger it, so a unit test with random data passes.
  - vLLM's v0.20.1/0.20.2 notes list fixes in this kernel ("inter-CTA init race on RadixRowState", #41444; a capture-time memset, #41665). This build still fails, so a newer image needs `tools/det_replay_topk.py` run against R34's dumps before the overlay is dropped.
  - Whether it matters for quality: the swapped blocks sit right at the cut, and the gate and needles pass with and without the fix.
- **The tracer is reusable.** `patch/det_trace.py` with `DET_TRACE=1` (base launcher) fingerprints every module call. `DET_TRACE_FUNCS` wraps named functions, including custom ops called by attribute. `DET_TRACE_QSA` adds logical views where buffers carry stale regions, and `DET_TRACE_DUMP` saves tensors. `tools/det_trace_diff.py --outputs` compares records. Two traps:
  - Whole-pool KV buffers, block tables and `torch.empty` output buffers differ between runs without meaning anything, because physical blocks differ. Compare module outputs or logical views.
  - The trace of a 1,664-token chunk is ~7 MB per rank in JSON.
- **The stale QSA indexer output buffer is harmless.** `[T, 2051]` int32 is fully overwritten (R27).
- **`cache_salt` gives cold repeats of a long prompt on one boot** (`tools/long_repeat_probe.py`). Before this, a repeat of a long prompt always hit the prefix cache.
- **Concurrency rows drift with the box state.** R24 measured 325-329 tok/s at 6 streams at 02:20; R36/R37 measure 301-309 at 06:00-06:15 with or without the fixes. Concurrency A/Bs need same-session baselines.
- **The weight loader logs "Auto-prefetch is disabled ... checkpoint size (116.21 GiB) exceeds 90% of available RAM (110.18 GiB)".** That is the T11 open question about a sequential prefetch: vLLM has one, and it is off here only by that size test. 116 GiB includes the PLE table files.
- **What determinism covers.** The same request with the same batch composition gives the same bits. Concurrent requests change the batch shapes (GEMM M, MoE blocks), so a request's output can still depend on what runs next to it. R35 and R36 gave identical acceptance at 2 and 4 streams, but that is not a guarantee.

## Open questions

- Should the prefix and tuned lanes get `QSA_DET=1`, given that the image's top-k is wrong at depth there too? The bounds say those lanes stay unchanged; the user decides. It is a two-variable switch (`MOE_DET=1 QSA_DET=1`).
- Should the gate's thresholds be tightened now that same-config repeats are exact? A numerics-neutral change should be IDENTICAL against `quality-baseline-det-2026-10-18.json`; a numerics-changing one (T6, T10) still needs the margin and drift thresholds against R14.
- Which part of `persistent_topk` fails on clustered scores? Not needed while QSADET bypasses it, but worth reporting upstream with R34's dumps.

## Next step

**T10** (TP2 x PP2 feasibility; Fable 5.1 / xhigh for the feasibility stage, per the board), as INDEX lists. On the hostpin lane, T10's quality claim can now be checked bitwise. Expect FP-sized divergences against both baselines (the all-reduce sums differ), but run-to-run noise is zero, so every difference is the change's. The server was **stopped** after R38 (06:37); GPUs at 15-92 MiB.
