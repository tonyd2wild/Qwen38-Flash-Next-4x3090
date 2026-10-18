# 02: Time accounting, floors and the portfolio re-rank (T2)

Status: done · Model: Opus 5.5 (the board says Opus for the floors and the re-rank, Sonnet for the scripts; the user ran the whole task on Opus) · Effort: high · Started: 2026-10-17 20:41 · Wall time: 1:35 (microbenchmarks, four boots)

## Goal

Account every millisecond of a prefill step, a decode step and an incremental turn. Set each component against a measured floor on this box, then re-rank the portfolio by the gaps. Add T1's nondeterminism probe.

## Hypothesis and kill test

From PLAN: the all-reduce (59% of the deep prefill step) runs far below the PCIe rate, and NCCL env tuning (P1) should get 8.5 MB under ~1.5 ms; the 190-390 ms of host-bound idle per step is probably the PLE gather. Kill test for P1: `nccl_bench.py` finds no setting under ~1.5 ms. **P1 is killed**. The reason is not NCCL. The host side of this box moves only ~29 GB/s in total, and NCCL already runs at 82% of that floor. The PLE hypothesis holds: the idle is one gap per step, and it is the gather.

## Setup

- Branch / commit: `perf-comm-ple` @ `7749103`
- Ledger rows: R15 (prefix lane; turn overhead and probe), R16 (prefix lane flags + `--profiler-config`; decode and 6-stream profiles), R17 (MTP=0; nondeterminism), R18 (tuned lane; nondeterminism). Microbenchmarks ran with no server.
- Box state before: GPUs free (15-92 MiB), host server down, page cache warm for the weights but **only 17% of the PLE table resident** (mincore sample)

## What I did

New tools, all under `vllm-w4a16/tools/`, with usage in their header comments:

- `trace_breakdown.py`: sums a torch-profiler trace by kernel category per step kind.
  - A step is accounted from its start to the next step's start. The annotation ends before the MTP draft and the sampler, so those would otherwise be lost.
  - It reports busy time, idle, and the idle's attribution: the innermost CPU op running before the next launch, and the previous -> next GPU op pattern.
- `nccl_bench.py`: torchrun with 4 processes, one fresh container per NCCL setting.
  - Measures all-reduce and all-gather at decode and prefill sizes, eager and CUDA-graph.
  - Kills a setting that hangs after `--timeout`.
- `floor_bench.py`: bf16 GEMM at the lane's shapes, HBM copy, and pinned PCIe. For PCIe, all four cards run in the same wall-clock slots, so concurrent runs really overlap.
- `ple_gather_bench.py`: the staged gather (`PLEMmapTable.gather_cpu`) with 1 and 4 processes, a pinned-memory UVA GPU gather prototype, and page-cache residency of the table.
- `det_probe.py` and `det_probe_topk.py`: bitwise repeatability of cuBLAS, NCCL and the QSA top-k. `index_add_` is the positive control.
- `turn_overhead.py`: the same warm prompt sent as chat, as completions text, and as completions token ids, with the server histograms. `--probe N` repeats a short prompt with logprobs.
- `repeat_probe.py`: per-position `prompt_logprobs` across repeats.

```bash
python3 vllm-w4a16/tools/trace_breakdown.py .cache/ple/prof-pc1          # and prof-pc1-shallow, prof-pc0; summaries in results/trace-breakdown-T2.txt
GPU_ARGS="--device nvidia.com/gpu=all" python3 vllm-w4a16/tools/nccl_bench.py --label T2 --settings base,simple,ll128,shmcpy,ch8,simple_ch8,nop2p,tree
GPU_ARGS=... python3 vllm-w4a16/tools/nccl_bench.py --label T2-tree --settings tree          # tree crashed the first driver (fixed)
GPU_ARGS=... python3 vllm-w4a16/tools/nccl_bench.py --label T2-extra --timeout 240 --settings 'shmcpy_cumem=NCCL_SHM_USE_CUDA_MEMCPY=1:NCCL_CUMEM_ENABLE=1,buff16=NCCL_BUFFSIZE=16777216,nthr512=NCCL_NTHREADS=512,ch16'
GPU_ARGS=... python3 vllm-w4a16/tools/floor_bench.py --label T2; GPU_ARGS=... python3 vllm-w4a16/tools/floor_bench.py --label T2-pcie --pcie-only
GPU_ARGS=... python3 vllm-w4a16/tools/ple_gather_bench.py --label T2 --modes resident,staged,hostpin; ... --label T2-take --modes take
GPU_ARGS=... python3 vllm-w4a16/tools/det_probe.py
docker run --rm --device nvidia.com/gpu=all -v $PWD/vllm-w4a16/tools:/tools:ro --entrypoint python3 qwen38fn-vllm:8a728663 /tools/det_probe_topk.py
# R15 (the profiler env var turned out to be ignored)
GPU_ARGS=... DOCKER_EXTRA="-e VLLM_TORCH_PROFILER_DIR=/root/.cache/prof-t2" bash vllm-w4a16/launch/launch-qwen38fn-w4a16-prefix.sh
URL=http://localhost:8090 python3 vllm-w4a16/tools/turn_overhead.py --label R15 --depth 128000 --reps 2 --probe 10
# R16: the prefix lane's knobs through the base launcher, plus the profiler
TEXTRA='--quantization gptq_marlin --enable-expert-parallel --mm-processor-kwargs {"max_pixels":1048576} --mamba-ssm-cache-dtype bfloat16 --skip-mm-profiling'
SPEC_EXTRA='"disable_eagle_block_drop":true' PREFIX_CACHE=1 KV_BYTES=2400000000 LM_ONLY=0 NCCL_MODE=nvl PLE_MODE=staged GRAPHS=nocompile MTP=3 TP=4 \
  GMU=0.95 SEQS=6 CHUNK=2048 MAXLEN=262144 KV_DTYPE=fp8_e5m2 CAPTURE_SIZES=4,8,12,16,20,24 \
  EXTRA="$TEXTRA --profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/root/.cache/prof-t2\",\"torch_profiler_with_stack\":false}" \
  GPU_ARGS=... bash vllm-w4a16/launch/qwen38fn-w4a16-3090-tp4.sh
#   warm-up, then start_profile / code prompt 192 tokens / stop_profile, start_profile / 6 streams x 128 tokens / stop_profile
# R17: same base-launcher line with MTP=0 and no SPEC_EXTRA, no profiler; R18: launch-qwen38fn-w4a16-tuned.sh unchanged
URL=http://localhost:8090 python3 vllm-w4a16/tools/repeat_probe.py --label R17-mtp0     # and R18-tuned
URL=http://localhost:8090 python3 vllm-w4a16/tools/turn_overhead.py --label R17-mtp0 --depth 0 --probe 10
```

Boot logs are in `.cache/ple/r15-boot.log` through `r18-boot.log`.

## Measurements

The numbers live in the notes file, section "Time accounting and floors (T2)": the measured ceilings, the NCCL table, the prefill and decode floor tables, the turn overhead, and the re-ranked portfolio. The headline facts:

| Fact | Number | Source |
|---|---|---|
| Deep prefill step (1,664 tokens) | period 994 ms: all-reduce 471, **PLE gather idle 199**, QSA + top-k 116, GEMM 96, Marlin 43, HC 25, all-gather 14, rest 30 | R8 `prof-pc1`, `results/trace-breakdown-T2.txt` |
| Shallow prefill step | 1,209 ms, PLE idle **413** (the table's rows are cold early in a prompt) | `prof-pc1-shallow` |
| PLE gather, 26,624 rows | staged 280-610 ms in the bench (Python-bound: fresh and repeated ids cost the same); GPU UVA gather 0.43-0.82 ms with 4 GPUs; numpy gather from a resident mapping 1.7 ms | `results/ple-gather-bench-T2-20261017-212841.json`, `ple-gather-bench-T2-take-20261017-220718.json` |
| All-reduce 8.5 MB | 4.33 ms in every NCCL setting (Simple 4.36, LL128 8.9, Tree 5.17); copy-engine SHM hangs | `results/nccl-bench-T2-*.json` |
| Host path, 4 cards at once | 31.2 H2D / 28.9 D2H / 14.4 GB/s per direction both ways; DRAM copy 21-28 GB/s; 4 of 8 channels populated | `results/floor-bench-T2-pcie-20261017-212620.json`, EDAC |
| GEMM peak | 75.8 TFLOPS bf16 (43-73 at lane shapes); HBM 844 GB/s | `results/floor-bench-T2-20261017-212430.json` |
| Decode step, 1 stream | 22.4 ms (R8) / 23.0 profiled (R16, 59 steps): all-reduce 4.3-5.7, all-gather 2.1, GEMM 8.7 (draft 2.2), Marlin + glue 2.9, PLE 1.3; memory floor 6.6 ms | R8, R16 traces |
| Decode step, 6 streams | 41.7 ms per 16 tokens: all-reduce 16.0 (106 x 150 us), PLE 4.6, all-gather 4.4 | R16 trace |
| Warm 128K TTFT | 0.72 s = 0.30 prefill + **0.34 tokenization** + 0.08 client side; with token ids 0.42 s | R15 `results/turn-overhead-R15-*.json` |

## Result

**Done.** The floors are measured on this box, not taken from datasheets. Three of them change the plan.

1. **The all-reduce is at the host's floor, so P1 is dead and P2's prefill half nearly so.**
   - Without P2P, every byte of a 4-rank all-reduce crosses host memory twice: 12 x 8.5 MB = 102 MB per call.
   - The four cards together move only 28.8 GB/s through the host. That gives a floor of 3.55 ms against NCCL's 4.33 ms.
   - No NCCL setting helped, and the reason is the box, not NCCL.
   - What can still move prefill comm: fewer bytes through the host (P8 TP2 x PP2 moves a third), or a faster host path. The faster host path means P13 P2P or the memory configuration, both the user's call.
2. **The PLE gather is the largest single fixable item.**
   - It costs 20% of a deep prefill step and 34% of a shallow one, 6% of a decode step and 11% at 6 streams.
   - A GPU gather from host RAM costs under 1 ms per chunk.
   - Today's path is slow because of a per-row Python loop on a thread pool, run by all four ranks at once. I/O bandwidth is not the reason.
3. **Tokenization costs ~0.3 s per turn at 128K, server-side.**
   - It's on every agent turn, warm or incremental.
   - Together with the Triton cache living outside the mounted cache dir, this makes P9 cheap and real.

The re-ranked portfolio is in the notes file. The new order is: P4 (T5), then P9 (T11), P3 (T4), P8 (T10, promoted), P2 decode path (T6, demoted), P5 (T7), P1 reduced to a per-size protocol choice (T3), and P10 unparked as a candidate.

## Quality gate

Not applicable: no candidate in this task. The four boots ran stock lanes. **The nondeterminism probe does bear on the gate**:

- On the prefix lane (R15), the same 33-token prompt with `prompt_logprobs`, one request at a time on an idle server, gives 6 distinct runs out of 6. The same holds with MTP off (R17) and on the tuned lane without prefix caching (R18).
- Runs differ from prompt position 0, with spreads up to 0.39-0.69 nat.
- Some values recur exactly, so the outcomes form a discrete set. Floating-point reordering would give a smooth spread of tiny differences instead.
- The gate's thresholds were calibrated on this noise (T1), so they hold. But the noise is wide, and its source should be found before a candidate that changes numerics is called lossless on the gate alone.

Ruled out:

| Candidate | How |
|---|---|
| cuBLAS bf16 GEMM, same and shifted addresses, with or without `CUBLAS_WORKSPACE_CONFIG=:4096:8` | `det_probe.py`: 1 distinct output in 50 |
| NCCL bf16 all-reduce at 29 and 1,664 tokens | `det_probe.py`: 1 in 50 |
| Marlin atomic reduction | off at this commit (`should_use_atomic_add_reduce` needs `VLLM_MARLIN_USE_ATOMIC_ADD`) |
| QSA top-k on a short context | `det_probe_topk.py`: 1 in 20 |
| Speculative decoding | R17 |
| Prefix caching / align mode | R18 |

Left: Marlin MoE under expert parallel, the staged PLE path, the QSA overlays (a split-k merge that reads unwritten partials would fit the discrete pattern), the HC and GDN kernels, and async scheduling.

## Side findings

- **`persistent_topk` at depth is nondeterministic.** On fixed logits (1,664 queries at 64K) it returned its indices in a different order in all 20 runs, and a different selected set in 1 of 20. That alone changes long-context outputs run to run, through both the attention summation order and the selection.
- **The profiler recipe in AGENTS.md was stale.** This image ignores `VLLM_TORCH_PROFILER_DIR`. It needs `--profiler-config`, and since the wrappers overwrite `EXTRA`, you call the base launcher (R16 command above). AGENTS.md is fixed. The default `torch_profiler_with_stack` is true; I turned it off. Under the profiler, decode is ~14% slower.
- **The Triton cache is not persisted.** It sits at `/root/.triton/cache` (93 MB after a warm-up), outside the mounted `/root/.cache`, so every boot JIT-compiles from scratch.
  - First-time shapes cost 0.4-1.8 s per kernel: `_qsa_sparse_paged_gqa_splitk_kernel`, `_ple_conv_kernel`, `_ple_conv_writeback_kernel`, `layer_norm_fwd_kernel`.
  - The JIT monitor warns once per kernel name, but a new specialization of the same kernel also recompiles (a 65-token prefill step took 2.1 s, 1.8 s of it in front of the QSA split-k kernel).
  - `-e TRITON_CACHE_DIR=/root/.cache/triton` is a one-flag T11 item.
- **Align mode rounds every chunk down to a multiple of 832.** The 2,048-token budget carries 1,664 tokens per step (19% fewer).
- **Every step ends with a ~45 ms tail.** That's after the annotation: the draft over the chunk, an all-gather and the sampler. It accounts for ~5% of a deep step.
- **The R13 "incremental turns have no big fixed overhead" note was wrong.** Client minus server TTFT is small, but server TTFT minus prefill grows ~2.7 us per prompt token. That's tokenization (R15, token-id prompts: 23 ms).
- **R15's cold 128K prefill took 135 s (946 tok/s), against R13's 164 s (777).** R15 started from an almost empty pool, and its preemptions were not recorded. `turn_overhead.py` now records them. T4 should check whether the preemption count depends on how much of the pool earlier requests hold.
- **Small prefill steps are launch-bound.** A 190-token step takes 237 ms, 136 ms of it with the CPU launching ~2,900 eager kernels (`qwen_gdn_attention_core_fused_norm_packed`, `ChunkGatedDeltaRuleFunction`, `moe_forward_shared` show as the host ops). That is the warm-TTFT floor per step today.
- **Ranks drift out of step in small prefills.** In the R16 traces, short prefill steps spent 190-254 ms in 106 tiny all-reduces: rank 0 waited for ranks that were compiling or launching.
- **The concurrent PCIe numbers differ by card.** GPU2 and GPU3 (behind one host bridge, `PHB`) got 4.4 GB/s each H2D, against 9.2 and 13.3 for GPU0 and GPU1.
- **`floor_bench.py`'s first 4-card PCIe run was not slot-aligned** (`floor-bench-T2-20261017-212430.json`). Use the `T2-pcie` file for the concurrent numbers.
- **PLE layer naming.** The PLE table belongs to `layers.1` in the checkpoint's tensor names, while `ple_layer_ids` in config.json says `[2]`.
- **Stale NVLink comment.** The launcher's comment about NVLink pairs describes the old box. Here `nvidia-smi nvlink -s` reports all links inactive.

## Open questions

- **The source of the temperature-0 nondeterminism.**
  - Proposed as T14: Opus 5.5 / xhigh, before any candidate that changes numerics.
  - Next tests, one boot each with `repeat_probe.py`: `--no-async-scheduling`; `PLE_MODE=mmap`; no expert parallel (if it fits); and an offline test of the QSA split-k merge with poisoned scratch buffers.
- ~~Why does the host copy at only 21-28 GB/s?~~ **Closed after the entry was written:** the user confirmed the box has 4 x 32 GiB DIMMs (4 of 8 channels), and that will not change. Every later plan treats ~29 GB/s through the host as a fixed ceiling.
- **Can P8's pipeline overlap a single request's chunks well enough** for the ~2x prefill estimate? And what does it cost decode, with two stages in series and the memory time doubled? That is T10's feasibility question.
- **P4 design choice (T5).**
  - A pinned table with a UVA gather removes the host sync but needs 51 GB pinned (the host allows it; the sandbox's 15.7 GiB memlock does not).
  - A resident unpinned mapping with a C-speed gather keeps one sync per step but needs no pinning.
  - Either way the table must be made resident: today only 17% of it is in the page cache.

## Next step

**The user chose T5 next** (2026-10-17). Recommended order after the re-rank: T5 (P4: PLE gather, Fable 5.1 design / xhigh, then Opus implementation), T11 (P9: tokenization and the Triton cache, Sonnet 5.5 / high), T4 (P3, Opus 5.5 / xhigh), then T14 (nondeterminism, Opus 5.5 / xhigh) before the first numerics-changing candidate. T3 shrinks to a short per-size protocol test; T10 is promoted (its condition is met), T6 keeps only its decode path. The user decides the order. The server was **stopped** (R18 at 22:02); GPUs free.
