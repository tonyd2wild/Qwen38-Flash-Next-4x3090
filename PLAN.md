# Plan: TTFT and tok/s for Qwen3.8-Flash-Next on 4x RTX 3090

Brief: [`fable-initial-prompt.md`](fable-initial-prompt.md). Live status: [`journal/INDEX.md`](journal/INDEX.md). Numbers: [`vllm-w4a16/results/perf-2026-10-notes.md`](vllm-w4a16/results/perf-2026-10-notes.md). Written 2026-10-17 by Fable 5.1 after a read-only exploration of the repo, the saved R8/R9 profiler traces, the vLLM source inside the image, and the checkpoint author's runtime snapshot. Approved by the user the same day.

## Context

The prefix-cache lane (`vllm-w4a16/launch/launch-qwen38fn-w4a16-prefix.sh`: vLLM 8a728663, TP4 + expert parallel, MTP3, fp8 e5m2 KV, align-mode prefix caching) is the daily driver for a coding agent. Conventional tuning is done (boots R1-R12 in `vllm-w4a16/README.md`). The brief asks for multiples, lossless only, engine-side only, measured on this box, shipped as a new lane plus a host script. Work is tackled one task at a time, by whichever model the task board below recommends, and every task leaves a journal entry so the next model can pick up cold.

What the exploration settled before any boot:

| Fact | Evidence | Consequence |
|---|---|---|
| **Prefill at depth is 59% NCCL all-reduce.** A 1664-token step takes 0.95 s; `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` takes 0.47 s of it over ~107 calls (3.4-4.4 ms each for an 8.5 MB message, about 2.5 GB/s). Everything else sums to about 0.3 s per step | `.cache/ple/prof-pc1/profiler_out_0.txt` (R8), `prof-pc0` (R9), `prof-pc1-shallow` | Ceiling without the comm tax: 4-5K tok/s (today 1.4K). NCCL runs the low-latency protocol on multi-megabyte messages through host memory; there is no GPU-to-GPU P2P on this board (`nvidia-smi topo -p2p r` is `CNS` for every pair) |
| Decode target graph = 15.1 ms per 4-token step (observed step about 21 ms including the draft, sampling and host gaps). Memory floor about 4 ms (about 3.2 GiB per card per token) | same traces, rows `execute_context_0(0)_generation_1(4)` | The rest is ~107 small collectives plus a ~1,900-kernel chain; the levers are collective latency, kernel count and tokens per step |
| Async scheduling is **already on** (`mtp` is in `EagleModelTypes`, so the default resolves to True) but `vllm-w4a16/patch/model_state.py` syncs GPU to CPU every step: `compute_ngram_ids` on the GPU, `.to("cpu")`, then a 32-thread `preadv` pool reads 16 rows per token | `vllm/config/vllm.py:1270-1320` in the image; `patch/ple_layer.py:535-555`; CPU side of the trace: `aten::copy_` 31% of CPU time, `cudaEventSynchronize` 22% | The host gather sits in every step's critical path. The disk-table design came from a 31 GB-RAM box; this box has 128 GB and already holds 105 GB of page cache |
| **Cold prompts above ~100K get preempted because one Mamba page per group leaks per prefill chunk.** The pool is 360 pages (2.4e9 B / 6.65 MB); at 100K a request needs 120 attention pages, and 60 chunks x 4 Mamba groups is 240 | R10 (155K prompt = 155 steps for 256,446 tokens, KV 99% then 0%, 2 preemptions); the checkpoint author's v0.4.0 note: "fixes a KV-cache leak that kept one recurrent-state block per prefill chunk alive in each Mamba layer group"; `MambaManager.remove_skipped_blocks` exists at this commit | A small KV-manager fix recovers 843 -> ~1,395 tok/s on a 155K cold prefill with caching on, and frees pool for seats at depth |
| The checkpoint author's 2x3090 runtime (TP2 + EP2, 4-8K chunks, PLE table in pinned host RAM) prefills at 3,000-4,200 tok/s on half the GPUs | `~/models/qwen38fn-w4a16-fp8ple/README.md`, `runtime/serve-container.sh` | Confirms the 4-way all-reduce over PCIe is the prefill tax |
| Image `qwen38fn-vllm:8a728663` is in the sandbox's podman store (built 2026-10-17 14:18, 13.4 GB); `torchrun`, `nvcc`, `ninja`, `cuda-python`, Triton 3.7.1, torch 2.13.0+cu130, NCCL 2.29.7 inside | `docker images`, `docker run --rm --entrypoint bash ...` | No build needed; microbenchmarks and custom kernels run in the container without a server |

Bounds honoured throughout: endpoint drop-in (port 8090, model id `qwen3.8-flash-next`, `qwen3_xml` tools, `qwen3` reasoning parser with thinking off, vision, 262,144 context, 6 seats); lossless only; no host, driver or weight changes; tuned and prefix lanes untouched; new lanes in new files; branch `perf-comm-ple` off `source-build-prefix-cache`, never pushed.

## Deliverables

0. **Scaffolding** (first commit): `PLAN.md` (this file), `AGENTS.md` (what every model reads first), `CLAUDE.md` (`@AGENTS.md`), `journal/INDEX.md` (task board, live status, GPU owner), `journal/TEMPLATE.md`, `journal/NN-<slug>.md` per task, and `vllm-w4a16/results/perf-2026-10-notes.md` (scoreboard before/after, ledger R13+, open questions). The journal holds the narrative; the notes file holds the numbers; no third copy.
1. Branch `perf-comm-ple` (no push).
2. Tools, new files under `vllm-w4a16/tools/`, stdlib/torch only, all taking `URL=`:
   - `scoreboard.py`: `bench-host.py`'s metrics verbatim (same decode and TTFT definitions, same `/metrics` hit counters) plus incremental turns at depth (16K/64K/128K cached + 1K/4K/16K new), a code-generation decode run (count-to-100 kept as the draft's best case, plus a real code prompt), a ~200K needle, and 2/4/6 concurrent streams (aggregate and per-stream decode, TTFT p50/p99). Asserts `vllm:num_requests_running` and `vllm:num_requests_waiting` are 0 before and after every run and aborts otherwise.
   - `quality_gate.py`: freezes ~20 prompts (code, prose, tool calls, long-context QA) at temperature 0 with `logprobs`; `--record` writes the baseline JSON, `--compare` reports the first divergent token and the logprob gap per prompt, parses every tool call, runs the needle at ~30K/~100K/~200K, a 155K cold prefill (no OOM, no preemption), and checks the boot log's pool >= 262,144 tokens.
   - `nccl_bench.py` (torchrun, 4 processes): all-reduce and all-gather latency and bandwidth at 20 KB, 160 KB, 8.5 MB, 10.5 MB, 42 MB under candidate `NCCL_*` settings, with `NCCL_DEBUG=INFO,TUNING` to show the chosen protocol.
   - `ple_gather_bench.py`: times `PLEMmapTable.gather_cpu` (`patch/ple_mmap.py`) for 64 and 32,768 warm ids, plus a pinned-memory / zero-copy gather prototype.
   - `trace_breakdown.py`: sums the saved torch-profiler JSON traces by category (NCCL, GEMM, Marlin MoE, GDN, QSA, HC/elementwise, PLE, copies) for prefill and decode steps.
3. Engine changes as mountable overlays and a new PLE mode (new files under `vllm-w4a16/patch/`, mounted only when the new lane's knobs are set): `allreduce_host.py` plus an overlay of `distributed/device_communicators/cuda_communicator.py` (P2); `ple_hostpin.py` and `PLE_MODE=hostpin` variants of `ple_layer.py` / `model_state.py` (P4); `kv_mamba_align_fix.py` overlay of `v1/core/single_type_kv_cache_manager.py` and/or the retention flag (P3).
4. New lane `vllm-w4a16/launch/launch-qwen38fn-w4a16-fast.sh` (wrapper in the existing style); new knobs in `qwen38fn-w4a16-3090-tp4.sh` whose defaults leave the old lanes byte-for-byte unchanged (`NCCL_TUNE`, `ALLREDUCE`, `PLE_MODE=hostpin`, `RETENTION`; `SPEC_EXTRA` exists).
5. `run-host-fast.sh` in the style of `run-host-docker.sh`: preflight, optional clean host baseline of the prefix lane, boot the fast lane, scoreboard + quality gate + needle, one pasteable log. The sandbox server is stopped before hand-over.
6. Lane README section (before/after scoreboard, quality-gate results, ledger R13+) and the top-level README headline.

## Task board

Models: Fable 5.1 (`claude-fable-5-1`; the strongest, for decisions where reasoning depth decides the outcome), Opus 5.5 (diagnosis, analysis, non-trivial implementation), Sonnet 5.5 (scripts, sweeps, running measurements, documentation drafts). Haiku 5.5 is not recommended here: every task touches a 5-minute boot cycle where a mistake costs more than the model saves. Effort is the `/effort` level. "Planning stage" means: write the Design section of the journal entry (approach, alternatives, kill test, test plan) and re-read it against the kill test before writing code; it is never a separate conversation unless noted.

| # | Task (portfolio ref) | Depends on | Model | Effort | Own planning stage? | Why |
|---|---|---|---|---|---|---|
| T0 | Scaffolding: branch, `PLAN.md`, `AGENTS.md`, `CLAUDE.md`, `journal/`, notes file | - | Sonnet 5.5 (done by Fable 5.1, who held the exploration context) | medium | no | Mechanical; content specified here |
| T1 | Baseline: `scoreboard.py`, `quality_gate.py`, boot the prefix lane, record the "before" column and the frozen quality baseline | T0 | Sonnet 5.5 | high | no | Careful script work; metric definitions must stay comparable to `bench-host.py`; no design freedom |
| T2 | Time accounting: `trace_breakdown.py`, `nccl_bench.py`, `ple_gather_bench.py`, incremental-turn breakdown, floor table, portfolio re-rank | T1 | Opus 5.5 for the floors and the re-rank; Sonnet 5.5 for the scripts | high | no, but it ends with a written re-ranking | The floors need physical reasoning; the scripts do not |
| T3 | P1 NCCL tuning: microbench matrix, then 1-2 boots | T2 | Opus 5.5 | medium | no | Empirical; needs NCCL semantics but no design |
| T4 | P3 Mamba align-mode page leak: instrument, diagnose, fix, 155K validation | T1 | Opus 5.5; escalate to Fable 5.1 if the first instrumented boot does not explain the leak | xhigh | short: hypotheses and the instrumentation list in the entry before the boot | Subtle allocator logic; one wrong patch wastes boots |
| T5 | P4 PLE hostpin: pinned-memory kill test, mapping strategy, gather kernel, `PLE_MODE=hostpin`, boot | T2 | Fable 5.1 for the design and kill tests; Opus 5.5 for implementation and boot | design xhigh, implementation high | yes | Mapping and pinning across 4 ranks under the sandbox memlock limit is a real design fork |
| T6 | P2 host-staged all-reduce: design doc, standalone 4-process prototype with correctness tests, overlay integration, boots (decode sizes first, then all sizes) | T2 (NCCL latency numbers), T3 | Fable 5.1 (design at max, prototype at xhigh); Opus 5.5 for integration and boots | max / xhigh, then high | yes, as its own entry | Hardest item: CUDA kernels, cross-process memory ordering, graph capture, failure modes |
| T7a | P5 flags: MTP depth sweep, reduced draft vocabulary | T1 | Sonnet 5.5 | medium | no | Boot and measure |
| T7b | P5 hybrid MTP + n-gram/suffix proposer | T7a, offline hit-rate estimate | Opus 5.5 | high | yes | Touches the speculator; acceptance semantics must stay lossless |
| T8 | P6 chunk 4096, only if memory was freed | T6 or T5 | Sonnet 5.5 | medium | no | One boot with the free-memory probe |
| T9 | P7 torch.compile diagnosis, timebox one boot plus 2 hours | T5 | Opus 5.5 | xhigh | no | Debugging an unknown failure; stop at the timebox and journal the traceback |
| T10 | P8 TP2 x PP2, conditional: comm still above 25% of the prefill step after T3/T6 | T3, T6 | Fable 5.1 feasibility (xhigh), then Opus 5.5 implementation (high) | xhigh / high | yes | Layout change with scheduler, PLE guard, MTP and pool implications |
| T11 | P9 fixed overheads for warm and incremental TTFT | T2 | Opus 5.5 (changed from Sonnet 5.5 after T5, user agreed 2026-10-17) | high | short: a Design section for the tokenization part | Scope grew after T5. A lossless tokenization cache needs a token-id equality argument at the seams. Moving the hostpin setup after weight loading changes a cross-rank broadcast, where a mistake hangs a boot. Only the `TRITON_CACHE_DIR` flag is Sonnet-sized |
| T12 | P11 draft KV-group identification (drop `disable_eagle_block_drop`) | T1 | Opus 5.5 | medium | no | Small, well-located patch |
| T13 | Recipe: fast lane, `run-host-fast.sh`, final sandbox scoreboard and gate, README and ledger, hand-over | all kept items | Sonnet 5.5 for scripts; Opus 5.5 for README prose and the before/after tables | high | no | The README's style is exacting; numbers must be transcribed exactly |
| - | P10 QSA kernel tuning (parked); P13 PCIe P2P diagnostics (read-only, then ask the user) | - | Fable 5.1 / Sonnet 5.5 | - | - | Only if time remains / the user decides |

Rules of thumb: effort `max` only where a design decision is irreversible for days of work (T6 design, T10 feasibility); `xhigh` to diagnose behaviour we do not understand (T4, T9); `high` by default for implementation and analysis; `medium` for sweeps and boots; journal writing and INDEX updates at `low`-`medium` whatever the model. Tasks run sequentially because one process owns the GPUs at a time (a server or a microbenchmark, never both). Recommended order: T0, T1, T2, T3, T4, T5, T6, T7a, T12, T8, T9, T7b, T10, T11, T13, re-ranked after T2 and after each kept win.

## Per-task workflow

1. Read `AGENTS.md`, this file, `journal/INDEX.md`, then the entries of your task's dependencies. Set the recommended model and effort.
2. Check the box: `nvidia-smi` shows under 1 GB used per GPU; `curl -s -m 3 localhost:8090/v1/models` and `curl -s -m 3 host.docker.internal:8090/v1/models` both fail (if the host server answers, stop and ask the user; it cannot be stopped from the sandbox). Claim the task in `journal/INDEX.md` (status, model, time).
3. If the task has a planning stage, write the Design section first and re-read it against the kill test before coding.
4. One change per boot. Every boot gets a ledger row (R13 onward) in the notes file, failures included.
5. Measure with `scoreboard.py` (running/waiting = 0 checks) and `quality_gate.py --compare`. Record the exact commands and flags in the entry.
6. Finish the entry: Result (kept or killed, and why), Side findings, Open questions, Next step. Update `journal/INDEX.md` and the notes file. Commit on `perf-comm-ple` (no push). Leave the GPUs free (`docker stop vllm_qwen38fn_3090`) unless the next task explicitly reuses the server, and say so in `INDEX.md`.

## Phase 1: Baseline (T1, sandbox, about 2 hours)

1. Boot the prefix lane: `GPU_ARGS="--device nvidia.com/gpu=all" bash vllm-w4a16/launch/launch-qwen38fn-w4a16-prefix.sh`; confirm overlays 7/7, `Using 'MARLIN' WNA16 MoE backend`, pool 299,482.
2. `scoreboard.py` (full) is the "before" column (sandbox). Record `/metrics` acceptance per position for the real prompts.
3. `quality_gate.py --record` writes `vllm-w4a16/results/quality-baseline-2026-10-17.json`.
4. Ledger R13 = this boot; journal entry `01-baseline.md`.

## Phase 2: Time accounting (T2, mostly without boots)

- Prefill shallow and deep from the saved traces via `trace_breakdown.py`; each component next to its floor: GEMM and Marlin against 35.6 TFLOPS per card (bf16 inputs, fp32 accumulate); HC, GDN and QSA kernels against 936 GB/s; all-reduce against the PCIe 4.0 x16 host path (22-25 GB/s per direction; a 4-ring moves 1.5 x S per rank each way); host overhead from the CPU side of the trace.
- Decode from the 16 graph-replay steps in the same traces, split the same way; per-collective latency from `nccl_bench.py`.
- Incremental turn: one request at 64K cached + 4K new, with the `/metrics` histograms (`request_queue_time`, `request_prefill_time`, `time_to_first_token`) against the client TTFT, to split API/tokenizer overhead, the tail-block recompute (a hit rounds down to the 832-token block), prefill proper and the first step.
- PLE gather cost: `ple_gather_bench.py` on the warm table.
- The floor table goes into the notes file; the biggest gaps re-rank the portfolio; journal entry `02-time-accounting.md`.

## Phase 3: Portfolio (ranked by expected gain x odds / effort)

Re-ranked after T2 on measured floors: see "Portfolio re-ranked after T2" in `vllm-w4a16/results/perf-2026-10-notes.md` and `journal/02-time-accounting.md`. The table below is the original ranking; two of its premises did not hold on this box: bf16 GEMMs peak at 75.8 TFLOPS, not 35.6, and the all-reduce is bound by the host side at ~29 GB/s for all four cards, not by the NCCL protocol.

| # | Idea | Mechanism | Ceiling on this box | Effort | Risk | Cheapest kill test |
|---|---|---|---|---|---|---|
| P1 | **NCCL protocol and transport tuning** | NCCL chose RING_LL for 8.5 MB messages over host shared memory. Try `NCCL_PROTO=Simple`, `NCCL_ALGO=Ring` or `Tree`, `NCCL_SHM_USE_CUDA_MEMCPY=1`, `NCCL_BUFFSIZE=16777216`, `NCCL_NTHREADS`, `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`, `NCCL_P2P_DISABLE=1` against `NCCL_P2P_LEVEL=NVL` | all-reduce 4.4 -> ~1 ms, so the prefill step 0.95 -> ~0.55 s (1.7-2x); decode a few percent | hours (microbench, then 1-2 boots) | low; the summation order may change, so run the quality gate | `nccl_bench.py`: no setting gets 8.5 MB under ~1.5 ms |
| P2 | **Host-staged custom all-reduce (own kernels)** | Every byte crosses host memory anyway. A shared pinned buffer (a `/dev/shm` file, `cudaHostRegister` Mapped+Portable in each rank, per-block sequence flags, double-buffered slots). Up to 256 KB: a one-shot zero-copy kernel, fp32 accumulate in rank order (~10-15 us). Megabytes: reduce-scatter plus all-gather through the staging area (~0.8-1.2 ms for 8.5 MB). CUDA-graph capturable. Inserted ahead of pynccl in `CudaCommunicator.all_reduce` via an overlay | decode: 107 x (NCCL latency - ~12 us), so 3-5 ms off a 21 ms step (1.2-1.3x); prefill: P1's ceiling, and additive if NCCL stays slow | 2-4 days | medium: cross-PCIe memory ordering, GeForce zero-copy throughput, graph capture, deadlock if a rank dies (add timeouts) | a standalone 4-process prototype; if the one-shot latency is not below NCCL's measured LL latency at 20 KB, drop the decode path |
| P3 | **Fix the align-mode Mamba page leak** | Instrument free blocks per group per step during a 155K cold prefill; check `MambaManager.remove_skipped_blocks` / `last_state_block_idx`, `_needs_internal_checkpoint`, and `--prefix-cache-retention-interval` (default 0 at this commit); patch so the previous step's state page is freed | 155K cold with caching on: 843 -> ~1,395 tok/s (1.65x); 0 preemptions; pool freed for seats at depth; concurrent long prompts safe | 0.5-1 day | low | an instrumented boot: `num_preemptions_total` stays 0 and KV usage stays flat across a 155K prefill |
| P4 | **PLE table in host RAM with a GPU-side gather (`PLE_MODE=hostpin`)** | Map the 51.2 GB table once (a shared file-backed or memfd mapping registered per rank; fallback: rank-split rows plus the byte-wise all-reduce already in `_ple_mmap_gather_into`), gather rows by UVA inside the CUDA graph; no `.to("cpu")`, no thread pool. Same bytes, lossless | prefill: the measured gather cost (estimated 50-200 ms per chunk -> ~5 ms); decode: removes the per-step sync (0.5-1.5 ms of 21) and lets async scheduling overlap | 1-2 days | medium: pinning 51 GB (sandbox memlock 15.7 GiB; the host runs with `memlock=-1`), 4 ranks sharing pages, file-backed registration on ext4 | `ple_gather_bench.py` (if today's gather is under 10 ms per chunk, the prefill half is moot) plus a 20 GiB pinned-allocation test |
| P5 | **Speculation breadth**: MTP depth 4-6, reduced draft vocabulary (`patch/mtp_draft_vocab.py`, `QWEN4EXP_DRAFT_VOCAB`), an MTP + n-gram/suffix hybrid proposer | More accepted tokens per ~21 ms step; the target verifies everything | count 189 -> ~250; code 145 -> 180-200; copied spans by multiples | flags: one boot each; hybrid: 1-2 days | low | acceptance per position on code prompts; offline n-gram hit rate on agent transcripts |
| P6 | **Chunk 2048 -> 4096** once memory allows | Amortises per-collective latency and per-step host work | prefill 10-20% | one boot | OOM (R7 at 2496 left 0.03 GiB free); the pool must stay >= 262,144 | boot plus a 155K free-memory probe |
| P7 | **torch.compile (piecewise or full)** | "only nocompile boots" is undiagnosed; with the table off the GPU, the PR #55272 duplication issue is moot | decode graph 15 -> ~11-12 ms (1.3x) | unknown | medium-high | one `GRAPHS=default` boot, keep the traceback |
| P8 | **TP2 x PP2** | 2-party rings move 1.0 x S against 1.5 x S per rank, half the collectives per rank, chunks pipeline across stages (the scheduler advances `num_computed_tokens` at schedule time). Needs the `model_state.py` PP guard relaxed (the author's version uses `get_pp_indices`), MTP on the last stage, pool re-validation | prefill +15-30% after P1/P2; decode within 10% either way | 2-3 days | medium-high | only if comm is still above 25% of the prefill step after P1/P2 |
| P9 | Fixed-overhead cuts for warm and incremental TTFT | measured in Phase 2 | 0.1-0.4 s per turn | low | low | the breakdown shows under 0.2 s outside prefill |
| P10 | QSA sparse-attention kernel (12% of deep prefill at ~10 TFLOPS) | Triton block tuning for sm86 | up to 10% prefill | days | medium | parked |
| P11 | Draft KV-group identification (`kv_cache_utils.py:2138-2160` in the image) | restores the trailing-block drop for the draft group only | ~1.5% count-to-100 | hours | low | the boot shows no "every group treated as a draft group" warning |
| P12 | Replicated HC weights (1.3 GiB per card, ~1.4 ms of reads per step) | sharding them adds collectives | negative for decode; noted only | - | - | - |
| P13 | **PCIe P2P on these GeForce cards** (driver, open kernel module, IOMMU/ACS) | direct GPU-to-GPU paths for NCCL and `custom_all_reduce` | the largest single comm fix | host-side | **out of bounds; needs the user's approval** | read-only diagnostics only |

Expected if P1-P4 land: prefill ~1.4K -> ~3K tok/s at depth (cold 128K about 45 s, an incremental 4K turn at 128K about 1.5-2 s); decode ~189 -> ~230-250 tok/s on count-to-100, 110 -> ~135 on prose, more with P5 on code. Report honestly whatever lands.

## Phase 4: Experiments (one change per boot, A/B against the same-day baseline, ledger R13 onward)

Microbenchmarks first, with no server running: `nccl_bench.py`, `ple_gather_bench.py`, the pinned-memory test, the all-reduce prototype. Then boots: R14 the NCCL winner (P1); R15 P3 instrumentation on a 155K prefill, R16 the fix; R17 `PLE_MODE=hostpin`; R18 `ALLREDUCE=host` for messages up to 256 KB, R19 all sizes; R20 the MTP depth sweep, R21 the draft vocabulary; R22 chunk 4096 if memory allows; R23 the `GRAPHS=default` diagnosis; P8 only if warranted. Every boot logs flags, pool, free memory per card during a 155K prefill, scoreboard deltas and the quality-gate verdict. Killed ideas get their journal entry too.

## Phase 5: Recipe (T13)

- `launch-qwen38fn-w4a16-fast.sh` = the prefix lane plus the winners (knobs only; overlays mounted conditionally and hash-checked like the existing ones).
- Final sandbox scoreboard and quality gate; stop the sandbox server; `run-host-fast.sh` for the user (baseline prefix lane, then the fast lane, one log).
- README: lane section, before/after table (sandbox and host columns), quality-gate results, ledger R13+, update log; the headline comes from the host run.

## Implementation notes (files, reuse)

- Launcher knobs: extend `vllm-w4a16/launch/qwen38fn-w4a16-3090-tp4.sh` in its existing style (env -> `docker run` args and mounts): `NCCL_TUNE` adds `-e NCCL_*`; `ALLREDUCE=host` mounts the communicator overlay plus `allreduce_host.py` (`--ipc host` already shares `/dev/shm`); `PLE_MODE=hostpin` is a new case next to `staged`; `RETENTION` passes `--prefix-cache-retention-interval`. The launcher's overlay hash check covers new mounts automatically.
- Custom all-reduce: CUDA kernels via `torch.utils.cpp_extension.load_inline` at import (nvcc and ninja are in the image; cache under `CACHE_HOST`, already mounted at `/root/.cache`). Hook point: `CudaCommunicator.all_reduce` (`vllm/distributed/device_communicators/cuda_communicator.py:305` in the image), checked before `pynccl_comm.all_reduce`; pynccl stays for all-gather, broadcast and PP.
- PLE hostpin: reuse `PLEMmapTable`'s header parsing and `read_rows_contiguous` (`patch/ple_mmap.py`) to populate the mapping; reuse `_ple_init_staging` and the forward view in `patch/ple_layer.py`; `compute_ngram_ids` already runs on the GPU (`models/qwen4_exp/nvidia/ops/ple.py`). The gather kernel writes into the existing staging buffer so `forward` is unchanged.
- Mamba leak: overlay `vllm/v1/core/single_type_kv_cache_manager.py` (`MambaManager.remove_skipped_blocks` / `allocate_new_blocks`, lines 1540-1800 in the image) after instrumenting; compare with the author's `runtime/vllm-overlay/v1/core/sched/scheduler.py::_mamba_block_aligned_split`.
- Tools: start from `bench-host.py` (metric definitions kept), `vllm-w4a16/tools/prefix_cache_3090.py` (turn structure, `/metrics` reads), `tools/needle_3090.py`, `tools/sixagent_decode.py` (concurrency pattern).
- Notes and ledger follow `vllm-w4a16/README.md` ("Boot ledger (2026-10-17, rebuilt image)") formatting.

## Verification

- Every candidate boot: `quality_gate.py --compare` against the frozen baseline (the first divergent token and the logprob gap must be floating-point-reordering sized; tool calls parse; needle correct at 30K/100K/200K; 155K cold prefill with no OOM; pool >= 262,144), the scoreboard with running/waiting = 0 before and after each run, `num_preemptions_total` tracked.
- Final: sandbox before/after table; the host run via `run-host-fast.sh` gives the demo numbers; sandbox against host on the same recipe is recorded (nobody has compared them yet).
- Regression guards: the tuned and prefix launchers are unchanged in behaviour (diff their `docker run` lines with the new knobs unset); a vision smoke test and a tool-call request on the fast lane.
- Scaffolding check: a fresh session given only `AGENTS.md` can find the next task, its dependencies' results and the exact commands that reproduce the last measurement.

## Needs the user (batched for one sitting)

1. Run `run-host-fast.sh` on the host at hand-over (baseline plus fast lane, about 1.5 hours, one log back).
2. Decide on P13 (PCIe P2P): read-only diagnostics will be presented; no host change without approval.
3. ~~If the sandbox cannot pin the full 51 GB table (memlock 15.7 GiB), P4 is validated here at reduced size and at full size only in the host run.~~ Moot after T5: the memlock limit does not apply to `cudaHostRegister`, and R19 ran the full table in the sandbox. The host run must leave ~48 GiB of RAM free for it.

## Risks and open questions (tracked in the notes file and the journal)

- NCCL env tuning may already recover most of the prefill comm; then P2 is justified mainly by decode latency, which the microbench must prove first.
- Pinning 51 GB on the host with 4 ranks sharing one mapping; fallbacks are listed in P4.
- Memory headroom (0.24-0.38 GiB free during long prefills) constrains chunk size and any new GPU buffers; every boot re-checks free memory during a 155K prefill.
- Decode multiples beyond ~1.3x need compile-level fusion (P7) or wider speculation (P5); both are in the portfolio, neither is promised.
