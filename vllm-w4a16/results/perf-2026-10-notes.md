# TTFT and tok/s work, 2026-10: scoreboard, ledger, open questions

The numbers file for the work planned in [`PLAN.md`](../../PLAN.md). Narrative per task is in [`journal/`](../../journal/INDEX.md). The lane README gets the final copy at the end (T13). Keep the metric definitions of `bench-host.py`: decode = completion tokens after the first / time after the first token; cold = fresh random prefix; warm = the same prompt again; incremental = a cached conversation plus N new tokens; TTFT is measured at the client from the HTTP request to the first streamed content token; single stream on a quiet server unless noted; cache hits read from `/metrics` around each request.

## Scoreboard

Columns: `host 10-17` = `current_benchmark.md` on the host under Docker (prefix lane; **the 1K and 16K rows and the decode block overlapped another client**, only the 64K and 128K rows are clean); `sandbox before` = the prefix lane in the sandbox, boot R13 (T1 fills it); `sandbox R19` = the prefix lane with `PLE_MODE=hostpin` (T5, kept); `sandbox R22` = the hostpin lane with `TRITON_CACHE=1` and `TOKSPLIT=1` (T11, kept); `sandbox after` = the fast lane in the sandbox; `host after` = the fast lane on the host (`run-host-fast.sh`).

| Metric | host 10-17 | sandbox before (R13) | sandbox R19: hostpin (T5) | sandbox R22: + Triton cache, toksplit (T11) | sandbox after | host after |
|---|---:|---:|---:|---:|---:|---:|
| Decode, count to 100, median of 5 (tok/s) | 143.6 (contaminated; R3 quiet: 188.8) | 186.0 (accept len 4.00) | 204.0 (accept len 4.00) | 207.5 (accept len 4.00) | | |
| Decode, prose 512 tokens (tok/s) | 110.4 (contaminated) | 124.2 (accept len 2.65; 1 run) | 144.9 (accept len 2.78; 1 run) | 148.0 (accept len 2.82; 1 run) | | |
| Decode, code prompt (tok/s) | n/a | 163.2 (accept len 3.48; median of 3) | 180.5 (accept len 3.48; median of 3) | 183.0 (accept len 3.50; median of 3) | | |
| Cold TTFT 1K / prefill tok/s | 2.63 s / 394 (contaminated) | 0.81 s / 1,274 | 0.50 s / 2,067 | 0.50 s / 2,077 | | |
| Cold TTFT 16K / prefill tok/s | 18.89 s / 845 (contaminated) | 11.41 s / 1,400 | 7.47 s / 2,135 | 7.26 s / 2,198 | | |
| Cold TTFT 64K / prefill tok/s | 45.63 s / 1,399 | 45.78 s / 1,396 | 29.78 s / 2,142 | 29.20 s / 2,189 | | |
| Cold TTFT 128K / prefill tok/s | 152.62 s / 837 (2 preemptions) | 164.16 s / 777 (2 preemptions) | 107.70 s / 1,184 (2 preemptions) | 105.22 s / 1,213 (2 preemptions) | | |
| Warm TTFT 1K / 16K / 64K / 128K | 1.87 (cont.) / 0.25 / 0.67 / 0.85 s | 0.21 / 0.28 / 0.71 / 0.72 s | 0.16 / 0.24 / 0.50 / 0.66 s | 0.16 / 0.18 / 0.42 / **0.38** s | | |
| Decode at 64K / 128K context (tok/s) | 125.0 / 136.6 | 119.2 / 116.8 | 125.1 / 158.1 | 131.3 / 144.4 | | |
| Incremental TTFT, 16K cached + 1K / 4K new | n/a | 0.98 / 3.23 s | 0.67 / 2.14 s | 0.61 / 2.04 s | | |
| Incremental TTFT, 64K cached + 4K / 16K new | n/a (R3 at ~70K + ~4K: 4.0 s and 5.2 s) | 3.61 / 12.26 s | 2.43 / 8.11 s | 2.26 / 7.80 s | | |
| Incremental TTFT, 128K cached + 4K / 16K new | n/a | 3.73 / 12.49 s | 2.62 / 8.35 s | 2.27 / 7.90 s | | |
| Needle ~200K: correct / TTFT / prefill tok/s | R5: correct / 275 s / 726 | correct (199,406 tok) / 267 s / 746 (4 preemptions) | correct (199,410 tok) / 175 s / 1,141 (4 preemptions) | correct (199,404 tok) / 171 s / 1,168 (4 preemptions) | | |
| 155K cold prefill tok/s, preemptions, min free GiB per card | R3: 843, 2, 0.38 (GPU2 0.24) | 838, 2, 0.68 (0.81 / 0.81 / 0.68 / 0.81; nvidia-smi poll, not comparable to R3's 0.38, see notes) | 1,278, 2, 0.60 (0.66 / 0.66 / 0.60 / 0.66) | 1,310, 2, 0.60 (0.66 / 0.66 / 0.60 / 0.66) | | |
| Concurrency 2 / 4 / 6: aggregate tok/s | n/a | 159.4 / 224.5 / 261.9 | 159.7 / 279.8 / 301.2 | 165.0 / 259.5 / 337.1 | | |
| Concurrency 2 / 4 / 6: per-stream decode median (tok/s) | n/a | 89.7 / 62.8 / 51.2 | 104.1 / 80.8 / 59.4 | 111.2 / 83.6 / 62.3 | | |
| Concurrency 2 / 4 / 6: TTFT p50 / p99 (s) | n/a | 1.00 / 1.42, 0.84 / 1.58, 1.57 / 2.38 | 0.69 / 3.02, 0.80 / 1.09, 1.04 / 2.13 | 0.66 / 2.93, 0.31 / 1.07, 0.57 / 1.56 | | |
| KV pool (tokens at 262,144 context) | 299,482 | 299,482 | 299,482 | 299,482 | | |

Notes on the R19 column (T5, `journal/05-p4-ple-hostpin.md`): one change from R13, `PLE_MODE` staged -> hostpin (`launch-qwen38fn-w4a16-hostpin.sh`); `results/scoreboard-R19-20261017-230308.json`, gate `results/quality-compare-R19-hostpin-20261017-225612.json` PASS. Prefill 1.53x at 1K-64K, 1.52x at 128K/155K (the preemptions stay: T4), decode +10% (count) / +17% (prose) / +11% (code), 6 streams +15%. Ready in 8.5 min instead of ~4.5 (the shared table takes ~2 min to fill and register during weight loading).

Notes on the R22 column (T11, `journal/14-p9-fixed-overheads.md`): the hostpin lane plus three changes, attributed boot by boot:
- R20: the hostpin boot fix.
- R21: the Triton cache, warm.
- R22: TOKSPLIT.

Files: `results/scoreboard-R22-20261018-004644.json`, gate `results/quality-compare-R22-toksplit-20261018-003757.json` PASS.
- **Warm and incremental turns:** about 2.6 us per prompt token less: warm 128K 0.66 -> 0.38 s, 128K + 4K 2.62 -> 2.27 s, 128K + 16K 8.35 -> 7.90 s. `turn_overhead.py`: chat 0.65 -> 0.36 s, level with token ids.
- **Decode and cold prefill:** within the boot-to-boot spread.
- **Boot:** ready in 5:23 instead of ~8.5 min.
- **First requests after a boot:** they no longer pay ~10 s of Triton JIT (`first-contact-R20/R21-*.json`).

Notes on the R13 column (detail in `journal/01-baseline.md`): measured with `vllm-w4a16/tools/scoreboard.py --label R13` (`results/scoreboard-R13-20261017-192716.json`). Boot-to-boot spread (R14 against R13): decode ±2.5%, cold prefill ±1.5%, warm TTFT ±0.1 s. A smaller win needs repeated runs. Acceptance mean length: count 4.00, prose 2.65, code 3.48. Incremental rows: the server's prefill time accounts for all but 20-80 ms of the client TTFT, at 1,340-1,380 tok/s on the uncached tokens. The min-free column comes from an nvidia-smi poll, which sees the caching allocator's reserved peak (flat 0.81 / 0.81 / 0.68 / 0.81 GiB in every long run), not transients, so it is not comparable to R3's 0.38 / 0.24. The needle and 155K prompts are real token counts; the older tools' 0.75 tokens-per-word filler undercounted by 1.4x.

Quality gate: `vllm-w4a16/tools/quality_gate.py --compare` against `results/quality-baseline-2026-10-17.json`, run first after every candidate boot. Thresholds come from the R14 noise passes: SUSPECT at a first divergence above a 2.0 nat baseline margin or a prefix drift above 1.0 nat; REVIEW if the run's p99 \|dlogprob\| exceeds 0.25 nat (noise: 0.119-0.124).

## Time accounting and floors (T2)

Detail and reasoning in `journal/02-time-accounting.md`. Traces: `.cache/ple/prof-*` (gitignored); summaries in `results/trace-breakdown-T2.txt` (`tools/trace_breakdown.py`, rank 0; a step is accounted from its start to the next step's start, so the MTP draft and the sampler are included).

Measured ceilings of this box (no server running): bf16 GEMM 75.8 TFLOPS at 8192^3 and 43-73 at the lane's per-rank prefill shapes; HBM copy 844 GB/s (`results/floor-bench-T2-20261017-212430.json`). Pinned copies, one card alone: H2D 25.5, D2H 22.8, both at once 13.7 GB/s each way. All four cards at once: **31.2 H2D, 28.9 D2H, 14.4 GB/s per direction in total when both directions run** (`results/floor-bench-T2-pcie-20261017-212620.json`). Host DRAM copy 21-28 GB/s at 1 to 32 processes; EDAC shows 4 of 8 memory channels populated (32 GB dual-rank RDIMMs in channels 0, 3, 5, 6). Every link trains at Gen4 x16. So the host side, not the links, caps everything that crosses host memory. (PLAN's 35.6 TFLOPS is the TF32 rate; bf16 with fp32 accumulate is 71 nominal.)

NCCL 2.29.7 on this box (`tools/nccl_bench.py`, `results/nccl-bench-T2-*.json`): transport SHM for every pair; all-reduce 20 KB 39 us in a CUDA graph (54 eager), 120 / 160 KB 135-180 us, 8.5 MB 4.33 ms, 10.5 MB 5.38 ms (busbw 2.9 GB/s). No env setting moves the big messages: `NCCL_PROTO=Simple` 4.36 ms (but 95-110 us at 120-160 KB, -35%; 66 us at 20 KB, +70%), LL128 8.9 ms, Tree 5.17 ms, `MIN_NCHANNELS` 8 / 16 4.6-4.8 ms, `NTHREADS=512` 4.33, `BUFFSIZE=16M` 4.37, `P2P_DISABLE=1` 4.39. `NCCL_SHM_USE_CUDA_MEMCPY=1` (copy-engine SHM) hangs at the 1.3 MB message with `NCCL_CUMEM_ENABLE` 0 or 1.

**The all-reduce floor.** Without GPU P2P a 4-rank all-reduce of S bytes moves 6S out of the GPUs and 6S back in through host memory, whatever the algorithm (ring or reduce-scatter + all-gather). For 8.5 MB that is 102 MB; at the measured 28.8 GB/s for all four cards both ways it is 3.55 ms. NCCL takes 4.33 ms: **82% of the host-path floor**. Prefill comm can only shrink by moving fewer bytes through the host (P8 TP2xPP2: a 2-rank all-reduce moves 4S, a third), by a faster host path (P13 P2P, which needs the user; the memory configuration is fixed at 4 x 32 GiB, confirmed by the user), or not at all.

### Prefill, per deep 1,664-token chunk (R8 `prof-pc1`, 15 steps; period 994 ms = 1,674 tok/s at the step level)

| Component | Measured (ms/step) | Floor | Floor (ms/step) | Gap (ms) | Lever |
|---|---:|---|---:|---:|---|
| NCCL all-reduce, 106 x 8.5 MB | 471 | 102 MB per call through the host at 28.8 GB/s | 376 | 95 | P2 (at most this gap); P8 (traffic / 3); P13 / memory (user) |
| NCCL all-gather, 10 calls (draft) | 14 | same path | ~11 | 3 | - |
| **PLE host gather (GPU idle between the ids' D2H and the rows' H2D)** | **199** (shallow steps **413**, cache off 233 per 1,978) | GPU gather by UVA from pinned host RAM, 4 GPUs at once: 0.43-0.82 ms; or a C-speed gather from a resident mapping: 1.7 ms + 0.4 ms H2D | 1-2 | **~197 (shallow ~411)** | **P4** |
| Dense GEMMs (`gemm`) | 96 | 4.58 TFLOP per card (of which the replicated HC mixes 2.13) at the measured shape rates (~55 TFLOPS), at peak 60 ms | 60-83 | 13-36 | - (P12: sharding HC adds collectives) |
| QSA sparse attention + top-k | 95 + 21 | gather of 2,048 selected fp8 K/V rows per query per layer, ~20 GB per step without reuse at 844 GB/s; 0.3 TFLOP | ~20 | ~95 | P10 (parked; now the third-largest gap, and it grows with depth) |
| Marlin W4A16 MoE | 43 | 1.96 TFLOP at 75.8 (weights 14.6 GiB at 844 GB/s: 19 ms, overlapped) | 26 | 17 | - |
| HC glue (`hc_combine_norm`, `gate_mix`, `silu`) | 25 | bandwidth over the 10,240-wide state (rough) | ~10 | ~15 | - |
| GDN, MoE glue, elementwise, copies | 23 | (measured; small) | 23 | 0 | - |
| Other idle | 5 | | 0 | 5 | - |
| **Sum** | **994** | | **~540** | **~455** | step-level ceiling ~3,100 tok/s |

### Decode

| Component (ms per step) | 1 stream, 4 tok/step (R16 code prompt, 59 steps, profiled; R8 unprofiled period 22.4) | 6 streams, 16 tok/step (R16, 43 steps, profiled) | Floor |
|---|---:|---:|---|
| Period | 23.0 | 41.7 | memory: target 4.28 GB per card per step (5.1 ms at 844 GB/s) + draft ~1.3 GB (1.5 ms) = 6.6 ms at 1 stream; 9.0 GB (10.6 ms) + draft at 24 tokens |
| NCCL all-reduce, 106 calls | 5.7 (unprofiled 4.3 = 106 x 40 us) | 16.0 (106 x 150 us) | 12 x message through the host: 20 KB -> 8 us + latency; 120-160 KB -> 50-66 us |
| NCCL all-gather, 10 calls (draft logits) | 2.1 | 4.4 | ~0.5 / ~1 |
| PLE stall | 1.3 | 4.6 | 0.015 (UVA gather of 64 / 384 rows) |
| GEMMs (target + draft) | 8.7 (2.2 in the draft) | 9.6 | dense weights 3.10 GB per card = 3.7 ms |
| Marlin MoE + glue | 2.9 | 4.7 | MoE weights 1.18 GB = 1.4 ms (1 stream) |
| Kernels per step | 2,392 | 2,411 | each costs ~2-4 us in a graph even when tiny: P7 |

### Turn overhead outside prefill (R13 scoreboard, R15 `tools/turn_overhead.py`)

Server TTFT minus server prefill time is 10 ms at 1K, 63 ms at 16K, 143-189 ms at 64K and **340-378 ms at 128K, cold and warm alike**; client minus server TTFT adds 5 / 15 / 50 / 80 ms. R15, the same warm 128K prompt three ways: chat 0.69-0.72 s client TTFT (server-prefill gap 311-343 ms), completions with text 0.68 (299-306), **completions with token ids 0.42 s (23 ms)**. The gap is server-side tokenization of the whole prompt on every turn. Warm 128K TTFT = 0.30 s prefill (a 236-token tail step) + 0.34 s tokenization + 0.08 s client side. Also: the Triton cache lives at `/root/.triton/cache` inside the container (93 MB), outside the mounted `/root/.cache`, so every boot JIT-compiles again; first-time shapes cost 0.4-1.8 s per kernel (R16 traces; the JIT monitor warns once per kernel name only).

### Portfolio re-ranked after T2

| Rank | Item | Measured ceiling | Odds | Effort | Change from PLAN |
|---|---|---|---|---|---|
| 1 | **P4 PLE gather off the host (T5)** | prefill deep -197 ms of 994 (1.25x), shallow -411 of 1,209 (1.5x); decode -1.3 ms of 22.4 (6%), -4.6 of 41.7 at 6 streams (11%) | high: bytes identical, 250-1000x faster in the bench | 1-2 days | up from 4; the resident unpinned variant (1.7 ms) needs no memlock, so the sandbox can validate it |
| 2 | **P9 turn overheads (T11)** | warm 128K 0.72 -> ~0.42 s; incremental 128K + 4K 3.73 -> ~3.4 s; first-shape JIT spikes after every boot (TRITON_CACHE_DIR, one flag) | high | hours | up from 9 |
| 3 | P3 Mamba page leak (T4) | 128K / 155K cold 777-849 -> ~1,395 tok/s at the same per-step cost; 0 preemptions | medium-high | 0.5-1 day | unchanged |
| 4 | P8 TP2 x PP2 (T10) | comm bytes / 3: prefill ~2x on top of P4 if chunks pipeline; decode likely a few ms slower per step (two stages in series) | medium | 2-3 days | up from 8; its condition (comm > 25% of the step after P1/P2) is met by measurement, since P1 is dead and P2 can take at most 95 of 471 ms |
| 5 | P2 host-staged all-reduce (T6), decode sizes | 1 stream: 106 x (40 - ~15 us) = ~2.6 ms of 22.4 (12%); 6 streams: 106 x (150 - ~70 us) = ~8 ms of 41.7 (20%); prefill at most 95 ms (10%) | medium | 2-4 days | down from 2; prefill half nearly moot |
| 6 | P5 speculation breadth (T7a / T7b) | code acceptance 0.93 / 0.85 / 0.70 per position: depth 4-6 pays on code | medium | 1 boot each / days | unchanged |
| 7 | P1 NCCL tuning (T3), reduced | big messages: none (killed by the microbench); 120-160 KB: Simple is 35% faster but 20 KB is 70% slower, so it needs a per-size choice (tuner plugin) and a 6-stream boot | low-medium | hours | down from 1 |
| 8 | P10 QSA kernel | ~95 ms per deep step (10%), more at depth | medium | days | unparked candidate |
| 9 | P7 compile, P11 draft KV group, P6 chunk 4096 | P7: kernel count; P11 ~1.5%; P6: align mode already rounds the 2,048 budget down to 1,664 tokens per step (2 x 832), a larger budget mostly amortises the ~45 ms per-step tail | low | - | P6 down |
| - | P13 PCIe P2P (host memory channels: closed, the user keeps 4 x 32 GiB) | the comm floor itself | - | host | needs the user |

**After T5 (R19):** P4 is done and kept. The deep prefill step is now ~777 ms per 1,664 tokens (2,142 tok/s at 64K), so the all-reduce (471 ms, unchanged) is ~60% of it and P8 (T10) gains weight; QSA (~116 ms) is the next compute item. The order stands: T11 (P9), T4 (P3; 128K/155K are now preemption-bound: 1,184-1,278 against 2,142 at 64K), T14, then T10.

**After T11 (R22):** P9 is done and kept: tokenization, Triton JIT and hostpin boot time.
- Frontend cost per turn is now ~20-30 ms at any depth.
- The next fixed cost is the ~45 ms per-step tail, then the launch-bound small steps (P7).
- The order stands: T4 (P3; 128K/155K cold at 1,213-1,310 tok/s against 2,189 at 64K, still 2 preemptions each), T14, then T10.

## Boot ledger (R13 onward; R1-R12 are in the lane README)

| Boot | Change | Result |
|---|---|---|
| R13 | (T1) prefix lane, sandbox: `scoreboard.py` / `quality_gate.py` smoke tests, then the full scoreboard (`results/scoreboard-R13-20261017-192716.{json,log}`, 17 min) | ready ~4 min, overlays 7/7, MARLIN, pool 299,482; count 186.0, 64K cold 1,396 tok/s, 128K cold 777 (2 preemptions), needle at 199,406 correct; 10 preemptions over the run, all in the >100K cold prompts; same-boot temperature-0 repeats are not bit-identical (see open questions) |
| R14 | (T1) prefix lane, fresh boot: `quality_gate.py --record` (`results/quality-baseline-2026-10-17.json`), two same-boot `--compare` noise passes, `scoreboard.py --sections decode,ttft` | gate PASS: tool calls parse, needles 29,958 / 99,726 / 199,393 correct, 155K 849 tok/s with 2 preemptions, pool 299,482; same-boot repeats diverge on 6-9 of 20 prompts at near-ties (p99 \|dlogprob\| 0.12), thresholds calibrated on this; count 190.6 (R13 186.0), 64K cold 1,403 |
| R15 | (T2) prefix lane, sandbox, `DOCKER_EXTRA=-e VLLM_TORCH_PROFILER_DIR=...` (ignored: this build does not know the variable, so no profile); `tools/turn_overhead.py --depth 128000 --reps 2 --probe 10` (`results/turn-overhead-R15-20261017-213810.json`) | pool 299,482; 128K cold 135.0 s server prefill (946 tok/s; preemptions not recorded by that run); warm 128K: chat 0.69-0.72 s, completions with token ids 0.42 s (tokenization ~0.3 s); probe: 10 of 10 distinct top-1 logprobs (-0.0185 to -0.0323) |
| R16 | (T2) prefix lane flags through the base launcher plus `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/.cache/prof-t2","torch_profiler_with_stack":false}'`; profiles of a code decode (192 tokens) and 6 concurrent streams (128 tokens) | args identical to R15 apart from the profiler; pool 299,482; code decode 140.7 tok/s under the profiler (163 unprofiled); traces in `.cache/ple/prof-t2`; runtime Triton JIT spikes of 0.4-1.8 s seen in the traces |
| R17 | (T2) prefix lane flags with `MTP=0` (one change), `tools/repeat_probe.py`, `turn_overhead.py --probe 10` | pool 337,597; still not repeatable: 6 of 6 distinct runs, spread up to 0.52 nat, from prompt position 0; probe 10 of 10 distinct. Speculative decoding is not the cause |
| R18 | (T2) tuned lane (`launch-qwen38fn-w4a16-tuned.sh`, no prefix caching), same probes | pool 303,079; still not repeatable: 6 of 6 distinct, spread up to 0.39 nat, from position 0. Prefix caching / align mode is not the cause |
| R19 | (T5) `PLE_MODE=hostpin` (new wrapper `launch-qwen38fn-w4a16-hostpin.sh` = prefix lane knobs with hostpin; one change): the PLE table copied by TP rank 0 into one shared memfd (47.7 GiB), registered by all four ranks, rows gathered on the GPU by UVA inside the forward and the decode graph; `quality_gate.py --compare`, then `scoreboard.py --label R19` | ready 8.5 min (setup: leader 41.8 s with 2 registration retries, followers 114 s, weight loading 193 s against 74), overlays 7/7 (ple_hostpin instead of model_state), MARLIN, pool 299,482, ~96 MiB per card for the mapping, idle free 0.75-0.83 GiB; gate PASS (14 identical, 6 FP-sized, p99 0.102 nat, needles correct, 155K 1,282 tok/s with 2 preemptions, min free 0.60 GiB); count 204.0, prose 144.9, code 180.5, 64K cold 2,142 tok/s (R13 1,396), 128K cold 1,184 (777), warm 64K 0.50 s (0.71), 128K + 16K incremental 8.35 s (12.49), 6 streams 301.2 tok/s (261.9). **Kept** |
| R20 | (T11) hostpin lane with the boot fix in `ple_hostpin.py`: the leader registers before handing the buffer out, the followers start at different 4 GiB chunks, `tp.barrier` after registration, the PLE files' page cache dropped after the fill. `TRITON_CACHE=1` on an empty cache dir, which behaves like no knob. `TOKSPLIT=0`. Ran `first_contact.py`, the gate and `turn_overhead.py --depth 128000` | ready 7:25. Setup 81 s on all ranks, 0 retries; weight loading 185 s (page cache 18 GiB before boot). Overlays 7/7, pool 299,482. First-contact JIT tax 10.3 s (1K prompt 3.91 s, against 0.50 s on pass 2). Gate PASS: 13 identical, 7 FP-sized, p99 0.118, needles correct, 155K 1,312 tok/s. Warm 128K: chat 0.64-0.65 s, ids 0.38 s, server - prefill 310 ms |
| R21 | (T11) same, after DONTNEED moved into every 64 MiB copy, on a warm Triton cache (94 MB). Ran `first_contact.py` and the gate | ready 5:27. Setup 45 s, 0 retries; weight loading 173 s; profile + KV phase 2 s (R20: 81 s), graph capture 2 s (9 s). First-contact JIT tax 0.23 s. Gate REVIEW: 1 SUSPECT, `tool_search` token 9, margin 3.75. 12 idle repeats of that request on this boot spread its margin 1.25-5.25 nat: T14 noise, not the cache. Needles correct, 155K 1,277 tok/s |
| R22 | (T11) plus `TOKSPLIT=1` (the hostpin wrapper's new default; `renderers_base.py` and `tok_split.py` mounted, `RAYON_NUM_THREADS=8`). Ran the gate, `turn_overhead.py` and the full `scoreboard.py --label R22` | ready 5:23. TOKSPLIT self-test passed at startup in 1.31 s. Setup 53 s, 0 retries; weight loading 157 s; overlays 9/9, pool 299,482. Gate PASS: 12 identical, 8 FP-sized, p99 0.116, needles correct, 155K 1,308 tok/s, prompt token counts equal to R20's. Warm 128K chat 0.36 s (ids 0.37-0.39 s), server - prefill 26 ms. Scoreboard: count 207.5, 64K cold 2,189, warm 128K 0.38 s, 128K + 4K / 16K 2.27 / 7.90 s, 6 streams 337.1. **Kept** |

## Open questions

- **No lane on this box is bit-reproducible at temperature 0, and the spread is larger than floating-point reordering** (T1, T2). With `prompt_logprobs`, the same 33-token prompt on an idle server gives 6 different runs out of 6 on the prefix lane (R15), with MTP off (R17) and on the tuned lane without prefix caching (R18); runs differ from prompt position 0 (the prediction from one token), with spreads up to 0.39-0.69 nat, and repeated values recur (a discrete set of outcomes). Ruled out by microbenchmarks (`tools/det_probe.py`, `det_probe_topk.py`): cuBLAS bf16 GEMMs at shifted addresses (with or without `CUBLAS_WORKSPACE_CONFIG`), the NCCL bf16 all-reduce, Marlin atomics (off at this commit), and the QSA top-k for short contexts. Ruled out by boots: speculative decoding, prefix caching. Left: Marlin MoE with expert parallel, the staged PLE path, the QSA overlays (split-k merge reading unwritten partials would fit), HC and GDN kernels, async scheduling. A stale-memory read would be a correctness bug, not noise; proposed as its own task (T14). Separately, `persistent_topk` at depth returns its indices in a different order every run and changed the selected set once in 20 runs (1,664 queries at 64K).
- NCCL copy-engine SHM (`NCCL_SHM_USE_CUDA_MEMCPY=1`) hangs at 1.3 MB on this box; a bug or a missing prerequisite? Only matters if T3 wants that path.
- Host DRAM copy at 21-28 GB/s with 4 of 8 channels populated caps the all-reduce. **Closed 2026-10-17: the user confirmed 4 x 32 GiB DIMMs and no hardware change.** Every plan treats ~29 GB/s through the host (all four cards) as a fixed ceiling.
- ~~Does `cudaHostRegister` on a file-backed shared mapping work here, and does it count against the sandbox memlock (T5)?~~ **Answered by T5:** no for file pages (the 3090 has no `cudaHostRegisterReadOnly`, so a read-only mapping cannot be registered); yes for a shared memfd, and the 15.7 GiB memlock is not enforced on `cudaHostRegister` (47.7 GiB per process registered in the sandbox). The host's `--ulimit memlock=-1` is therefore not what makes it work.
- ~~The hostpin boot takes 8.5 min instead of ~4.5.~~ **Answered by T11:** two causes.
  - Concurrent pinning of the same pages was slow and failed transiently. Fixed by registering on the leader first, rotating the followers' chunk order, and a barrier.
  - The PLE files' page cache pushed out the weights. Fixed by dropping it per copy.
  - Now ready in 5:23.
  - The rest is the ~50 s setup plus weight reads from NVMe at 0.3-0.6 GB/s through mmap faults. Would a sequential prefetch of the weight files close it?
- Hostpin boots now drop the PLE files from the page cache. A staged-lane boot right after one starts with the table cold (slower first long prefills).
- Hostpin keeps ~48 GiB of host RAM pinned for as long as the server runs (MemAvailable 119 -> 59 GiB while serving). The host run must not run anything else that needs that RAM.
- Why the previous step's Mamba page is not freed in align mode at this commit (T4).
- P13: PCIe P2P on the GeForce cards is a host change; diagnostics to be presented, user decides.
