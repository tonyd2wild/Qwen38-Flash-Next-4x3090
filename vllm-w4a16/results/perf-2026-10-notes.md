# TTFT and tok/s work, 2026-10: scoreboard, ledger, open questions

The numbers file for the work planned in [`PLAN.md`](../../PLAN.md). Narrative per task is in [`journal/`](../../journal/INDEX.md). The lane README gets the final copy at the end (T13). Keep the metric definitions of `bench-host.py`: decode = completion tokens after the first / time after the first token; cold = fresh random prefix; warm = the same prompt again; incremental = a cached conversation plus N new tokens; TTFT is measured at the client from the HTTP request to the first streamed content token; single stream on a quiet server unless noted; cache hits read from `/metrics` around each request.

## Scoreboard

Columns: `host 10-17` = `current_benchmark.md` on the host under Docker (prefix lane; **the 1K and 16K rows and the decode block overlapped another client**, only the 64K and 128K rows are clean); `sandbox before` = the prefix lane in the sandbox, boot R13 (T1 fills it); `sandbox after` = the fast lane in the sandbox; `host after` = the fast lane on the host (`run-host-fast.sh`).

| Metric | host 10-17 | sandbox before (R13) | sandbox after | host after |
|---|---:|---:|---:|---:|
| Decode, count to 100, median of 5 (tok/s) | 143.6 (contaminated; R3 quiet: 188.8) | 186.0 (accept len 4.00) | | |
| Decode, prose 512 tokens (tok/s) | 110.4 (contaminated) | 124.2 (accept len 2.65; 1 run) | | |
| Decode, code prompt (tok/s) | n/a | 163.2 (accept len 3.48; median of 3) | | |
| Cold TTFT 1K / prefill tok/s | 2.63 s / 394 (contaminated) | 0.81 s / 1,274 | | |
| Cold TTFT 16K / prefill tok/s | 18.89 s / 845 (contaminated) | 11.41 s / 1,400 | | |
| Cold TTFT 64K / prefill tok/s | 45.63 s / 1,399 | 45.78 s / 1,396 | | |
| Cold TTFT 128K / prefill tok/s | 152.62 s / 837 (2 preemptions) | 164.16 s / 777 (2 preemptions) | | |
| Warm TTFT 1K / 16K / 64K / 128K | 1.87 (cont.) / 0.25 / 0.67 / 0.85 s | 0.21 / 0.28 / 0.71 / 0.72 s | | |
| Decode at 64K / 128K context (tok/s) | 125.0 / 136.6 | 119.2 / 116.8 | | |
| Incremental TTFT, 16K cached + 1K / 4K new | n/a | 0.98 / 3.23 s | | |
| Incremental TTFT, 64K cached + 4K / 16K new | n/a (R3 at ~70K + ~4K: 4.0 s and 5.2 s) | 3.61 / 12.26 s | | |
| Incremental TTFT, 128K cached + 4K / 16K new | n/a | 3.73 / 12.49 s | | |
| Needle ~200K: correct / TTFT / prefill tok/s | R5: correct / 275 s / 726 | correct (199,406 tok) / 267 s / 746 (4 preemptions) | | |
| 155K cold prefill tok/s, preemptions, min free GiB per card | R3: 843, 2, 0.38 (GPU2 0.24) | 838, 2, 0.68 (0.81 / 0.81 / 0.68 / 0.81; nvidia-smi poll, not comparable to R3's 0.38, see notes) | | |
| Concurrency 2 / 4 / 6: aggregate tok/s | n/a | 159.4 / 224.5 / 261.9 | | |
| Concurrency 2 / 4 / 6: per-stream decode median (tok/s) | n/a | 89.7 / 62.8 / 51.2 | | |
| Concurrency 2 / 4 / 6: TTFT p50 / p99 (s) | n/a | 1.00 / 1.42, 0.84 / 1.58, 1.57 / 2.38 | | |
| KV pool (tokens at 262,144 context) | 299,482 | 299,482 | | |

Notes on the R13 column (detail in `journal/01-baseline.md`): measured with `vllm-w4a16/tools/scoreboard.py --label R13` (`results/scoreboard-R13-20261017-192716.json`). Boot-to-boot spread (R14 against R13): decode ±2.5%, cold prefill ±1.5%, warm TTFT ±0.1 s. A smaller win needs repeated runs. Acceptance mean length: count 4.00, prose 2.65, code 3.48. Incremental rows: the server's prefill time accounts for all but 20-80 ms of the client TTFT, at 1,340-1,380 tok/s on the uncached tokens. The min-free column comes from an nvidia-smi poll, which sees the caching allocator's reserved peak (flat 0.81 / 0.81 / 0.68 / 0.81 GiB in every long run), not transients, so it is not comparable to R3's 0.38 / 0.24. The needle and 155K prompts are real token counts; the older tools' 0.75 tokens-per-word filler undercounted by 1.4x.

Quality gate: `vllm-w4a16/tools/quality_gate.py --compare` against `results/quality-baseline-2026-10-17.json`, run first after every candidate boot. Thresholds come from the R14 noise passes: SUSPECT at a first divergence above a 2.0 nat baseline margin or a prefix drift above 1.0 nat; REVIEW if the run's p99 \|dlogprob\| exceeds 0.25 nat (noise: 0.119-0.124).

## Time accounting (preliminary, from the R8/R9 traces; T2 redoes it by category from the JSON and adds the floors)

Per prefill step at depth (prof-pc1, 1664 tokens, caching on): span 948 ms = NCCL all-reduce 475 ms (107 ops, 3.4-4.4 ms each for 8.5 MB, ~2.5 GB/s, protocol `RING_LL`) + GPU compute ~280 ms (`aten::mm` 96, QSA 96, top-k 21, Marlin MoE 43, HC glue 25, GDN ~17, MoE glue ~8, all-gather 15) + host-bound idle ~190 ms. Shallow steps (prof-pc1-shallow): 1,165 ms with ~390 ms idle. Caching off (prof-pc0, 2048 tokens): 1,148 ms with 620 ms NCCL. Per token at depth: NCCL 0.29 ms, compute 0.17 ms, idle 0.11-0.23 ms.

Decode (count-to-100, MTP3): target graph replay 15.1-15.5 ms per 4-token step; observed ~21 ms per step. Floors and the split of the graph are T2's job.

Floors to fill in (T2): GEMM vs 35.6 TFLOPS per card (bf16, fp32 accumulate); bandwidth-bound kernels vs 936 GB/s; all-reduce vs the PCIe 4.0 host path (20-25 GB/s per direction, a 4-ring moves 1.5 x S per rank each way); per-collective latency from `nccl_bench.py`; host overhead from the CPU side of the trace.

## Boot ledger (R13 onward; R1-R12 are in the lane README)

| Boot | Change | Result |
|---|---|---|
| R13 | (T1) prefix lane, sandbox: `scoreboard.py` / `quality_gate.py` smoke tests, then the full scoreboard (`results/scoreboard-R13-20261017-192716.{json,log}`, 17 min) | ready ~4 min, overlays 7/7, MARLIN, pool 299,482; count 186.0, 64K cold 1,396 tok/s, 128K cold 777 (2 preemptions), needle at 199,406 correct; 10 preemptions over the run, all in the >100K cold prompts; same-boot temperature-0 repeats are not bit-identical (see open questions) |
| R14 | (T1) prefix lane, fresh boot: `quality_gate.py --record` (`results/quality-baseline-2026-10-17.json`), two same-boot `--compare` noise passes, `scoreboard.py --sections decode,ttft` | gate PASS: tool calls parse, needles 29,958 / 99,726 / 199,393 correct, 155K 849 tok/s with 2 preemptions, pool 299,482; same-boot repeats diverge on 6-9 of 20 prompts at near-ties (p99 \|dlogprob\| 0.12), thresholds calibrated on this; count 190.6 (R13 186.0), 64K cold 1,403 |

## Open questions

- **The prefix lane is not bit-reproducible at temperature 0** (T1). The same 29-token prompt with `max_tokens=1` gives top-1 logprobs -0.58 to -0.71 across four sends on one boot, and 6-9 of the gate's 20 prompts diverge on a same-boot repeat, at margins of 0-0.5 nat (once 1.25). Source unknown: reduction atomics, MoE router near-ties, or the PLE path. Probe in T2 (`PLE_MODE=none`, `--no-async-scheduling`, MTP=0; ten repeats each). The lossless claim rests on this noise floor.
- How much of the 190-390 ms per-step host-bound idle is the PLE gather (T2 `ple_gather_bench.py`).
- NCCL LL latency at decode sizes on the SHM path (T2); decides P2's decode path.
- Does `cudaHostRegister` on a file-backed shared mapping work here, and does it count against the sandbox memlock (T5).
- Why the previous step's Mamba page is not freed in align mode at this commit (T4).
- P13: PCIe P2P on the GeForce cards is a host change; diagnostics to be presented, user decides.
