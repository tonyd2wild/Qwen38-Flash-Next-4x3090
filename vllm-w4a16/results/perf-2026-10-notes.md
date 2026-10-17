# TTFT and tok/s work, 2026-10: scoreboard, ledger, open questions

The numbers file for the work planned in [`PLAN.md`](../../PLAN.md). Narrative per task is in [`journal/`](../../journal/INDEX.md). The lane README gets the final copy at the end (T13). Keep the metric definitions of `bench-host.py`: decode = completion tokens after the first / time after the first token; cold = fresh random prefix; warm = the same prompt again; incremental = a cached conversation plus N new tokens; TTFT is measured at the client from the HTTP request to the first streamed content token; single stream on a quiet server unless noted; cache hits read from `/metrics` around each request.

## Scoreboard

Columns: `host 10-17` = `current_benchmark.md` on the host under Docker (prefix lane; **the 1K and 16K rows and the decode block overlapped another client**, only the 64K and 128K rows are clean); `sandbox before` = the prefix lane in the sandbox, boot R13 (T1 fills it); `sandbox after` = the fast lane in the sandbox; `host after` = the fast lane on the host (`run-host-fast.sh`).

| Metric | host 10-17 | sandbox before (R13) | sandbox after | host after |
|---|---:|---:|---:|---:|
| Decode, count to 100, median of 5 (tok/s) | 143.6 (contaminated; R3 quiet: 188.8) | | | |
| Decode, prose 512 tokens (tok/s) | 110.4 (contaminated) | | | |
| Decode, code prompt (tok/s) | n/a | | | |
| Cold TTFT 1K / prefill tok/s | 2.63 s / 394 (contaminated) | | | |
| Cold TTFT 16K / prefill tok/s | 18.89 s / 845 (contaminated) | | | |
| Cold TTFT 64K / prefill tok/s | 45.63 s / 1,399 | | | |
| Cold TTFT 128K / prefill tok/s | 152.62 s / 837 (2 preemptions) | | | |
| Warm TTFT 1K / 16K / 64K / 128K | 1.87 (cont.) / 0.25 / 0.67 / 0.85 s | | | |
| Decode at 64K / 128K context (tok/s) | 125.0 / 136.6 | | | |
| Incremental TTFT, 16K cached + 1K / 4K new | n/a | | | |
| Incremental TTFT, 64K cached + 4K / 16K new | n/a (R3 at ~70K + ~4K: 4.0 s and 5.2 s) | | | |
| Incremental TTFT, 128K cached + 4K / 16K new | n/a | | | |
| Needle ~200K: correct / TTFT / prefill tok/s | R5: correct / 275 s / 726 | | | |
| 155K cold prefill tok/s, preemptions, min free GiB per card | R3: 843, 2, 0.38 (GPU2 0.24) | | | |
| Concurrency 2 / 4 / 6: aggregate tok/s | n/a | | | |
| Concurrency 2 / 4 / 6: per-stream decode median (tok/s) | n/a | | | |
| Concurrency 2 / 4 / 6: TTFT p50 / p99 (s) | n/a | | | |
| KV pool (tokens at 262,144 context) | 299,482 | | | |

## Time accounting (preliminary, from the R8/R9 traces; T2 redoes it by category from the JSON and adds the floors)

Per prefill step at depth (prof-pc1, 1664 tokens, caching on): span 948 ms = NCCL all-reduce 475 ms (107 ops, 3.4-4.4 ms each for 8.5 MB, ~2.5 GB/s, protocol `RING_LL`) + GPU compute ~280 ms (`aten::mm` 96, QSA 96, top-k 21, Marlin MoE 43, HC glue 25, GDN ~17, MoE glue ~8, all-gather 15) + host-bound idle ~190 ms. Shallow steps (prof-pc1-shallow): 1,165 ms with ~390 ms idle. Caching off (prof-pc0, 2048 tokens): 1,148 ms with 620 ms NCCL. Per token at depth: NCCL 0.29 ms, compute 0.17 ms, idle 0.11-0.23 ms.

Decode (count-to-100, MTP3): target graph replay 15.1-15.5 ms per 4-token step; observed ~21 ms per step. Floors and the split of the graph are T2's job.

Floors to fill in (T2): GEMM vs 35.6 TFLOPS per card (bf16, fp32 accumulate); bandwidth-bound kernels vs 936 GB/s; all-reduce vs the PCIe 4.0 host path (20-25 GB/s per direction, a 4-ring moves 1.5 x S per rank each way); per-collective latency from `nccl_bench.py`; host overhead from the CPU side of the trace.

## Boot ledger (R13 onward; R1-R12 are in the lane README)

| Boot | Change | Result |
|---|---|---|
| R13 | (T1) prefix lane, sandbox, scoreboard and quality baseline | |

## Open questions

- How much of the 190-390 ms per-step host-bound idle is the PLE gather (T2 `ple_gather_bench.py`).
- NCCL LL latency at decode sizes on the SHM path (T2); decides P2's decode path.
- Does `cudaHostRegister` on a file-backed shared mapping work here, and does it count against the sandbox memlock (T5).
- Why the previous step's Mamba page is not freed in align mode at this commit (T4).
- P13: PCIe P2P on the GeForce cards is a host change; diagnostics to be presented, user decides.
