# 01: Baseline tools and the "before" column (T1)

Status: done · Model: Opus 5.5 (the board recommends Sonnet 5.5; the user chose Opus for this task) · Effort: high · Started: 2026-10-17 19:17 · Wall time: 1:15 (two boots)

## Goal

Build `scoreboard.py` and `quality_gate.py` (plus the shared `benchlib.py`), boot the prefix lane, fill the "sandbox before (R13)" column of the notes file and freeze the temperature-0 quality baseline that every candidate's `--compare` runs against.

## Hypothesis and kill test

No hypothesis: this is a measurement task. Comparability check instead: R13's count-to-100 and 64K cold/warm rows should land near the README's quiet numbers (188.8 tok/s; ~1,377-1,399 tok/s at 64K; warm 0.67-0.81 s). They did (186.0; 1,396; 0.71 s).

## Setup

- Branch / commit: `perf-comm-ple` @ `d4157e0`
- Lane and flags: `vllm-w4a16/launch/launch-qwen38fn-w4a16-prefix.sh` unchanged, `GPU_ARGS="--device nvidia.com/gpu=all"`, image `qwen38fn-vllm:8a728663`
- Ledger rows: R13 (tool smoke tests and the full scoreboard), R14 (fresh boot: the frozen quality record, two same-boot noise passes, decode + TTFT again for boot-to-boot spread)
- Box state before: GPUs free (15-92 MiB), host server down (`host.docker.internal:8090` no answer at 19:17), page cache warm (yes)

## The tools

All under `vllm-w4a16/tools/`, stdlib only, `URL=` (default `http://localhost:8090`). They are documented in their header comments.

- `benchlib.py`: shared code. `chat()` is `bench-host.py`'s request (temperature 0, thinking off, streamed TTFT to the first content token, decode = (completion tokens - 1) / (last - first token)), with `/metrics` deltas attached (cache hits, preemptions, spec-decode drafts and per-position acceptance, server latency histogram sums). `assert_idle()` aborts with exit 3 unless `num_requests_running` and `num_requests_waiting` are 0. It polls for up to 10 s so a just-finished request can drain, and it runs before and after every measured request. Also: `warmup()` (short prompt, count to 100, ~8K prompt; T0 found that the first count after a boot is slow from Triton JIT), `GpuMemPoller` (nvidia-smi every 0.5 s, minimum free per card), the agent `SYSTEM`/`TOOLS`/`TASKS` copied from `sixagent_load.py`, and the two fillers.
- `scoreboard.py --label RNN [--sections decode,ttft,incremental,needle200k,cold155k,concurrency] [--quick]`: the notes file's rows, with `results/scoreboard-<label>-<time>.{json,log}` and a final block of the rows in notes-file order, ready to paste. The full run takes 17 min.
- `quality_gate.py --record [PATH] | --compare [PATH] | --rescore COMPARE_JSON`: 20 frozen prompts (5 code, 4 prose, 5 tool calls, 3 long-context QA at 8K/16K/30K, 1 vision (a stdlib-generated PNG), thinking off and on). They are sent non-streamed with `logprobs`, `top_logprobs: 5` and `return_tokens_as_token_ids`. The drop-in checks follow: tool calls parse to a known tool with JSON arguments, no reasoning by default, the reasoning split with `enable_thinking`, needles at 30K/100K/200K, a 155K cold prefill, and boot-log facts (pool ≥ 262,144, `max_num_seqs` 6, `max_model_len` 262,144, `qwen3_xml`, `qwen3`, thinking off by default) plus `/v1/models`. A full run takes about 10 min. Exit 0 PASS, 1 FAIL, 2 REVIEW. `--rescore` re-judges a saved compare report with the current thresholds and sends no requests.
- Baseline: `vllm-w4a16/results/quality-baseline-2026-10-17.json` (R14, fresh boot, cold cache).

**Protocol for every later candidate boot:** run `quality_gate.py --compare` first after the boot (it does its own warm-up), then `scoreboard.py`. The prefix cache is part of the compute path, so the compare has to see the same cold cache the record saw. The gate stores per-prompt hits and prints `cache a->b` when they differ. In a full run the 200K needle and the 155K prefill evict the long prompts, so the R14 passes went back to cold for every prompt except the shared 832-token tool prefix, which is hit identically in record and compare.

## What I did

```bash
# R13, launched 19:18:20, "Application startup complete" by 19:22; pool 299,482; overlays 7/7; MARLIN
GPU_ARGS="--device nvidia.com/gpu=all" bash vllm-w4a16/launch/launch-qwen38fn-w4a16-prefix.sh
docker logs vllm_qwen38fn_3090 > .cache/ple/r13-boot.log 2>&1
# smoke tests (output in the session scratchpad, not committed)
python3 vllm-w4a16/tools/scoreboard.py --quick --label smoke --out-dir $SP
python3 vllm-w4a16/tools/scoreboard.py --quick --sections decode --label guard --no-warmup   # with a 1500-token curl running: ABORT [start], exit 3
python3 vllm-w4a16/tools/quality_gate.py --record $SP/qg-smoke.json --skip-long --label smoke --out-dir $SP
python3 vllm-w4a16/tools/quality_gate.py --compare $SP/qg-smoke.json --skip-long --label smoke --out-dir $SP
# the "before" column, 19:27-19:44
URL=http://localhost:8090 python3 vllm-w4a16/tools/scoreboard.py --label R13
docker stop -t 60 vllm_qwen38fn_3090
# R14, launched 19:45; fresh cache for the frozen record
GPU_ARGS="--device nvidia.com/gpu=all" bash vllm-w4a16/launch/launch-qwen38fn-w4a16-prefix.sh
python3 vllm-w4a16/tools/quality_gate.py --record vllm-w4a16/results/quality-baseline-2026-10-17.json --label R14
python3 vllm-w4a16/tools/quality_gate.py --compare --label R14-sameboot1
python3 vllm-w4a16/tools/quality_gate.py --compare --label R14-sameboot2
python3 vllm-w4a16/tools/quality_gate.py --rescore vllm-w4a16/results/quality-compare-R14-sameboot{1,2}-*.json   # after recalibrating
URL=http://localhost:8090 python3 vllm-w4a16/tools/scoreboard.py --sections decode,ttft --label R14
docker logs vllm_qwen38fn_3090 > .cache/ple/r14-boot.log 2>&1; docker stop -t 60 vllm_qwen38fn_3090
```

A first `scoreboard.py --label R14` call was interrupted during warm-up (an accidental tool rejection; the stub log was deleted). The rerun started on an idle server, and its guard never fired.

## Measurements

The full column is in the notes file. These are the rows later tasks will A/B against (sources: `results/scoreboard-R13-20261017-192716.json`, `results/scoreboard-R14-20261017-202351.json`).

| Metric | README / earlier | R13 | R14 | R14 - R13 |
|---|---:|---:|---:|---:|
| Decode, count to 100, median of 5 (tok/s) | 188.8 (R3) | 186.0 | 190.6 | +2.5% |
| Decode, prose 512 tokens (tok/s) | 109.5 (2026-09-06) | 124.2 | 126.1 | +1.5% |
| Decode, code prompt, median of 3 (tok/s) | 145.8 (other prompt) | 163.2 | 167.2 | +2.5% |
| Cold TTFT 1K / prefill tok/s | - | 0.81 s / 1,274 | 0.81 s / 1,271 | 0 |
| Cold TTFT 16K / prefill tok/s | - | 11.41 s / 1,400 | 11.56 s / 1,380 | -1.4% |
| Cold TTFT 64K / prefill tok/s | 1,377-1,399 | 45.78 s / 1,396 | 45.45 s / 1,403 | +0.5% |
| Cold TTFT 128K / prefill tok/s | 837 (host) | 164.16 s / 777, 2 preemptions | 164.17 s / 778, 2 preemptions | 0 |
| Warm TTFT 1K / 16K / 64K / 128K (s) | 0.25 / 0.67 / 0.85 (host) | 0.21 / 0.28 / 0.71 / 0.72 | 0.21 / 0.24 / 0.62 / 0.74 | ±0.09 |
| Decode at 64K / 128K (tok/s) | 125.0 / 136.6 (host) | 119.2 / 116.8 | 118.3 / 111.5 | |
| Incremental 16K + 1K / 4K (s) | - | 0.98 / 3.23 | | |
| Incremental 64K + 4K / 16K (s) | 4.0 / 5.2 at ~70K + ~4K (R3, other shape) | 3.61 / 12.26 | | |
| Incremental 128K + 4K / 16K (s) | - | 3.73 / 12.49 | | |
| Needle ~200K | R5: correct, 275 s | correct, 199,406 tok, 267 s, 746 tok/s, 4 preemptions | R14 gate: correct, 266 s | |
| 155K cold prefill tok/s, preemptions | R3: 843, 2 | 838, 2 | gate: 849 / 988 / 831, 2 each | |
| Concurrency 2 / 4 / 6: aggregate tok/s | - | 159.4 / 224.5 / 261.9 | | |
| Concurrency 2 / 4 / 6: per-stream decode median | - | 89.7 / 62.8 / 51.2 | | |
| Concurrency 2 / 4 / 6: TTFT p50 / p99 (s) | - | 1.00 / 1.42, 0.84 / 1.58, 1.57 / 2.38 | | |
| Acceptance mean length: count / prose / code | 3.39 (R12 log) | 4.00 / 2.65 / 3.48 | 4.00 / 2.65 / 3.48 | |

The noise budget for an A/B, judged from R13 against R14: decode ±2.5%, cold prefill ±1.5%, warm TTFT ±0.1 s. A claimed win below these needs repeated runs.

## Result

**Done; kept.** The tools work. Their numbers agree with the README's quiet runs, so the scoreboard keeps `bench-host.py`'s definitions. The "before" column and the frozen baseline are committed. The gate's thresholds are calibrated on measured noise, not guessed (next section).

## Quality gate

R14 record: PASS. 20/20 prompts recorded, all 5 tool calls parse to the expected tool, the vision prompt answers "Left red, right blue.", thinking off by default (no reasoning) and the reasoning split when on, long QA correct at 8K/16K/30K. Needles: 29,958 tok in 21.7 s, 99,726 in 72.0 s, 199,393 in 266.4 s, all correct; the 200K needle had 4 preemptions. 155K cold: 154,545 tok in 182.1 s (849 tok/s), 2 preemptions, no error. Pool 299,482.

**The lane is not bit-reproducible at temperature 0, even on the same boot.** This is the central finding of T1, and the gate is built around it:

- Same boot, no cache hits, short prompts: 6-9 of the 20 prompts diverge on a repeat (R13 smoke: 9; R14 pass 1: 9; pass 2: 6). It is never a semantic change; the outputs are rewordings ("I'll create the simple hello world script." vs "I'll create the script file with a simple hello world program").
- It starts in prefill. The same 29-token prompt with `max_tokens=1`, sent four times, gives top-1 logprobs of -0.640 / -0.709 / -0.582 / -0.696. So the variation is in the forward pass itself, not in speculation or sampling.
- Divergences happen at near-ties. The baseline's top1-top2 margin at the first divergent token was 0-0.5 nat in 23 of 24 cases (logprobs move in steps of 0.125, so the logits are effectively bf16). The exception is `tool_write` token 4 in R14 pass 1, at a 1.25 nat margin: token 5148 scored -2.08 / -0.34 / -1.69 in the three runs with a common-prefix drift of 0.015.
- Over common prefixes (2,503 positions, R14 passes 1 + 2), |dlogprob| has p50 0.000, p90 0.021, p99 0.122, max 0.451.

So the thresholds are: SUSPECT if a first divergence sits at a baseline margin above 2.0 nat, or if one prompt's prefix drift exceeds 1.0 nat; REVIEW if the run's p99 |dlogprob| over all common prefixes exceeds 0.25 nat (about 2x noise). Rescored with these, both R14 passes are PASS (p99 0.124 and 0.119). Pass 1 had been REVIEW under my first guess of 1.0 / 0.5, because of `tool_write`. A synthetic check confirms the aggregate catches a small systematic change: subtracting |N(0, 0.15)| from every logprob of pass 2 gives REVIEW at p99 0.403, while no per-prompt check fires. A gross error (wrong all-reduce, broken kernel) shows up as early high-margin divergences and failed checks.

## Side findings

- **The fillers' token ratio.** `needle_3090.py` and `prefix_cache_3090.py` assume 0.75 tokens per word, but their text tokenizes at about 1.05. Their "50K" was the 69,834-token prompt in the README, and a "200K" target there would be ~280K. `benchlib.filler_needle` divides by 1.05, so the new tools' sizes are real tokens (needle 199,393-199,406; 155K = 154,447-154,569).
- **Incremental turns cost about new tokens / prefill rate, with no big fixed overhead.** 64K + 16K: 16,591 uncached tokens in 12.0 s server prefill time = 1,380 tok/s. 16K + 4K: 4,218 in 3.15 s = 1,340 tok/s. The server's queue time is 0.000 s, and client TTFT minus server TTFT is 20-80 ms. For P9 (T11) the fixed-overhead target looks small; the lever is prefill speed. One oddity: 16K + 1K is 0.98 s for 1,228 uncached tokens (0.91 s server prefill, 1,350 tok/s), about the same as a 1K cold prompt (0.81 s).
- **The cache hits include the shared tool prefix.** The 8-tool system prompt is 1,302-1,317 tokens, and its first 832-token block is reused across requests. That is the only hit in the gate's cold run.
- **The minimum free memory from `nvidia-smi` is flat** at 0.81 / 0.81 / 0.68 / 0.81 GiB in every long run (needles, 155K, the 128K row). The PyTorch caching allocator keeps its reservation, so nvidia-smi sees the reserved peak, not transient activations. That is why it reads higher than R3's 0.38 / 0.24, which came from some other method. Chunk or buffer changes (T5, T8) need `torch.cuda.mem_get_info` inside the workers, or the boot log's allocator stats, not this poller.
- Preemptions are deterministic: 2 for every cold prompt from 128K to 155K, 4 at 200K, 0 at 100K and below. This is consistent with the Mamba page leak (T4); the needle at 99,726 tokens just fits.
- MTP acceptance on the code prompt is high (per position 0.93 / 0.85 / 0.70, mean length 3.48), so code sits near count-to-100 and well above prose (0.75 / 0.53 / 0.37). This bears on T7a (depth 4-6 should pay off on code) and T7b.
- The PLE staged gather (`patch/ple_mmap.py:162`, `patch/ple_layer.py:535-555`) reuses one pinned host buffer per call, and the next overwrite is guarded by the synchronising `ids.to("cpu")` on the same stream. I read it to rule out a host-buffer race as the source of the nondeterminism; it looks safe. T5's hostpin design removes it anyway.
- `assert_idle` was tested live: a background 1500-token request makes the scoreboard abort at start with exit 3.

## Open questions

- **Where does the run-to-run nondeterminism come from?** Candidates: nondeterministic reductions (atomics in Marlin MoE or a split-k kernel, a QSA split-k merge), MoE router near-ties flipping an expert (that would explain single logits moving 1.7 nat), or something in the staged PLE path. Cheap tests for T2, each repeating the `max_tokens=1` probe ten times on one boot: `PLE_MODE=none` (is the PLE path involved; this changes outputs, so it is a diagnostic only), `--no-async-scheduling`, MTP=0, and one forward with the profiler to list kernels that use atomics. It matters because a later "lossless" claim rests on this noise floor, and the noise floor is wide enough to hide a small real error.
- Whether the decode spread between boots (+2.5% R14 vs R13) comes from boot state (allocator layout, NUMA, clocks). It sets the smallest decode win we can claim from one boot each side.

## Next step

T2 (time accounting), Opus 5.5 for the floors and the re-rank, Sonnet 5.5 for `trace_breakdown.py`, `nccl_bench.py` and `ple_gather_bench.py`. Add the nondeterminism probe above to T2's list. T4 (Mamba leak) depends only on T1, so it can go first if T2 waits. Its kill test is already measurable with these tools: `scoreboard.py --sections cold155k` reports preemptions (2 today), and `quality_gate.py` reports them for the needles. The server was **stopped** (GPUs free, 15-92 MiB, at 20:28).
