# The tuned lane, 2026-10-01

Launcher: `launch/launch-qwen38fn-w4a16-tuned.sh`. Same box, image, checkpoint and patch as boot 19. Two flags added: `--skip-mm-profiling` and `--kv-cache-memory-bytes 2400000000`.

## Memory and context

| | Boot 19 (2026-09-06) | Tuned (2026-10-01) |
|---|---:|---:|
| Context | 262,144 | 262,144 |
| KV pool | 279,462 tokens (1.07x) | **303,079 tokens (1.16x)** |
| KV budget | profiler's choice at gmu 0.95 | pinned at 2.4e9 bytes (about 2.24 GiB per card) |
| Free per card for transients | | about 0.65 GiB |
| Vision | on, 1 MP cap | on, 1 MP cap (live image probe passed) |
| Tools, thinking off by default | yes | yes |

Engine log line: `GPU KV cache size: 303,079 tokens, Maximum concurrency for 262,144 tokens per request: 1.16x`.

Long-prefill check: a prompt of about 155K tokens prefilled at 2,249 tok/s, 69 s to first token, no OOM.

Why the headroom matters: on the same night an experimental lane on this box (EXL3, parked) booted fine with the pool auto-sized at gmu 0.95, then died on a 155K-token prefill with `torch.OutOfMemoryError ... Tried to allocate 512.00 MiB`. vLLM's boot profiler only measures short-context activations. The fix is to pin `KV_BYTES` below what vLLM would take on its own and validate with a long prefill. W4A16's transients are leaner, and the 155K probe passed with 0.65 GiB per card free.

Walk-up rule: +50 MB of `KV_BYTES` is about +6.6K pool tokens (about 7.9 KB per token per GPU). Climb only with a 155K+ token prefill after each step.

## Tony's tool suite, 69 scenarios, thinking off ([2wild-model-eval](https://github.com/tonyd2wild/2wild-model-eval))

| Run | Points | Pass / partial / fail | Quality | Responsiveness | Median turn | Decode during suite |
|---|---:|---|---:|---:|---:|---:|
| boot 19, run 1 (2026-09-07) | | 60 / 4 / 5 | 89.9 | | 616 ms | |
| boot 19, run 2 (2026-09-07) | | 62 / 3 / 4 | 92.0 | | | |
| **tuned (2026-10-01)** | **126 / 138** | **61 / 4 / 4** | **91.3** | **99.1** | **681 ms** | **88.1 tok/s** |

Same model and the same quality band as boot 19: the tuning moved memory, not answers. Category scores at 100: tool selection, parameter precision, multi-step chains, structured output, format compliance, hallucinated tools, context retention, contradictory parameters, toolset scale (50 tools). Where it lost points: one missed tool call (TC-15), one fund transfer to an unknown account with no checks (TC-17), one invented amount instead of asking (TC-41), one over-refused security education question (TC-59), and four partials on ambiguity and empty results.

## 2Wild Agency Benchmark, 100 tasks, thinking off

| Run | Passed / 100 | Decode | Prefill | Wall |
|---|---:|---:|---:|---:|
| boot 19 (2026-09-07) | 67 | | | 466 s |
| **tuned (2026-10-01)** | **67** | **116.1 tok/s** | **about 701 tok/s** | **357 s** |

By category (of 10): discount/affiliate 9, security 9, sneaker pipeline 9, agentic tool use 8, YouTube 7, Shopify 6, social captions 6, fleet/infra 6, SolePlay KPI 5, posting rail 2. Same total as boot 19, with a different spread (YouTube up 3, Shopify down 1, captions down 1, posting rail down 1). Judge: GLM-5.3-Flash, as before, but served from a different stack than the September run, so treat per-category moves of one task as noise.

## Not re-measured on the tuned lane

Count to 100, the real-prompt single-stream set and the six-agent load were run on boot 19 and not repeated here. The decode figures above are suite averages over short agent turns, not comparable to the count-to-100 headline.
