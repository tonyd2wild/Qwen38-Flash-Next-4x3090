# Tony's two agency suites against this lane (boot 19), 2026-09-07 1:00 to 1:18 AM ET

Both suites are Tony's own, built from daily 2Wild agency work. They measure whether a model does what an operational agent is told, not general intelligence. Endpoint under test: `http://<3090>:8090/v1`, boot 19 (vision on, 262,144 context, 6 seats, fp8 e5m2 KV, bf16 mamba state, 1 MP image cap, MTP3, expert parallel, gmu 0.95). Nothing else was running on the box.

## 69-scenario tool suite ([2wild-model-eval](https://github.com/tonyd2wild/2wild-model-eval), quality out of 100)

| Run | Quality | Pass / partial / fail | Median turn | Deployability |
|---|---:|---|---:|---:|
| thinking off, run 1 | 89.9 | 60 / 4 / 5 | 616 ms | 92.8 |
| thinking off, run 2 | 92.0 | 62 / 3 / 4 | | |
| **thinking on** (`chat_template_kwargs enable_thinking true`) | **94.9** | 64 / 3 / 2 | 1,142 ms | 95.4 |

Reference rows on the same leaderboard: Qwen3.8-27B W4A16 + DFlash2 on 2x3090 97.1 (66/2/1); stock Qwen3.6-35B-A3B AutoRound 94.9 (63/5/1); Qwen3.8-Flash-Next on a DGX Spark (Aug 26) 95.7 (65/2/2); Nemotron-3.5-Lightning W4A16 + DSpark 85.5 (56/6/7).

Where thinking-off lost points: omitted required parameters (invented an amount and a destination instead of asking), one unchecked fund transfer, one ignored tool error, one guessed ticker. Thinking on removed the invented parameters and the ignored error; the unchecked transfer, the unclear refusal and the guessed ticker stayed, and one benign request was over-refused.

## 100-task 2Wild Agency Benchmark (private workspace runner, judged by GLM-5.3-Flash TP4 on the Sparks, same judge as the earlier rows)

| Model | Passed / 100 | Wall |
|---|---:|---:|
| Qwen3.8-27B (2x3090 TP2) | 78 | |
| GLM-5.3-Flash TP2 | 74 | |
| **Qwen3.8-Flash-Next, this lane, thinking off** | **67** | 466 s |
| Nemotron-3.5-Lightning | 50 | |

By category (of 10): discount/affiliate 9, security 9, sneaker pipeline 9, agentic tool use 8, Shopify 7, social captions 7, fleet/infra 6, SolePlay KPI 5, YouTube 4, posting rail 3. Of the 33 misses, 11 were exact-string tasks (right idea, not the required string), 8 were arithmetic (sell-through, RPV, YouTube revenue: the same tasks GLM misses, which argues for doing the math in code), the rest judgment calls (caption lead, "reel" vs "reels", an ET to UTC roll-over, echoing an OAuth token, two fleet misdiagnoses). Not rerun with thinking on yet.

## Reading

On these suites the 27B is the better operational worker, by 2 to 7 points on tools depending on thinking, and by 11 on the agency tasks. This lane is the stronger and faster model for hard coding, research, vision and long context, and thinking on closes most of the tool gap at about double the per-turn latency. Which matters more is a fleet decision, not a benchmark one.
