# Six concurrent agents at gpu-memory-utilization 0.95 (boot 19), 2026-09-06 9:20 PM ET

Asked for on X after the repo recorded the gmu 0.97 OOM and the move to 0.95: rerun the six-agent workload at 0.95 with prompt lengths, p99 TTFT and any OOMs. Boot 19 = vision on, full 262,144 context, 6 seats, fp8 e5m2 KV, bf16 mamba state, 1 MP image cap, MTP3, expert parallel, gmu 0.95, pool 279,462 tokens. Harness: `tools/sixagent_load.py` and `tools/sixagent_decode.py` from the Mac over Tailscale, streaming chat completions with 8 tool schemas in every request (so prompts look like an agent's), temperature 0.2. Raw per-request records in the two JSON files beside this note. The engine log was checked after each run.

## A. Agent turns (short answers, growing prompts)

6 agents x 3 turns = 18 requests, each turn appends a fake tool result to the conversation. `tool_choice: none` so the answer streams as text.

| | |
|---|---|
| Prompt tokens | 1,310 min, 2,541 median, 3,776 max (45,721 total) |
| Output tokens | 22 to 112 per turn (1,163 total): the model answers briefly and hands back to the tools, as an agent should |
| TTFT | p50 1.82 s, p90 3.45 s, p99 7.53 s (worst case = 6 prompts of 2.5K to 3.8K tokens landing together on a 2,048-token prefill chunk) |
| Wall | 22.3 s for all 18 requests |
| Errors / OOM / preemptions | 0 / 0 / 0 (`docker logs` grep for OutOfMemory, EngineDead, preempt) |

Per-stream decode is not meaningful on 22-token answers, so it is not reported for this run.

## B. Six concurrent long outputs (sustained decode)

6 agents x 2 rounds = 12 requests, each asked for a 600-word implementation plan with tools attached but `tool_choice: none`, max_tokens 700 (all 12 hit the cap).

| | |
|---|---|
| Prompt tokens | 1,369 to 1,382 |
| Output tokens | 700 each, 8,400 total |
| Wall | 33.3 s |
| Aggregate output over the wall (includes the prefill gaps) | **252.6 tok/s** |
| Engine log, 10 s windows at 6 running | 247.9 and 222.9 tok/s; peak window 369.2 tok/s |
| Per-stream decode | 39.2 min, **48.0 median**, 66.3 max tok/s |
| TTFT | p50 1.45 s, p90 3.60 s, p99 4.03 s |
| Errors / OOM / preemptions | 0 / 0 / 0 |

For comparison, Tony's earlier run from the coding-agent latency monitor on boot 12b (gmu 0.97, text only) reported 317.2 tok/s sustained aggregate, 569.8 peak burst, 52.9 tok/s per agent, TTFT 0.47 s; different harness (tool round-trips driven by the monitor), so treat the two as the same ballpark rather than a matched pair. What changed at 0.95 is headroom, not speed: single-stream count-to-100 stayed at 191.7 to 194.6 across boots 13 to 19.

## Fleet note

While these ran, 33 Hermes agent profiles and the JARVIS dashboard were pointed at the same endpoint; the engine log showed up to 6 running requests and no preemption, so the 279,462-token pool covered the mix. Each live seat also pins 16 mamba state blocks (about 190 MB of pool) regardless of prompt length; six long conversations at once will queue behind each other rather than crash.
