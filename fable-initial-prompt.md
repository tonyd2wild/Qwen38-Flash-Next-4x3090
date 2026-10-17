# Brief: TTFT and tok/s for Qwen3.8-Flash-Next on 4x RTX 3090

You're taking over performance work on the local inference server in this repo: Qwen3.8-Flash-Next served by vLLM on four RTX 3090s. I want the biggest gains you can find in time to first token (TTFT) and in decode speed (tok/s). The server will eventually back a coding agent, so long, growing contexts with heavy prefix reuse are the workload that matters most.

The conventional tuning is done: the lane README's ledgers record about 30 boots of flag, memory and layout changes. What I want from you is the non-obvious: changes that move the numbers by multiples, not percent. Take the small wins too, but hunt for the big ones. Treat every design decision this stack inherited as a hypothesis, and measure every claim on this box.

## Scoreboard

Single stream on a quiet server (one client at a time), unless noted. TTFT comes in three kinds:
- **cold**: nothing cached
- **warm**: the exact same prompt again
- **incremental**: a cached conversation plus N new tokens, i.e. an agent turn

The yardstick is `./bench-host.py --sizes 1000,16000,64000,128000 --runs 5 --gen 512`; `current_benchmark.md` shows its output. Keep its metric definitions so the numbers stay comparable. Add incremental turns at depth (16K-128K cached, 1K-16K new), a code-generation decode run (count-to-100 is the draft's best case), and a ~200K needle. Secondary: 2-6 concurrent streams (aggregate and per-stream decode, TTFT p50/p99). Report them, and don't trade them away silently.

## Bounds

- **The engine is yours to change; the client, and anything between it and the engine, is not.** The endpoint stays drop-in: OpenAI-compatible on port 8090, model id `qwen3.8-flash-next`, tool calling (`qwen3_xml`), the reasoning parser with thinking off by default, vision, and a 262,144-token context. Every gain must reach an unmodified client.
- **In bounds:** launch flags, the container, vLLM overlays and patches, a vLLM source build, another engine, your own kernels.
- **Out of bounds unless you make a very strong case and I approve first:** any change to the host (drivers, kernel modules or their configuration, BIOS, sysctl, clocks and power limits), and any change to the model weights (target, MTP draft, n-gram table). Lossless changes to how the weights are stored, laid out, loaded or executed are fine.
- **Lossless only.** A change passes when all of these hold:
  - Temperature-0 outputs match the baseline on a fixed set of ~20 prompts (code, prose, tool calls, long-context QA) that you freeze at baseline. Wherever they differ, show it's floating-point reordering: the first divergent token and the logprob gap.
  - The needle test passes at ~30K, ~100K and ~200K tokens.
  - Tool calls still parse.
  - A 155K-token cold prefill doesn't OOM.
  - A 262,144-token request still fits the KV pool.
- Trading away a feature (vision, context length, seats) can only be a separately labeled option, never the new default.
- Work on a new branch off `source-build-prefix-cache`, and don't push. Keep the tuned and prefix lanes working; new lanes go in new files, in the repo's existing style.

## Where you run

- You're in a container on the GPU box (`claude-box`). The repo is at `/workspace` and the checkpoint at `~/models/qwen38fn-w4a16-fp8ple` (don't modify it). The four GPUs are visible here.
- `docker` here is rootless podman 5.4.2 inside the sandbox, not the host's Docker. Boots R1-R12 in the lane README ran this way. Pass `GPU_ARGS="--device nvidia.com/gpu=all"` (CDI). The image `qwen38fn-vllm:8a728663` isn't in this store yet: build it from `vllm-w4a16/container/Containerfile` first (the README says ~7 min cold). A boot takes 4-5 min.
- Only one server can hold the GPUs at a time. Before a boot, check that every GPU has under 1 GB in use. The host's own server answers at `http://host.docker.internal:8090`, not at localhost. If it's up, ask me to stop it; you can't from here. Your own server answers at `http://localhost:8090`.
- The demo runs on the host under Docker 29.2.1 (nvidia runtime), so the final numbers must come from there. Give me a script in the style of `run-host-docker.sh` that runs the lane and the scoreboard and writes a log I paste back, and stop your sandbox server first. Known differences:
  - The sandbox's memlock limit is 15.7 GiB and can't be raised from inside it. Rootful Docker on the host can apply the launcher's `--ulimit memlock=-1:-1`.
  - `/dev/shm` here is 32 GB.
  - Nobody has compared the sandbox and the host on the same recipe yet.
- Batch what you need from me (host runs, approvals) so I can do it in one sitting.

## The box

| | |
|---|---|
| GPUs | 4x RTX 3090 24 GB (sm_86), each on PCIe 4.0 x16. **No NVLink.** Driver 595.91.07 |
| GPU to GPU | `nvidia-smi topo -m`: GPU2-GPU3 `PHB`, every other pair `NODE`. `nvidia-smi topo -p2p r`: `CNS` (not supported) for every pair |
| CPU | AMD EPYC 7542: 32 cores / 64 threads, one socket, one NUMA node |
| RAM | 128 GB (125.6 GiB visible); no hugepages, THP `madvise` |
| Storage | WD_BLACK SN850X 4 TB NVMe under dm-crypt + LVM (model and repo); 1.8 TB free |

## The current lane (prefix cache)

`vllm-w4a16/launch/launch-qwen38fn-w4a16-prefix.sh` wraps `launch-qwen38fn-w4a16-tuned.sh`, which wraps `qwen38fn-w4a16-3090-tp4.sh`:
- vLLM at commit 8a728663, built from `container/Containerfile`, plus the overlays in `vllm-w4a16/patch/` (the launcher hash-checks them).
- TP4 with expert parallel: each card owns 128 whole experts, on the Marlin W4A16 MoE kernel. `--disable-custom-all-reduce`, `NCCL_P2P_LEVEL=NVL`.
- An MTP draft of 3 tokens (the INT4 g32 draft in the checkpoint's `runtime/mtp-int4-g32`).
- KV cache in fp8 e5m2 (an overlay variant, since e4m3 doesn't compile on sm_86); Mamba state in bf16.
- CUDA graphs FULL_DECODE_ONLY with torch.compile off, capture sizes 4-24. Chunked prefill at 2,048 tokens, 6 seats, 262,144 context. The KV budget is pinned at 2.4e9 bytes (a pool of 299,482 tokens).
- Prefix caching in Mamba align mode, with `disable_eagle_block_drop` (the lane README explains why).
- The 47.7 GB FP8 n-gram table stays in the safetensors files. Each step, the step's n-gram ids are copied to the host, a 32-thread pool reads the rows with `preadv` (16 rows of 160 B per token), and the rows are copied into a fixed GPU staging buffer before the forward or graph replay (`patch/model_state.py`, `patch/ple_layer.py`, `patch/ple_mmap.py`).

## The model (from `config.json`)

- 48 layers: 36 Gated DeltaNet linear-attention layers and 12 full-attention layers (every 4th). The full-attention layers use QSA sparse attention with an indexer (top-2,048 budget, compress ratio 4).
- Hidden size 2,560; 24 query heads, 2 KV heads, head dim 256.
- MoE with 512 experts, top-10, expert width 640, plus one shared expert.
- 4-stream hyper-connections.
- A 3-gram embedding table, injected at layer 2.
- Vocab 248,320. A one-layer MTP head and a 27-layer vision tower.
- About 19 GB of weights per card. 0.83-0.94 GiB stays free per card after boot, dropping to 0.24-0.38 GiB during long prefills.

## Measured so far

On this box unless marked.

| What | Result | Source |
|---|---|---|
| Count to 100, single stream | 188.8 tok/s; 1,804 of 1,806 draft tokens accepted | R3 (sandbox) |
| Cold prefill at 70K-84K, caching on | 1,359-1,377 tok/s | lane README |
| Cold 64K / 128K | 45.6 s at 1,399 tok/s / 152.6 s at 837 tok/s, 2 preemptions | `current_benchmark.md` (host) |
| Warm 64K / 128K | 0.67 s / 0.85 s | `current_benchmark.md` (host) |
| Exact repeat, 69,834 tokens | TTFT 51.4 s cold, 0.81 s warm | R3 |
| 3 turns, about 4K new tokens each, at about 70K depth | TTFT 0.8 / 4.0 / 5.2 s | R3 |
| 155K cold, caching on / off | 843 / 1,395 tok/s. With caching on it ran 155 steps covering 256,446 tokens, with 2 preemptions | R3, R4, R10 |
| Deep prefill step, profiled | 0.95 s per 1,664 tokens, about half of it NCCL all-reduce (RING_LL). Caching off: 1.15 s per 2,048 tokens, the same cost per token | R8, R9 |
| Original box (NVLink pairs, 31 GB RAM), reference only | 155K prefill 2,249 tok/s; count to 100 193.3 tok/s; no draft 55.8 tok/s (18 ms per step); prose 109.5, chat 108.4, code 145.8 tok/s | READMEs |

## Anomalies

- **`current_benchmark.md` is partly contaminated.** It ran at the same time as `run-host-docker.sh`'s test phase (`host-run-prefix-20261017-173506.log`): both started at 17:55:37-38. The 1K row's 48,256 "warm hits" are exactly that script's four prefix-test hits (11,648 + 10,816 + 10,816 + 14,976). The decode block and the 1K and 16K rows, including the 7.4 and 2.9 tok/s decode@ctx, are two clients sharing the server. The same overlap made that script's PREFIX_CACHE check fail: the denominator of turn 3's hit rate includes the bench's 16K request. Only the 64K and 128K rows ran alone.
- As a result, there is no clean single-stream decode number under host Docker yet.
- With caching on, cold prompts above about 100K get preempted and recomputed (see the 155K row); below that there's no penalty. `--mamba-block-size 4992` changed nothing (R11).
- Both bench scripts read cache hits from server-wide `/metrics` counters, so any concurrent request pollutes them.

## Inherited from another box

The lane was built and tuned on a different machine: two NVLink pairs (0-1, 2-3) and 31 GB of RAM. This box has no NVLink and 128 GB. Decisions made for that machine include reading the n-gram table from disk every step (31 GB of RAM couldn't hold it) and the `NCCL_MODE=nvl` default. The headline numbers in the top-level README come from that machine too.

## Already tried

Don't repeat these without new evidence. All on the original box unless marked R.
- FP8 e4m3 and NVFP4 KV: no sm_86 kernels. e5m2 works through the overlay.
- torch.compile modes: only `GRAPHS=nocompile` boots.
- Mamba cache mode `all` raises at this commit. `--mamba-block-size` has no effect (R11).
- An auto-sized KV pool can boot and then OOM on a long prefill (seen on the EXL3 lane). Pin `KV_BYTES` and validate with a 155K+ prefill.
- `CHUNK=2496` (R7): 757 tok/s, GPU2 down to 0.03 GiB free, PLE staging sized for 2,048. `CHUNK=1664` with caching off (R6): 1,350 tok/s.
- External draft models (llama.cpp lane) were a loss on this MoE. As of September, no EAGLE or DFlash drafter exists for this model.
- EXL3: parked. Under TP the plugin had to dequantize the dense layers to BF16.
- `NCCL_P2P_LEVEL=NVL` vs `NCCL_P2P_DISABLE=1` on the NVLink box: noise.

## How to work

1. **Baseline.** Build the image, boot the prefix lane in the sandbox, and run the scoreboard. Check that `vllm:num_requests_running` and `vllm:num_requests_waiting` are 0 before and after every run. This is the "before" column.
2. **Account for the time.** Take a cold prefill step (shallow and deep), a decode step, and an incremental turn (from the HTTP request to the first streamed token). Break each one down by component, and set each component next to its floor on this hardware: FLOPs, memory bandwidth, PCIe bandwidth, per-collective latency, host overhead. The biggest gaps between measured and floor are where the multiples are.
3. **Ideas.** Before building anything big, write a portfolio: many ideas across the whole stack, including ones that sound unreasonable. For each one, give the mechanism, a back-of-envelope ceiling, the cost, the risk, and the cheapest experiment that could kill it. Rank by expected gain x odds / effort. Show me the portfolio, then start on the top items without waiting, except those that need my approval.
4. **Experiments.** One change per boot where you can, A/B against a same-day baseline. Kill ideas fast. Log every boot, failures included, as ledger rows R13 onward in the lane README's style.
5. **Recipe.** Ship the winning combination as a new lane (a launcher plus whatever it needs), with a before/after scoreboard, the quality-gate results, and the host script.

Keep a running notes file under `vllm-w4a16/results/` with the scoreboard, the ledger and open questions, so the work survives context compaction.

## Read first

- `README.md`, then `vllm-w4a16/README.md` (boot ledgers, R1-R12, caveats)
- `current_benchmark.md`, `bench-host.py`, `run-host-docker.sh`, `host-run-prefix-20261017-173506.log`
- `vllm-w4a16/launch/`, `patch/`, `tools/`, `research/`, `results/`
- `~/models/qwen38fn-w4a16-fp8ple/config.json`, and `runtime/`: the checkpoint author's own runtime snapshot (Dockerfile, overlay, launch script, lockfile) for a different vLLM build, validated on 2x RTX 3090 with 128 GB RAM
- The vLLM source inside the image, at `/usr/local/lib/python3.12/dist-packages/vllm`
