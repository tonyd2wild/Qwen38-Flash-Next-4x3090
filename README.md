# Qwen3.8-Flash-Next on 4× RTX 3090 (llama.cpp)

Serve **Qwen3.8-Flash-Next** (125B-A3B MoE + 51B n-gram embedding + 4B MTP head, arch `qwen4exp`) on a single 4× RTX 3090 box — **two concurrent users at the full native 262,144-token context each**, with n-gram speculative decoding for a free +45% on agentic work.

Deployed and battle-tested 2026-08-26, day-0 of the model's release.

## Results

| Metric | Value |
|---|---|
| Quant | unsloth **UD-IQ4_XS** GGUF (93.7 GB on disk, ~60 GB in VRAM) |
| Context | **2 dedicated slots × 262,144 tokens** (`-c 524288 --parallel 2`) |
| Decode, freeform | 40–49 tok/s solo · ~30 tok/s each with 2 concurrent users |
| Decode, copy/edit/tool tasks | **62 tok/s median with `--spec-type ngram-mod`** (43 without, +45%) |
| Prefill | ~311 tok/s |
| TTFT (warm) | ~150–170 ms |
| Load time | ~85 s |
| VRAM | 20.5–22.6 GB used per card at full config |
| KV pool | f16, ~12 KB/token/card → ~6.2 GB/card (~25 GB total) for the 524K pool |

For reference: this beats a 2× DGX Spark TP2 deployment of the same model in NVFP4 with MTP4 speculation (~33 tok/s) — Ampere bandwidth is still formidable.

## Why this is non-obvious

1. **No released llama.cpp knows `qwen4exp`.** Every stock build — including the official `ghcr.io/ggml-org/llama.cpp:server-cuda` image — fails with `unknown model architecture: 'qwen4exp'`. You must build [unsloth's PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) (branch `qwen4exp/qwen3.8-flash-next` on `unslothai/llama.cpp`).
2. **The model is bigger than it looks, and smaller than it loads.** The GGUF is ~94 GB but only ~60 GB lands in VRAM: the 51B n-gram embedding table is lookup-only (no matmuls), so llama.cpp leaves it memory-mapped on NVMe and the OS page cache serves the hot rows. This is why a "180B-param" model fits 96 GB of VRAM with room for a half-million-token KV pool. Corollary: the first requests after a restart run ~25% slower until the page cache warms.
3. **KV is nearly free on this architecture.** Most layers are SSM/linear-attention with fixed-size state; only a few real attention layers pay per-token KV rent. Measured: doubling context 65K→131K cost just ~0.78 GB/card. That's how 2×262K slots fit.

## Build

```bash
./build-llama-qwen4exp.sh
```

Read the trap comments in the script. The big one: the build container **must** run with `--gpus all` or the final link fails on `undefined reference to cuMemCreate` — the driver library only exists inside the container when the NVIDIA runtime injects it.

## Launch

```bash
./launch-qwen38fn-gguf.sh
```

Runs the server in the same CUDA container the binary was built against (`--network host`, port 8090, OpenAI-compatible). Key flags:

- `-ngl 999 --tensor-split 1,1,1,1` — all layers on GPU, spread across 4 cards (layer split; `--split-mode row` is **not supported** for this arch: `device CUDA0 does not support split buffers`)
- `-c 524288 --parallel 2` — two dedicated full-context slots
- `--jinja` — the model's embedded chat template
- `--spec-type ngram-mod` — see below

## Speculative decoding: what works and what doesn't

- **`--spec-type ngram-mod` works and is free.** Drafts from contextual repetition, no draft model needed. On copy/edit/tool-call output (i.e. what agents generate all day): 43 → 62 tok/s median. Freeform prose: neutral, no penalty. Credit to [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark) for proving this flag on this model. Their single-Spark setup required `--parallel 1` with spec decode; on this 4×3090 build **concurrency + ngram-mod is stable** (verified under sustained 2-user load).
- **`--spec-type draft-mtp` does not work *yet* — but not for the reason you'd think.** llama.cpp has had an MTP framework since May 2026, and it correctly initializes for this model, then fails with `model doesn't contain MTP layers`: unsloth's GGUFs currently ship without the 4B MTP head. The day an MTP-bearing GGUF appears, it's a flag flip and (based on Qwen3.6 precedent) roughly another 1.7×.
- **External draft models are a dead end** for this MoE (expert-scaling dynamics — see 0xBakeer's notes).
- **Quantized KV cache aborts** on this arch. Keep KV f16.

## Ops notes

- **Thinking model**: hidden reasoning consumes `max_tokens`. Small budgets return empty `content` with all tokens spent in `reasoning_content`. Budget generously or disable thinking via chat-template kwargs.
- **Downloading the GGUF**: Hugging Face repeatedly stalled/throttled both `hf download` and `hf_transfer` on day-0 (three separate wedges, including one at 98%). What actually worked: `aria2c -x8 -s8 -c` against the resolve URL — 8 connections, true resume, ~30 MB/s sustained. Verify byte counts against the HF API tree listing before trusting any shard.
- `-fa on` (flash-attn) is neutral here — auto already does the right thing, and the arch is SSM-heavy anyway.
- System RAM matters less than you'd think: this box has 31 GB and the process RSS is ~3 GB. The n-gram table pages through the OS cache.

## Files

| File | What |
|---|---|
| `build-llama-qwen4exp.sh` | Container build of the PR branch, with the traps documented |
| `launch-qwen38fn-gguf.sh` | The exact production launcher |
| `bench_decode.py` | 3-run decode/TTFT bench (freeform) |
| `bench_agentic.py` | Copy/edit-style bench (shows the ngram-mod gain) |

## Credits

- [unsloth](https://huggingface.co/unsloth) / danielhanchen — the `qwen4exp` llama.cpp PR and the dynamic GGUF quants
- [0xBakeer](https://github.com/0xBakeer/qwen38-flash-next-spark) — first public proof of `ngram-mod` speculation on this model, and the PLE-offload analysis
- The 2Wild fleet (Kai on the Mac, Knox on the 5080) — deploy, debugging, and the benching

## Hardware

4× NVIDIA RTX 3090 24 GB (NVLink pairs 0↔1 and 2↔3, PCIe between pairs), 31 GB system RAM, NVMe storage, Ubuntu + Docker with NVIDIA runtime. Driver 580.173.02.
