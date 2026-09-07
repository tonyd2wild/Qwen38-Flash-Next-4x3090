# Qwen3.8-Flash-Next on 4x RTX 3090: 193 tok/s, full 262K context, in vLLM

The 125B-A6B hybrid MoE with its 51B n-gram table and the MTP draft, served from one 4x RTX 3090 box (96 GB VRAM, 31 GB host RAM) by upstream vLLM. Measured 2026-09-06, every number from the box.

| | Default lane: vLLM ([`vllm-w4a16/`](vllm-w4a16/)) | Lane 2: llama.cpp (below) |
|---|---|---|
| Count to 100, single stream, temperature 0 | **193.3 tok/s median** (no draft: 55.8) | 96 to 102 tok/s on copy/edit tasks, 40 to 49 freeform |
| Real prompts, single stream (prose / chat / code) | **109.5 / 108.4 / 145.8 tok/s** ([log](vllm-w4a16/results/realprompts_3090_262k_s6_2026-09-06.txt)) | 40 to 49 freeform (above) |
| Context | **262,144 native**, 6 seats, KV pool 362,077 tokens (fp8 e5m2) | 2 x 262,144 slots (f16 KV) |
| Quant | Intel AutoRound W4A16 experts (Marlin), BF16 attention, FP8 n-gram table | unsloth UD-IQ4_XS GGUF |
| Speculation | albucino's INT4 MTP draft, 3 tokens, expert parallel | unsloth MTP head |
| Where the 47.7 GB table lives | on the NVMe, 16 rows per token read per step by our patch, inside CUDA graphs | left out of VRAM by llama.cpp's lookup-only path |
| Serving | OpenAI-compatible vLLM on :8090, tools and reasoning parsers | llama.cpp server on :8090 |

## Default lane: vLLM in four lines

```bash
hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE --local-dir ~/models/qwen38fn-w4a16-fp8ple   # 129 GB, the draft is inside
cp -R vllm-w4a16/patch ~/patches/qwen4exp-ple-mmap
LM_ONLY=1 NCCL_MODE=nvl PLE_MODE=staged GRAPHS=nocompile MTP=3 TP=4 GMU=0.97 SEQS=6 CHUNK=2048 MAXLEN=262144 \
KV_DTYPE=fp8_e5m2 CAPTURE_SIZES=4,8,12,16,20,24 EXTRA="--quantization gptq_marlin --enable-expert-parallel" bash vllm-w4a16/launch/qwen38fn-w4a16-3090-tp4.sh
```

Endpoint `http://<box>:8090/v1`, model `qwen3.8-flash-next`. Load is about 2 minutes. The lane README has the boot-by-boot ladder (55.8 with no draft, 103.5 with the draft, 193.5 once expert parallel put the experts on Marlin, the memory passes that took the pool from 46K to 300K tokens), the Ampere findings (FP8 e4m3 KV does not compile on 3090s; e5m2 does through our overlay variant and passed the needle test at 7K, 28K and 53K), the caveats, and the credits.

Pick the lane by need: vLLM for single-stream speed, 6 seats, tools and the OpenAI API; llama.cpp for two dedicated full-context slots on a GGUF with no patching.

## Lane 2: llama.cpp (the original lane, kept intact)

Serve **Qwen3.8-Flash-Next** (125B-A3B MoE + 51B n-gram embedding + 4B MTP head, arch `qwen4exp`) on a single 4× RTX 3090 box — at **up to 102 tok/s with MTP speculative decoding**, or **two concurrent users at the full native 262,144-token context each**.

Deployed day-0 of the model's release (2026-08-26) and updated 2026-09-02 when working MTP heads landed. Every number here was measured on the box, not copied from a model card.

### Results

| Metric | Value |
|---|---|
| Quant | unsloth **UD-IQ4_XS** GGUF (93.7 GB on disk, ~60 GB in VRAM) |
| Context | **2 dedicated slots × 262,144 tokens** (`-c 524288 --parallel 2`) |
| Decode, freeform | 40–49 tok/s solo · ~30 tok/s each with 2 concurrent users |
| Decode, copy/edit/tool tasks | **96-102 tok/s with unsloth MTP head** (58.8 without, up to 1.74x); 62 tok/s with `ngram-mod` on IQ4_XS |
| Prefill | ~311 tok/s |
| TTFT (warm) | ~150–170 ms |
| Load time | ~85 s |
| VRAM | 20.5–22.6 GB used per card at full config |
| KV pool | f16, ~12 KB/token/card → ~6.2 GB/card (~25 GB total) for the 524K pool |

For reference: this beats a 2× DGX Spark TP2 deployment of the same model in NVFP4 with MTP4 speculation (~33 tok/s) — Ampere bandwidth is still formidable.

### Why this is non-obvious

1. **No released llama.cpp knows `qwen4exp`.** Every stock build — including the official `ghcr.io/ggml-org/llama.cpp:server-cuda` image — fails with `unknown model architecture: 'qwen4exp'`. You must build [unsloth's PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) (branch `qwen4exp/qwen3.8-flash-next` on `unslothai/llama.cpp`).
2. **The model is bigger than it looks, and smaller than it loads.** The GGUF is ~94 GB but only ~60 GB lands in VRAM: the 51B n-gram embedding table is lookup-only (no matmuls), so llama.cpp leaves it memory-mapped on NVMe and the OS page cache serves the hot rows. This is why a "180B-param" model fits 96 GB of VRAM with room for a half-million-token KV pool. Corollary: the first requests after a restart run ~25% slower until the page cache warms.
3. **KV is nearly free on this architecture.** Most layers are SSM/linear-attention with fixed-size state; only a few real attention layers pay per-token KV rent. Measured: doubling context 65K→131K cost just ~0.78 GB/card. That's how 2×262K slots fit.

### Build

```bash
./build-llama-qwen4exp.sh
```

Read the trap comments in the script. The big one: the build container **must** run with `--gpus all` or the final link fails on `undefined reference to cuMemCreate` — the driver library only exists inside the container when the NVIDIA runtime injects it.

### Launch

```bash
./launch-qwen38fn-gguf.sh
```

Runs the server in the same CUDA container the binary was built against (`--network host`, port 8090, OpenAI-compatible). Key flags:

- `-ngl 999 --tensor-split 1,1,1,1` — all layers on GPU, spread across 4 cards (layer split; `--split-mode row` is **not supported** for this arch: `device CUDA0 does not support split buffers`)
- `-c 524288 --parallel 2` — two dedicated full-context slots
- `--jinja` — the model's embedded chat template
- `--spec-type ngram-mod` — see below

### Speculative decoding: MTP is the winner (updated 2026-09-02)

**Use unsloth's MTP head.** On 2026-09-02 unsloth published real MTP draft heads in the
[`MTP/` folder](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/MTP) of the GGUF repo.
They are a different class of artifact from the third-party heads published in the days after release,
and they turn speculative decoding from a 2x *loss* into a **1.4-1.7x win** on this box.

Measured here, same binary, same model, same prompts, temperature 0, 3 runs each, `--parallel 1`:

| task | MTP off | MTP on | gain |
|---|---:|---:|---|
| count to 100 | 58.8 tok/s | **102.3 tok/s** | **1.74x** |
| code generation | 58.8 | **96.1** | 1.63x |
| freeform prose | 58.7 | **81.6** | 1.39x |
| JSON edit | 58.4 | **95.5** | 1.64x |

Draft acceptance ran **0.66 on prose to 0.92 on structured output**. For reference, unsloth measured
1.67x on a B200; four 3090s beat that ratio, because a slower target model makes each accepted draft
token worth relatively more.

Getting the pieces (no build required — unsloth ship prebuilt binaries):

```bash
# 1. the MTP head (2.6 GB)
hf download unsloth/Qwen3.8-Flash-Next-GGUF --local-dir models/qwen38fn-gguf     --include "*mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf*"

# 2. a binary that knows qwen4exp MTP (stock llama.cpp does not)
#    https://github.com/unslothai/llama.cpp/releases  tag b10715-mix-86bd2d3 or newer
curl -LO https://github.com/unslothai/llama.cpp/releases/download/b10715-mix-86bd2d3/app-b10715-mix-86bd2d3-linux-x64-cuda12-portable.tar.gz
tar xzf app-b10715-mix-86bd2d3-linux-x64-cuda12-portable.tar.gz
```

Then run — and read trap 2 below before you trust the speed you get:

```bash
./launch-mtp-q4kxl.sh          # MTP on
SPEC=0 ./launch-mtp-q4kxl.sh   # same everything, speculation off (for honest A/B)
python3 bench_mtp.py count100 3
```

### Four traps, each of which cost a boot cycle

1. **`--no-repack` is mandatory for K-quants on a low-RAM box.** `UD-Q4_K_XL` tries to repack weights
   into a **41.3 GiB** CPU buffer; with 31 GB of system RAM it dies with
   `failed to allocate CPU_REPACK buffer of size 44354764800`. IQ-quants (`UD-IQ4_XS`) never trigger
   this, which is why the older lane worked without the flag.
2. **unsloth's `cuda12-portable` tarball ships no CUDA runtime.** `libggml-cuda.so` needs
   `libcudart.so.12` and `libcublas.so.12`, neither of which is in the archive. When they are missing the
   backend fails to `dlopen` **silently** — the server starts, answers requests, and runs entirely on CPU.
   There is no error in the log. Point `LD_LIBRARY_PATH` at a CUDA runtime and
   **always confirm with `nvidia-smi` that VRAM is actually in use after boot.**
3. **A `shared-` head prints a scary error and then works.** `borrow_shared_tensor: this model is a draft
   head without its own 'token_embd.weight'` plus `failed to fit params` are expected: the auto-fit loads
   the head alone to measure it, before the model it borrows from exists. Pass `-c` and `-ngl` yourself.
4. **Pass `-md` explicitly.** The heads live in an `MTP/` subfolder that sidecar auto-discovery does not
   search, so `--spec-type draft-mtp` alone silently runs with no draft at all.

### Context cost

`UD-Q4_K_XL` + MTP fits **131,072 tokens** on 96 GB of VRAM (88.8 GiB resident, and the allocator retries
its way in — one card ends with 61 MiB free). The context drop versus the 2x262K ngram-mod lane is
**mostly the quant, not MTP**: Q4_K_XL is ~10 GB larger than IQ4_XS, while the shared MTP head is only
~2.6 GB. Pairing the head with `UD-IQ4_XS` should buy most of that context back.

### What we tested before this, and why it failed

Third-party heads published in the first days after release (`quimmedes`, `agentionai`) drafted at only
**~0.35 acceptance** and netted **~20 tok/s against a 43 tok/s baseline — a 2x loss**. That held across
head quant (Q4_K_M and Q8_0), context size, `--parallel`, `-kvu`, and batch sizes. A separate head
(`dzannotti`) with a fuller MTP block **segfaults the CUDA backend at load** (its author tested only
ROCm and Vulkan). The lesson worth keeping: **a speculative-decoding claim means nothing without the
hardware and sampler it was measured on**, and acceptance rate is the number to check first — the log
line `draft acceptance = 0.66139 (325 accepted / 491 generated)` tells you within one request whether a
head is worth keeping.

### ngram-mod: still useful, no extra file

`--spec-type ngram-mod` drafts from repetition in the context with no draft model at all: 43 -> 62 tok/s
on copy/edit/tool output, neutral on prose. It was the best option on this box before the unsloth heads
existed, and it remains the right choice when you cannot spare the ~2.6 GB.

### Other speculation notes

- **External draft models** are a dead end for this MoE (expert-scaling dynamics).
- **Quantized KV cache** (`-ctk/-ctv q8_0`) aborts on this arch in the mainline-derived builds. Keep KV at f16.
- **MTP is for low concurrency.** unsloth measure a net loss (0.81-0.87x) at concurrency 8; a busy model has
  no idle capacity for a draft to exploit. All numbers above are `--parallel 1`.

### Ops notes

- **Thinking model**: hidden reasoning consumes `max_tokens`. Small budgets return empty `content` with all tokens spent in `reasoning_content`. Budget generously or disable thinking via chat-template kwargs.
- **Downloading the GGUF**: Hugging Face repeatedly stalled/throttled both `hf download` and `hf_transfer` on day-0 (three separate wedges, including one at 98%). What actually worked: `aria2c -x8 -s8 -c` against the resolve URL — 8 connections, true resume, ~30 MB/s sustained. Verify byte counts against the HF API tree listing before trusting any shard.
- `-fa on` (flash-attn) is neutral here — auto already does the right thing, and the arch is SSM-heavy anyway.
- System RAM matters less than you'd think: this box has 31 GB and the process RSS is ~3 GB. The n-gram table pages through the OS cache.

### Files


| File | What |
|---|---|
| `build-llama-qwen4exp.sh` | Container build of the PR branch, with the traps documented |
| `launch-qwen38fn-gguf.sh` | The exact production launcher |
| `bench_decode.py` | 3-run decode/TTFT bench (freeform) |
| `bench_agentic.py` | Copy/edit-style bench (shows the ngram-mod gain) |
| `launch-mtp-q4kxl.sh` | MTP launcher (`SPEC=0` disables speculation for A/B) |
| `bench_mtp.py` | Four-task bench: count100 / code / prose / jsonedit |

### Credits

- [unsloth](https://huggingface.co/unsloth) / danielhanchen — the `qwen4exp` llama.cpp PR and the dynamic GGUF quants
- [0xBakeer](https://github.com/0xBakeer/qwen38-flash-next-spark) — first public proof of `ngram-mod` speculation on this model, and the PLE-offload analysis
- The 2Wild fleet (Kai on the Mac, Knox on the 5080) — deploy, debugging, and the benching

### Hardware

4× NVIDIA RTX 3090 24 GB (NVLink pairs 0↔1 and 2↔3, PCIe between pairs), 31 GB system RAM, NVMe storage, Ubuntu + Docker with NVIDIA runtime. Driver 580.173.02.
