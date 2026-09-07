# Lane 2: vLLM, W4A16 experts + FP8 n-gram table on the SSD, MTP3, expert parallel

The same 4x RTX 3090 box, served by upstream vLLM instead of llama.cpp. Built the evening of 2026-09-06, every number measured on the box.

**Count to 100, single stream, temperature 0: 193.3 tok/s median**, 391 tokens in about 2.2 s, TTFT 150 to 200 ms. Without the draft the same box does 55.8. Context 262,144 (the model's native maximum) with 6 seats, KV pool 362,077 tokens in fp8 e5m2, 18.8 GB per card.

Endpoint on the box: `http://<3090>:8090/v1`, model id `qwen3.8-flash-next`, OpenAI compatible, thinking off by default, tool parser `qwen3_xml`.

## What runs

| Piece | What | Where it comes from |
|---|---|---|
| Routed experts | INT4 group-128, AutoRound, GPTQ layout, served by the Marlin W4A16 MoE kernel | Intel `Qwen3.8-Flash-Next-W4A16-AutoRound`, as assembled by albucino |
| Attention, GDN, hyper-connections, router, embeddings, lm_head | BF16, untouched | same |
| 51B n-gram (PLE) table | FP8 E4M3 with one global scale, **left on the NVMe**: our patch reads the 16 rows each token needs per step, inside CUDA graphs | RadixArk FP8 table, as packed by albucino; the disk reader is ours (`patch/`) |
| MTP draft | 1-layer head, experts INT4 group-32, loaded as a **separate draft directory** by stock vLLM (`runtime/mtp-int4-g32` in the same download) | albucino |
| KV cache | FP8 e5m2 through our variant of the attention overlay (e4m3 does not compile on Ampere; BF16 also works with a 1.6x smaller pool) | |
| Engine | `vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c` (multi-arch, amd64 layer) plus the overlays in `patch/upstream-overlays/` | |

Checkpoint: `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` (129 GB). Download it with `hf download` into `/home/<you>/models/qwen38fn-w4a16-fp8ple`; the draft ships inside it.

## Why it fits, and why it is fast

The box has 96 GB of VRAM and 31 GB of host RAM. The 47.7 GB table cannot live in either, so it stays on the SSD and the model state fetches 16 rows per token into a fixed GPU buffer before each forward (the "staged gather" in `patch/ple_layer.py` and `patch/model_state.py`, the same code that runs one DGX Spark in the sister repo). The non-table weights are 68 GB, split four ways.

The speed came from three things, measured one boot at a time (ledger below):

1. **The draft.** No draft: 55.8 tok/s (one token per step, 18 ms per step over PCIe). MTP3 with albucino's INT4 draft: 103.5.
2. **Expert parallel.** With `--enable-expert-parallel` each card owns 128 whole experts, so the expert width per card is 640 instead of 160, which is what the Marlin kernel needs (`intermediate_per_partition % 128 == 0`). Without it vLLM quietly runs the experts on an untuned Triton WNA16 kernel. Marlin took the draft config from 103.5 to 193.5 tok/s and freed 1.3 GB per card (no qzeros, no tile padding).
3. **Vision tower off** (`--language-model-only`): 0.2 GB per card and no image profiling pass, which together with gmu 0.97 took the KV pool from 96K to 186K tokens at 64K context.

## The recipe

```bash
# patch dir on the box: ~/patches/qwen4exp-ple-mmap (contents of patch/)
LM_ONLY=1 NCCL_MODE=nvl PLE_MODE=staged GRAPHS=nocompile MTP=3 TP=4 GMU=0.95 SEQS=6 CHUNK=2048 \
MAXLEN=262144 KV_DTYPE=fp8_e5m2 CAPTURE_SIZES=4,8,12,16,20,24 \
EXTRA="--quantization gptq_marlin --enable-expert-parallel" \
bash launch/qwen38fn-w4a16-3090-tp4.sh
```

Knobs in the launcher: `PLE_MODE` (staged | mmap | none), `GRAPHS` (nocompile = CUDA graphs for decode with torch.compile off, the only mode that boots here), `MTP` (0 or N; capture sizes must be multiples of N+1), `DRAFT_DIR`, `TP`, `SEQS`, `CHUNK`, `GMU`, `MAXLEN`, `KV_DTYPE`, `KV_BYTES`, `LM_ONLY`, `NCCL_MODE` (nvl | nop2p), `EXTRA`. `--quantization gptq_marlin` is passed but vLLM resolves the AutoRound config to its INC path anyway; the flag is harmless.

## Ampere findings (each cost one boot)

| Boot | Change | Result |
|---|---|---|
| 1 | first try, patch as shipped for the Sparks | OOM at 23.8 GiB per card: the patch only took the disk path when it saw NVIDIA's ModelOpt quant config. Fixed: the disk path is taken whenever it is requested (`patch/ple_layer.py`). |
| 2 | fix above, FP8 e4m3 KV | Triton: `type fp8e4nv not supported in this architecture`. The sparse-attention kernel's FP8 path needs compute capability 8.9. |
| 3 | BF16 KV, 131K | Kernels compile and run. Refused at KV sizing: 1.04 GiB per card available, 131K needs 1.65. |
| 4 | 64K, chunk 2048 | **First working boot.** 114K pool, count-to-100 55.8 tok/s, no draft. |
| 5 | NCCL_P2P_LEVEL=NVL | 56.4, noise. The two NVLink pairs (0-1 at 28 GB/s, 2-3 at 56 GB/s) are joined only by PCIe through the host bridge, so the PCIe hop sets the pace of a 4-way all-reduce. |
| 6 | fp8_e5m2 KV | Refused at init by the attention layer's allowlist (BF16, e4m3, nvfp4). |
| 8 | MTP3 draft from albucino's folder, 1 seat, 32K | Loads on the nightly with no overlay changes. 103.5 tok/s, 100% acceptance on counting, 4 tokens per step. Pool 45,762. |
| 9 | plus expert parallel | `Using 'MARLIN' WNA16 MoE backend`. 193.5 tok/s. 19.0 GB per card, pool 96,044. |
| 10 | plus language-model-only, gmu 0.97, 64K, 2 seats | 194.2 tok/s. 18.8 GB per card, pool 186,016. |
| 11 | fp8_e5m2 KV through a variant of the attention overlay (`patch/upstream-overlays/*_e5m2.py`) | Runs on Ampere. 191.8 tok/s, pool 299,431 at 64K. Needle in a haystack answered correctly at 7K, 28K and 53K (prefill 1,265 / 1,699 / 2,347 tok/s). |
| 12 | max context 262,144, 6 seats | 193.3 tok/s, pool 362,077. **The default.** |
| 13 | gmu 0.95 instead of 0.97 (after a CUDA OOM under fleet load at 0.97: 310 MiB free, 512 MiB asked) | 193.3 tok/s. Pool 299,800 at 262K (1.14x), about 800 MiB headroom per card. **Serving default.** |
| 14 | vision tower on (`LM_ONLY=0`), 131K | 196.1 tok/s (prose 110.3). Pool 160,199 at 131K (1.22x). Image described correctly. **Tony's daily-driver boot.** |
| 15 | vision + `--mm-processor-kwargs '{"max_pixels":1048576}'` | 194.6 tok/s. **Pool 160,199 to 243,608 at 131K (1.86x)**: the profiler was running a 16.7 MP image; the cap drops the encoder budget from 16,384 to 2,048 tokens. Image test still correct. |
| 16 | `--kv-cache-memory-bytes` 2.01 GiB explicit (vLLM had used 1.65, offered 2.31) | Pool **fell** to 238,312: the explicit budget still has the CUDA graph reserve taken out of it. Dropped. |
| 17 | `--mamba-ssm-cache-dtype bfloat16` (checkpoint pins fp32) | 192.6 tok/s, prose 107.3. **Pool 265,888 at 131K (2.03x)**, +9%; attention block 1600 to 832. Needle correct at 9.8K / 39K / 74K prompt tokens; image correct. |

Not possible on these cards, and why: FP8 e4m3 and NVFP4 KV (Triton and SM100 kernels), NVFP4 or FP8 expert checkpoints (no Ampere kernels), a DFlash2 or EAGLE drafter for this model (none exists), applying albucino's own vLLM overlay (27 whole-file replacements against a different vLLM tree; reuse his checkpoint and flag shapes, not his files).

## Caveats

Counting to 100 is the easiest possible text for a draft (100% acceptance). Prose, code and long context will accept fewer draft tokens; the same 40-prompt harness the Spark lanes use lives in the sister repo and has not been run on this box yet. e5m2 KV keeps 2 mantissa bits; the needle test passed at every rung we ran, and BF16 KV is one knob away (`KV_DTYPE=auto`, pool 186K at 64K). The box holds one model at a time: this lane cannot coexist with the 27B and 35B lanes.

## More pool: the 2026-09-06 evening hunt

Three research passes (saved under `research/`: upstream knobs, an on-box memory audit of boot 14, a survey of every Flash-Next checkpoint on the Hub) fed the boots from 15 on. What held: the vision cost was the profiling image, not the tower (boot 15); vLLM prints its own KV budget suggestion and `KV_BYTES` can take part of the 5% gmu never requests (boot 16); the DeltaNet state is pinned to fp32 by the checkpoint and `--mamba-ssm-cache-dtype bfloat16` halves it (untested here). What did not: sharing the draft's embeddings and lm_head (vLLM already does), prefix caching, encoder data-parallel, block-size flags, swap, CPU offload. The Hub has one materially smaller checkpoint (ranxianglei's 296-expert prune, roughly 12.6 GiB per card) at a quality cost we have not measured.

## Vision

**The default recipe ships without vision.** `LM_ONLY=1` passes `--language-model-only`, which drops the vision tower at load. Qwen3.8-Flash-Next is a vision-language model; the tower in this checkpoint is 27 layers, 1152 wide, **0.84 GiB in BF16**, and vLLM loads it whole on every tensor-parallel rank, so vision costs 0.84 GiB of KV budget per card plus vLLM's image-encoding reservation. That is the whole reason the default drops it: with the tower resident, the 262,144-token pool does not fit on 24 GB cards.

Vision on = the same launcher with `LM_ONLY=0 MAXLEN=131072` (boot 14, ledger row 39): **pool 160,199 tokens at 131K, 1.22x**, encoder cache 16,384 tokens, load 102 s. Adding `--mm-processor-kwargs '{"max_pixels":1048576}'` (boot 15) lifts that to **243,608** because the profiler stops running a 16.7 MP image, and `--mamba-ssm-cache-dtype bfloat16` (boot 17) to **265,888**. A 547x900 screenshot cost 513 prompt tokens and was described correctly in 3.4 s; count-to-100 stayed at 196.1 tok/s and prose at 110.3 (two runs each, box quiet). vLLM warns that the MTP draft does not take image embeddings, so drafting is text-only on image turns and the target verifies as usual. Text-only at the same settings (boot 13, row 38) is 299,800 tokens at 262K, so the tower and its encoder cache cost about 140K tokens of pool. Image requests go through the normal chat completions `image_url` content parts. Clients pointed at the text-only boot must not send images (they return HTTP 400).

## Real prompts, not just the count

Same 262K/6-seat config, streaming, temperature 0, max_tokens 450, two runs each (`results/realprompts_3090_262k_s6_2026-09-06.txt`):

| Prompt | tok/s (median of 2) | TTFT |
|---|---|---|
| prose, 400-word explainer | 109.5 | 0.15 to 0.21 s |
| chat, 300-word summary | 108.4 | 0.17 to 0.21 s |
| code, Python function with docstring | 145.8 | 0.16 to 0.19 s |
| count to 100 | 193.3 (ladder median) | 0.16 s |

The count test is the MTP draft's best case (long runs of predictable tokens). Prose lands at a bit over half of it, code in between.

**Six parallel coding agents** (our coding-agent latency monitor, each agent doing tool round-trips, run by Tony on 2026-09-06 at 8:06 PM ET): 12,189 tokens in 39 s, **317.2 tok/s sustained aggregate** (313.3 wall-average), **569.8 tok/s peak burst**, per-agent 54.1 high / 51.6 low / 52.9 average, TTFT 0.47 s average. The engine log for that minute shows 545 to 561 tok/s generation throughput at 6 running requests.

## Ledger rows (from the sister repo's `kv_pool_ledger.md`)

    | 30 | **4x RTX 3090 (x86, Ampere) TP4, albucino W4A16 + FP8 PLE on SSD (our patch)** | 0.96 | 65,536 | bf16 (FP8 e4m3 KV does not compile on Ampere) | FULL_DECODE_ONLY, no compile, capture 1,2 | 0 (no draft in this checkpoint) | 114,135 | n/a | 1.74x @64K | gptq_marlin, seqs 2, chunk 2048, staged gather, NCCL_P2P_DISABLE=1; weights 19.76 GiB/card, load 161 s; boot 4 of the evening (boots 1-3: table alloc OOM → selector fix; FP8 KV Triton dtype; 131K did not fit) | count-to-100 single stream 55.8 tok/s median (53.1/55.8/56.1), 18 ms per step, TTFT 150 ms |
    | 31 | 4x3090 TP4, as row 30 + NCCL_P2P_LEVEL=NVL (NVLink inside the 0-1 and 2-3 pairs) | 0.96 | 65,536 | bf16 | FULL_DECODE_ONLY, capture 1,2 | 0 | 112,663 | n/a | 1.72x @64K | only change vs row 30: NCCL transport | count-to-100 56.4 (53.7/56.4/56.9) vs 55.8: noise; the PCIe hop between the pairs sets the pace |
    | 32 | 4x3090 TP4, as row 31 + --kv-cache-dtype fp8_e5m2 | 0.96 | 65,536 | fp8_e5m2 | n/a | 0 | n/a | n/a | n/a | REFUSED at model init: `Qwen4Exp QSA requires a BF16, FP8-E4M3 or NVFP4 main KV cache`; e4m3 fails in Triton on Ampere (row 30 notes) → BF16 is the only KV format for this model on 3090s | n/a |
    | 33 | **4x3090 TP4 + MTP3 (albucino INT4 g32 draft, separate dir)** | 0.96 | 32,768 | bf16 | FULL_DECODE_ONLY, capture 4 | 3 | 45,762 | n/a | 1.40x @32K | as row 31 + `--speculative-config {mtp, 3, model: runtime/mtp-int4-g32}`, seqs 1; weights+draft 20.28 GiB/card; draft experts on Marlin (padded), target still Triton WNA16 | **count-to-100 103.5 tok/s median (99.5/103.5/104.5), acceptance 100%, 4 tokens/step**; no-draft 55.8 |
    | 34 | **4x3090 TP4 + MTP3 + `--enable-expert-parallel`** (target experts on MARLIN WNA16) | 0.96 | 32,768 | bf16 | FULL_DECODE_ONLY, capture 4 | 3 | 96,044 | n/a | 2.93x @32K | as row 33 + EP; log `Using 'MARLIN' WNA16 MoE backend`; weights+draft 19.01 GiB/card (Marlin drops qzeros), load 116 s | **count-to-100 193.5 tok/s median (177.1/193.5/197.9)**; ladder no-draft 55.8 → MTP3 103.5 → MTP3+EP 193.5 |
    | 35 | **4x3090 TP4 + MTP3 + EP + `--language-model-only`, gmu 0.97, 64K, seqs 2** (serving config) | 0.97 | 65,536 | bf16 | FULL_DECODE_ONLY, capture 4,8 | 3 | 186,016 | n/a | 2.84x @64K | as row 34 + LM_ONLY=1 (vision tower dropped), gmu 0.97; weights+draft 18.83 GiB/card, load 113 s, available KV 2.96 GiB/card | count-to-100 194.2 tok/s median (161.8/194.2/196.4), same as row 34 |
    | 36 | 4x3090 TP4 + MTP3 + EP + LM only, **fp8_e5m2 KV via our overlay variant** | 0.97 | 65,536 | fp8_e5m2 | FULL_DECODE_ONLY, capture 4,8 | 3 | 299,431 | n/a | 4.57x @64K | as row 35 with KV_DTYPE=fp8_e5m2 (upstream-overlays/{qsa,ops_qsa}_e5m2.py); needle 7K/28K/53K all correct, prefill 1,265/1,699/2,347 | count-to-100 191.8 (174.8/191.8/191.9) = same as BF16; pool 1.6x |
    | 37 | **4x3090 DEFAULT: TP4 + MTP3 + EP + LM only, fp8_e5m2 KV, 262,144 ctx, 6 seats** | 0.97 | 262,144 | fp8_e5m2 | FULL_DECODE_ONLY, capture 4..24 | 3 | 362,077 | n/a | 1.38x @262K | as row 36 with MAXLEN 262144, SEQS 6, CAPTURE_SIZES 4,8,12,16,20,24; 18.83 GiB/card, load 107 s, available KV 2.79 GiB/card | count-to-100 193.3 tok/s median (182.7/193.3/197.8); headline of tonyd2wild/Qwen38-Flash-Next-4x3090 |
    | 38 | **4x3090 TEXT-ONLY SERVING: as row 37 with gmu 0.95** (after death 1: CUDA OOM under fleet load at gmu 0.97, 310 MiB free) | 0.95 | 262,144 | fp8_e5m2 | FULL_DECODE_ONLY, capture 4..24 | 3 | 299,800 | n/a | 1.14x @262K | only change = GMU 0.95; init engine 87.8 s; ~800 MiB headroom per card | count-to-100 193.3 (one reading, same as row 37) |
    | 39 | **4x3090 VISION ON (Tony's daily driver): as row 38 with LM_ONLY=0, 131,072 ctx** | 0.95 | 131,072 | fp8_e5m2 | FULL_DECODE_ONLY, capture 4..24 | 3 | 160,199 | n/a | 1.22x @131K | vision tower 0.84 GiB BF16 per rank + encoder cache 16,384 tokens; init engine 101.9 s; draft warns it ignores image embeddings (text-only draft inputs, target verifies) | image test correct (KPI dashboard screenshot, 513 tok in, 3.4 s); count-to-100 196.1 median quiet (196.2/196.1/191.6), prose 110.3; first runs 57.8/151.9 overlapped fleet restarts |
    | 40 | **4x3090 VISION + 1 MP image cap: as row 39 + `--mm-processor-kwargs {"max_pixels":1048576}`** | 0.95 | 131,072 | fp8_e5m2 | FULL_DECODE_ONLY, capture 4..24 | 3 | 243,608 | n/a | 1.86x @131K | encoder budget 16,384 → 2,048 tokens (profiled with 2 images of the new max); worker print: consumed 19.73 GiB, peak activation 0.59, graphs 0.26, KV 1.65 GiB, suggests 2.31 GiB to fully utilize; init engine 96 s | +83,409 tokens (+52%) from one flag; image test still correct (513 tok in, 3.4 s); count-to-100 194.6 (195.9/194.6/194.4) |
    | 41 | 4x3090 vision + 1 MP cap + **KV_BYTES=2161310925 (2.01 GiB explicit)** (as row 40, only change) | n/a (skipped) | 131,072 | fp8_e5m2 | FULL_DECODE_ONLY, capture 4..24 | 3 | 238,312 | n/a | 1.82x @131K | vLLM "reserved 2.01 GiB for KV Cache as specified... skipped memory profiling"; init engine 94.8 s | **LOST pool vs row 40 (243,608)**: the explicit budget still has the cudagraph estimate taken out of it, so 2.01 GiB manual < the profiler's own allocation at gmu 0.95; not kept, not tested further |
    | 42 | **4x3090 vision + 1 MP cap + `--mamba-ssm-cache-dtype bfloat16`** (as row 40, only change; checkpoint pins fp32) | 0.95 | 131,072 | fp8_e5m2 | FULL_DECODE_ONLY, capture 4..24 | 3 | 265,888 | n/a | 2.03x @131K | attention block 1600 → 832 tokens, mamba page padded 0.48%; worker print unchanged (19.73 consumed, 0.59 peak, 0.26 graphs, 1.65 KV); init engine 99.4 s | +22,280 tokens (+9.1%) vs row 40; needle CORRECT at 9,826 / 39,127 / 74,013 prompt tokens (prefill 1,398 / 2,124 / 2,196 tok/s); image test correct; count-to-100 192.6 (192.3/193.0/192.6), prose 107.3 (row 40: 194.6 / row 39: 110.3, within noise) |

## Files

- `launch/qwen38fn-w4a16-3090-tp4.sh`: the launcher (docker, TP4, knobs above).
- `patch/`: our disk-backed PLE table (`ple_layer.py`, `ple_mmap.py`, `model_state.py` for the staged gather), `mtp_draft_vocab.py` (reduced draft vocabulary, not used on this lane yet), `compilation.py` (piecewise mode, unused here), and `upstream-overlays/` (unmodified upstream vLLM PR files, see credits) plus the `_e5m2` variants of two of them for Ampere.
- `tools/count100_3090.sh`: Tony's speed test, three count-to-100 runs, median.
- `results/`: the count-to-100 logs for every working boot.

## Credits

- **albucino (Dominik Bucko)**: the checkpoint assembly (Intel AutoRound W4A16 target + RadixArk FP8 PLE table + the INT4 group-32 MTP draft) and the documentation of running this model on 3090s, including the BF16-KV and expert-parallel layout we confirmed here. His runtime is not used; his checkpoint and draft are.
- **Intel** for the AutoRound W4A16 quantization; **RadixArk** for the FP8 n-gram table.
- **andreasgru** (vLLM PR #54846, FP8 KV on the sparse-attention path, with Nanetnounou's RFC) and **peakcrosser7** (PR #55375, PLE conv state stride fix): the upstream overlays in `patch/upstream-overlays/`, unchanged except the `_e5m2` variants marked as ours.
- **Trosfy** (PR #54129) for the shape of the staged gather; **gau-nernst** (PR #55272) for the finding that torch.compile duplicates the table; **blazux** for the FLA shared-memory gate and the piecewise mode on GB10; **pangoleen** and **MiaAI-Lab** for measurements we compared against.
- The one-Spark, TP2 and TP4 lanes on NVIDIA's own NVFP4 checkpoint: https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark

## Update log

- 2026-09-06 7:50 PM ET: lane added after boot 10.
- 2026-09-06 8:00 PM ET: e5m2 KV validated (boot 11), full 262K context with 6 seats (boot 12) made the default; the vLLM lane became the repo's headline. Still untested: the TP2 x PP2 layout (tensor parallel inside each NVLink pair, pipeline across the PCIe hop) and the 40-prompt harness.
