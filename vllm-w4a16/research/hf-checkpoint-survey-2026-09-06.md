# Qwen3.8-Flash-Next checkpoints on the Hub, sized for 4x3090 in vLLM (2026-09-06)

Sub-agent pass over the HF search "Qwen 3.8 Flash" (168 Flash-Next repos, 4 queries plus per-repo file lists). Sizes are summed from safetensors headers unless marked; per-card estimates use (body - vision) / 4 + vision + ~1.0 GiB draft, calibrated to albucino's measured 18.8 GiB per card. Pool at ~130K tokens per GiB.

| Repo (last modified) | Format / parts quantized | PLE table | MTP | Total GiB | Runs on Ampere in vLLM |
|---|---|---|---|---|---|
| albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE (09-05) | Intel AutoRound INT4 g128 experts 58.45; BF16 GDN 3.89, QSA 1.74, embed 1.18, lm_head 1.18, HC/routers 0.71, shared 0.44, vision 0.84 | FP8 + scale, 47.7 | INT4 draft 3.87, separate dir | 120.1 | yes (our baseline) |
| Intel/Qwen3.8-Flash-Next-W4A16-AutoRound (08-31), -RTN | same body, BF16 PLE 95.4, BF16 MTP 4.86 | BF16 | BF16 | 168.8 | yes, bigger |
| ranxianglei/Qwen3.8-Flash-Next-W4A16-Modular (09-04) | Intel experts pruned 512 to 296 per layer, 33.79; side BF16 as Intel; vision BF16 | FP8 47.7, no scale key | none | 91.5 | likely (same AutoRound layout); SGLang-only so far |
| Saren/Qwen3.8-Flash-Next-AutoRound-hybrid (08-29) | Intel experts; GDN/QSA/shared block-FP8 (1.95/0.56/0.22); lm_head INT8 GPTQ 0.61 | separate repo | BF16 4.86 | 70.1 | plausible (Marlin FP8 W8A16 + two small vLLM patches), no 3090 report |
| edougawa/Qwen3.8-Flash-Next-W4A16-PLE4-MTP-Spark (08-31) | compressed-tensors RTN INT4: experts 58.0, GDN 1.00 (incl. in_proj_a/b), QSA 0.69, lm_head 0.31, shared 0.11, MTP 1.27; vision BF16 | INT4 per-row 24.5 (custom) | INT4 | 88.4 | maybe; "not tested" anywhere, PLE needs their patch |
| VnimanieAI / aixiaoma W4A16 (08-27, 09-02), devan-carlin (XPU), arnomatic PLE8 (AMD) | c-t INT4 experts 58.0 + QSA INT4 0.28; rest BF16 | BF16 (arnomatic INT8 g32) | BF16 | 167.5 / 122.8 | yes, no gain |
| wtdcode AWQ-W4A16 (08-27), yobo2u AWQ (SGLang patch) | experts only INT4 | BF16 | BF16 | 168.4 | yes, no gain |
| davetha DERISKED-W4A16-AWQ (08-30) | abliterated; experts + QSA INT4 | BF16 | BF16 | 167.5 | yes, no gain |
| cyankiwi AWQ-INT4 g32 asym (08-28); Jon-Nielsen -FP8PLE (09-05) | experts 65.0 (zero points) | BF16 / FP8 | BF16 | 175.4 / 127.7 | yes, bigger |
| leoncca AWQ-g32 (09-03) | experts 65.8, FP8 KV scales (unused on sm86) | FP8 | BF16 | 128.7 | yes, bigger |
| btbtyler09 GPTQ-4bit g32 (09-02) | experts 65.6 + QSA + shared INT4 | BF16 | BF16 | 174.8 | yes, bigger (card: ~20 GB per GPU at TP4) |
| HaberstrohSystems int2-mixed (09-03) | AutoRound W2 experts 30.1, GDN/QSA/lm_head INT8, no vision, no MTP | FP8 .bin 47.7 | none | 83.9 | no (2-bit MoE only in their SGLang patch) |
| HamboneLabs uint3-g64 (08-29) | uint3 experts 45.7, NVFP4 tail | BF16 .bin | NVFP4 | 147 | no (GB10 kernels) |
| Soomin33 FP6-INT8, textclf TQ-4bit, Jab1718 selective-int8 (128E) | custom packs | | | 145 / 160 / 37 | no |
| tcclaviger, MJPansa MXFP4-FP8 | MXFP4 experts 59.8 + FP8 side | FP8 | FP8 | 117 | no (AMD image); no gain anyway |
| Qwen FP8 + 8 derivatives | FP8 experts | BF16 | BF16 | 173 | no (~29 GiB per card) |
| NVFP4 family (~30 repos: RadixArk, nvidia, Inferact, primitive-ai, Mia, huginnfork x6, edougawa, r0b0tlab, patdev, lovedheart x3, skjortan, ...) | NVFP4 | various | various | 96 to 178 | no (sm86) |
| BF16 sources: Qwen, unsloth, abliterations, REAP-288/384 (sh0wie), REAM-288 (WaveCut), REAM-60Pct 308E (Akicou) | BF16 | BF16 | various | 232 to 335 | quantize ourselves |
| Sidecars: primitive-ai PLE-quant (FP8/INT4/NVFP4 tables), Saren ple-table-fp8, hampsonw MTP-INT4-Experts 1.39 | host-side or draft only | | | | n/a |
| GGUF ~45, MLX ~20, EXL3 (turboderp), halogen .hgn | | | | | no |

albucino's card, discussions and GitHub issues (5): no planned variants; his overlay targets 2x3090 TP2+EP2 only.

## Shortlist (per card, estimated)

1. ranxianglei Modular-296E: body 43.7 GiB, about 12.6 GiB per card with albucino's draft (11.6 without), -6.2 GiB, about +800K tokens. Same AutoRound/GPTQ layout (Marlin MoE), `num_experts 296`, FP8 PLE in the index (`ple_embedding_dtype float8_e4m3fn`, 128 shards). Caveats: the FP8 table ships without `weight_scale` (albucino's has one), so the dequant convention in their SGLang fork must be checked before our disk patch reads it; experts were pruned on their coding-agent traffic; no MTP; quality only shown on their own 5-prompt recovery suite, SGLang only.
2. Saren hybrid: body 65.2 GiB, about 17.9 GiB per card, -0.9, about +115K. Needs `quant_config` passed to `ParallelLMHead` (upstream `nvidia/model.py:649` omits it, also `mtp.py`) and Saren's GPTQ-to-Fp8 dispatch shim; on sm86 the FP8 side layers would run through Marlin W8A16 with block-to-group scale conversion (`marlin_utils_fp8.py:193-214`). Code exists, no 3090 report.
3. edougawa W4A16-PLE4: about 16.7 GiB per card, about +270K, untested, custom PLE pack, INT4 on in_proj_a/b.
4. Everything else is albucino-sized or bigger.

## Making our own

Measured BF16 side layers in albucino: 9.98 GiB total = 3.13 GiB per card (2.29 sharded + 0.84 vision, itself sharded at TP4 in vLLM).

| Move | GiB per card | Pool | How / risk |
|---|---|---|---|
| FP8-block GDN in_proj_qkv/z/out_proj + QSA qkvo + shared, INT8 lm_head (Saren `tools/`, CPU only, 1 to 2 h on the 3090 box, one shard in RAM at a time) | -0.8 | +105K | needs the two vLLM patches above; Marlin FP8 on sm86 unverified for this model |
| Same set at INT4 GPTQ (AutoRound `--layer_config`, or compressed-tensors RTN) | -1.3 | +170K | vLLM wires quant_config into `in_proj_qkvz` / `in_proj_ba` / `out_proj` (`qwen_gdn_linear_attn.py:422-494`, fused halves must share bits and group); keep in_proj_a/b [48,2560], indexer, HC, routers, embed BF16; every Hub quant kept GDN at 8 bits or more citing recurrent-state error compounding (only edougawa went 4-bit, untested); the 640-wide shared expert needs g32 or EP at TP4 |
| Drop the draft / the vision tower | -1.0 / -0.84 | +130K / +110K | `language_model_only` upstream; vision is already off in the text lane |
| Expert pruning 512 to 296 (ranxianglei's keep-set, or profile our own traffic with their CLI; tensor surgery on albucino, no requant) | -6.2 | +800K | quality on unprofiled tasks; reversible per layer |
| W2/W3 experts | -5 to -7 | | no Ampere vLLM MoE kernel below 4 bits today |

Calibrated passes (GPTQ/AutoRound) need the source staged: 335 GB BF16 does not fit the 3090 box (31 GB RAM); a DGX Spark (121 GB) holds Intel's 73 GB body plus activations. RTN and FP8 conversions are data-free and run on the 3090 box CPU. Tool traps: AutoRound skips 48-wide tensors and needs the `Qwen4ExpTextRMSNorm.group_size` patch (Haberstroh `scripts/07`); llm-compressor AWQ cannot linearize the experts under 256 GB RAM (VnimanieAI); do not fold AWQ smoothing into norms (hyper-connection sigmoid gate, davetha).

Sources: HF API and model cards; vLLM main (`vllm/models/qwen4_exp/nvidia/model.py`, `layers/mamba/gdn/qwen_gdn_linear_attn.py`, `quantization/fp8.py`, `utils/marlin_utils_fp8.py`); github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound; github.com/DominikBucko/qwen38-flash-next-2x3090; vllm issue #17579; auto-round issues #1496, #1650.
