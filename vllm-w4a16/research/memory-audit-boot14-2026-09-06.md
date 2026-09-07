# Boot 14 memory audit (vision on, 131K, gmu 0.95), 4x3090 TP4+EP, 2026-09-06

Read-only pass by a sub-agent over `flash14.log`, `flash13.log` (text-only), the checkpoint safetensors headers and the vLLM source inside the container. Per card, GiB. "measured" = read from the log; "estimated" = summed from tensor headers.

| Item | GiB | Basis |
|---|---|---|
| Card total / free at init / requested (gmu 0.95) | 23.56 / 23.04 / 22.38 | measured |
| Routed experts, W4 Marlin, EP (58.45 GiB total / 4) | ~14.6 | estimated |
| BF16 side layers: DeltaNet 0.97, attention ~0.33, hyperconnection 1.19 (replicated, `ReplicatedLinear`, not quantized by AutoRound), shared expert 0.11, router 0.12, PLE proj 0.06 | ~2.8 | estimated |
| embed 0.30 + lm_head 0.30 (sharded) | 0.59 | estimated |
| Vision tower (0.84 GiB total, TP-sharded by Column/RowParallel) | ~0.21 | measured delta: 18.83 text-only vs 19.01 with the tower |
| MTP draft: experts 0.33 + its own embed 0.30 + its own lm_head 0.30 (byte-identical to the target's) + dense ~0.12 | ~1.05 | estimated |
| Weights total | 19.01 | measured |
| Non-torch (NCCL, context, workspaces) | 0.70 | measured |
| Profile peak (2048-token chunk + one max-size image) | 0.98 | measured (1.31 reported minus the 0.33 graph estimate); text-only boot 13 = 0.25 |
| Encoder cache 16,384 tokens x 2560 x bf16 | 0.08 transient | measured; the tower's profile peak ~0.73 |
| CUDA graphs (6 sizes) | 0.26 actual, 0.33 reserved | measured |
| Idle headroom (23.04 - 22.38) | 0.66 | measured (+0.52 pre-init context outside the budget) |
| KV pool | 1.36 = 160,199 tokens | measured; ~117.8K tokens per GiB at this geometry |
| DeltaNet state per seat: 36 x (conv 30 KB + SSM 12x128x128 fp32 = 786 KB) + PLE conv 240 KB | 29.7 MB real, ~192 MB of pool | fp32 forced by the checkpoint's `mamba_ssm_dtype: float32` |

Pool geometry (reconstructed, matches the log's 160,199 tokens, 1.22x, "Mamba groups [0,1,2,3]", block 1600): a block is ~12.0 MB = 13 attention layers (12 + the draft's) x (819,200 B e5m2 KV + 102,400 B indexer) plus rings; 121 blocks. A 131K request needs 82 attention blocks + 16 mamba blocks (4 groups x (1 + 3 speculative)) + 1 ring = 99, so 121/99 = 1.22x. Every live seat pins 16 blocks regardless of length: attention capacity ~ (121 - 16N) x 1600 tokens, so 1 seat 168K, 2 seats 142K, 4 seats 91K, 6 seats ~40K in total.

Ranked changes (per card):
1. `--mm-processor-kwargs '{"max_pixels":1048576}'`: +0.6 to 0.7 GiB, +70K to 80K tokens (the profiler otherwise runs a 16.7 MP image, `preprocessor_config.json` longest_edge 16,777,216). Low risk, large images are downscaled.
2. `--kv-cache-memory-bytes` at vLLM's printed "to fully utilize" figure (1,729,576,448 on boot 14): +0.25 GiB, +29K tokens; re-read the figure after any other change.
3. `--max-num-seqs 4`: pool unchanged, pinned mamba blocks 96 to 64.
4. MTP depth: each speculative token costs one block per mamba group per seat. MTP=2 frees 24 blocks at 6 seats. Sharing the target's embed and lm_head with the draft (patch `qwen4_exp/nvidia/mtp.py` around lines 177 and 404) would free 0.59 GiB, about +79K tokens, at no speed cost. Moderate risk (loader change).
5. `--mamba-ssm-cache-dtype bfloat16`: SSM 786 to 393 KB per layer, block would fall to ~832 tokens; +8 to 10% tokens. vLLM warns but honors it. Needs a long-context A/B.
6. gmu 0.96: +0.24 GiB (+28K), overlaps 2. gmu 0.97 leaves ~0.19 GiB and died under load.
7. `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`: +0.07 GiB. Fewer capture sizes: ~0.0125 GiB each. `--enforce-eager` saves 0.33 but hurts decode.
8. `--max-num-batched-tokens 1024`: maybe -0.1 GiB, only binding after 1; slower prefill.
9. Requantize (new checkpoint): hyperconnection 1.19 GiB BF16 replicated; FP8 Marlin (W8A16 runs on sm86) ~-0.6, INT4 ~-0.9; lm_head INT4 via `lm_head_quantized` ~-0.22; vision FP8 ~-0.1. High effort.
10. Keep prefix caching off: on would set `mamba_cache_mode=all` (checkpoints per 1600-token block ~18 KB per token vs 6.9 KB attention) and MTP blocks mamba prefix hits anyway.
11. Block packing waste (vLLM side, no flag): GDN groups fill 9.83 of 12.0 MB per block, the PLE group 0.82 of 12.0 MB (4 blocks per seat for 240 KB of state), the ring group 1 block per request; a packing fix could return 4 to 5 blocks per seat.

Not verified: the exact KVCacheConfig (not logged at INFO), Marlin repack size, draft dense sharding, the ViT peak breakdown, NCCL's share of the 0.70 non-torch, whether the max_pixels override also drops the video budget (code suggests yes).

Correction (checked against the source and the ladder after the audit): item 4's "share the target's embed and lm_head with the draft" is already what vLLM does. `vllm/v1/worker/gpu/spec_decode/eagle/utils.py` (`load_eagle_model`, `_should_share`) deletes the draft's own `embed_tokens` and `lm_head` after load and points them at the target's whenever the draft model does not declare `has_own_embed_tokens` / `has_own_lm_head` (Qwen4ExpMTP declares neither). The ladder agrees: the draft added 0.52 GiB per card (row 30 19.76 GiB without it, row 33 20.28 GiB with it), which is its experts and dense layers, not 1.05 GiB. So the 0.59 GiB in the draft line above is not resident and there is nothing to win there; the audit's per-item sum should be read with that 0.59 removed (which also closes its 0.24 GiB gap to the measured 19.01).
