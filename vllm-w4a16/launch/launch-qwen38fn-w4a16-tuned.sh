#!/usr/bin/env bash
# Qwen3.8-Flash-Next W4A16 on 4x RTX 3090, the tuned lane (2026-10-01, the daily driver on our box).
# Boot 19 flags + --skip-mm-profiling + a pinned KV budget. KV_BYTES 2.4e9 gives a pool of 303,079 tokens
# at 262,144 context (1.16x) with about 0.65 GiB per card left free for long-prefill transients.
# Validated with a ~155K-token prefill (2,249 tok/s, 69 s TTFT, no OOM).
#
# Wrapper over qwen38fn-w4a16-3090-tp4.sh in this folder; it does not modify it.
# Set MODEL_HOST=<model_dir> (the albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE download) and
# PATCH_DIR=<patch_dir> (a copy of ../patch) if yours differ from the launcher defaults.
#
# Raising KV_BYTES: +50 MB is about +6.6K pool tokens. Climb in 50 MB steps and run a 155K+ token
# prefill after each one. The boot-time profiler only measures short-context activations, so a pool
# that boots fine can still OOM on a long prefill (see the lane README, "Tuned lane").
KV_BYTES="${KV_BYTES:-2400000000}" LM_ONLY=0 NCCL_MODE=nvl PLE_MODE=staged GRAPHS=nocompile MTP=3 TP=4 \
GMU=0.95 SEQS=6 CHUNK=2048 MAXLEN=262144 KV_DTYPE=fp8_e5m2 CAPTURE_SIZES=4,8,12,16,20,24 \
EXTRA="--quantization gptq_marlin --enable-expert-parallel --mm-processor-kwargs {\"max_pixels\":1048576} --mamba-ssm-cache-dtype bfloat16 --skip-mm-profiling" \
exec bash "$(dirname "$0")/qwen38fn-w4a16-3090-tp4.sh"
