#!/usr/bin/env bash
# The prefix lane (launch-qwen38fn-w4a16-prefix.sh: the tuned lane's flags with prefix caching in mamba align mode)
# with PLE_MODE=hostpin instead of staged: the 51.2 GB n-gram table is copied once into a shared memfd that every
# rank pins (cudaHostRegister; one copy in host RAM for the four ranks, filled by rank 0 from the model-plefp8 files)
# and the GPU gathers the rows through UVA inside the forward and the decode CUDA graph, so no step waits
# for a host-side gather. Same bytes as staged (lossless). See journal/05-p4-ple-hostpin.md (T5).
#
# Needs about 51 GB of host RAM to stay pinned while the server runs, and an unlimited memlock in the container
# (the base launcher passes --ulimit memlock=-1:-1). Rank 0 reads the table from the files at every boot (page
# cache or disk) while the other ranks wait.
# The tuned and prefix wrappers are not changed by this file; the knobs below are theirs, with PLE_MODE=hostpin.
# T11 (journal/14-p9-fixed-overheads.md) adds TRITON_CACHE=1 (Triton kernels survive a reboot) and TOKSPLIT=1 (exact
# split tokenization with a piece cache); both can be turned off from the environment.
# T4 (journal/04-p3-mamba-leak.md) adds MAMBA_FIX=1: align-mode Mamba state blocks are freed once no in-flight step reads
# them, instead of one per prefill chunk staying held until the request ends (cold prompts above ~101K tokens no longer
# get preempted). MAMBA_FIX=0 turns it off; MAMBA_TRACE=1 adds the diagnosis log lines.
# T14 (journal/16-nondeterminism.md) adds MOE_DET=1 and QSA_DET=1: each expert's tokens in a fixed order before the
# Marlin MoE GEMMs, and the QSA top-k as the exact top-k in ascending order (the image's persistent_topk returns a new
# order every call and, at depth, a slightly wrong set in a few rows). With both, temperature-0 outputs repeat bit for
# bit, run to run and boot to boot. MOE_DET=0 / QSA_DET=0 turn them off.
KV_BYTES="${KV_BYTES:-2400000000}" LM_ONLY=0 NCCL_MODE=nvl PLE_MODE=hostpin GRAPHS=nocompile MTP=3 TP=4 \
GMU=0.95 SEQS=6 CHUNK=2048 MAXLEN=262144 KV_DTYPE=fp8_e5m2 CAPTURE_SIZES=4,8,12,16,20,24 \
SPEC_EXTRA="${SPEC_EXTRA-\"disable_eagle_block_drop\":true}" PREFIX_CACHE=1 \
TRITON_CACHE="${TRITON_CACHE:-1}" TOKSPLIT="${TOKSPLIT:-1}" MAMBA_FIX="${MAMBA_FIX:-1}" MOE_DET="${MOE_DET:-1}" QSA_DET="${QSA_DET:-1}" \
EXTRA="--quantization gptq_marlin --enable-expert-parallel --mm-processor-kwargs {\"max_pixels\":1048576} --mamba-ssm-cache-dtype bfloat16 --skip-mm-profiling" \
exec bash "$(dirname "$0")/qwen38fn-w4a16-3090-tp4.sh"
