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
KV_BYTES="${KV_BYTES:-2400000000}" LM_ONLY=0 NCCL_MODE=nvl PLE_MODE=hostpin GRAPHS=nocompile MTP=3 TP=4 \
GMU=0.95 SEQS=6 CHUNK=2048 MAXLEN=262144 KV_DTYPE=fp8_e5m2 CAPTURE_SIZES=4,8,12,16,20,24 \
SPEC_EXTRA="${SPEC_EXTRA-\"disable_eagle_block_drop\":true}" PREFIX_CACHE=1 \
EXTRA="--quantization gptq_marlin --enable-expert-parallel --mm-processor-kwargs {\"max_pixels\":1048576} --mamba-ssm-cache-dtype bfloat16 --skip-mm-profiling" \
exec bash "$(dirname "$0")/qwen38fn-w4a16-3090-tp4.sh"
