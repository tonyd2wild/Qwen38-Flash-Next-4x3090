#!/usr/bin/env bash
# Build llama-server with qwen4exp support (unsloth PR #27742) inside a CUDA container.
# No system CUDA/nvcc needed on the host — only Docker with the NVIDIA runtime.
#
# TRAPS (each of these cost us a failed build or a confusing error):
#  1. The build container MUST run with --gpus all. Without it there is no
#     libcuda.so for the final link and it dies with:
#       undefined reference to `cuMemCreate' (etc.)
#  2. Run the built binary inside the SAME image (or one with matching CUDA libs)
#     and set LD_LIBRARY_PATH to the build dirs, or it fails on
#     libllama-server-impl.so / libcudart.so.12.
#  3. -DCMAKE_CUDA_ARCHITECTURES=86 for RTX 3090 (Ampere).
set -euo pipefail

SRC="$HOME/llama-qwen4exp"
IMAGE="nvidia/cuda:12.4.1-devel-ubuntu22.04"

[ -d "$SRC" ] || git clone --depth 1 -b qwen4exp/qwen3.8-flash-next \
  https://github.com/unslothai/llama.cpp.git "$SRC"

docker run --rm --gpus all -v "$SRC:/src" -w /src "$IMAGE" bash -c '
  apt-get update -qq && apt-get install -y -qq cmake git build-essential >/dev/null &&
  git config --global --add safe.directory /src &&
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF &&
  cmake --build build -j $(nproc) --target llama-server &&
  echo BUILD-DONE'
