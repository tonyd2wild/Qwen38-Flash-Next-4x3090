#!/usr/bin/env bash
# Qwen3.8-Flash-Next UD-IQ3_XXS GGUF across all 4 RTX 3090s, llama.cpp (unsloth PR #27742 build).
# Mutually exclusive with the DFlash2 lanes; stop them first, docker start them to restore.
set -euo pipefail
docker rm -f llama_qwen38fn 2>/dev/null || true
docker run --gpus all -d --name llama_qwen38fn --restart no \
  --network host \
  -v /home/tony/models:/models:ro \
  -v /home/tony/llama-qwen4exp:/src:ro \
  -e LD_LIBRARY_PATH=/src/build/bin:/src/build/src:/src/build/common \
  nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  /src/build/bin/llama-server \
    -m /models/qwen38fn-gguf/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
    -ngl 999 --tensor-split 1,1,1,1 -c 524288 --parallel 2 \
    --host 0.0.0.0 --port 8090 --jinja --spec-type ngram-mod \
    -a qwen3.8-flash-next-gguf
echo launched llama_qwen38fn
sleep 2
docker ps --format "{{.Names}} {{.Status}}" | grep llama_qwen38fn
