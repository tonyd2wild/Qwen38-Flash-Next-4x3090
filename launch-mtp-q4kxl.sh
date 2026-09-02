#!/usr/bin/env bash
# Qwen3.8-Flash-Next UD-Q4_K_XL + unsloth shared-Q8_0 MTP head, --parallel 1.
# Uses unsloth prebuilt CUDA12-portable llama-server (build 10715).
set -euo pipefail
BIN=/home/tony/llama-unsloth-mtp/llama-server
export LD_LIBRARY_PATH=/home/tony/llama-unsloth-mtp:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/cufft/lib:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/curand/lib:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/nccl/lib:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/home/tony/personaplex-venv/lib/python3.12/site-packages/nvidia/nvtx/lib:${LD_LIBRARY_PATH:-}
MAIN=/home/tony/models/qwen38fn-gguf/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
HEAD=/home/tony/models/qwen38fn-gguf/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
SPEC="${SPEC:-1}"
SPECARGS=""
[ "$SPEC" = "1" ] && SPECARGS="-md $HEAD --spec-type draft-mtp --spec-draft-n-max 2"
exec "$BIN" -m "$MAIN" $SPECARGS \
  --no-repack -ngl 999 --tensor-split 1,1,1,1 \
  --parallel 1 -c 131072 \
  --host 0.0.0.0 --port 8090 --jinja \
  -a qwen3.8-flash-next-gguf
