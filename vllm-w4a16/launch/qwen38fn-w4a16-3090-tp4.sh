#!/usr/bin/env bash
# Qwen3.8-Flash-Next on the 4x RTX 3090 box (x86, Ampere), vLLM TP4, albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE
# (Intel AutoRound W4A16 experts in GPTQ layout + RadixArk FP8 n-gram table) with OUR disk-table patch so the 47.7 GB
# table never has to be resident (the box has 31 GB of RAM). Kai / 2Wild, 2026-09-06. Experimental.
# Knobs: NCCL_MODE (nvl|nop2p), KV_BYTES, LM_ONLY (0|1), PLE_MODE (staged|mmap|none), GRAPHS (nocompile|eager|piecewise|default), MTP (0|N; N needs a draft the checkpoint
# does not carry in-tree, see albucino's runtime/mtp-int4-g32), SEQS, CHUNK, GMU, MAXLEN, KV_DTYPE, TP, MODULE (vLLM subpackage
# that implements this checkpoint; nvidia = the ModelOpt-era layout our overlays target), OVERLAYS, EXTRA, DOCKER_EXTRA.
set -euo pipefail
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c}"
NAME="${NAME:-vllm_qwen38fn_3090}"
MODEL_HOST="${MODEL_HOST:-/home/tony/models/qwen38fn-w4a16-fp8ple}"
PATCH_DIR="${PATCH_DIR:-$HOME/patches/qwen4exp-ple-mmap}"
CACHE_HOST="${CACHE_HOST:-/home/tony/qwen38fn-vllm-cache}"; mkdir -p "$CACHE_HOST"
PLE_MODE="${PLE_MODE:-staged}"; GRAPHS="${GRAPHS:-nocompile}"; CAPTURE_SIZES="${CAPTURE_SIZES-4,8,12,16}"
MTP="${MTP:-0}"; SEQS="${SEQS:-4}"; CHUNK="${CHUNK-4096}"; GMU="${GMU:-0.92}"; MAXLEN="${MAXLEN:-262144}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"; TP="${TP:-4}"; PORT="${PORT:-8090}"; MODULE="${MODULE:-nvidia}"; OVERLAYS="${OVERLAYS:-1}"
VP=/usr/local/lib/python3.12/dist-packages/vllm; MP="$VP/models/qwen4_exp/$MODULE"
test -f "$MODEL_HOST/config.json" || { echo "MODEL MISSING at $MODEL_HOST" >&2; exit 3; }
PLE_ENV=(); case "$PLE_MODE" in
  mmap)    PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_MMAP_THREADS="${PLE_WORKERS:-32}"
           -v "$PATCH_DIR/ple_layer.py:$MP/ple_layer.py:ro" -v "$PATCH_DIR/ple_mmap.py:$MP/ops/ple_mmap.py:ro") ;;
  staged)  PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_STAGED=1 -e QWEN4EXP_PLE_MMAP_THREADS="${PLE_WORKERS:-32}"
           -v "$PATCH_DIR/ple_layer.py:$MP/ple_layer.py:ro" -v "$PATCH_DIR/ple_mmap.py:$MP/ops/ple_mmap.py:ro"
           -v "$PATCH_DIR/model_state.py:$MP/model_state.py:ro") ;;
  none)    ;;
  *) echo "PLE_MODE must be staged|mmap|none" >&2; exit 2 ;;
esac
OVERLAY_MOUNT=(); if [ "$OVERLAYS" = "1" ]; then
  QSA_SFX=""; if [ "$KV_DTYPE" = "fp8_e5m2" ]; then QSA_SFX="_e5m2"; fi   # Ampere: e5m2 variant of the QSA overlay (no fp8e4nv in Triton)
  OVERLAY_MOUNT=(-v "$PATCH_DIR/upstream-overlays/ops_ple.py:$MP/ops/ple.py:ro" -v "$PATCH_DIR/upstream-overlays/ops_qsa$QSA_SFX.py:$MP/ops/qsa.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/qsa$QSA_SFX.py:$MP/qsa.py:ro" -v "$PATCH_DIR/upstream-overlays/platforms_interface.py:$VP/platforms/interface.py:ro"); fi
GRAPH_ARGS=(); GRAPH_MOUNT=(); case "$GRAPHS" in
  eager)     GRAPH_ARGS=(--enforce-eager) ;;
  piecewise) GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}'); GRAPH_MOUNT=(-v "$PATCH_DIR/compilation.py:$VP/config/compilation.py:ro") ;;
  nocompile) if [ -n "${CAPTURE_SIZES:-}" ]; then GRAPH_ARGS=(--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}"); else GRAPH_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'); fi ;;
  default)   ;;
  *) echo "GRAPHS must be eager|piecewise|nocompile|default" >&2; exit 2 ;;
esac
KV_ARGS=(); if [ "$KV_DTYPE" != "auto" ]; then KV_ARGS=(--kv-cache-dtype "$KV_DTYPE"); fi
# KV_BYTES=<bytes>: hand vLLM an explicit KV budget (it prints the number it could have used at boot); LM_ONLY=1 drops the vision tower
if [ -n "${KV_BYTES:-}" ]; then KV_ARGS+=(--kv-cache-memory-bytes "$KV_BYTES"); fi
LM_ONLY="${LM_ONLY:-0}"; LMO_ARGS=(); if [ "$LM_ONLY" = "1" ]; then LMO_ARGS=(--language-model-only); fi
# NCCL transport on this board: two NVLink pairs (0-1 NV2, 2-3 NV4) joined only by PCIe through the host bridge.
#   NCCL_MODE=nvl   -> NCCL_P2P_LEVEL=NVL: P2P over NVLink inside a pair, shared memory across pairs (default)
#   NCCL_MODE=nop2p -> NCCL_P2P_DISABLE=1 (what the 27B lane used; also disables NVLink transport)
NCCL_MODE="${NCCL_MODE:-nvl}"; case "$NCCL_MODE" in
  nvl)   NCCL_ENV=(-e NCCL_P2P_LEVEL=NVL) ;;
  nop2p) NCCL_ENV=(-e NCCL_P2P_DISABLE=1) ;;
  *) echo "NCCL_MODE must be nvl|nop2p" >&2; exit 2 ;;
esac
# MTP draft: this checkpoint carries no mtp.* tensors; albucino ships a separate INT4 g32 draft under runtime/mtp-int4-g32
# (inside the model mount). vLLM 8a728663 accepts a separate draft dir for method "mtp" (compressed-tensors, Marlin g32 on sm86).
# Capture sizes must be multiples of MTP+1. Do not pass use_local_argmax_reduction (our Qwen4ExpMTP has no get_top_tokens).
DRAFT_DIR="${DRAFT_DIR:-/models/qwen38fn/runtime/mtp-int4-g32}"
SPEC=(); if [ "$MTP" != "0" ]; then SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"model\":\"$DRAFT_DIR\"}"); fi
docker rm -f "$NAME" 2>/dev/null || true
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
docker run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 \
  -v "$MODEL_HOST:/models/qwen38fn:ro" -v "$CACHE_HOST:/root/.cache" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCH_CUDA_ARCH_LIST=8.6 \
  "${NCCL_ENV[@]}" -e NCCL_CUMEM_ENABLE=0 -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "${PLE_ENV[@]}" "${GRAPH_MOUNT[@]}" "${OVERLAY_MOUNT[@]}" ${DOCKER_EXTRA:-} \
  "$IMAGE" \
    /models/qwen38fn --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --tensor-parallel-size "$TP" --disable-custom-all-reduce \
    --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GMU" ${CHUNK:+--max-num-batched-tokens $CHUNK} \
    --no-enable-prefix-caching \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --default-chat-template-kwargs "{\"enable_thinking\": false}" \
    "${SPEC[@]}" "${GRAPH_ARGS[@]}" "${KV_ARGS[@]}" "${LMO_ARGS[@]}" ${EXTRA:-}
echo "launched $NAME image=$IMAGE module=$MODULE ple=$PLE_MODE graphs=$GRAPHS kv=$KV_DTYPE tp=$TP gmu=$GMU maxlen=$MAXLEN seqs=$SEQS mtp=$MTP"
sleep 3; docker ps --format "{{.Names}} {{.Status}}" | grep "$NAME" || { echo "$NAME exited"; docker logs "$NAME" 2>&1 | tail -5; exit 1; }
