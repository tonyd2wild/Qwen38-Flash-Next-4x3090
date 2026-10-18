#!/usr/bin/env bash
# Qwen3.8-Flash-Next on the 4x RTX 3090 box (x86, Ampere), vLLM TP4, albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE
# (Intel AutoRound W4A16 experts in GPTQ layout + RadixArk FP8 n-gram table) with OUR disk-table patch so the 47.7 GB
# table never has to be resident (the box has 31 GB of RAM). Kai / 2Wild, 2026-09-06. Experimental.
# Knobs: NCCL_MODE (nvl|nop2p), KV_BYTES, LM_ONLY (0|1), PLE_MODE (staged|mmap|hostpin|none; hostpin copies the table into one pinned, shared host buffer
# and gathers rows on the GPU, needs memlock for 51.2 GB), GRAPHS (nocompile|eager|piecewise|default), MTP (0|N; N needs a draft the checkpoint
# does not carry in-tree, see albucino's runtime/mtp-int4-g32), SEQS, CHUNK, GMU, MAXLEN, KV_DTYPE, TP, MODULE (vLLM subpackage
# that implements this checkpoint; nvidia = the ModelOpt-era layout our overlays target), OVERLAYS, EXTRA, DOCKER_EXTRA,
# PREFIX_CACHE (0|1; 1 = --enable-prefix-caching --mamba-cache-mode align), GPU_ARGS (default --gpus all),
# TRITON_CACHE (0|1; 1 = keep Triton's kernel cache in the mounted cache dir across boots), TOKSPLIT (0|1; 1 = exact
# split tokenization with a piece cache for long prompts, TOKSPLIT_THREADS rayon threads, default 8; journal/14),
# MAMBA_FIX (0|1; 1 = free every align-mode Mamba state block once no in-flight step reads it; journal/04), MAMBA_TRACE
# (0|1; 1 = MAMBATRACE log lines from the KV manager for prompts of >= 32K tokens; journal/04), DET_TRACE (0|1; 1 = the
# T14 activation tracer as a vLLM general plugin, diagnostic only, DET_TRACE_MAX_TOKENS default 512, DET_TRACE_FUNCS
# functions to wrap as mod:attr,...; journal/16), MOE_DET (0|1; 1 = each expert's token list in a fixed order after
# moe_align_block_size, so the Marlin MoE GEMMs give the same bits run to run; journal/16), QSA_DET (0|1; 1 = the QSA
# selection as the exact top-k in ascending order, ties to the lowest index, instead of persistent_topk's set and order).
set -euo pipefail
# The upstream nightly-8a728663 tag was pruned; build this image from ../container/Containerfile (same commit, same paths).
IMAGE="${IMAGE:-qwen38fn-vllm:8a728663}"
NAME="${NAME:-vllm_qwen38fn_3090}"
LANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_HOST="${MODEL_HOST:-$HOME/models/qwen38fn-w4a16-fp8ple}"
PATCH_DIR="${PATCH_DIR:-$LANE_DIR/patch}"
CACHE_HOST="${CACHE_HOST:-$(cd "$LANE_DIR/.." && pwd)/.cache/ple}"; mkdir -p "$CACHE_HOST"
PLE_MODE="${PLE_MODE:-staged}"; GRAPHS="${GRAPHS:-nocompile}"; CAPTURE_SIZES="${CAPTURE_SIZES-4,8,12,16}"
MTP="${MTP:-0}"; SEQS="${SEQS:-4}"; CHUNK="${CHUNK-4096}"; GMU="${GMU:-0.92}"; MAXLEN="${MAXLEN:-262144}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"; TP="${TP:-4}"; PORT="${PORT:-8090}"; MODULE="${MODULE:-nvidia}"; OVERLAYS="${OVERLAYS:-1}"
PREFIX_CACHE="${PREFIX_CACHE:-0}"; GPU_ARGS="${GPU_ARGS:---gpus all}"; TRITON_CACHE="${TRITON_CACHE:-0}"; TOKSPLIT="${TOKSPLIT:-0}"
MAMBA_FIX="${MAMBA_FIX:-0}"; MAMBA_TRACE="${MAMBA_TRACE:-0}"; DET_TRACE="${DET_TRACE:-0}"; MOE_DET="${MOE_DET:-0}"; QSA_DET="${QSA_DET:-0}"
VP=/usr/local/lib/python3.12/dist-packages/vllm; MP="$VP/models/qwen4_exp/$MODULE"
test -f "$MODEL_HOST/config.json" || { echo "MODEL MISSING at $MODEL_HOST" >&2; exit 3; }
PLE_ENV=(); case "$PLE_MODE" in
  mmap)    PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_MMAP_THREADS="${PLE_WORKERS:-32}"
           -v "$PATCH_DIR/ple_layer.py:$MP/ple_layer.py:ro" -v "$PATCH_DIR/ple_mmap.py:$MP/ops/ple_mmap.py:ro") ;;
  staged)  PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_STAGED=1 -e QWEN4EXP_PLE_MMAP_THREADS="${PLE_WORKERS:-32}"
           -v "$PATCH_DIR/ple_layer.py:$MP/ple_layer.py:ro" -v "$PATCH_DIR/ple_mmap.py:$MP/ops/ple_mmap.py:ro"
           -v "$PATCH_DIR/model_state.py:$MP/model_state.py:ro") ;;
  hostpin) PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_HOSTPIN=1
           -v "$PATCH_DIR/ple_layer.py:$MP/ple_layer.py:ro" -v "$PATCH_DIR/ple_mmap.py:$MP/ops/ple_mmap.py:ro"
           -v "$PATCH_DIR/ple_hostpin.py:$MP/ops/ple_hostpin.py:ro") ;;
  none)    ;;
  *) echo "PLE_MODE must be staged|mmap|hostpin|none" >&2; exit 2 ;;
esac
# Triton compiles into /root/.triton/cache by default, which lives in the container and is gone at the next boot.
case "$TRITON_CACHE" in
  0) TRITON_ENV=() ;;
  1) TRITON_ENV=(-e TRITON_CACHE_DIR=/root/.cache/triton) ;;
  *) echo "TRITON_CACHE must be 0|1" >&2; exit 2 ;;
esac
# TOKSPLIT=1: the API server tokenizes long prompts in exactly-equal pieces, cached across turns (patch/tok_split.py).
case "$TOKSPLIT" in
  0) TOK_ENV=() ;;
  1) TOK_ENV=(-e QWEN4EXP_TOKSPLIT=1 -e RAYON_NUM_THREADS="${TOKSPLIT_THREADS:-8}"
              -v "$PATCH_DIR/renderers_base.py:$VP/renderers/base.py:ro" -v "$PATCH_DIR/tok_split.py:$VP/renderers/tok_split.py:ro") ;;
  *) echo "TOKSPLIT must be 0|1" >&2; exit 2 ;;
esac
# MAMBA_FIX / MAMBA_TRACE: the KV-manager overlay (patch/single_type_kv_cache_manager.py); with both 0 nothing is mounted.
case "$MAMBA_FIX$MAMBA_TRACE" in
  00) KVM_ENV=() ;;
  01|10|11) KVM_ENV=(-e QWEN4EXP_MAMBA_FIX="$MAMBA_FIX" -e QWEN4EXP_MAMBA_TRACE="$MAMBA_TRACE"
                     -v "$PATCH_DIR/single_type_kv_cache_manager.py:$VP/v1/core/single_type_kv_cache_manager.py:ro") ;;
  *) echo "MAMBA_FIX and MAMBA_TRACE must be 0|1" >&2; exit 2 ;;
esac
# DET_TRACE=1: patch/det_trace.py as vllm/det_trace.py, loaded in every process through the entry point in the mounted
# dist-info directory; it records fingerprints only while $CACHE_HOST/det-trace/on exists (tools/det_trace_diff.py).
case "$DET_TRACE" in
  0) DET_ENV=(); DET_DIST=() ;;
  1) DET_ENV=(-e QWEN4EXP_DET_TRACE=1 -e QWEN4EXP_DET_TRACE_MAX_TOKENS="${DET_TRACE_MAX_TOKENS:-512}" -e QWEN4EXP_DET_TRACE_FUNCS="${DET_TRACE_FUNCS:-}"
              -e QWEN4EXP_DET_TRACE_QSA="${DET_TRACE_QSA:-0}" -e QWEN4EXP_DET_TRACE_DUMP="${DET_TRACE_DUMP:-}"
              -v "$PATCH_DIR/det_trace.py:$VP/det_trace.py:ro")
     DET_DIST=(-v "$PATCH_DIR/qwen4exp_det_trace-0.1.dist-info:${VP%/vllm}/qwen4exp_det_trace-0.1.dist-info:ro") ;;
  *) echo "DET_TRACE must be 0|1" >&2; exit 2 ;;
esac
# MOE_DET=1: the overlay patch/moe_align_block_size.py re-sorts each expert's tokens after the atomic align kernel.
case "$MOE_DET" in
  0) MOED_ENV=() ;;
  1) MOED_ENV=(-e QWEN4EXP_MOE_DET=1 -v "$PATCH_DIR/moe_align_block_size.py:$VP/model_executor/layers/fused_moe/moe_align_block_size.py:ro") ;;
  *) echo "MOE_DET must be 0|1" >&2; exit 2 ;;
esac
# QSA_DET=1: the overlay patch/qsa_indexer.py selects the exact top-k itself (radix select; decode rows check persistent_topk).
case "$QSA_DET" in
  0) QSAD_ENV=() ;;
  1) QSAD_ENV=(-e QWEN4EXP_QSA_DET=1 -v "$PATCH_DIR/qsa_indexer.py:$MP/ops/qsa_indexer.py:ro") ;;
  *) echo "QSA_DET must be 0|1" >&2; exit 2 ;;
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
# SPEC_EXTRA: more speculative-config keys as a JSON fragment, e.g. '"disable_eagle_block_drop":true'.
SPEC=(); if [ "$MTP" != "0" ]; then SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"model\":\"$DRAFT_DIR\"${SPEC_EXTRA:+,$SPEC_EXTRA}}"); fi
# Prefix caching for Qwen4Exp at this commit works only in mamba align mode ("all" raises); align needs chunked
# prefill, which CHUNK (--max-num-batched-tokens) already turns on.
case "$PREFIX_CACHE" in
  0) PC_ARGS=(--no-enable-prefix-caching) ;;
  1) [ -n "$CHUNK" ] || { echo "PREFIX_CACHE=1 needs CHUNK (chunked prefill)" >&2; exit 2; }
     PC_ARGS=(--enable-prefix-caching --mamba-cache-mode align) ;;
  *) echo "PREFIX_CACHE must be 0|1" >&2; exit 2 ;;
esac
docker stop -t 60 "$NAME" >/dev/null 2>&1 || true; docker rm -f "$NAME" >/dev/null 2>&1 || true
# A removed container can leave its server running (seen once under rootless podman); never test the wrong server.
if curl -s -m 3 -o /dev/null "http://localhost:$PORT/v1/models"; then echo "port $PORT still answers after removing $NAME; stop that server first" >&2; exit 5; fi
# --ipc host shares the host /dev/shm, so no --shm-size: Docker ignores it with --ipc host and podman rejects the pair.
docker run $GPU_ARGS -d --name "$NAME" --restart no \
  --network host --ipc host --ulimit memlock=-1:-1 \
  -v "$MODEL_HOST:/models/qwen38fn:ro" -v "$CACHE_HOST:/root/.cache" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCH_CUDA_ARCH_LIST=8.6 \
  "${NCCL_ENV[@]}" -e NCCL_CUMEM_ENABLE=0 -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "${PLE_ENV[@]}" "${GRAPH_MOUNT[@]}" "${OVERLAY_MOUNT[@]}" "${TRITON_ENV[@]}" "${TOK_ENV[@]}" "${KVM_ENV[@]}" "${DET_ENV[@]}" "${DET_DIST[@]}" "${MOED_ENV[@]}" "${QSAD_ENV[@]}" ${DOCKER_EXTRA:-} \
  "$IMAGE" \
    /models/qwen38fn --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP" --disable-custom-all-reduce \
    --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GMU" ${CHUNK:+--max-num-batched-tokens $CHUNK} \
    "${PC_ARGS[@]}" \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --default-chat-template-kwargs "{\"enable_thinking\": false}" \
    "${SPEC[@]}" "${GRAPH_ARGS[@]}" "${KV_ARGS[@]}" "${LMO_ARGS[@]}" ${EXTRA:-}
echo "launched $NAME image=$IMAGE module=$MODULE ple=$PLE_MODE graphs=$GRAPHS kv=$KV_DTYPE tp=$TP gmu=$GMU maxlen=$MAXLEN seqs=$SEQS mtp=$MTP prefix_cache=$PREFIX_CACHE triton_cache=$TRITON_CACHE toksplit=$TOKSPLIT mamba_fix=$MAMBA_FIX mamba_trace=$MAMBA_TRACE det_trace=$DET_TRACE moe_det=$MOE_DET qsa_det=$QSA_DET"
sleep 3; docker ps --format "{{.Names}} {{.Status}}" | grep "$NAME" || { echo "$NAME exited"; docker logs "$NAME" 2>&1 | tail -5; exit 1; }
# Each overlay must be the file Python actually imports: compare the mounted source with the module file in the container.
OVL=(); for m in "${PLE_ENV[@]}" "${OVERLAY_MOUNT[@]}" "${GRAPH_MOUNT[@]}" "${TOK_ENV[@]}" "${KVM_ENV[@]}" "${DET_ENV[@]}" "${MOED_ENV[@]}" "${QSAD_ENV[@]}"; do case "$m" in /*:*:ro) OVL+=("$m") ;; esac; done
for m in "${OVL[@]}"; do src="${m%%:*}"; dst="${m#*:}"; dst="${dst%:ro}"
  mod="${dst#$VP/}"; mod="vllm.${mod%.py}"; mod="${mod//\//.}"
  got=$(docker exec "$NAME" python3 -c "import hashlib,importlib.util as u; f=u.find_spec('$mod').origin; print(f, hashlib.sha256(open(f,'rb').read()).hexdigest())")
  want=$(sha256sum "$src" | cut -d' ' -f1)
  if [ "$got" = "$dst $want" ]; then echo "overlay ok  $mod <- $src"; else echo "OVERLAY MISMATCH $mod: imports '$got', want $dst $want" >&2; exit 4; fi
done
# The tracer's dist-info is a directory, not a module: check that Python sees its entry point instead.
if [ "$DET_TRACE" = "1" ]; then
  ep=$(docker exec "$NAME" python3 -c "from importlib.metadata import entry_points as e; print([p.value for p in e(group='vllm.general_plugins') if p.name=='qwen4exp_det_trace'])")
  if [ "$ep" = "['vllm.det_trace:register']" ]; then echo "plugin ok   qwen4exp_det_trace -> vllm.det_trace:register"; else echo "PLUGIN MISSING qwen4exp_det_trace: $ep" >&2; exit 4; fi
fi
