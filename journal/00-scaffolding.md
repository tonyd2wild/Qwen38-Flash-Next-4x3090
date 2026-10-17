# 00: Scaffolding and exploration hand-over (T0)

Status: done · Model: Fable 5.1 · Effort: max · Started: 2026-10-17 17:40 · Wall time: ~1:00 (exploration + plan + scaffolding)

## Goal

Create the branch, `PLAN.md`, `AGENTS.md`, `CLAUDE.md`, the journal, and the notes file, and write down everything the exploration established so that T1 onward can run on cheaper models without re-deriving it.

## Setup

- Branch / commit: `perf-comm-ple` created from `source-build-prefix-cache` @ `27c144b`
- No boot in this task. GPUs were idle (15-91 MiB used), no server on `localhost:8090` or `host.docker.internal:8090` at 17:58.

## What I did

```
git checkout -b perf-comm-ple
# wrote PLAN.md, AGENTS.md, CLAUDE.md (@AGENTS.md), journal/{INDEX,TEMPLATE,00-scaffolding}.md,
# vllm-w4a16/results/perf-2026-10-notes.md; committed with the brief and current_benchmark.md
```

Read-only exploration behind the plan (all commands are repeatable): `README.md`, `vllm-w4a16/README.md`, `current_benchmark.md`, `bench-host.py`, `run-host-docker.sh`, `host-run-prefix-20261017-173506.log`, every file under `vllm-w4a16/{launch,patch,tools,research,results}`, `~/models/qwen38fn-w4a16-fp8ple/{config.json,README.md,runtime/**}`, the three profiler summaries under `.cache/ple/prof-*/profiler_out_0.txt`, `docker logs vllm_qwen38fn_3090` (the exited R12 container), and the vLLM source inside the image via `docker run --rm --entrypoint bash qwen38fn-vllm:8a728663 -c 'sed -n ...'`.

## Findings (the hand-over)

### Box and sandbox facts (verified 2026-10-17 17:58)

- 4x RTX 3090, PCIe gen4 x16 each (links idle at gen1 when the GPUs sleep; they renegotiate under load), driver 595.91.07 **open kernel module**, CUDA 13.2. `nvidia-smi topo -p2p r` and `-p2p w`: `CNS` (chipset not supported) for every pair, so there is no GPU-to-GPU P2P; NCCL's intra-node transport is host shared memory for every pair. GPU2-GPU3 share a host bridge (`PHB`), the others are `NODE`.
- CPU AMD EPYC 7542 (32c/64t, one NUMA node), 125 GiB RAM of which 105 GB was page cache (the model files are warm), THP `madvise`, no hugepages. NVMe under dm-crypt + LVM, 1.8 TB free.
- Sandbox: rootless podman 5.4.2 (`docker` is an alias), cgroup cpu `max`, pids 2048, memlock 15.7 GiB (`ulimit -l` = 16468616 KB), `/dev/shm` 32 GB, `/home/dev/models` mounted read-only, `/workspace` read-write. `host.docker.internal` resolves to 192.168.100.2.
- Image `qwen38fn-vllm:8a728663` (id `e4b068fe5cfa`, 13.4 GB, built 14:18): vLLM 0.28.1rc1.dev388+g8a728663c at `/usr/local/lib/python3.12/dist-packages/vllm`, torch 2.13.0+cu130, triton 3.7.1, transformers 5.20.1, NCCL 2.29.7 (`nvidia/nccl/lib/libnccl.so.2`), flashinfer 0.6.18, `torchrun`, `nvcc` 13.0, `g++`, `ninja`, `cuda-python` (`cuda.bindings`); no cupy, no nsys.
- The Qwen4Exp model package is `vllm/models/qwen4_exp/{common,nvidia,amd}/` (not `model_executor/models`); the launcher's `MP` points at the `nvidia` variant. `nvidia/low_latency_gemm.py` has skinny-GEMM plans for sm90/sm100 only; on sm86 the standard cuBLAS path runs.
- The last sandbox boot (R12, 16:05-16:09): weights 19.01 GiB per card, initial free 23.13 GiB (GPU2 23.05), KV reserved 2.24 GiB (pinned), CUDA graphs 0.23 GiB, weight load 74 s, `init engine` 117 s, ready about 255 s after launch. Allocator warnings `expandable_segments: memory mapping failed with OOM ... 20971520 bytes` during loading are benign (transient).
- Async scheduling is on by default for this lane: `vllm/config/vllm.py:1270-1320` enables it unless the speculative method is outside `EagleModelTypes`, and `MTPModelTypes` (which `EagleModelTypes` includes) contains `"mtp"` (`vllm/config/speculative.py:56-67`). The boot log shows no "Async scheduling not supported" warning.
- The first requests after a boot JIT-compile Triton kernels (`layer_norm_fwd_kernel`, `_qsa_sparse_paged_gqa_splitk_kernel`, `_compute_local_logits_stats_kernel`, `_rejection_kernel`, `_resample_kernel`, `_qsa_merge_splitk_kernel`): the first count-to-100 run after boot reads 82-86 tok/s instead of 143. **Warm up with a short prompt, a count-to-100 and one long prompt before measuring.**
- `generation_config.json` overrides the sampling defaults (temperature 1.0, top_k 20, top_p 0.95); every bench passes `temperature: 0` explicitly, keep doing that.
- `tools/count100_3090.sh` and `tools/prefix_cache_3090.py` default to the Tailscale IP `100.113.64.18`; always pass `URL=http://localhost:8090`.
- `current_benchmark.md` is partly contaminated (its 1K/16K rows and decode block overlapped `run-host-docker.sh`'s tests; only the 64K and 128K rows ran alone). Nothing may talk to the server during a measurement.

### The lane's per-step anatomy (from the R8/R9 traces, rank 0)

Per prefill step, divided from the profiler totals (prof-pc1 = caching on, deep, 15 steps of 1664 tokens; prof-pc1-shallow = caching on, first 24 steps of the 84K prompt; prof-pc0 = caching off, 9 steps of 2048):

| Component | deep (per 1664 tok) | shallow (per 1664 tok) | caching off (per 2048 tok) | Notes |
|---|---:|---:|---:|---:|
| Step span (`execute_context_1(...)`) | 948 ms | 1,165 ms | 1,148 ms | wall of the step on the GPU timeline |
| GPU busy (self CUDA total minus decode/partial steps) | ~755 ms | ~776 ms | ~924 ms | so the GPU is **idle 190-390 ms per step (20-33%)**, host-bound |
| NCCL `AllReduce_Sum_bf16_RING_LL` | 475 ms (107 `vllm::all_reduce` ops) | 495 ms (104 ops) | 620 ms (112 ops) | 3.4-4.4 ms per 8.5-10.5 MB message = ~2.5 GB/s; **the single largest item** |
| NCCL `AllGather_RING_LL` | 15 ms (4-5 ops) | 15 ms | 19 ms | |
| `aten::mm` (cuBLAS/cutlass bf16 GEMMs: HC mixes, GDN/QSA projections, draft) | 96 ms | 97 ms | 116 ms | |
| QSA attention (`qwen4_exp_qsa_with_output`, 12 layers + draft) | 96 ms (sparse split-k kernel 79, prefill kernel 16) | 87 ms | 125 ms | ~10 TFLOPS achieved |
| QSA indexer top-k (`persistent_topk`) | 21 ms | 12 ms | 28 ms | |
| Marlin W4A16 MoE (`moe_wna16_marlin_gemm`) | 43 ms | 44 ms | 54 ms | |
| HC glue (`hc_combine_norm`, `hc_gate_mix`, `hc_silu`) | 25 ms | 25 ms | 33 ms | |
| GDN chunked scan (`ChunkGatedDeltaRuleFunction` + fused norm op) | ~17 ms | ~17 ms | ~17 ms | small |
| MoE glue (`silu_and_mul`, `moe_sum`, `topk_softmax`, align) | ~8 ms | ~9 ms | ~9 ms | |

Per token at depth: NCCL 0.29 ms, GPU compute 0.17 ms, host-bound idle 0.11-0.23 ms = 0.57-0.70 ms per token = 1,430-1,750 tok/s, which matches the measured 1,377-1,399. The all-reduces are on the 2,560-wide block outputs (attention out-projection and MoE combine), two per layer, plus the draft layer and the PLE projection; the hyper-connection mixes run replicated on every rank over the 10,240-wide state (their weights are `ReplicatedLinear`, ~1.3 GiB per card).

Decode (count-to-100, 4 tokens per step with MTP3): the target CUDA graph replay `execute_context_0(0)_generation_1(4)` is 15.1-15.5 ms; the observed step is ~21 ms (189 tok/s = 4 tokens per 21 ms). Acceptance from the R12 log: count-like request mean acceptance length 3.39 (positions 0.885 / 0.782 / 0.721); two real prompts 2.16 and 2.37 (positions 0.58 / 0.375 / 0.205 and 0.657 / 0.418 / 0.299). So prose at 110 tok/s is ~2.3 accepted tokens per 21 ms step: for prose and code the step time is the lever; MTP depth beyond 3 adds little on prose (third position already 0.2-0.3) and maybe 0.4-0.6 tokens on code.

### Where the time hides (mechanisms, with code pointers inside the image)

- NCCL protocol: the kernels are `RING_LL` for every message size. LL carries 8 B of data per 16 B line and is meant for small messages; NCCL's tuning picked it anyway on the SHM path. Env knobs to test in T2/T3: `NCCL_PROTO`, `NCCL_ALGO`, `NCCL_SHM_USE_CUDA_MEMCPY`, `NCCL_BUFFSIZE`, `NCCL_NTHREADS`, `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS`, `NCCL_P2P_DISABLE=1` vs the launcher's `NCCL_P2P_LEVEL=NVL` (equivalent on a box with no NVLink). `NCCL_CUMEM_ENABLE=0` is set by the launcher.
- All-reduce dispatch: `vllm/distributed/communication_op.py` -> `GroupCoordinator.all_reduce` (`parallel_state.py:722`, a custom op `torch.ops.vllm.all_reduce`) -> `CudaCommunicator.all_reduce` (`device_communicators/cuda_communicator.py:305-375`): tries FlashInfer, NCCL symmetric memory, quick-reduce, FlashInfer PCIe-IPC, custom all-reduce, torch symm-mem, then `pynccl_comm.all_reduce`. The boot log confirms only `PYNCCL` is active (`SymmMemCommunicator: Device capability 8.6 not supported`; `FlashInfer All Reduce is disabled ... world_size=4`; custom all-reduce disabled by `--disable-custom-all-reduce`, and it needs P2P anyway: `custom_all_reduce.py:87-259`). `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` is the opt-in for the PCIe-IPC backend (`envs.py:1898`); CUDA IPC between devices needs P2P, so expect it to fall back, but it is a one-flag test.
- PLE staged gather: `patch/model_state.py:128-142` calls `_ple_stage_rows` each step; `patch/ple_layer.py:535-555` computes the n-gram ids on the GPU, then `.to("cpu")` (a full sync), then `PLEMmapTable.gather_cpu` (`patch/ple_mmap.py:162-186`): `ids.tolist()`, then one `preadv` per row inside a Python loop, 8 rows per thread-pool task (`QWEN4EXP_PLE_MMAP_CHUNK`), 32 threads (`QWEN4EXP_PLE_MMAP_THREADS`). A 1664-token chunk is 26,624 rows (16 heads x 160 B); a decode step is 64 rows. The staging buffer is `max_num_tokens x 16 x 160 B` = 5.2 MB on each GPU; the forward only views it (`ple_layer.py:496-503`). The author's runtime instead computes the PLE in a separate CPU process and signals the GPU with `cuStreamWaitValue32` inside the graph (`runtime/vllm-overlay/model_executor/layers/ple_offload_layer.py`, `v1/ple_offload/`); our P4 design is simpler (the table in host RAM, a GPU gather by UVA).
- Table geometry: 320 M rows of 160 B (FP8 e4m3, one global scale), 128 shards of equal rows in `model-plefp8-0000{0..9}.safetensors` (51.2 GB); `PLEMmapTable` reads the safetensors headers and opens one fd per file. `hybrid_sources.json` and `runtime/repro.lock.json` carry the provenance.
- Align-mode Mamba allocation (`vllm/v1/core/single_type_kv_cache_manager.py`): `MambaManager` at 1389; `remove_skipped_blocks` at ~1553 frees `blocks[last_state_block_idx]` when it is older than two steps; `_needs_internal_checkpoint` at ~1582; `get_num_blocks_to_allocate` align branch at ~1640; `allocate_new_blocks` align branch at ~1700-1795 (`last_state_block_idx`, null-block padding, speculative block relocation). `reachable_block_mask` at ~1500 implements `--prefix-cache-retention-interval` (`config/cache.py:61`: default 0 = keep only the reachable boundaries; `engine/arg_utils.py:1258`). The leak arithmetic (one page per group per chunk) says something in this path does not free; T4 instruments it. The author's fix lives in a different vLLM version (`runtime/vllm-overlay/v1/core/sched/scheduler.py:382-440`, `_mamba_block_aligned_split`, plus their KV manager, not shipped) so it is a reference, not a patch to copy.
- Pool geometry: block 832 tokens (attention page padded to the Mamba page: `interface.py:953`, `Padding mamba page size by 0.48%`), 2.4e9 B per card = 360 pages, "299,482 tokens" is the attention-equivalent. Each live seat pins 4 groups x (1 + 3 speculative) pages regardless of length.
- Spec decode: `vllm/v1/worker/gpu/spec_decode/` (`speculator.py`, `mtp/`, `eagle/`, `rejection_sampler.py`); `vllm/v1/spec_decode/` has `ngram_proposer.py`, `ngram_proposer_gpu.py`, `suffix_decoding.py`, `custom_class_proposer.py` (any class with a `propose` method, `method="custom_class"`). The draft-group warning (`kv_cache_utils.py:2138-2160`: "no KV cache group could be identified as the draft model's") is why the lane needs `disable_eagle_block_drop` (P11 fixes the identification instead). `Fused multi-step draft decode is not supported by attention backend(s) QWEN4_EXP_EXP_QSA_STATE; falling back to rebuilding attention metadata between draft steps` (`speculator.py:120`) is a known cost of each draft step.
- Pipeline parallelism: `vllm/v1/worker/gpu/pp_utils.py`, `model_runner.py:213` (`use_pp`); the scheduler advances `num_computed_tokens` at schedule time (`v1/core/sched/scheduler.py:1460-1475`, "allowing us to schedule the prefill request again immediately in the next scheduling step"), so one request's chunks can be in flight on two stages; `max_concurrent_batches` at `config/vllm.py:567`. The only hard PP blocker in our patch is the guard at `patch/model_state.py:45-51`; the author's `model_state.py` (`runtime/vllm-overlay/models/qwen3_8_flash_next/nvidia/model_state.py`) uses `get_pp_indices` to run the PLE only on the stage that owns layer 2. `Qwen4ExpModel.forward` (`models/qwen4_exp/nvidia/model.py:474-558`) already handles intermediate tensors.
- Mixed-precision facts for the floors: 3090 bf16 tensor peak 71 TFLOPS with bf16 accumulate, 35.6 with fp32 accumulate (what cuBLAS uses here); 936 GB/s HBM; PCIe 4.0 x16 ~25 GB/s per direction raw, 20-22 practical; 4-GPU ring all-reduce moves 2(n-1)/n = 1.5 x S per rank each way.

### Memory accounting (per card)

Weights 19.01 GiB (routed experts ~14.6, BF16 side layers ~2.8 of which HC ~1.19 replicated, embed+lm_head 0.59 sharded, vision ~0.21, draft ~0.5), KV 2.24 GiB pinned, CUDA graphs 0.23, non-torch (NCCL buffers, context, workspaces) ~0.7, free 0.83-0.94 at idle, 0.24-0.38 at the trough of a long prefill (GPU2 lowest; it carries ~140 MiB of something from the host). Any new GPU buffer or a bigger chunk has to come out of that; the 262,144-token pool is the hard floor (`vllm-w4a16/research/memory-audit-boot14-2026-09-06.md` has the full audit).

### Side findings worth a later look

- The deep steps (0.95 s) are faster than the shallow steps (1.17 s) in the same 84K run; the difference is host-bound idle (190 vs 390 ms), not kernels. Prime suspect: the PLE gather with cold table pages early in the prompt (`POSIX_FADV_RANDOM` is set, so nothing readahead-warms them). `ple_gather_bench.py` in T2 should measure warm and cold.
- `Command Buffer Full` (CUPTI) shows 4,000-6,900 events per profile: the CPU launch queue backs up behind the NCCL kernels; with faster collectives this goes away by itself.
- The reduced-vocabulary draft patch (`patch/mtp_draft_vocab.py`, `QWEN4EXP_DRAFT_VOCAB`) exists but is not mounted by any lane; the draft's `lm_head` is 248,320 x 2,560 bf16 = 1.27 GB read per draft step across the 4 ranks (~0.35 ms per step per rank, three times per decode step).
- `max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings` warning: with MTP3 the scheduler budget per step is 2048 including draft slots; chunk changes interact with this.
- The host run log (`host-run-prefix-20261017-173506.log`) shows the host's `ulimit -l` at 16468616 too (the launcher's `--ulimit memlock=-1:-1` applies inside the container), Docker 29.2.1 with the nvidia runtime, and a clean 255 s boot with the same pool. The host image build took ~15 min there.

## Open questions

- Which component of the 190-390 ms per-step idle is the PLE gather versus scheduler/prepare overhead (T2).
- NCCL's LL latency at decode sizes on this SHM path (T2 microbench) decides whether P2's decode path is worth building.
- Whether `cudaHostRegister` on a file-backed `MAP_SHARED` mapping works on ext4 here, and whether it counts against the sandbox memlock (T5 kill tests).
- Why `remove_skipped_blocks` does not free the previous step's Mamba page in practice (T4).

## Next step

T1 (Sonnet 5.5, effort high): write `vllm-w4a16/tools/scoreboard.py` and `vllm-w4a16/tools/quality_gate.py`, boot the prefix lane, warm up, run both, fill the "before" column in the notes file, add ledger row R13, write `01-baseline.md`. The server was **not** left running by T0.
