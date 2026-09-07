# More KV pool on 4x3090: upstream knobs (vLLM 8a728663), 2026-09-06

Research pass by a sub-agent reading the vLLM tree at our commit; nothing here was measured on the box until the ladder says so. Per rank (TP4). Tokens at ~130K per GiB with fp8 e5m2 KV.

| # | Knob | Expected gain (est.) | Risk | Evidence |
|---|---|---|---|---|
| 1 | `--kv-cache-memory-bytes N` (launcher `KV_BYTES`), picking a number below the worker's printed "to fully utilize gpu memory" value | up to (1-gmu) x 24 GiB - 150 MiB ~ 1.0 GiB at gmu 0.95; we would claim part of it and keep the rest as headroom | taking it all reproduces the gmu 0.97 crash (512 MiB asked, 310 MiB free) | `vllm/v1/worker/gpu_worker.py:534-556`, `:813-869`; `docs/configuration/optimization.md` |
| 2 | Vision boots: `--mm-processor-kwargs '{"max_pixels":1048576}'` (or `--limit-mm-per-prompt` with width/height) | profiling image 16.7 MP (16,384 encoder tokens) shrinks to ~1 MP; ViT MLP peak alone ~0.55 GiB at the default; 0.3 to 0.8 GiB | images above the cap are downscaled | `vllm/multimodal/encoder_budget.py:104-119`, `vllm/v1/core/encoder_cache_manager.py:333-340`, `vllm/model_executor/models/qwen3_vl.py:957-963`, `vllm/v1/worker/gpu/model_runner.py:862-877` |
| 3 | `--skip-mm-profiling` | as #2 without downscaling | a large image OOMs at runtime | `vllm/config/multimodal.py:220-226` |
| 4 | Fewer `cudagraph_capture_sizes` | 50 to 200 MiB for four fewer graphs | padded decode batches | `vllm/v1/worker/gpu/cudagraph_utils.py:738-848`, `gpu_worker.py:572-617` |
| 5 | `--max-num-batched-tokens 1024` | 100 to 250 MiB (activation peak scales with the chunk) | slower prefill | `vllm/utils/mem_utils.py:318-326`, `vllm/config/vllm.py:1909-1946` |
| 6 | `--mamba-ssm-cache-dtype bfloat16` (checkpoint says fp32) | GDN page 817,152 B to 423,936 B, attention block 1600 to 832, per-request state 121 MB to 63 MB (~58 MB per seat) | precision drift on long recurrences (NemotronH pins fp32 for this reason) | `vllm/model_executor/models/config.py:811-813`, `vllm/model_executor/layers/mamba/mamba_utils.py:98-110, 264-285`, `vllm/platforms/interface.py:871-960` |
| 7 | `num_speculative_tokens` 3 to 1 | mamba slots per request 4 to 2 (-61 MB), smaller QSA ring | lower acceptance gain | `vllm/v1/core/single_type_kv_cache_manager.py:1630-1635`, `vllm/v1/kv_cache_interface.py:922-931` |
| 8 | `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | pool grows by the graph estimate, then the graphs eat the headroom | reproduces the 0.97 crash; avoid | `gpu_worker.py:580-616`, `vllm/envs.py:2139`, issue #45178 |
| 9 | `--mm-encoder-tp-mode data` | negative: replicates the 0.83 GB ViT per rank instead of a quarter; keep `weights` | | `vllm/config/multimodal.py:182-194` |
| 10 | `--enable-prefix-caching` | negative: forces `mamba_cache_mode=align` (2+spec slots per request plus checkpoints); keep off | | `models/config.py:634-657`, `kv_cache_interface.py:928-930` |
| 11 | `--mamba-block-size`, `--block-size` | none: only legal with prefix caching; block size cannot go below the mamba page | | `vllm/config/vllm.py:2858-2869`, `interface.py:943-944` |
| 12 | `--num-gpu-blocks-override` | only overrides the count after profiling; forcing more OOMs | | `kv_cache_utils.py:1073-1080` |
| 13 | `--swap-space` | gone at this commit | | `vllm/engine/arg_utils.py` |
| 14 | `--cpu-offload-gb` | live (UVA weight offload) but streams weights over PCIe each step | decode collapses | `vllm/config/offload.py:23-32` |

Notes:
- Pool = ceil(gmu x 24 GiB) - non_kv_cache_memory - cudagraph estimate (`vllm/v1/worker/utils.py:536-556`, `gpu_worker.py:613-617`, `mem_utils.py:318-326`). Nothing is counted twice; the cudagraph estimate is the only extra reservation; the last 5% is never requested.
- Why 1600 tokens: `_align_hybrid_block_size` sets the attention block so one block covers the larger of the GDN page and the PLE page. GDN at TP4 with 3 draft tokens: conv (10240/4) x (4-1+3) x bf16 = 30,720 B plus SSM (48/4) x 128 x 128 x fp32 = 786,432 B = 817,152 B; attention per token per layer = 2 x 1 KV head x 256 x 1 B (e5m2) = 512 B; 817,152 / 512 = 1596, rounded to 1600, padded 0.25% (matches the log). With prefix caching off `mamba_block_size = max_model_len`, so max-model-len does not change the waste; the bf16 state is the only lever on the page.
- Spec decoding: each request holds 1 + num_speculative_tokens mamba slots, so at MTP 3 about 4 x 37 layers x 819,200 B = 121 MB per request. The draft's own attention layer takes its own KV group.
- Multimodal: `encoder_cache_size = max(max_num_batched_tokens, max_tokens_per_item)`; 16,384 comes from the processor default `longest_edge` 16,777,216 px. Profiling runs the ViT at that size; that peak, not the cache, is what shrinks the pool.
- `expandable_segments:True` is already set; it only reduces virtual fragmentation. 512 MiB asked with 310 MiB free was real exhaustion.

Sources: vllm-project/vllm issues #37121, #45178, #40696; docs.vllm.ai hybrid KV cache manager design; docs.vllm.ai configuration/optimization.
