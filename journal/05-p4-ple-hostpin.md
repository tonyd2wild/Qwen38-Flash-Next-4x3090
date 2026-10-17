# 05: PLE table in host RAM with a GPU-side gather (T5, portfolio P4)

Status: done · Model: Opus 5.5 (the board says Fable 5.1 / xhigh for the design and Opus 5.5 / high for the implementation; the user ran the whole task on Opus 5.5) · Effort: high · Started: 2026-10-17 22:21 · Wall time: 1:10 (kill tests, one boot)

## Goal

Take the PLE n-gram row gather out of the host's critical path.

- Today every prefill step has one GPU-idle gap: 199 ms on a deep step and 413 ms on a shallow one, while each rank's Python thread pool runs one `preadv` per row.
- Every decode step syncs the GPU to the CPU for the same reason: PLE is 1.3 ms of a 22.4 ms step with 1 stream, 4.6 ms of 41.7 ms with 6 streams.

Success is measured by scoreboard rows:
- cold TTFT and prefill tok/s at 16K/64K/128K;
- incremental turns at depth;
- decode (count, prose, code, 2/4/6 streams), which must not regress.

The quality gate must show the outputs are unchanged: the bytes gathered are the same.

## Hypothesis and kill test

**Mechanism.**
- In a GPU gather of 160-byte rows from host memory through UVA, the ids never leave the GPU. So there is no `.to("cpu")`, no thread pool, and no GPU->CPU sync.
- The kernel runs inside the forward. For decode, that means inside the captured CUDA graph.

**Ceiling.**
- T2 measured a UVA gather of 26,624 rows at 0.43-0.82 ms per GPU with all 4 GPUs at once, and 384 rows at 15 us. That was from a 1-4 GiB pinned `cudaHostAlloc` slice (`results/ple-gather-bench-T2-20261017-212841.json`).
- Expected deep step: 994 -> ~800 ms (+24% prefill tok/s at depth). Shallow step: 1,209 -> ~800 ms (+50%).
- Expected decode: -1.3 ms per step with 1 stream, -4.6 ms with 6 streams, plus whatever async scheduling regains once the step has no sync.

**Kill tests.** All are cheap, with no server running, and they come before any lane code is booted.
- **K1 (feasibility).** Four processes, one per GPU, each map the 10 `model-plefp8-*.safetensors` files read-only and `cudaHostRegister` them (`Portable | Mapped | ReadOnly`). The test runs in the sandbox, where memlock is 15.7 GiB and the table is 51.2 GB. It passes if:
  - every registration succeeds;
  - the GPU memory cost of the mapping is at most 100 MiB per card (free memory at the trough of a long prefill is 0.24-0.38 GiB, so more than that needs a decision);
  - host RAM holds one copy, not four (`MemAvailable` drops by ~51 GB once).
- **K2 (speed).** A UVA gather of random ids over the whole 320M-row table, 4 GPUs at once, with bytes equal to `PLEMmapTable.gather_cpu`. It passes if 26,624 rows take under 10 ms per GPU (today 199-413 ms of idle) and 384 rows under 0.1 ms. The risk being tested is the GPU TLB: random 160 B reads over 51 GB of 4 KiB pages.
- **K3 (the boot).** It passes if deep prefill gains at least 15%. That means:
  - R13's 64K cold row (1,396 tok/s) goes to 1,600 tok/s or more;
  - decode count-to-100 is no worse than R13's 186 by more than run-to-run noise;
  - free GPU memory during the 155K cold prefill stays above 0.15 GiB per card.

If K1 fails on the sandbox's memlock but the mechanism works, the full-size variant can only be validated on the host. The Design below says what runs here in that case. If K2 shows more than 50 ms, P4 hostpin is killed and the unpinned fallback (Alternative B) is next.

## Setup

- Branch / commit: `perf-comm-ple` @ `95c8dc2`
- Lane and flags: the prefix lane (`launch-qwen38fn-w4a16-prefix.sh` knobs) with `PLE_MODE=hostpin` in place of `staged`; image `qwen38fn-vllm:8a728663`
- Ledger rows: R19 (one boot); kill tests K1/K2 ran with no server
- Box state before: GPUs free (15-91 MiB), host server down (22:21), page cache 104 GB of 125 GiB, PLE table ~17% resident at T2

## Design

> **Outcome note, added after K1.** Approach A failed K1: the 3090 has no `cudaHostRegisterReadOnly`, so a read-only file mapping cannot be registered. The implementation is the fallback A'' below: one shared memfd, filled by TP rank 0. Everything else in this section held: the UVA gather inside the forward, no staging, the launcher case and the wrapper. Gather with `index_select`, because the memfd copy makes the rows contiguous and 8-byte aligned. The Design text is left as written before coding.

### Approach (A): register the checkpoint's own file pages, gather by UVA

1. **Map and pin.** Each rank `mmap`s the 10 table files read-only (`MAP_SHARED`), populates the pages (`madvise(MADV_POPULATE_READ)`), and registers each mapping with `cudaHostRegister(Portable | Mapped | ReadOnly)`.
   - The pinned pages are the page cache's pages of the checkpoint files. The four ranks pin the same physical pages: one 51.2 GB copy in RAM, no copy step, no `/dev/shm` file. `/dev/shm` is 32 GB in the sandbox, too small for the table.
   - Pinning also makes the table resident for as long as the server runs, which cures the shallow-step penalty (T2: only 17% of the table was in the page cache).
2. **Address table.** The device address of row 0 of each of the 128 shards goes into a GPU int64 tensor. `cudaHostGetDevicePointer` gives the device address. With UVA on Linux x86 it equals the host pointer, and the code checks that.
   - The shards are not 4 KiB-aligned inside the files, and the file sizes are odd (5,200,027,199 bytes), so the headers are not padded. A row can therefore start at any byte. The kernel reads bytes, not words.
3. **Gather kernel** (Triton, in a new `patch/ple_hostpin.py`).
   - It maps each id to a shard and a local row: `shard = id // shard_rows`, `local = id - shard * shard_rows`.
   - It loads the 160 bytes through the shard's base pointer (an int64 cast to a pointer).
   - It writes them into the uint8 output the existing code already expects.
   - It is pure GPU work, so it is legal inside CUDA graph capture. The pointers are stable because the pages are pinned.
4. **Where it runs.** The stock (non-staged) mmap path of `patch/ple_layer.py` already computes the ids on the GPU inside `forward` and calls the custom op `qwen4_exp_ple_mmap_gather` -> `_ple_mmap_gather_into`. Hostpin adds one branch at the top of `_ple_mmap_gather_into`: if the hostpin table exists, launch the kernel and return. The table is built eagerly in `load_weights`, next to the existing `PLEMmapTable` construction.
   - Nothing else changes: no staging buffer, no model-state overlay (with `QWEN4EXP_PLE_STAGED` unset, the stock `model_state.py` is used).
   - The decode graph captures the id computation and the gather together.
   - The prefill forward runs both eagerly with no sync.
5. **Launcher.**
   - `PLE_MODE=hostpin` is a new case in `qwen38fn-w4a16-3090-tp4.sh`: `QWEN4EXP_PLE_MMAP=1`, `QWEN4EXP_PLE_HOSTPIN=1`, and mounts of `ple_layer.py`, `ple_mmap.py` and `ple_hostpin.py`. The hash check covers the new mount automatically.
   - A new wrapper `launch-qwen38fn-w4a16-hostpin.sh` = the prefix lane's knobs with `PLE_MODE=hostpin`.
   - The tuned and prefix wrappers are not touched. The guarded branch in `ple_layer.py` is inert unless `QWEN4EXP_PLE_HOSTPIN=1`, and `ple_hostpin.py` is only imported then.

### Alternatives considered

- **(A') A private pinned copy per rank** (`cudaHostAlloc`). 4 x 51.2 GB does not fit in 125 GiB. Dead.
- **(A'') One shared anonymous copy:** a `memfd` filled once, opened by the other ranks through `/proc/<pid>/fd`, registered per rank. It costs 51 GB of RAM on top of the evictable page cache, plus a cross-process hand-off at load. It is the fallback if K1 shows that file-backed pages cannot be pinned (a `FOLL_LONGTERM` refusal on the filesystem, or a driver refusal).
- **(D) A rank-split pinned table:** each rank pins its own quarter (12.8 GB, under the sandbox memlock), gathers its own rows, zeroes the rest, and a byte-wise all-reduce fills them in (the existing `QWEN4EXP_PLE_RESIDENT` arithmetic, moved to host memory).
  - It always fits in the sandbox.
  - But it adds a collective to every step: 4.26 MB per prefill chunk is ~2.2 ms through the host floor, and a decode step pays one more small all-reduce (40-100 us).
  - It is the fallback if K1 fails only because of memlock, and it is the variant the sandbox can validate in that case.
- **(B) A resident unpinned mapping with a C-speed CPU gather:** `numpy.take` from a touched mapping measured 1.65 ms for 26,624 rows (T2). It needs no pinning and no GPU mapping, but:
  - it keeps the GPU->CPU sync in every step, which blocks async scheduling in decode;
  - it cannot keep the table resident under memory pressure without `mlock` (memlock again);
  - a page fault in the gather path serializes on `mmap_lock` (the reason `ple_mmap.py` moved to `preadv`).
  
  It is the fallback if K2 fails.
- **(E) The checkpoint author's design:** a separate CPU process computes the PLE and signals the GPU with `cuStreamWaitValue32` inside the graph. It is more machinery, for a gain that UVA already captures. Not pursued.

### Failure modes and how the code handles them

- **Registration fails** (memlock, driver, filesystem). Raise at load with the CUDA error string. The lane does not silently fall back to staged: a fallback would make an A/B lie.
- **Memory.** Pinning 51.2 GB of page cache leaves ~70 GB for everything else.
  - The 76 GB of target weights will no longer all fit in the page cache across boots, so later boots may read some weights from NVMe.
  - The boot time goes in the ledger.
  - The weight-loader order reads `model-0000*` before `model-plefp8-*`, so a boot's own weights are loaded before the table is pinned.
- **GPU page tables.** Mapping 51.2 GB of 4 KiB host pages costs GPU memory for page tables, which K1 measures. If it is above the budget, register only the shards that are in use. That doesn't work: all shards are used, since ids are hashes. The fallback is D, which maps 12.8 GB per GPU.
- **Shutdown.** Process exit releases the registrations. There is no explicit unregister, matching the existing pinned staging buffers.

### Test plan

1. **Code.** `ple_hostpin.py` (table + kernel), the guarded branch in `ple_layer.py`, the launcher case and the wrapper. `py_compile` the Python files and `bash -n` the shell files. Diff the `docker run` lines of the prefix and tuned lanes with the new knob unset.
2. **K1 and K2.** A new `filepin` mode in `tools/ple_gather_bench.py`. It imports the real `ple_hostpin.py` and measures:
   - registration time, success and VRAM cost per card, with 1 process and then 4;
   - `MemAvailable` before and after;
   - the gather time over the full table, compared with T2's `hostpin` slice numbers;
   - bytes equal to `gather_cpu` for 4,096 random ids per rank.
3. **K3, boot R19.** Run `launch-qwen38fn-w4a16-hostpin.sh`:
   - boot log: the overlays, the registration log line, pool 299,482, free memory per card;
   - `quality_gate.py --compare` against `quality-baseline-2026-10-17.json`;
   - `scoreboard.py --label R19`, compared with R13 (same day).
   - One change only: `PLE_MODE` staged -> hostpin.
4. **If K3 passes:** keep it as a lane knob for T13's fast lane. If prefill gains but decode does not, record the decode step from a trace (R16 recipe) to see whether async scheduling is now overlapping.

**Re-read against the kill test** (done before coding).
- Every kill test is observable without a boot except K3, and K1/K2 use the code the lane will import, not a separate prototype.
- The one sandbox/host difference that can flip the verdict, memlock, has a defined outcome:
  - if K1 fails only on memlock, D is the sandbox variant;
  - A is reported as host-only, and the journal says so.
- Lossless holds by construction (the same bytes reach the same consumer) and is checked bytewise in K2 and by the gate in K3.

## What I did

1. **Approach A first** (file-backed pages, `patch/ple_hostpin.py` v1 with a Triton byte-gather kernel), plus a `filepin` mode in `tools/ple_gather_bench.py` that imports the patch file the way the lane does.
   ```bash
   GPU_ARGS="--device nvidia.com/gpu=all" python3 vllm-w4a16/tools/ple_gather_bench.py --label T5-filepin --modes resident,filepin
   ```
   **K1 failed for A:** `cudaHostRegister` on the read-only file mappings returned 801 `operation not supported` (`results/ple-gather-bench-T5-filepin-20261017-222905.json`).
2. **A registration probe** (a scratch script, 64 MiB regions, in the image), to find out what the driver accepts:

   | Attribute or mapping | Result |
   |---|---|
   | `cudaDevAttrHostRegisterReadOnlySupported` | **0** |
   | `cudaDevAttrHostRegisterSupported` | 1 |
   | `CanUseHostPointerForRegisteredMem` | 1 |
   | `PageableMemoryAccess` | 1 |
   | read-only file mapping (shared or private), any flags | fails: `invalid argument` without ReadOnly, 801 with it |
   | private writable file mapping | registers, but pinning breaks copy-on-write, so each rank would hold its own copy |
   | anonymous memory | registers |
   | shared `memfd` | registers |

   So A is impossible on this GPU, and the design's fallback A'' (one shared memfd) became the implementation.
3. **`patch/ple_hostpin.py` v2** (the version that ships):
   - TP rank 0 creates a `memfd`, sizes it to 51.2 GB (`rows_total x 160`) and fills it with 64 MiB `preadv`s on 8 threads, row-contiguous.
   - The other ranks receive `(pid, fd)` through `GroupCoordinator.broadcast_object` and open `/proc/<pid>/fd/<fd>`.
   - Every rank maps the buffer `MAP_SHARED` and registers it (`Portable | Mapped`) in 4 GiB pieces, each retried up to 5 times on a transient failure.
   - It checks that the device pointer equals the host pointer.
   - The gather is `torch.index_select` over a `__cuda_array_interface__` int64 view `[320M, 20]` of the pinned buffer. The rows are 8-byte aligned now, so the Triton byte kernel went away.

   Two bench runs found bugs in the bench and the CAI view, not in the mechanism:
   - The view's read-only flag must be False.
   - In one run three of four concurrent full-size registrations failed with `invalid argument` (`ple-gather-bench-T5-filepin-20261017-223138.json`). A probe of four processes sharing a 4/20/47.7 GiB memfd, concurrent and sequential, whole and in 4 GiB chunks, registered every time. The failure is transient, so the chunked retry went in.
4. **The lane wiring:**
   - `patch/ple_layer.py` gets `_ple_hostpin_enabled()` (`QWEN4EXP_PLE_HOSTPIN=1`), one branch at the top of `_ple_mmap_gather_into`, and the table's construction in `load_weights`.
   - `launch/qwen38fn-w4a16-3090-tp4.sh` gets a `PLE_MODE=hostpin` case (mounts `ple_layer.py`, `ple_mmap.py`, `ple_hostpin.py`; no `model_state.py`, no `QWEN4EXP_PLE_STAGED`).
   - The new `launch/launch-qwen38fn-w4a16-hostpin.sh` = the prefix lane's knobs with `PLE_MODE=hostpin`.
5. **K1/K2 on v2:**
   ```bash
   GPU_ARGS="--device nvidia.com/gpu=all" python3 vllm-w4a16/tools/ple_gather_bench.py --label T5-filepin --modes filepin
   ```
   `results/ple-gather-bench-T5-filepin-20261017-224528.json`. Both pass (Measurements).
6. **R19** (one change from R13/R14: `PLE_MODE` staged -> hostpin):
   ```bash
   GPU_ARGS="--device nvidia.com/gpu=all" bash vllm-w4a16/launch/launch-qwen38fn-w4a16-hostpin.sh     # 22:47:22, ready 22:55:57
   URL=http://localhost:8090 python3 vllm-w4a16/tools/quality_gate.py --compare --label R19-hostpin
   URL=http://localhost:8090 python3 vllm-w4a16/tools/scoreboard.py --label R19
   ```
   The boot log is in `.cache/ple/r19-boot.log`.

## Measurements

**Kill tests K1/K2** (`results/ple-gather-bench-T5-filepin-20261017-224528.json`, no server running, the real `patch/ple_hostpin.py`):

| Check | 1 process | 4 processes (one shared buffer) | Pass mark |
|---|---:|---:|---|
| Registration of 47.7 GiB | ok (memlock 15.7 GiB not enforced) | ok on all four, 1 retry | all succeed |
| Setup: fill / register | 13.0 s / 7.1 s | leader 12.5 / 8.7 s; followers register in 19.9 s | - |
| GPU memory for the mapping | 94 MiB | 94 MiB per card | <= 100 MiB |
| MemAvailable | 118.6 -> 69.9 GiB | 117.0 -> 67.1 GiB (one copy, not four) | one copy |
| Bytes against `gather_cpu` (4,098 ids incl. rows 0 and last; eager and graph replay) | equal | equal on every rank | equal |
| Gather 26,624 rows (one 1,664-token chunk), random over 320M rows | 0.879 ms | 0.871-0.880 ms (graph 0.85) | < 10 ms |
| Gather 384 rows (6 seats x 4 tokens) | 0.046 ms eager, 0.016 graph | 0.046-0.051 eager, 0.017-0.021 graph | < 0.1 ms |
| Gather 64 rows | 0.05-0.07 eager, 0.017-0.018 graph | same | - |

The random rows over the full 51 GB table cost no more than T2's 1-4 GiB slice: 0.88 ms against 0.43-0.82. The GPU's TLB is not a problem at these sizes.

**K3: boot R19 against R13** (same day, one change). The scoreboard is `results/scoreboard-R19-20261017-230308.json`; R13 is `results/scoreboard-R13-20261017-192716.json`. R13/R14 boot-to-boot spread was ±2.5% on decode and ±1.5% on cold prefill.

| Metric | Baseline (R13) | This boot (R19) | Delta |
|---|---:|---:|---:|
| Decode, count to 100, median of 5 (tok/s) | 186.0 | 204.0 | +9.7% |
| Decode, prose 512 tokens (tok/s) | 124.2 | 144.9 | +16.7% |
| Decode, code prompt (tok/s) | 163.2 | 180.5 | +10.6% |
| Cold TTFT 1K / prefill tok/s | 0.81 s / 1,274 | 0.50 s / 2,067 | 1.62x |
| Cold TTFT 16K / prefill tok/s | 11.41 s / 1,400 | 7.47 s / 2,135 | 1.53x |
| Cold TTFT 64K / prefill tok/s | 45.78 s / 1,396 | 29.78 s / 2,142 | **1.53x** |
| Cold TTFT 128K / prefill tok/s | 164.16 s / 777 (2 preemptions) | 107.70 s / 1,184 (2 preemptions) | 1.52x |
| Warm TTFT 1K / 16K / 64K / 128K (s) | 0.21 / 0.28 / 0.71 / 0.72 | 0.16 / 0.24 / 0.50 / 0.66 | -0.05 to -0.21 s |
| Decode at 64K / 128K context (tok/s) | 119.2 / 116.8 | 125.1 / 158.1 | +5% / +35% |
| Incremental TTFT, 16K cached + 1K / 4K new (s) | 0.98 / 3.23 | 0.67 / 2.14 | -32% / -34% |
| Incremental TTFT, 64K cached + 4K / 16K new (s) | 3.61 / 12.26 | 2.43 / 8.11 | -33% / -34% |
| Incremental TTFT, 128K cached + 4K / 16K new (s) | 3.73 / 12.49 | 2.62 / 8.35 | -30% / -33% |
| Needle ~200K: correct / TTFT / prefill tok/s | correct / 267 s / 746 (4 preemptions) | correct / 175 s / 1,141 (4 preemptions) | 1.53x |
| 155K cold prefill tok/s, preemptions, min free GiB | 838, 2, 0.68 | 1,278, 2, 0.60 | 1.53x; -80 MiB free (the mapping's page tables) |
| Concurrency 2 / 4 / 6: aggregate tok/s | 159.4 / 224.5 / 261.9 | 159.7 / 279.8 / 301.2 | 0% / +25% / +15% |
| Concurrency 2 / 4 / 6: per-stream decode median | 89.7 / 62.8 / 51.2 | 104.1 / 80.8 / 59.4 | +16% / +29% / +16% |
| Concurrency 2 / 4 / 6: TTFT p50 / p99 (s) | 1.00 / 1.42, 0.84 / 1.58, 1.57 / 2.38 | 0.69 / 3.02, 0.80 / 1.09, 1.04 / 2.13 | p50 lower; one 2-stream outlier (3.02 s) |
| KV pool | 299,482 | 299,482 | = |
| Ready after launch | ~4-4.5 min | 8.5 min | +4 min (see Side findings) |

## Result

**Kept.**
- **Prefill.** Every cold prefill row gains 1.52-1.62x, more than the 1.25x the T2 floor table predicted for a deep step. The staged gather did not only add its own ~200-400 ms per step. Its `.to("cpu")` also forced the host to wait for the end of every step before it could launch the next one, and that serialisation is gone too. A deep 1,664-token step is now ~777 ms (64K at 2,142 tok/s).
- **Decode and turns.** Decode gains 10-17% and up to 35% at 128K context. Incremental and warm turns get 30-34% faster.
- **Correctness.** The bytes reaching the model are identical, and the gate passes within the calibrated noise.
- **Costs.**
  - ~48 GiB of host RAM stays pinned while the server runs.
  - ~95 MiB of GPU memory goes to page tables (the 155K minimum free drops from 0.68 to 0.60 GiB).
  - The boot is 4 minutes longer.

128K and 155K cold are still preemption-bound: 1,184-1,278 tok/s against 2,142 at 64K, with 2 preemptions each. That is T4's job, and it is now worth more in absolute terms.

## Quality gate

`results/quality-compare-R19-hostpin-20261017-225612.json`: **PASS**.
- **Frozen prompts:** 14 IDENTICAL, 6 FP-SIZED; p99 |dlogprob| 0.102 over 1,574 positions (R14 same-boot noise: 0.12).
- **First divergences:**

  | Prompt | Token | Baseline margin |
  |---|---:|---:|
  | `code_sql` | 105 | 0.375 |
  | `prose_explain` | 29 | 0.25 |
  | `prose_summary` | 116 | 0.125 |
  | `prose_email` | 42 | 0.125 |
  | `prose_compare` | 89 | 0.0 |
  | `tool_write` | 4 | 1.25 |

  All are below the 2.0-nat SUSPECT threshold, which T1 calibrated on same-boot repeats of the baseline lane. Every one is the known run-to-run nondeterminism (T14), not hostpin: the gathered bytes are equal by test.
- **Tool calls:** all 5 parse to the expected tool.
- **Thinking:** off by default, and the split works when it is on.
- **Vision:** IDENTICAL.
- **Needles:** 29,958 / 99,726 / 199,393 tokens, all correct.
- **155K cold prefill:** 1,282 tok/s with 2 preemptions, min free 0.66 / 0.66 / 0.60 / 0.66 GiB.
- **Boot facts:** pool 299,482; boot-log facts all as baseline.

## Side findings

- **What the driver accepts.** On this GPU and driver (595.91.07 open module, kernel 7.0): `cudaDevAttrHostRegisterReadOnlySupported` = 0, `PageableMemoryAccess` = 1 (HMM). `cudaHostRegister` accepts anonymous and shared-memfd memory but not read-only file mappings. A writable private file mapping registers, but each pin breaks copy-on-write.
- **Memlock does not limit `cudaHostRegister`.** The 15.7 GiB RLIMIT_MEMLOCK is not enforced on it: 47.7 GiB per process registered in the sandbox. So the sandbox *can* validate full-size pinned designs. The PLAN's worry (P4 risk, "Needs the user" item 3) does not apply, and nothing here depends on the host's `--ulimit memlock=-1`.
- **Shmem huge pages are off** (`shmem_enabled` = never; changing it is a host change). So the memfd uses 4 KiB pages: 12.5M PTEs, ~95 MiB of GPU page tables per card. A driver-allocated host buffer could use 2 MiB pages: `cuMemCreate` with a host location, exported as a POSIX fd and passed between ranks over a Unix socket. That would cut the page tables to under 1 MiB per card. Only worth it if a later task needs that GPU memory (T8 chunk 4096).
- **Registration can fail transiently.** It returned `invalid argument` in one bench run (3 of 4 ranks, full size at once), on 1 of 4 the next time, and on 2 chunks at R19's boot. Isolated probes never failed. Registering in 4 GiB pieces with retries makes it reliable. Each piece is a separate registration, and `index_select` crosses them fine, because adjacent registrations get adjacent device addresses.
- **Boot time.**
  - At boot, `load_weights` for the PLE layer runs at the *start* of weight loading, not the end.
  - The followers' registration took 99 s, against 20 s in the bench, and total weight loading took 193 s against 74 s. The ranks pin while others read weights, and host RAM drops to ~59 GiB available.
  - Cheap fixes to try: build the table after loading (first forward or a post-load hook), or put a TP barrier after registration so no rank reads weights while others pin.
  - Boot time is not a scoreboard row, but it affects the user (T11/T13).
- **What hostpin no longer needs.** `QWEN4EXP_PLE_STAGED` and the `model_state.py` overlay are not used: the stock model state plus the mmap-mode custom op are enough once the gather has no host side. The staging buffer (5.2 MB per card) is gone too. The PLEMmapTable thread pool (48 idle threads) is still created; it is harmless.
- **The other lanes.** They mount the edited `ple_layer.py`, whose new code is inert unless `QWEN4EXP_PLE_HOSTPIN=1`: the branch and the import only run then. I did not boot them in T5; their `docker run` lines are unchanged.
- **Decode at 128K context** gained the most (116.8 -> 158.1 tok/s). That suggests the staged path's per-step sync cost grew with context, perhaps through larger host-side prepare work that async scheduling now overlaps. Worth a trace in T11 if decode at depth matters.

## Open questions

- Is the boot-time cost (+4 min) contention between pinning and weight loading, or memory pressure? See Side findings for the two candidate fixes.
- **Host RAM headroom on the host run.** The host is the same box and has 125 GiB of RAM; anything else resident there (the host's own services, the page cache of the target weights) competes with the 48 GiB pinned.
- Does `PageableMemoryAccess` = 1 (HMM) let a GPU kernel read the file pages *without* registration, with no copy and no pinning? Not tried: page faults on first touch and eviction under memory pressure make it riskier than the memfd. A curiosity, not a plan.

## Next step

**T11** (P9 fixed overheads: tokenization ~0.3 s per 128K turn, `TRITON_CACHE_DIR`, and possibly the hostpin boot time), then T4 (P3; 128K/155K cold are now preemption-bound), then T14, then T10. T13's fast lane takes `PLE_MODE=hostpin` (wrapper `launch-qwen38fn-w4a16-hostpin.sh`). The server was **stopped** at 23:15; GPUs free (15-92 MiB).
