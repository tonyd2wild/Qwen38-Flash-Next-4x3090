# Journal index: TTFT and tok/s work on the 4x3090 lane

Plan: [`PLAN.md`](../PLAN.md) · Rules: [`AGENTS.md`](../AGENTS.md) · Numbers: [`vllm-w4a16/results/perf-2026-10-notes.md`](../vllm-w4a16/results/perf-2026-10-notes.md) · Entry template: [`TEMPLATE.md`](TEMPLATE.md)

**GPU owner right now:** nobody. T4 stopped its last server (R25) at 02:45 on 2026-10-18; GPUs at 15-92 MiB. Last checked: 2026-10-18 02:45 by Opus 5.5 (T4).

**Next task:** **T14** (temperature-0 nondeterminism root cause), **Opus 5.5 / xhigh**, before the first numerics-changing candidate. Then T10.
- Measure on the hostpin lane (`launch-qwen38fn-w4a16-hostpin.sh`, now defaulting to `TRITON_CACHE=1 TOKSPLIT=1 MAMBA_FIX=1`; ready in 5-7 min). R24 is the scoreboard baseline (`results/scoreboard-R24-20261018-021245.json`); R25 is the shipping config.
- The sharpest probe is the gate's `tool_search` token 9: `tools/tool_search_probe.py` (the R21/R25 repeat script, now in the repo).
  - The baseline picks token 79114 at a 3.75 nat margin.
  - Gate runs pick 28 at -0.38687 against 79114 at -1.13687 on R21 and on R25, the same to 5 digits, with different code (R25 has the Mamba fix, R21 not).
  - R22 and R24 pick 79114.
  - 12 idle repeats right after the gate on the same boot give 79114 with margins 1.25-5.25 (R21) and 5.3-7.8 nat (R25).
  - So the outcome takes a few discrete values, and the same value comes back bit-exactly on different boots. That suggests state carried between requests (page or buffer contents, batch layout, the order in which blocks are reused) rather than random reduction order. A hypothesis, not a finding: R24 ran the same gate sequence and got the other value. The ruled-out list in the T14 row and in the notes file's first open question still applies.
- Known noise in the gate to expect: `tool_write` token 4 (margin 1.25), `vision_halves` token 0 (0.875), prose divergences at margins 0-0.125.
- `tools/mamba_leak_sim.py` is a CPU-only harness for the image's KV manager with this lane's groups, useful if T14 suspects block reuse.

On every candidate boot, run `quality_gate.py --compare` first, then `scoreboard.py`.

## Task board

Status values: `todo`, `in progress (model, since)`, `done`, `killed`. Update the row when you claim a task and when you finish it. The one-line result is for the next model; the entry has the detail.

| # | Task | Portfolio | Depends on | Model / effort | Entry | Status | One-line result |
|---|---|---|---|---|---|---|---|
| T0 | Scaffolding: branch, PLAN.md, AGENTS.md, CLAUDE.md, journal, notes file | - | - | Fable 5.1 / max (board says Sonnet; done by the planner) | [00-scaffolding.md](00-scaffolding.md) | done | Scaffolding committed; exploration findings and code pointers recorded in the entry |
| T1 | Baseline: `scoreboard.py`, `quality_gate.py`, boot the prefix lane, record "before" + frozen quality baseline | - | T0 | Sonnet 5.5 / high (run by Opus 5.5 / high, user's choice) | [01-baseline.md](01-baseline.md) | done (Opus 5.5, 19:17-20:32) | Tools in `vllm-w4a16/tools/`; before column R13 (count 186.0, 64K cold 1,396 tok/s, 128K 777 with 2 preemptions); baseline `quality-baseline-2026-10-17.json` (R14). The lane is not bit-reproducible at temperature 0, so the gate thresholds are calibrated on measured noise |
| T2 | Time accounting: `trace_breakdown.py`, `nccl_bench.py`, `ple_gather_bench.py`, incremental-turn breakdown, floor table, portfolio re-rank | - | T1 | Opus 5.5 / high (Sonnet for scripts; all run by Opus 5.5) | [02-time-accounting.md](02-time-accounting.md) | done (Opus 5.5, 20:41-22:15) | Deep prefill step 994 ms: all-reduce 471 (82% of the host-path floor: 4 cards move only ~29 GB/s through host memory, so NCCL tuning cannot help), PLE gather 199 (shallow 413; floor ~1 ms), QSA 116. Tokenization ~0.3 s per 128K turn. Re-rank: P4, P9, P3, P8 up; P1, P2 down. No lane is bit-repeatable at temperature 0 (not spec decode, not prefix caching): T14 |
| T3 | NCCL protocol and transport tuning (reduced by T2: big messages are at the host floor; only a per-size protocol choice for 120-160 KB decode messages is left) | P1 | T2 | Opus 5.5 / medium | 03-p1-nccl-tuning.md | todo | |
| T4 | Align-mode Mamba page leak: instrument, diagnose, fix, 155K validation | P3 | T1 | Opus 5.5 / xhigh (escalate to Fable if the first boot does not explain it) | [04-p3-mamba-leak.md](04-p3-mamba-leak.md) | done (Opus 5.5 / xhigh, 01:18-02:50) | **Kept** (R23-R25). Under async scheduling the align-mode state free ran one chunk behind and never fired during prefill. One Mamba state block per chunk per group stayed held, and a lone request ran the 384-block pool dry at 101,504 tokens. `MAMBA_FIX=1` (overlay of `v1/core/single_type_kv_cache_manager.py`) keeps each slot listed until the image's own test frees it (6 blocks per group). 0 preemptions up to 200K. 128K cold 1,213 -> 2,149 tok/s, 155K 1,310 -> 2,132, needle 171 -> 94 s. Gate PASS (R24); other rows unchanged |
| T5 | PLE table in host RAM with a GPU-side gather (`PLE_MODE=hostpin`) | P4 | T2 | Fable 5.1 design / xhigh, Opus 5.5 implementation / high (all run by Opus 5.5, user's choice) | [05-p4-ple-hostpin.md](05-p4-ple-hostpin.md) | done (Opus 5.5, 22:21-23:30) | **Kept** (R19). Table copied once into a shared memfd, registered by all 4 ranks (file pages cannot be: no ReadOnly register on the 3090), UVA `index_select` in the forward and decode graph. Prefill 1.53x (64K cold 1,396 -> 2,142 tok/s), decode +10-17%, incremental turns -30-34%, gate PASS. Costs: 48 GiB host RAM pinned, ~95 MiB GPU page tables, boot 8.5 min |
| T6 | Host-staged custom all-reduce: design entry, prototype, integration, boots | P2 | T2, T3 | Fable 5.1 design / max, prototype / xhigh; Opus 5.5 integration / high | 06-p2-allreduce-design.md, 07-p2-allreduce.md | todo | |
| T7a | Speculation flags: MTP depth sweep, reduced draft vocabulary | P5 | T1 | Sonnet 5.5 / medium | 08-p5-spec-flags.md | todo | |
| T12 | Draft KV-group identification (drop `disable_eagle_block_drop`) | P11 | T1 | Opus 5.5 / medium | 09-p11-draft-kv-group.md | todo | |
| T8 | Chunk 2048 -> 4096 (only if memory was freed) | P6 | T5 or T6 | Sonnet 5.5 / medium | 10-p6-chunk-4096.md | todo | |
| T9 | torch.compile diagnosis (timebox one boot + 2 h) | P7 | T5 | Opus 5.5 / xhigh | 11-p7-compile.md | todo | |
| T7b | Hybrid MTP + n-gram/suffix proposer | P5 | T7a | Opus 5.5 / high | 12-p5-hybrid-proposer.md | todo | |
| T10 | TP2 x PP2 (condition met by T2: comm is 47% of the step and at the host floor; a 2-rank all-reduce moves a third of the bytes) | P8 | T3, T6 | Fable 5.1 feasibility / xhigh, Opus 5.5 implementation / high | 13-p8-tp2pp2.md | todo | |
| T11 | Fixed overheads for warm and incremental TTFT: tokenization (~0.3 s per 128K turn), `TRITON_CACHE_DIR`, hostpin boot time (8.5 min) | P9 | T2, T5 | Opus 5.5 / high (was Sonnet; changed after T5, user agreed) | [14-p9-fixed-overheads.md](14-p9-fixed-overheads.md) | done (Opus 5.5, 23:28-01:15) | **Kept** (R20-R22). `TOKSPLIT=1`: exact piece-wise tokenization (cuts before `<\|im_start\|>`, after `\n` before a letter, and between a letter and a space-letter; 0 mismatches over ~359K fuzzed cuts) with a piece cache; warm 128K 0.66 -> 0.38 s, 128K + 4K 2.62 -> 2.27 s. `TRITON_CACHE=1`: first-contact JIT 10.3 -> 0.2 s. Hostpin boot 8.5 -> 5.4 min (leader registers first, rotated chunks, barrier, PLE page cache dropped). Gate PASS; decode and cold unchanged |
| T13 | Recipe: fast lane, `run-host-fast.sh`, final scoreboard and gate, README and ledger, hand-over | - | all kept items | Sonnet 5.5 scripts / high, Opus 5.5 README / high | 15-recipe.md | todo | |
| T14 | Temperature-0 nondeterminism root cause (T2: not cuBLAS, NCCL, Marlin atomics, short-context top-k, spec decode or prefix caching; differs from prompt position 0, up to 0.4-0.7 nat) | - | T2 | Opus 5.5 / xhigh | 16-nondeterminism.md | todo | |
| - | P10 QSA kernel tuning (parked); P13 PCIe P2P read-only diagnostics, then ask the user | P10, P13 | - | Fable 5.1 / Sonnet 5.5 | - | parked | |

## Decisions and asks for the user (append here; batch them)

- 2026-10-17: P13 (PCIe P2P on the GeForce cards) is out of bounds without approval. Read-only diagnostics will be collected during T2 and presented; no host change until the user decides.
- 2026-10-17 (T2): the host side caps the all-reduce. Four cards move ~29 GB/s in total through host memory; a host DRAM copy runs at 21-28 GB/s with 4 of 8 memory channels populated (EDAC: channels 0, 3, 5, 6). NCCL is at 82% of that floor, so no software change gives prefill comm more than ~20%. **User, 2026-10-17: the box has 4 x 32 GiB DIMMs and that will not change.** Treat ~29 GB/s through the host as fixed: prefill comm can only shrink by moving fewer bytes (P8 TP2 x PP2). P13 (P2P is `CNS` for every pair; NVLink links inactive) is still open if it ever becomes relevant.
- 2026-10-17 (T5): the hostpin lane pins ~48 GiB of host RAM while it serves (MemAvailable 119 -> 59 GiB in the sandbox). The host run must leave that free. The sandbox memlock (15.7 GiB) turned out not to limit `cudaHostRegister`, so no host setting is needed.
- Hand-over will need one host sitting: `run-host-fast.sh` (baseline prefix lane, then the fast lane; about 1.5 hours; one log back).

## Ledger pointer

Boot rows R13 onward live in the notes file, in the README's style. The README gets the final copy at T13.
