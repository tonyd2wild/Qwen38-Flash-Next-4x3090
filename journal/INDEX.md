# Journal index: TTFT and tok/s work on the 4x3090 lane

Plan: [`PLAN.md`](../PLAN.md) · Rules: [`AGENTS.md`](../AGENTS.md) · Numbers: [`vllm-w4a16/results/perf-2026-10-notes.md`](../vllm-w4a16/results/perf-2026-10-notes.md) · Entry template: [`TEMPLATE.md`](TEMPLATE.md)

**GPU owner right now:** nobody. T2 stopped its last server (R18) at 22:02; GPUs at 15-92 MiB. Host server down at 21:32. Last checked: 2026-10-17 22:02 by Opus 5.5 (T2).

**Next task:** re-ranked by T2 (see `02-time-accounting.md` and the notes file): **T5** (PLE gather off the host, the largest measured lever: Fable 5.1 design / xhigh, then Opus 5.5), then T11 (tokenization ~0.3 s per 128K turn, Triton cache not persisted: Sonnet 5.5 / high), T4, and T14 (temperature-0 nondeterminism) before the first candidate that changes numerics. The user picks. On every candidate boot, run `quality_gate.py --compare` first, then `scoreboard.py`.

## Task board

Status values: `todo`, `in progress (model, since)`, `done`, `killed`. Update the row when you claim a task and when you finish it. The one-line result is for the next model; the entry has the detail.

| # | Task | Portfolio | Depends on | Model / effort | Entry | Status | One-line result |
|---|---|---|---|---|---|---|---|
| T0 | Scaffolding: branch, PLAN.md, AGENTS.md, CLAUDE.md, journal, notes file | - | - | Fable 5.1 / max (board says Sonnet; done by the planner) | [00-scaffolding.md](00-scaffolding.md) | done | Scaffolding committed; exploration findings and code pointers recorded in the entry |
| T1 | Baseline: `scoreboard.py`, `quality_gate.py`, boot the prefix lane, record "before" + frozen quality baseline | - | T0 | Sonnet 5.5 / high (run by Opus 5.5 / high, user's choice) | [01-baseline.md](01-baseline.md) | done (Opus 5.5, 19:17-20:32) | Tools in `vllm-w4a16/tools/`; before column R13 (count 186.0, 64K cold 1,396 tok/s, 128K 777 with 2 preemptions); baseline `quality-baseline-2026-10-17.json` (R14). The lane is not bit-reproducible at temperature 0, so the gate thresholds are calibrated on measured noise |
| T2 | Time accounting: `trace_breakdown.py`, `nccl_bench.py`, `ple_gather_bench.py`, incremental-turn breakdown, floor table, portfolio re-rank | - | T1 | Opus 5.5 / high (Sonnet for scripts; all run by Opus 5.5) | [02-time-accounting.md](02-time-accounting.md) | done (Opus 5.5, 20:41-22:15) | Deep prefill step 994 ms: all-reduce 471 (82% of the host-path floor: 4 cards move only ~29 GB/s through host memory, so NCCL tuning cannot help), PLE gather 199 (shallow 413; floor ~1 ms), QSA 116. Tokenization ~0.3 s per 128K turn. Re-rank: P4, P9, P3, P8 up; P1, P2 down. No lane is bit-repeatable at temperature 0 (not spec decode, not prefix caching): T14 |
| T3 | NCCL protocol and transport tuning (reduced by T2: big messages are at the host floor; only a per-size protocol choice for 120-160 KB decode messages is left) | P1 | T2 | Opus 5.5 / medium | 03-p1-nccl-tuning.md | todo | |
| T4 | Align-mode Mamba page leak: instrument, diagnose, fix, 155K validation | P3 | T1 | Opus 5.5 / xhigh (escalate to Fable if the first boot does not explain it) | 04-p3-mamba-leak.md | todo | |
| T5 | PLE table in host RAM with a GPU-side gather (`PLE_MODE=hostpin`) | P4 | T2 | Fable 5.1 design / xhigh, Opus 5.5 implementation / high | 05-p4-ple-hostpin.md | todo | |
| T6 | Host-staged custom all-reduce: design entry, prototype, integration, boots | P2 | T2, T3 | Fable 5.1 design / max, prototype / xhigh; Opus 5.5 integration / high | 06-p2-allreduce-design.md, 07-p2-allreduce.md | todo | |
| T7a | Speculation flags: MTP depth sweep, reduced draft vocabulary | P5 | T1 | Sonnet 5.5 / medium | 08-p5-spec-flags.md | todo | |
| T12 | Draft KV-group identification (drop `disable_eagle_block_drop`) | P11 | T1 | Opus 5.5 / medium | 09-p11-draft-kv-group.md | todo | |
| T8 | Chunk 2048 -> 4096 (only if memory was freed) | P6 | T5 or T6 | Sonnet 5.5 / medium | 10-p6-chunk-4096.md | todo | |
| T9 | torch.compile diagnosis (timebox one boot + 2 h) | P7 | T5 | Opus 5.5 / xhigh | 11-p7-compile.md | todo | |
| T7b | Hybrid MTP + n-gram/suffix proposer | P5 | T7a | Opus 5.5 / high | 12-p5-hybrid-proposer.md | todo | |
| T10 | TP2 x PP2 (condition met by T2: comm is 47% of the step and at the host floor; a 2-rank all-reduce moves a third of the bytes) | P8 | T3, T6 | Fable 5.1 feasibility / xhigh, Opus 5.5 implementation / high | 13-p8-tp2pp2.md | todo | |
| T11 | Fixed overheads for warm and incremental TTFT | P9 | T2 | Sonnet 5.5 / high | 14-p9-fixed-overheads.md | todo | |
| T13 | Recipe: fast lane, `run-host-fast.sh`, final scoreboard and gate, README and ledger, hand-over | - | all kept items | Sonnet 5.5 scripts / high, Opus 5.5 README / high | 15-recipe.md | todo | |
| T14 | Temperature-0 nondeterminism root cause (T2: not cuBLAS, NCCL, Marlin atomics, short-context top-k, spec decode or prefix caching; differs from prompt position 0, up to 0.4-0.7 nat) | - | T2 | Opus 5.5 / xhigh | 16-nondeterminism.md | todo | |
| - | P10 QSA kernel tuning (parked); P13 PCIe P2P read-only diagnostics, then ask the user | P10, P13 | - | Fable 5.1 / Sonnet 5.5 | - | parked | |

## Decisions and asks for the user (append here; batch them)

- 2026-10-17: P13 (PCIe P2P on the GeForce cards) is out of bounds without approval. Read-only diagnostics will be collected during T2 and presented; no host change until the user decides.
- 2026-10-17 (T2): the host side caps the all-reduce. Four cards move ~29 GB/s in total through host memory; a host DRAM copy runs at 21-28 GB/s with 4 of 8 memory channels populated (EDAC: channels 0, 3, 5, 6). NCCL is at 82% of that floor, so no software change gives prefill comm more than ~20%. Questions for the user, no change from here: are the DIMMs running at rated speed, would populating all 8 channels be possible, and P13 (P2P is `CNS` for every pair; NVLink links inactive).
- Hand-over will need one host sitting: `run-host-fast.sh` (baseline prefix lane, then the fast lane; about 1.5 hours; one log back).

## Ledger pointer

Boot rows R13 onward live in the notes file, in the README's style. The README gets the final copy at T13.
