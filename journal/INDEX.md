# Journal index: TTFT and tok/s work on the 4x3090 lane

Plan: [`PLAN.md`](../PLAN.md) · Rules: [`AGENTS.md`](../AGENTS.md) · Numbers: [`vllm-w4a16/results/perf-2026-10-notes.md`](../vllm-w4a16/results/perf-2026-10-notes.md) · Entry template: [`TEMPLATE.md`](TEMPLATE.md)

**GPU owner right now:** nobody. No server is running in the sandbox; the host server was down at 17:58 on 2026-10-17. Last checked: 2026-10-17 18:10 by Fable 5.1 (T0).

**Next task:** T1 (baseline) with Sonnet 5.5 at effort high. Start from `01-baseline.md` (copy the template).

## Task board

Status values: `todo`, `in progress (model, since)`, `done`, `killed`. Update the row when you claim a task and when you finish it. The one-line result is for the next model; the entry has the detail.

| # | Task | Portfolio | Depends on | Model / effort | Entry | Status | One-line result |
|---|---|---|---|---|---|---|---|
| T0 | Scaffolding: branch, PLAN.md, AGENTS.md, CLAUDE.md, journal, notes file | - | - | Fable 5.1 / max (board says Sonnet; done by the planner) | [00-scaffolding.md](00-scaffolding.md) | done | Scaffolding committed; exploration findings and code pointers recorded in the entry |
| T1 | Baseline: `scoreboard.py`, `quality_gate.py`, boot the prefix lane, record "before" + frozen quality baseline | - | T0 | Sonnet 5.5 / high | 01-baseline.md | todo | |
| T2 | Time accounting: `trace_breakdown.py`, `nccl_bench.py`, `ple_gather_bench.py`, incremental-turn breakdown, floor table, portfolio re-rank | - | T1 | Opus 5.5 / high (Sonnet for scripts) | 02-time-accounting.md | todo | |
| T3 | NCCL protocol and transport tuning | P1 | T2 | Opus 5.5 / medium | 03-p1-nccl-tuning.md | todo | |
| T4 | Align-mode Mamba page leak: instrument, diagnose, fix, 155K validation | P3 | T1 | Opus 5.5 / xhigh (escalate to Fable if the first boot does not explain it) | 04-p3-mamba-leak.md | todo | |
| T5 | PLE table in host RAM with a GPU-side gather (`PLE_MODE=hostpin`) | P4 | T2 | Fable 5.1 design / xhigh, Opus 5.5 implementation / high | 05-p4-ple-hostpin.md | todo | |
| T6 | Host-staged custom all-reduce: design entry, prototype, integration, boots | P2 | T2, T3 | Fable 5.1 design / max, prototype / xhigh; Opus 5.5 integration / high | 06-p2-allreduce-design.md, 07-p2-allreduce.md | todo | |
| T7a | Speculation flags: MTP depth sweep, reduced draft vocabulary | P5 | T1 | Sonnet 5.5 / medium | 08-p5-spec-flags.md | todo | |
| T12 | Draft KV-group identification (drop `disable_eagle_block_drop`) | P11 | T1 | Opus 5.5 / medium | 09-p11-draft-kv-group.md | todo | |
| T8 | Chunk 2048 -> 4096 (only if memory was freed) | P6 | T5 or T6 | Sonnet 5.5 / medium | 10-p6-chunk-4096.md | todo | |
| T9 | torch.compile diagnosis (timebox one boot + 2 h) | P7 | T5 | Opus 5.5 / xhigh | 11-p7-compile.md | todo | |
| T7b | Hybrid MTP + n-gram/suffix proposer | P5 | T7a | Opus 5.5 / high | 12-p5-hybrid-proposer.md | todo | |
| T10 | TP2 x PP2 (conditional: comm still > 25% of the prefill step after T3/T6) | P8 | T3, T6 | Fable 5.1 feasibility / xhigh, Opus 5.5 implementation / high | 13-p8-tp2pp2.md | todo | |
| T11 | Fixed overheads for warm and incremental TTFT | P9 | T2 | Sonnet 5.5 / high | 14-p9-fixed-overheads.md | todo | |
| T13 | Recipe: fast lane, `run-host-fast.sh`, final scoreboard and gate, README and ledger, hand-over | - | all kept items | Sonnet 5.5 scripts / high, Opus 5.5 README / high | 15-recipe.md | todo | |
| - | P10 QSA kernel tuning (parked); P13 PCIe P2P read-only diagnostics, then ask the user | P10, P13 | - | Fable 5.1 / Sonnet 5.5 | - | parked | |

## Decisions and asks for the user (append here; batch them)

- 2026-10-17: P13 (PCIe P2P on the GeForce cards) is out of bounds without approval. Read-only diagnostics will be collected during T2 and presented; no host change until the user decides.
- Hand-over will need one host sitting: `run-host-fast.sh` (baseline prefix lane, then the fast lane; about 1.5 hours; one log back).

## Ledger pointer

Boot rows R13 onward live in the notes file, in the README's style. The README gets the final copy at T13.
