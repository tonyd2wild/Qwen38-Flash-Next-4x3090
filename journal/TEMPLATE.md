# NN: <task title> (T<n>, portfolio P<m>)

Status: in progress | done | killed · Model: <model> · Effort: <level> · Started: <YYYY-MM-DD HH:MM> · Wall time: <h:mm>

## Goal

One or two sentences: what this task is meant to change and how success is measured (which scoreboard rows).

## Hypothesis and kill test

The mechanism we expect, the ceiling we computed, and the cheapest observation that would kill the idea.

## Setup

- Branch / commit: `perf-comm-ple` @ `<sha>`
- Lane and flags: launcher, knobs, `DOCKER_EXTRA`, image
- Ledger rows: R<n> (+ R<n+1> ...)
- Box state before: GPUs free (yes/no), host server down (yes/no), page cache warm (yes/no)

## Design (only for tasks with a planning stage)

Approach, alternatives considered and why they lost, data layout, failure modes, test plan. Re-read against the kill test before coding.

## What I did

Exact commands in order, including boots, microbenchmarks, scripts and their arguments. Enough for another model to reproduce without guessing.

## Measurements

Tables with the scoreboard's metric names. Each number cites its source (ledger row, log file path, trace directory).

| Metric | Baseline (R13) | This boot (R<n>) | Delta |
|---|---:|---:|---:|

## Result

**Kept** or **killed**, and the reasoning in two or three sentences. If killed: what the kill test showed, and whether a variant is worth a later look.

## Quality gate

`quality_gate.py --compare` verdict: first divergent token and logprob gap per prompt, tool calls parsed, needle 30K/100K/200K, 155K cold prefill (free memory per card at the minimum), pool tokens.

## Side findings

Anything a later task should know: measurements made in passing, code pointers (file:line in the image), traps, harness notes.

## Open questions

## Next step

Which task follows and what it should start with. Say whether the server was left running.
