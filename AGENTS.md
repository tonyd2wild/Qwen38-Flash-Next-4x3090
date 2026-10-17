# AGENTS.md: read this first

You are working on the vLLM lane that serves Qwen3.8-Flash-Next on this 4x RTX 3090 box. The goal is time to first token and decode tok/s for a coding agent, by multiples, lossless, engine-side only. One task at a time; every task leaves a journal entry so the next model can pick up cold.

## Read order

1. This file.
2. [`PLAN.md`](PLAN.md): context, task board (with the model and effort to use), portfolio, workflow, verification.
3. [`journal/INDEX.md`](journal/INDEX.md): what is done, in progress or killed, who owns the GPUs right now, and the next task.
4. The journal entries of the task you take and of every task it depends on. [`journal/00-scaffolding.md`](journal/00-scaffolding.md) carries the exploration findings and code pointers.
5. The brief, [`fable-initial-prompt.md`](fable-initial-prompt.md), and the lane README, [`vllm-w4a16/README.md`](vllm-w4a16/README.md) (boots R1-R12, caveats), when you need the background.

## Bounds (from the brief; never relax them)

- The endpoint stays drop-in: OpenAI-compatible on port 8090, model id `qwen3.8-flash-next`, `qwen3_xml` tool calling, `qwen3` reasoning parser with thinking off by default, vision, 262,144-token context, 6 seats. Every gain must reach an unmodified client.
- Lossless only: temperature-0 outputs match the frozen baseline except for floating-point reordering (show the first divergent token and the logprob gap); the needle passes at ~30K/~100K/~200K; tool calls parse; a 155K cold prefill does not OOM; a 262,144-token request fits the KV pool.
- In bounds: launch flags, the container, vLLM overlays and patches, a vLLM source build, another engine, your own kernels. Out of bounds without the user's explicit approval: anything on the host (drivers, kernel modules, BIOS, sysctl, clocks, power limits) and any change to the weights (target, MTP draft, n-gram table). Lossless changes to how the weights are stored, laid out, loaded or executed are fine.
- Trading away a feature can only be a separately labeled option, never the new default.
- Keep the tuned and prefix lanes working; new lanes go in new files, in the repo's style. Work on branch `perf-comm-ple`. Never push.

## The box, the sandbox, and how to run things

- You are in a container on the GPU box. Repo at `/workspace`, checkpoint at `~/models/qwen38fn-w4a16-fp8ple` (read-only; do not modify). `docker` here is rootless podman 5.4.2; pass `GPU_ARGS="--device nvidia.com/gpu=all"` to the launchers. The image `qwen38fn-vllm:8a728663` is already in this store.
- One process owns the GPUs at a time: a server or a microbenchmark, never both. Before any boot or GPU microbenchmark: `nvidia-smi --query-gpu=memory.used --format=csv` must show under 1 GB per GPU, and `curl -s -m 3 -o /dev/null -w '%{http_code}' http://localhost:8090/v1/models` must not return 200. The host's own server answers at `http://host.docker.internal:8090`; if it does, stop and ask the user, you cannot stop it from here. Your server answers at `http://localhost:8090`.
- Boot: `GPU_ARGS="--device nvidia.com/gpu=all" bash vllm-w4a16/launch/launch-qwen38fn-w4a16-prefix.sh` (or the lane you are testing). Ready in 250-285 s; wait for `Application startup complete` in `docker logs vllm_qwen38fn_3090`. Stop with `docker stop -t 60 vllm_qwen38fn_3090`. The launcher hash-checks every overlay against the file Python imports and refuses to start if the port still answers.
- Sandbox limits that differ from the host: memlock 15.7 GiB (cannot be raised here; the host applies `--ulimit memlock=-1:-1`), `/dev/shm` 32 GB, 125 GiB RAM, 64 threads. The demo runs on the host under Docker 29.2.1, so final numbers come from the host script.
- Profiling a boot: this image ignores `VLLM_TORCH_PROFILER_DIR` (the boot log says "Unknown vLLM environment variable" and `/start_profile` returns 404). Pass `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/.cache/prof-<name>","torch_profiler_with_stack":false}'` instead. The wrappers overwrite `EXTRA`, so call the base launcher with the lane's knobs; the exact R16 command is in `journal/02-time-accounting.md`. Then `curl -X POST localhost:8090/start_profile` / `stop_profile`; traces land in `.cache/ple/prof-<name>/` (gitignored) and `tools/trace_breakdown.py` sums them. The profiler slows decode by about 14%. The R8/R9/R16 traces are already there.
- Reading the vLLM source of the image: `docker run --rm --entrypoint bash qwen38fn-vllm:8a728663 -c 'sed -n 1,80p /usr/local/lib/python3.12/dist-packages/vllm/<file>'`. The Qwen4Exp model lives at `vllm/models/qwen4_exp/nvidia/` (not under `model_executor/models`).

## Measurement rules

- Both bench scripts read server-wide `/metrics` counters, so nothing else may talk to the server during a measurement. `scoreboard.py` asserts `vllm:num_requests_running` and `vllm:num_requests_waiting` are 0 before and after every run; keep that check in anything you add.
- Keep `bench-host.py`'s metric definitions (decode = completion tokens after the first / time after the first token; cold = fresh random prefix; warm = the same prompt again; incremental = cached conversation plus N new tokens) so numbers stay comparable with `current_benchmark.md` and the README.
- One change per boot, A/B against a same-day baseline. Every boot, failures included, is a ledger row (R13 onward) in `vllm-w4a16/results/perf-2026-10-notes.md`, in the README's row style.
- Run `quality_gate.py --compare` on every candidate before calling it a win.

## Per-task workflow

1. Read the files above. Switch to the model and effort the task board recommends (`/model`, `/effort`); if you are a different model, say so in the entry.
2. Check the box (above). Claim the task in `journal/INDEX.md`: status `in progress`, model, start time.
3. If the task board says the task has a planning stage, write the entry's Design section first (approach, alternatives considered, kill test, test plan) and re-read it against the kill test before writing code.
4. Do the work. Record exact commands, flags and ledger rows as you go, not at the end.
5. Finish the entry from `journal/TEMPLATE.md`: Result (kept or killed, and why), Quality gate, Side findings (anything a later task should know), Open questions, Next step, model, effort, wall time. Killed ideas get the same care; the "why" is what the next model needs.
6. Update `journal/INDEX.md` (status, one-line result, GPU owner) and the notes file (scoreboard columns, ledger, open questions). Commit on `perf-comm-ple` with the entry included, one commit per task where possible, no push. Leave the GPUs free unless `INDEX.md` says the next task reuses the server.

## Journal conventions

- `journal/NN-<slug>.md`, numbered in execution order; the names are pre-assigned in `INDEX.md`. Copy `journal/TEMPLATE.md`.
- Tables use the scoreboard's metric names. Quote numbers with their source (ledger row, log file, trace).
- The journal is the narrative; `vllm-w4a16/results/perf-2026-10-notes.md` is the numbers (scoreboard before/after, ledger, open questions); the README gets the final write-up. Do not keep a fourth copy anywhere.

## Model and effort (summary; the full table with reasons is in `PLAN.md`)

| Task | Model | Effort | Planning stage |
|---|---|---|---|
| T1 baseline tools and run, T7a spec flags, T8 chunk, T13 scripts | Sonnet 5.5 | high (medium for sweeps) | no |
| T2 floors and re-rank, T3 NCCL tuning, T4 Mamba leak, T7b hybrid proposer, T9 compile, T11 fixed overheads, T12 draft KV group, T13 README | Opus 5.5 | high (xhigh for T4, T9; medium for T3, T12) | T4 short, T7b yes, T11 short (tokenization) |
| T5 hostpin design, T6 all-reduce design and prototype, T10 TP2xPP2 feasibility | Fable 5.1 | xhigh (max for the T6 design) | yes |

Escalate one model up when a task's first boot does not explain what you see; never escalate for journal writing.
