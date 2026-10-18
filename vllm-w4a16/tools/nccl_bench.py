#!/usr/bin/env python3
# NCCL all-reduce and all-gather latency and bandwidth on the 4 GPUs, with no server running, under candidate NCCL_*
# settings. The lane's collectives: decode all-reduces 4-24 tokens x 2560 bf16 (20-120 KB), prefill all-reduces a chunk
# x 2560 bf16 (1664 tokens = 8.5 MB, 2048 = 10.5 MB), ~107 per step. Each setting runs in a fresh container (NCCL reads
# its env at init) with torchrun, 4 processes, bf16 sum, timed two ways: eager back-to-back calls (CUDA events around
# --iters calls, divided) and the same calls replayed from a CUDA graph (how decode runs). busbw = algbw x 2(n-1)/n
# for all-reduce, x (n-1)/n for all-gather. The NCCL INFO log of rank 0 (transport per channel, chosen algorithm and
# protocol) is kept next to the JSON.
# Usage (driver; needs the image and free GPUs; one process owns the GPUs at a time):
#   GPU_ARGS="--device nvidia.com/gpu=all" python3 vllm-w4a16/tools/nccl_bench.py --label T2
#   ... --settings base,simple,ll128          # presets below, or NAME=VAR=val:VAR=val
#   ... --settings 'mine=NCCL_PROTO=Simple:NCCL_MIN_NCHANNELS=8'
# Writes results/nccl-bench-<label>-<time>.json (+ .log with every setting's rank-0 NCCL lines).
# Worker mode (inside the container): torchrun --nproc-per-node 4 nccl_bench.py --worker --out FILE
import argparse, json, os, re, shlex, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
# Launcher env (qwen38fn-w4a16-3090-tp4.sh, NCCL_MODE=nvl) is the base of every preset.
BASE_ENV = {"NCCL_P2P_LEVEL": "NVL", "NCCL_CUMEM_ENABLE": "0"}
PRESETS = {
    "base": {},
    "nop2p": {"NCCL_P2P_DISABLE": "1"},
    "simple": {"NCCL_PROTO": "Simple"},
    "ll128": {"NCCL_PROTO": "LL128"},
    "ll": {"NCCL_PROTO": "LL"},
    "tree": {"NCCL_ALGO": "Tree"},
    "shmcpy": {"NCCL_SHM_USE_CUDA_MEMCPY": "1"},
    "ch8": {"NCCL_MIN_NCHANNELS": "8"},
    "ch16": {"NCCL_MIN_NCHANNELS": "16"},
    "simple_ch8": {"NCCL_PROTO": "Simple", "NCCL_MIN_NCHANNELS": "8"},
}
# bytes: decode 1 / 6 seats x 4 tokens, 32 tokens, prefill 832 / 1664 / 2048 / 4096 / 8192 tokens, x 2560 bf16
SIZES = [4 * 5120, 24 * 5120, 32 * 5120, 256 * 5120, 832 * 5120, 1664 * 5120, 2048 * 5120, 4096 * 5120, 8192 * 5120]
AG_SIZES = [4 * 5120, 8 * 62080 * 2, 1664 * 5120]  # gathered output bytes: hidden rows, draft logits shard, a chunk

ap = argparse.ArgumentParser()
ap.add_argument("--worker", action="store_true")
ap.add_argument("--out")
ap.add_argument("--label", default="run")
ap.add_argument("--settings", default="base")
ap.add_argument("--iters", type=int, default=0, help="calls per measurement (default: by size)")
ap.add_argument("--timeout", type=int, default=300, help="seconds per setting before it counts as hung")
ap.add_argument("--image", default=os.environ.get("IMAGE", "qwen38fn-vllm:8a728663"))
ap.add_argument("--out-dir", default=RESULTS)
a = ap.parse_args()


def worker():
    import torch, torch.distributed as dist
    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
    r, n = dist.get_rank(), dist.get_world_size()
    res = {"world": n, "allreduce": [], "allgather": [], "nccl_version": ".".join(map(str, torch.cuda.nccl.version()))}

    def timeit(fn, iters, graph):
        s = torch.cuda.Stream()
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s):
                for _ in range(3): fn()
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                for _ in range(iters): fn()
            run = g.replay
        else:
            run = lambda: [fn() for _ in range(iters)]
        for _ in range(3): run()
        torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
        times = []
        for _ in range(5):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); run(); e1.record(); torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1e3 / iters)
            dist.barrier()
        return statistics.median(times), min(times)

    for size in SIZES:
        x = torch.zeros((size // 2,), dtype=torch.bfloat16, device="cuda")  # zeros: repeated in-place sums stay finite
        iters = a.iters or (200 if size <= 256 * 5120 else 40 if size <= 2048 * 5120 else 20)
        row = {"bytes": size}
        for mode in ("eager", "graph"):
            try:
                med, mn = timeit(lambda: dist.all_reduce(x), iters, mode == "graph")
                row[mode + "_us"] = med; row[mode + "_min_us"] = mn
            except Exception as ex:  # graph capture of NCCL can fail under some settings; report, do not abort
                row[mode + "_error"] = repr(ex)[:200]; torch.cuda.synchronize()
        best = row.get("eager_us")
        if best:
            row["algbw_GBs"] = size / best / 1e3; row["busbw_GBs"] = row["algbw_GBs"] * 2 * (n - 1) / n
        res["allreduce"].append(row)
        if r == 0: print(json.dumps(row), flush=True)
    for size in AG_SIZES:
        inp = torch.ones((size // 2 // n,), dtype=torch.bfloat16, device="cuda")
        outp = torch.empty((size // 2,), dtype=torch.bfloat16, device="cuda")
        iters = a.iters or 100
        row = {"bytes": size}
        for mode in ("eager", "graph"):
            try:
                med, mn = timeit(lambda: dist.all_gather_into_tensor(outp, inp), iters, mode == "graph")
                row[mode + "_us"] = med
            except Exception as ex:
                row[mode + "_error"] = repr(ex)[:200]; torch.cuda.synchronize()
        if row.get("eager_us"):
            row["algbw_GBs"] = size / row["eager_us"] / 1e3; row["busbw_GBs"] = row["algbw_GBs"] * (n - 1) / n
        res["allgather"].append(row)
        if r == 0: print(json.dumps({"allgather": row}), flush=True)
    # correctness of one fresh all-reduce under this setting (lossless check of the transport, not of summation order)
    y = torch.arange(1024, dtype=torch.float32, device="cuda") * (r + 1)
    dist.all_reduce(y); torch.cuda.synchronize()
    res["fp32_sum_exact"] = bool(torch.equal(y, torch.arange(1024, dtype=torch.float32, device="cuda") * (n * (n + 1) // 2)))
    if r == 0:
        with open(a.out, "w") as f: json.dump(res, f)
    dist.destroy_process_group()


def parse_settings(spec):
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item: continue
        if "=" in item:
            name, rest = item.split("=", 1)
            env = dict(kv.split("=", 1) for kv in rest.split(":"))
            out.append((name, env))
        elif item in PRESETS: out.append((item, PRESETS[item]))
        else: sys.exit(f"unknown setting {item}; presets: {','.join(PRESETS)}")
    return out


def driver():
    gpu_args = shlex.split(os.environ.get("GPU_ARGS", "--gpus all"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, f"nccl-bench-{a.label}-{stamp}")
    allres = {"label": a.label, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "image": a.image, "settings": {}}
    logf = open(base + ".log", "w")
    tmpdir = os.path.join(os.path.dirname(HERE), "..", ".cache", "ple", "nccl-bench"); os.makedirs(tmpdir, exist_ok=True)
    tmpdir = os.path.abspath(tmpdir)
    for name, env in parse_settings(a.settings):
        full = dict(BASE_ENV); full.update(env)
        full.update({"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,TUNING"})
        envargs = sum((["-e", f"{k}={v}"] for k, v in full.items()), [])
        cname = f"nccl-bench-{name}"
        cmd = (["docker", "run", "--rm", "--name", cname] + gpu_args + ["--network", "host", "--ipc", "host", "--ulimit", "memlock=-1:-1",
               "-v", f"{HERE}:/tools:ro", "-v", f"{tmpdir}:/out"] + envargs +
               ["--entrypoint", "torchrun", a.image, "--nproc-per-node", "4", "--nnodes", "1", "--node-rank", "0", "--rdzv-backend", "static",
                "--master-addr", "127.0.0.1", "--master-port", "29511", "/tools/nccl_bench.py", "--worker",
                "--out", f"/out/{name}.json"] + (["--iters", str(a.iters)] if a.iters else []))
        print(f"== {name}: {' '.join(f'{k}={v}' for k, v in full.items() if not k.startswith('NCCL_DEBUG'))}", flush=True)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired as ex:  # NCCL_SHM_USE_CUDA_MEMCPY=1 hung at 1.3 MB on 2026-10-17
            subprocess.run(["docker", "stop", "-t", "5", cname], capture_output=True)
            out = (ex.stdout or b"").decode(errors="replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
            rows = [l for l in out.splitlines() if l.startswith("{")]
            print(f"   HUNG: no progress within {a.timeout} s; completed sizes: {rows}")
            allres["settings"][name] = {"env": env, "error": f"timeout after {a.timeout} s", "completed": rows}
            continue
        logf.write(f"===== {name} {full}\n")
        nccl = [l for l in (p.stdout + p.stderr).splitlines() if "NCCL" in l and ("] 0 " in l or " 0 NCCL" in l
                or re.search(r":\d+:\d+ \[0\]", l))]
        logf.write("\n".join(nccl[:400]) + "\n"); logf.flush()
        try:
            res = json.load(open(os.path.join(tmpdir, f"{name}.json")))
            os.remove(os.path.join(tmpdir, f"{name}.json"))
        except Exception:
            print(f"   FAILED (exit {p.returncode}); last lines:\n" + "\n".join((p.stdout + p.stderr).splitlines()[-15:]))
            allres["settings"][name] = {"env": env, "error": (p.stdout + p.stderr)[-3000:]}
            continue
        res["env"] = env; res["wall_s"] = time.time() - t0
        res["transport"] = sorted(set(m.group(1) for l in nccl for m in [re.search(r"via (\S+)", l)] if m))
        algo = sorted(set(l.partition("NCCL ")[2].strip()[:120] for l in nccl if re.search(r"[Aa]lgo|[Pp]roto", l)))
        res["tuning_lines"] = algo[:40]
        allres["settings"][name] = res
        print(f"   NCCL {res['nccl_version']}  transport {res['transport']}  fp32 exact {res['fp32_sum_exact']}  "
              f"({res['wall_s']:.0f} s)")
        print(f"   {'all-reduce bytes':>18}{'eager us':>10}{'graph us':>10}{'busbw GB/s':>12}")
        for row in res["allreduce"]:
            print(f"   {row['bytes']:>18,}{row.get('eager_us', float('nan')):>10.1f}"
                  f"{row.get('graph_us', float('nan')):>10.1f}{row.get('busbw_GBs', float('nan')):>12.2f}"
                  + (f"  graph: {row['graph_error'][:60]}" if "graph_error" in row else ""))
        for row in res["allgather"]:
            print(f"   {'AG ' + format(row['bytes'], ','):>18}{row.get('eager_us', float('nan')):>10.1f}"
                  f"{row.get('graph_us', float('nan')):>10.1f}{row.get('busbw_GBs', float('nan')):>12.2f}")
        with open(base + ".json", "w") as f: json.dump(allres, f, indent=1)
    with open(base + ".json", "w") as f: json.dump(allres, f, indent=1)
    print(f"wrote {base}.json and .log")


if a.worker: worker()
else: driver()
