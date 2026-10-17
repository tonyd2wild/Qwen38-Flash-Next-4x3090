#!/usr/bin/env python3
# Measured hardware ceilings for the floor table in results/perf-2026-10-notes.md, with no server running:
#   gemm   bf16 x bf16 -> bf16 torch.mm (cuBLAS, fp32 accumulate) TFLOPS at a square 8192 size and at the lane's
#          per-rank prefill shapes (M = 1664 tokens; TP4 slices of the GDN / QSA / shared-expert projections, the
#          replicated hyper-connection low-rank mixes)
#   hbm    device-to-device copy bandwidth (read + write counted) on 1 GiB
#   pcie   pinned host <-> device copy-engine bandwidth (32 MB copies for 3 s per mode: H2D, D2H, both at once), one
#          card alone, then all four cards in the same wall-clock slots (each card has its own x16 link; the host side
#          is shared), per card and in total
# Usage (driver; needs the image and free GPUs):
#   GPU_ARGS="--device nvidia.com/gpu=all" python3 vllm-w4a16/tools/floor_bench.py --label T2
# Writes results/floor-bench-<label>-<time>.json. Worker mode runs inside the container (--worker).
import argparse, json, os, shlex, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
ap = argparse.ArgumentParser()
ap.add_argument("--worker", choices=["gemm", "pcie"])
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--start-at", type=float, default=0.0, help="wall-clock start (aligns concurrent workers)")
ap.add_argument("--out")
ap.add_argument("--pcie-only", action="store_true")
ap.add_argument("--label", default="run")
ap.add_argument("--image", default=os.environ.get("IMAGE", "qwen38fn-vllm:8a728663"))
ap.add_argument("--out-dir", default=RESULTS)
a = ap.parse_args()

# (name, M, K, N) per rank; hidden 2560, GDN qkv 10240 / z 6144 / out 6144, QSA q+gate 12288 / o 6144, shared
# expert 2x640 / 640, HC state 4 x 2560 with low rank 320 (replicated on every rank)
SHAPES = [("square 8192", 8192, 8192, 8192), ("gdn in_proj_qkv", 1664, 2560, 2560), ("gdn in_proj_z", 1664, 2560, 1536),
          ("gdn out_proj", 1664, 1536, 2560), ("qsa q+gate", 1664, 2560, 3072), ("qsa o_proj", 1664, 1536, 2560),
          ("shared gate_up", 1664, 2560, 320), ("shared down", 1664, 160, 2560), ("hc mix down", 1664, 10240, 320),
          ("hc mix up", 1664, 320, 10240), ("decode gdn qkv M=4", 4, 2560, 2560), ("decode hc down M=4", 4, 10240, 320)]


def ev_time(fn, iters, reps=5):
    import torch
    for _ in range(3): fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters): fn()
        e1.record(); torch.cuda.synchronize(); out.append(e0.elapsed_time(e1) / iters)
    return statistics.median(out)  # ms


def w_gemm():
    import torch
    torch.cuda.set_device(a.rank)
    res = {"device": torch.cuda.get_device_name(), "gemm": [], "clock_note": "boost clocks as the driver sets them"}
    for name, m, k, n in SHAPES:
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda"); w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        ms = ev_time(lambda: torch.mm(x, w.t()), 5 if m >= 8192 else 50)
        flop = 2 * m * k * n; byts = 2 * (m * k + k * n + m * n)
        res["gemm"].append({"name": name, "M": m, "K": k, "N": n, "ms": ms, "TFLOPS": flop / ms / 1e9,
                            "GBs": byts / ms / 1e6})
    src = torch.empty(2**30, dtype=torch.uint8, device="cuda"); dst = torch.empty_like(src)
    ms = ev_time(lambda: dst.copy_(src), 10)
    res["hbm_copy_GBs"] = 2 * 2**30 / ms / 1e6
    json.dump(res, open(a.out, "w"))


def w_pcie():
    # Each mode runs for SLOT_S seconds in a wall-clock slot shared by all workers, so concurrent runs overlap fully.
    import torch
    torch.cuda.set_device(a.rank)
    res = {"rank": a.rank, "pcie": []}
    size = 32 * 2**20
    h = torch.empty(size, dtype=torch.uint8, pin_memory=True); h2 = torch.empty_like(h).pin_memory()
    d = torch.empty(size, dtype=torch.uint8, device="cuda"); d2 = torch.empty_like(d)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    for slot, mode in enumerate(("h2d", "d2h", "both")):
        t_start = a.start_at + slot * (SLOT_S + 1.0)
        while time.time() < t_start: time.sleep(0.0005)
        n = 0; t0 = time.time()
        while time.time() - t0 < SLOT_S:
            for _ in range(8):
                if mode in ("h2d", "both"):
                    with torch.cuda.stream(s1): d.copy_(h, non_blocking=True)
                if mode in ("d2h", "both"):
                    with torch.cuda.stream(s2): h2.copy_(d2, non_blocking=True)
            torch.cuda.synchronize(); n += 8
        dt = time.time() - t0
        res["pcie"].append({"mode": mode, "GBs_per_direction": n * size / dt / 1e9})
    json.dump(res, open(a.out, "w"))


SLOT_S = 3.0


def driver():
    gpu_args = shlex.split(os.environ.get("GPU_ARGS", "--gpus all"))
    tmp = os.path.abspath(os.path.join(os.path.dirname(HERE), "..", ".cache", "ple", "floor-bench")); os.makedirs(tmp, exist_ok=True)
    def run(mode, ranks):
        start = time.time() + 25  # container start + CUDA init before the aligned start
        cmd = ["docker", "run", "--rm"] + gpu_args + ["--ipc", "host", "--ulimit", "memlock=-1:-1", "-v", f"{HERE}:/tools:ro",
               "-v", f"{tmp}:/out", "--entrypoint", "bash", a.image, "-c",
               " & ".join(f"python3 /tools/floor_bench.py --worker {mode} --rank {r} --start-at {start} --out /out/{mode}{r}.json"
                          for r in ranks) + " & wait"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        outs = []
        for r in ranks:
            f = os.path.join(tmp, f"{mode}{r}.json")
            try: outs.append(json.load(open(f))); os.remove(f)
            except Exception: outs.append({"rank": r, "error": (p.stdout + p.stderr)[-1500:]})
        return outs
    R = {"label": a.label, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    R["gemm"] = run("gemm", [0])[0] if not a.pcie_only else {"gemm": [], "hbm_copy_GBs": 0, "device": "-"}
    g = R["gemm"]
    if "error" in g: print(g["error"]); sys.exit(1)
    print(f"{g['device']}: HBM copy {g['hbm_copy_GBs']:.0f} GB/s")
    for x in g["gemm"]:
        print(f"  {x['name']:<22} {x['M']:>5} x {x['K']:>5} x {x['N']:>5}  {x['ms']*1e3:9.1f} us  {x['TFLOPS']:6.1f} TFLOPS  {x['GBs']:6.0f} GB/s")
    R["pcie_1"] = run("pcie", [0]); R["pcie_4"] = run("pcie", [0, 1, 2, 3])
    for k in ("pcie_1", "pcie_4"):
        for o in R[k]:
            if "error" in o: print(k, o["error"][-500:]); continue
            print(f"  {k} rank {o['rank']}: " + "  ".join(f"{r['mode']} {r['GBs_per_direction']:.1f}" for r in o["pcie"])
                  + "  GB/s per direction (32 MB copies, 3 s per mode)")
        ok = [o for o in R[k] if "error" not in o]
        if ok: print(f"  {k} total: " + "  ".join(f"{m} {sum(o['pcie'][i]['GBs_per_direction'] for o in ok):.1f}"
                                            for i, m in enumerate(("h2d", "d2h", "both"))))
    path = os.path.join(a.out_dir, f"floor-bench-{a.label}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    json.dump(R, open(path, "w"), indent=1); print("wrote", path)


if a.worker == "gemm": w_gemm()
elif a.worker == "pcie": w_pcie()
else: driver()
