#!/usr/bin/env python3
# Bitwise run-to-run repeatability of the kernels a short prefill runs, with no server running, to locate the
# temperature-0 nondeterminism found in T1 (journal/01-baseline.md): the same prompt gives different top-1 logprobs on
# the same boot. Each test runs the same op on the same input --reps times and counts distinct output bit patterns:
#   gemm      bf16 torch.mm at the lane's per-rank shapes (M = 29 and 1664), output and input at shifted offsets
#             inside a larger buffer (the caching allocator hands out different addresses per request)
#   gemm_ws   the same with CUBLAS_WORKSPACE_CONFIG=:4096:8 (cuBLAS' deterministic-workspace setting)
#   index_add torch.index_add_ / index_put_(accumulate=True) (atomics), as a positive control
#   allreduce 4-process NCCL all-reduce of random bf16 (the launcher's NCCL env), sizes of a short prefill and a chunk
# Usage (driver; needs the image and free GPUs):
#   GPU_ARGS="--device nvidia.com/gpu=all" python3 vllm-w4a16/tools/det_probe.py
import argparse, json, os, shlex, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--worker", choices=["single", "allreduce"])
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--image", default=os.environ.get("IMAGE", "qwen38fn-vllm:8a728663"))
a = ap.parse_args()
SHAPES = [(29, 2560, 2560), (29, 10240, 320), (29, 320, 10240), (29, 1536, 2560), (1664, 2560, 2560), (1664, 10240, 320)]


def distinct(outs):
    import torch
    ref = outs[0]
    return 1 + len({bytes(o.view(torch.uint8).cpu().numpy().tobytes()) for o in outs[1:] if not torch.equal(o, ref)})


def w_single():
    import torch
    torch.manual_seed(0)
    res = {"cublas_ws": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "gemm": []}
    for m, k, n in SHAPES:
        x0 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda"); w0 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        outs_same, outs_shift = [], []
        for i in range(a.reps):
            outs_same.append(torch.mm(x0, w0.t()))
            # same values at a different address (shift by i*8 elements = 16 B steps, 256 B aligned every 16th)
            big = torch.empty(m * k + 64 * 8, dtype=torch.bfloat16, device="cuda")
            off = (i % 64) * 8
            x = big[off:off + m * k].view(m, k); x.copy_(x0)
            ob = torch.empty(m * n + 64 * 8, dtype=torch.bfloat16, device="cuda")
            o = ob[off:off + m * n].view(m, n)
            torch.mm(x, w0.t(), out=o); outs_shift.append(o.clone())
        torch.cuda.synchronize()
        res["gemm"].append({"M": m, "K": k, "N": n, "distinct_same_addr": distinct(outs_same),
                            "distinct_shifted": distinct(outs_shift),
                            "same_vs_shifted_equal": bool(torch.equal(outs_same[0], outs_shift[0]))})
    src = torch.randn(4096, 2560, device="cuda"); idx = torch.randint(0, 64, (4096,), device="cuda")
    outs = []
    for _ in range(a.reps):
        t = torch.zeros(64, 2560, device="cuda"); t.index_add_(0, idx, src); outs.append(t)
    res["index_add_fp32_distinct"] = distinct(outs)
    print(json.dumps(res))


def w_allreduce():
    import torch, torch.distributed as dist
    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    dist.init_process_group("nccl", device_id=torch.device("cuda", lr))
    r = dist.get_rank(); res = {}
    for toks in (29, 1664):
        g = torch.Generator(device="cuda").manual_seed(100 + r)
        x0 = torch.randn(toks * 2560, dtype=torch.bfloat16, device="cuda", generator=g)
        outs = []
        for i in range(a.reps):
            y = x0.clone(); dist.all_reduce(y); outs.append(y)
        torch.cuda.synchronize()
        res[toks] = distinct(outs)
    if r == 0: print(json.dumps({"allreduce_distinct": res}))
    dist.destroy_process_group()


def driver():
    gpu = shlex.split(os.environ.get("GPU_ARGS", "--gpus all"))
    base = ["docker", "run", "--rm"] + gpu + ["--network", "host", "--ipc", "host", "-v", f"{HERE}:/tools:ro",
            "-e", "NCCL_P2P_LEVEL=NVL", "-e", "NCCL_CUMEM_ENABLE=0"]
    for name, extra, cmd in [
        ("single", [], ["python3", "/tools/det_probe.py", "--worker", "single", "--reps", str(a.reps)]),
        ("single+CUBLAS_WORKSPACE_CONFIG", ["-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8"],
         ["python3", "/tools/det_probe.py", "--worker", "single", "--reps", str(a.reps)]),
        ("allreduce", [], ["torchrun", "--nproc-per-node", "4", "--nnodes", "1", "--node-rank", "0", "--rdzv-backend",
                           "static", "--master-addr", "127.0.0.1", "--master-port", "29512", "/tools/det_probe.py",
                           "--worker", "allreduce", "--reps", str(a.reps)])]:
        p = subprocess.run(base + extra + ["--entrypoint", cmd[0], a.image] + cmd[1:], capture_output=True, text=True,
                           timeout=600)
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        print(f"== {name}: {line[-1] if line else (p.stdout + p.stderr)[-800:]}", flush=True)


if a.worker == "single": w_single()
elif a.worker == "allreduce": w_allreduce()
else: driver()
