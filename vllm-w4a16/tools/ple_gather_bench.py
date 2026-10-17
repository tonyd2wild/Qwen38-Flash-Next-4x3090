#!/usr/bin/env python3
# Cost of the PLE n-gram row gather, today's path against a GPU-side gather from pinned host memory (the P4 /
# PLE_MODE=hostpin idea), with no server running. A step gathers ngram_heads (16) rows of 160 B per token, random
# over the 320M-row FP8 table (51.2 GB in model-plefp8-*.safetensors). Today (PLE_MODE=staged) every TP rank does the
# same gather on its own: PLEMmapTable.gather_cpu (patch/ple_mmap.py), one preadv per row on a 32-thread pool, into a
# pinned staging buffer, then one H2D copy. Measured here:
#   staged   gather_cpu + H2D for 64 (decode, 1 seat), 384 (6 seats x 4), 26,624 (a 1664-token chunk) and 32,768
#            (2048) rows; fresh random ids (pages may or may not be cached) and the same ids again (warm); 1 process
#            and --procs processes at once (the 4 ranks do this concurrently in the server)
#   hostpin  a slice of the table (--pin-gib, bounded by memlock: 15.7 GiB in the sandbox) read once into pinned
#            memory, then rows gathered by the GPU through UVA (torch index on a CUDA view of the pinned buffer), ids
#            already on the GPU, no host sync; and the same rows via H2D copy for reference
#   resident the fraction of sampled table pages in the page cache (mincore), which bounds the cold case
#   take     the unpinned alternative: numpy gather (C speed, one call) from a read-only mapping of one shard after
#            touching it, i.e. a resident table without pinning (--modes take runs only this)
#   filepin  the T5 implementation (patch/ple_hostpin.py, imported as the lane imports it): rank 0 copies the whole
#            table into one shared memfd, every process maps it and registers it with cudaHostRegister, then the UVA
#            gather over all 320M rows; setup time, GPU memory the mapping costs, MemAvailable, bytes against
#            gather_cpu, eager and CUDA-graph replay timings; 1 process, then --procs at once sharing one buffer
# Usage (driver; needs the image and free GPUs):
#   GPU_ARGS="--device nvidia.com/gpu=all" python3 vllm-w4a16/tools/ple_gather_bench.py --label T2
# Writes results/ple-gather-bench-<label>-<time>.json. Worker mode runs inside the container (--worker).
import argparse, ctypes, json, mmap, os, random, shlex, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
SIZES = [64, 384, 1664 * 16, 2048 * 16]

ap = argparse.ArgumentParser()
ap.add_argument("--worker", choices=["staged", "hostpin", "resident", "take", "filepin"])
ap.add_argument("--modes", default="resident,staged,hostpin,take")
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--nprocs", type=int, default=1, help="workers in this container (filepin barrier)")
ap.add_argument("--out")
ap.add_argument("--label", default="run")
ap.add_argument("--procs", type=int, default=4, help="concurrent staged gathers (the TP ranks)")
ap.add_argument("--pin-gib", type=float, default=4.0)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--threads", type=int, default=32, help="QWEN4EXP_PLE_MMAP_THREADS of the lane")
ap.add_argument("--image", default=os.environ.get("IMAGE", "qwen38fn-vllm:8a728663"))
ap.add_argument("--model", default=os.path.expanduser("~/models/qwen38fn-w4a16-fp8ple"))
ap.add_argument("--out-dir", default=RESULTS)
a = ap.parse_args()
MODEL_IN = "/models/qwen38fn"
PLE_LAYER = 1  # tensor names say layers.1 (config.json ple_layer_ids is [2])


def table():
    sys.path.insert(0, "/patch")
    from ple_mmap import PLEMmapTable
    return PLEMmapTable(MODEL_IN, PLE_LAYER, threads=a.threads)


def w_staged():
    import torch
    torch.cuda.set_device(a.rank)
    t = table(); rng = random.Random(1000 + a.rank)
    dev_buf = torch.empty((max(SIZES), t.row_bytes), dtype=torch.uint8, device="cuda")
    res = {"rank": a.rank, "rows_total": t.rows_total, "row_bytes": t.row_bytes, "threads": t.threads, "sizes": {}}
    # every rank gathers the same ids at the same moment in the server; use a shared seed per rep
    for n in SIZES:
        fresh, warm = [], []
        for rep in range(a.reps):
            ids = torch.tensor(random.Random(n * 100 + rep).sample(range(t.rows_total), n), dtype=torch.int64)
            for lst in (fresh, warm):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                rows = t.gather_cpu(ids)
                t1 = time.perf_counter()
                dev_buf[:n].copy_(rows, non_blocking=True); torch.cuda.synchronize()
                t2 = time.perf_counter()
                lst.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3))
        res["sizes"][n] = {
            "fresh_gather_ms": statistics.median(x[0] for x in fresh[1:]),
            "warm_gather_ms": statistics.median(x[0] for x in warm[1:]),
            "h2d_ms": statistics.median(x[1] for x in warm[1:]),
            "fresh_all": [round(x[0], 3) for x in fresh], "warm_all": [round(x[0], 3) for x in warm]}
        res["sizes"][n]["warm_us_per_row"] = res["sizes"][n]["warm_gather_ms"] * 1e3 / n
    json.dump(res, open(a.out, "w"))


def w_hostpin():
    import torch
    torch.cuda.set_device(a.rank)
    t = table()
    rows = int(a.pin_gib * 2**30 // t.row_bytes)
    rows = min(rows, t.shard_rows * (t.num_shards))
    t0 = time.perf_counter()
    host = t.read_rows_contiguous(0, rows)  # pinned when the allocation succeeds
    load_s = time.perf_counter() - t0
    res = {"rank": a.rank, "pinned": bool(host.is_pinned()), "rows": rows, "gib": rows * t.row_bytes / 2**30,
           "load_s": load_s, "sizes": {}}
    if not host.is_pinned():
        res["error"] = "pin_memory failed (memlock?)"; json.dump(res, open(a.out, "w")); return
    ptr = host.data_ptr()

    class View:  # a CUDA view of the pinned buffer through UVA (pinned host memory is mapped on UVA systems)
        __cuda_array_interface__ = {"shape": (rows, t.row_bytes // 8), "typestr": "<i8", "data": (ptr, False),
                                    "version": 3, "strides": None}
    gview = torch.as_tensor(View(), device="cuda")
    # correctness: the GPU gather returns the same bytes as the table
    probe = torch.tensor(random.Random(7).sample(range(rows), 64), dtype=torch.int64)
    ref = t.gather_cpu(probe).clone()
    got = gview[probe.cuda()].view(torch.uint8).view(64, t.row_bytes).cpu()
    res["bytes_equal"] = bool(torch.equal(ref, got))
    out = torch.empty((max(SIZES), t.row_bytes // 8), dtype=torch.int64, device="cuda")
    for n in SIZES:
        ids = torch.tensor(random.Random(n).sample(range(rows), n), dtype=torch.int64, device="cuda")
        def gather(): torch.index_select(gview, 0, ids, out=out[:n])
        for _ in range(3): gather()
        torch.cuda.synchronize()
        iters = 50 if n <= 384 else 10
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(a.reps):
            e0.record()
            for _ in range(iters): gather()
            e1.record(); torch.cuda.synchronize(); times.append(e0.elapsed_time(e1) / iters)
        g = statistics.median(times)
        # reference: an H2D copy of the same number of bytes from pinned memory (the staged path's copy)
        src = host[:n]; dst = torch.empty((n, t.row_bytes), dtype=torch.uint8, device="cuda")
        e0.record()
        for _ in range(iters): dst.copy_(src, non_blocking=True)
        e1.record(); torch.cuda.synchronize()
        res["sizes"][n] = {"uva_gather_ms": g, "uva_GBs": n * t.row_bytes / g / 1e6,
                           "h2d_same_bytes_ms": e0.elapsed_time(e1) / iters}
    json.dump(res, open(a.out, "w"))


def w_resident():
    import numpy as np
    t = table()
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    page = mmap.PAGESIZE
    maps = {fd: mmap.mmap(fd, 0, prot=mmap.PROT_READ) for fd in {fd for fd, _ in t.shard_base}}
    addr = {fd: np.frombuffer(m, dtype=np.uint8).__array_interface__["data"][0] for fd, m in maps.items()}
    rng = random.Random(3); vec = ctypes.create_string_buffer(1); hit = tot = 0
    for _ in range(200000):
        i = rng.randrange(t.rows_total); s, l = divmod(i, t.shard_rows); fd, off = t.shard_base[s]
        o = (off + l * t.row_bytes) // page * page
        tot += 1
        if libc.mincore(ctypes.c_void_p(addr[fd] + o), page, vec) == 0 and vec.raw[0] & 1: hit += 1
    json.dump({"sampled_rows": tot, "resident_fraction": hit / tot}, open(a.out, "w"))


def w_take():
    import numpy as np
    t = table(); fd, off = t.shard_base[0]; rows = t.shard_rows
    m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    arr = np.frombuffer(m, dtype=np.uint8, count=rows * t.row_bytes, offset=off).reshape(rows, t.row_bytes)
    t0 = time.perf_counter(); int(arr[::25].sum(dtype=np.uint64)); touch = time.perf_counter() - t0
    res = {"shard_rows": rows, "shard_gb": rows * t.row_bytes / 1e9, "touch_s": touch, "sizes": {}}
    for n in SIZES:
        ids = np.array(random.Random(n).sample(range(rows), n)); outb = np.empty((n, t.row_bytes), np.uint8)
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); np.take(arr, ids, axis=0, out=outb); ts.append(time.perf_counter() - t0)
        res["sizes"][n] = {"take_ms": statistics.median(ts) * 1e3}
    json.dump(res, open(a.out, "w"))


def meminfo():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":"); m[k] = int(v.split()[0]) * 1024
    return m


def w_filepin():
    import torch
    torch.cuda.set_device(a.rank)
    sys.path.insert(0, "/patch")
    from ple_hostpin import PLEHostPinTable
    t = table(); m0 = meminfo()
    res = {"rank": a.rank, "rows_total": t.rows_total, "memavail_before": m0["MemAvailable"], "cached_before": m0["Cached"]}
    share = "/out/filepin-share.json"

    def exchange(payload):  # what the lane does with a broadcast over the TP group, done with a file here
        if payload is not None:
            json.dump(payload, open(share + ".tmp", "w")); os.rename(share + ".tmp", share); return payload
        t0 = time.time()
        while not os.path.exists(share):
            if time.time() - t0 > 1800: raise TimeoutError("no leader")
            time.sleep(0.1)
        return json.load(open(share))
    try:
        hp = PLEHostPinTable(t, leader=a.rank == 0, exchange=exchange if a.nprocs > 1 else None)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"; json.dump(res, open(a.out, "w")); return
    m1 = meminfo()
    res.update(setup_s=hp.setup_s, fill_s=hp.fill_s, register_s=hp.register_s, vram_cost_mib=hp.vram_cost / 2**20,
               pinned_gib=hp.nbytes / 2**30, register_retries=hp.register_retries, memavail_after=m1["MemAvailable"], cached_after=m1["Cached"],
               describe=hp.describe(), sizes={})
    # barrier: every worker in the container has its mapping before anyone times the gather
    open(f"/out/filepin-ready{a.rank}", "w").close(); t0 = time.time()
    while sum(os.path.exists(f"/out/filepin-ready{r}") for r in range(a.nprocs)) < a.nprocs and time.time() - t0 < 900:
        time.sleep(0.05)
    rng = random.Random(11 + a.rank)
    probe = torch.tensor([rng.randrange(t.rows_total) for _ in range(4096)] + [0, t.rows_total - 1], dtype=torch.int64)
    ref = t.gather_cpu(probe).clone()
    got = torch.empty((probe.numel(), t.row_bytes), dtype=torch.uint8, device="cuda")
    hp.gather(probe.cuda(), got); torch.cuda.synchronize()
    res["bytes_equal"] = bool(torch.equal(ref, got.cpu()))
    out = torch.empty((max(SIZES), t.row_bytes), dtype=torch.uint8, device="cuda")
    for n in SIZES:
        idss = [torch.tensor([rng.randrange(t.rows_total) for _ in range(n)], dtype=torch.int64, device="cuda")
                for _ in range(a.reps + 1)]
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for k, ids in enumerate(idss):  # fresh random ids each rep: no GPU cache reuse between reps
            torch.cuda.synchronize(); e0.record(); hp.gather(ids, out[:n]); e1.record(); torch.cuda.synchronize()
            if k: times.append(e0.elapsed_time(e1))
        # the same gather captured in a CUDA graph and replayed (the decode path)
        gids = idss[0].clone(); g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            hp.gather(gids, out[:n]); torch.cuda.synchronize()
            with torch.cuda.graph(g): hp.gather(gids, out[:n])
        gt = []
        for ids in idss[1:]:
            gids.copy_(ids); torch.cuda.synchronize(); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            gt.append(e0.elapsed_time(e1))
        if n <= probe.numel():
            gids.copy_(probe[:n].cuda()); g.replay(); torch.cuda.synchronize(); res.setdefault("graph_bytes_equal", True)
            res["graph_bytes_equal"] &= bool(torch.equal(ref[:n], out[:n].cpu()))
        res["sizes"][n] = {"gather_ms": statistics.median(times), "graph_ms": statistics.median(gt),
                           "GBs": n * t.row_bytes / statistics.median(times) / 1e6, "all": [round(x, 3) for x in times]}
    json.dump(res, open(a.out, "w"))


def driver():
    gpu_args = shlex.split(os.environ.get("GPU_ARGS", "--gpus all"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tmp = os.path.abspath(os.path.join(os.path.dirname(HERE), "..", ".cache", "ple", "ple-bench")); os.makedirs(tmp, exist_ok=True)
    def run(mode, ranks, pin_gib=a.pin_gib):
        for f in [f"filepin-ready{r}" for r in ranks] + ["filepin-share.json"]:
            try: os.remove(os.path.join(tmp, f))
            except FileNotFoundError: pass
        cmd = ["docker", "run", "--rm"] + gpu_args + ["--ipc", "host", "--ulimit", "memlock=-1:-1",
               "-v", f"{a.model}:{MODEL_IN}:ro", "-v", f"{os.path.join(os.path.dirname(HERE), 'patch')}:/patch:ro",
               "-v", f"{HERE}:/tools:ro", "-v", f"{tmp}:/out", "--entrypoint", "bash", a.image, "-c",
               " & ".join(f"python3 /tools/ple_gather_bench.py --worker {mode} --rank {r} --out /out/{mode}{r}.json "
                          f"--reps {a.reps} --threads {a.threads} --pin-gib {pin_gib} --nprocs {len(ranks)}"
                          for r in ranks) + " & wait"]
        t0 = time.time(); p = subprocess.run(cmd, capture_output=True, text=True)
        outs = []
        for r in ranks:
            f = os.path.join(tmp, f"{mode}{r}.json")
            try: outs.append(json.load(open(f))); os.remove(f)
            except Exception: outs.append({"rank": r, "error": (p.stdout + p.stderr)[-1500:]})
        print(f"   ({time.time() - t0:.0f} s)", flush=True)
        return outs
    R = {"label": a.label, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "threads": a.threads}
    modes = a.modes.split(",")
    if "resident" in modes:
        print("== page-cache residency of the table (mincore sample)", flush=True)
        R["resident"] = run("resident", [0])[0]; print("  ", R["resident"])
    if "staged" in modes:
        print("== staged gather, 1 process", flush=True)
        R["staged_1"] = run("staged", [0])
        print(f"== staged gather, {a.procs} processes at once", flush=True)
        R[f"staged_{a.procs}"] = run("staged", list(range(a.procs)))
    if "hostpin" in modes:
        print(f"== hostpin prototype, {a.pin_gib} GiB pinned, 1 process", flush=True)
        R["hostpin_1"] = run("hostpin", [0])
        print(f"== hostpin prototype, {a.procs} processes at once ({a.pin_gib / a.procs:.1f} GiB each)", flush=True)
        R[f"hostpin_{a.procs}"] = run("hostpin", list(range(a.procs)), pin_gib=a.pin_gib / a.procs)  # memlock is shared
    if "take" in modes:
        print("== resident unpinned: numpy take from a touched mapping of one shard", flush=True)
        R["take"] = run("take", [0])[0]
        o = R["take"]
        print("  " + (o.get("error", "")[-400:] if "error" in o else
              f"shard {o['shard_gb']:.2f} GB touched in {o['touch_s']:.1f} s; " +
              "  ".join(f"{n}: {v['take_ms']:.3f} ms" for n, v in o["sizes"].items())))
    if "filepin" in modes:
        for k, ranks in (("filepin_1", [0]), (f"filepin_{a.procs}", list(range(a.procs)))):
            m0 = meminfo(); print(f"== filepin (patch/ple_hostpin.py), {len(ranks)} process(es); MemAvailable "
                                  f"{m0['MemAvailable'] / 2**30:.1f} GiB, Cached {m0['Cached'] / 2**30:.1f} GiB", flush=True)
            R[k] = run("filepin", ranks); m1 = meminfo()
            R[k + "_host_mem"] = {"before": m0, "after": m1}
            for o in R[k]:
                if "error" in o: print(f"  {k} rank {o['rank']}: ERROR {o['error'][-600:]}"); continue
                print(f"  {k} rank {o['rank']}: {o['describe']}; bytes equal {o['bytes_equal']} (graph "
                      f"{o.get('graph_bytes_equal')}); MemAvailable {o['memavail_before'] / 2**30:.1f} -> "
                      f"{o['memavail_after'] / 2**30:.1f} GiB\n    " + "  ".join(
                          f"{n}: {v['gather_ms']:.3f} ms ({v['GBs']:.1f} GB/s), graph {v['graph_ms']:.3f}"
                          for n, v in o["sizes"].items()))
            print(f"  host after: MemAvailable {m1['MemAvailable'] / 2**30:.1f} GiB, Cached {m1['Cached'] / 2**30:.1f} GiB")
    for k in ("staged_1", f"staged_{a.procs}"):
        for o in R.get(k, []):
            if "error" in o: print(k, "rank", o.get("rank"), "ERROR", o["error"][-400:]); continue
            print(f"  {k} rank {o['rank']}: " + "  ".join(
                f"{n}: fresh {v['fresh_gather_ms']:.2f} / warm {v['warm_gather_ms']:.2f} ms (+H2D {v['h2d_ms']:.2f})"
                for n, v in o["sizes"].items()))
    for k in ("hostpin_1", f"hostpin_{a.procs}"):
        for o in R.get(k, []):
            if "error" in o: print(k, "rank", o.get("rank"), "ERROR", o["error"][-400:]); continue
            print(f"  {k} rank {o['rank']}: pinned {o['pinned']} {o['gib']:.1f} GiB in {o['load_s']:.1f} s, "
                  f"bytes equal {o.get('bytes_equal')}  " + "  ".join(
                f"{n}: UVA {v['uva_gather_ms']:.3f} ms ({v['uva_GBs']:.1f} GB/s), H2D {v['h2d_same_bytes_ms']:.3f}"
                for n, v in o["sizes"].items()))
    path = os.path.join(a.out_dir, f"ple-gather-bench-{a.label}-{stamp}.json")
    json.dump(R, open(path, "w"), indent=1); print("wrote", path)


if a.worker == "staged": w_staged()
elif a.worker == "hostpin": w_hostpin()
elif a.worker == "resident": w_resident()
elif a.worker == "take": w_take()
elif a.worker == "filepin": w_filepin()
else: driver()
