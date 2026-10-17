# SPDX-License-Identifier: Apache-2.0
# 2026-10-17 (T5, journal/05-p4-ple-hostpin.md). PLE n-gram table served from pinned host RAM with a GPU-side gather
# (PLE_MODE=hostpin). One TP rank (the leader) copies the table's rows out of the model-plefp8-*.safetensors files
# into one shared memfd, contiguous (row i at byte i * row_bytes); every rank opens that memfd through
# /proc/<leader pid>/fd, maps it shared and registers it with cudaHostRegister (Portable | Mapped). The GPU gathers
# rows through UVA with index_select. One 51.2 GB copy in host RAM for all ranks; the ids never leave the GPU, so there
# is no host sync and the gather is legal inside a CUDA graph.
# Why a memfd and not the files' own page-cache pages: the 3090 reports cudaDevAttrHostRegisterReadOnlySupported = 0,
# a read-only file mapping cannot be registered without that flag (error 801/1), and a writable private mapping would
# be copied page by page on pinning (one copy per rank).
"""Pinned, shared PLE table in host RAM with a UVA row gather."""

from __future__ import annotations

import ctypes
import mmap
import os
import resource
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import torch

_HOST_REGISTER_FLAGS = 0x01 | 0x02  # cudaHostRegisterPortable | cudaHostRegisterMapped
_COPY_BYTES = 64 << 20  # one preadv per 64 MiB of rows while filling
_REGISTER_BYTES = 4 << 30  # cudaHostRegister in 4 GiB pieces, each retried on a transient failure
_REGISTER_TRIES = 5


def _cudart() -> ctypes.CDLL:
    # The libcudart torch already loaded (same runtime state, same current device); dlopen by its path returns it.
    torch.cuda.init()
    with open("/proc/self/maps") as f:
        paths = {line.split()[-1] for line in f if "libcudart.so" in line}
    if not paths:
        raise RuntimeError("PLE hostpin: libcudart is not loaded in this process")
    rt = ctypes.CDLL(sorted(paths)[0])
    rt.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    rt.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    rt.cudaGetErrorString.restype = ctypes.c_char_p
    return rt


def _fill(table, mm: mmap.mmap, threads: int) -> None:
    """Copy every row of the table, in order, into the writable shared mapping mm."""
    rb, sr, mv = table.row_bytes, table.shard_rows, memoryview(mm)
    jobs = []
    for s, (fd, off) in enumerate(table.shard_base):
        rows = min(sr, table.rows_total - s * sr)
        for a in range(0, rows * rb, _COPY_BYTES):
            jobs.append((fd, off + a, s * sr * rb + a, min(_COPY_BYTES, rows * rb - a)))

    def copy(job):
        fd, src, dst, n = job
        got = 0
        while got < n:
            r = os.preadv(fd, [mv[dst + got : dst + n]], src + got)
            if r <= 0:
                raise IOError(f"PLE hostpin: short read at {src + got} ({got} of {n} bytes)")
            got += r

    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="ple-hostpin-fill") as pool:
        list(pool.map(copy, jobs))
    mv.release()


class PLEHostPinTable:
    """The rows of a PLEMmapTable in one shared, pinned host buffer, gathered by the GPU.

    exchange(payload) is called once by every rank with the leader's (pid, fd) on the leader and None elsewhere, and
    must return the leader's payload on every rank (a broadcast; the leader fills the buffer before calling it).
    Without exchange the caller is a lone leader (a microbenchmark).
    """

    def __init__(self, table, *, leader: bool = True, exchange: Callable[[object], object] | None = None,
                 threads: int = 8) -> None:
        self.rows_total = int(table.rows_total)
        self.row_bytes = int(table.row_bytes)
        self.dtype = table.dtype
        self.nbytes = self.rows_total * self.row_bytes
        self.device = torch.device("cuda", torch.cuda.current_device())
        if self.row_bytes % 8:
            raise ValueError(f"PLE hostpin: row of {self.row_bytes} bytes is not a whole number of int64 words")
        rt = _cudart()
        free0 = torch.cuda.mem_get_info(self.device)[0]
        t0 = time.perf_counter()
        self.fill_s = 0.0
        if leader:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
            fd = libc.memfd_create(b"qwen4exp-ple-hostpin", 0)
            if fd < 0:
                raise OSError(ctypes.get_errno(), "PLE hostpin: memfd_create failed")
            os.ftruncate(fd, self.nbytes)
            self._mm = mmap.mmap(fd, self.nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            _fill(table, self._mm, threads)
            self.fill_s = time.perf_counter() - t0
            self._fd = fd  # kept open: the followers open it through /proc/<pid>/fd
            payload = (os.getpid(), fd, self.nbytes)
        else:
            payload = None
        if exchange is not None:
            payload = exchange(payload)
        if not leader:
            pid, lfd, nbytes = payload
            if nbytes != self.nbytes:
                raise ValueError(f"PLE hostpin: leader's buffer is {nbytes} bytes, expected {self.nbytes}")
            self._fd = os.open(f"/proc/{pid}/fd/{lfd}", os.O_RDWR)
            self._mm = mmap.mmap(self._fd, self.nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
        t1 = time.perf_counter()
        self.register_retries = 0
        for off in range(0, self.nbytes, _REGISTER_BYTES):
            # Adjacent registrations map adjacent device addresses, so the view below spans them. A registration can
            # fail transiently ("invalid argument") while four ranks pin the same pages and the page cache churns;
            # a retry of that chunk succeeds.
            n = min(_REGISTER_BYTES, self.nbytes - off)
            for attempt in range(_REGISTER_TRIES):
                err = rt.cudaHostRegister(addr + off, n, _HOST_REGISTER_FLAGS)
                if err == 0:
                    break
                rt.cudaGetLastError()
                self.register_retries += 1
                time.sleep(0.5 * (attempt + 1))
            else:
                raise RuntimeError(f"PLE hostpin: cudaHostRegister({n / 2**30:.1f} GiB at offset {off}) failed "
                                   f"{_REGISTER_TRIES} times: {rt.cudaGetErrorString(err).decode()} (error {err}); "
                                   f"RLIMIT_MEMLOCK {resource.getrlimit(resource.RLIMIT_MEMLOCK)}")
        self.register_s = time.perf_counter() - t1
        dptr = ctypes.c_void_p()
        err = rt.cudaHostGetDevicePointer(ctypes.byref(dptr), addr, 0)
        if err != 0 or dptr.value != addr:
            # UVA on Linux x86: registered memory has the same address on the device; the view below relies on it.
            raise RuntimeError(f"PLE hostpin: device pointer {dptr.value} != host pointer {addr} (error {err})")
        words = self.row_bytes // 8

        class _View:  # the pinned buffer as a CUDA tensor [rows, words] of int64 (no copy)
            __cuda_array_interface__ = {"shape": (self.rows_total, words), "typestr": "<i8",
                                        "data": (addr, False), "version": 3, "strides": None}

        self.rows = torch.as_tensor(_View(), device=self.device)
        torch.cuda.synchronize(self.device)
        self.setup_s = time.perf_counter() - t0
        self.vram_cost = free0 - torch.cuda.mem_get_info(self.device)[0]  # driver page tables for the mapping
        self.leader = leader

    def gather(self, ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Rows for int64 ids (on the GPU, any shape, flattened) into out: uint8 [n, row_bytes] on the same GPU."""
        ids = ids.reshape(-1)
        n = ids.numel()
        torch.index_select(self.rows, 0, ids, out=out[:n].view(torch.int64))
        return out

    def describe(self) -> str:
        return (f"PLEHostPinTable({'leader' if self.leader else 'follower'}, {self.nbytes / 2**30:.1f} GiB pinned "
                f"(shared memfd), {self.rows_total:,} rows x {self.row_bytes} B, setup {self.setup_s:.1f} s (fill "
                f"{self.fill_s:.1f}, register {self.register_s:.1f}, {self.register_retries} retries), GPU memory for the mapping "
                f"{self.vram_cost / 2**20:.0f} MiB on {self.device})")


__all__ = ["PLEHostPinTable"]
