#!/usr/bin/env python3
"""tier_probe.py — measure parallel read bandwidth per storage tier.

Parallel direct-read probe per tier (PMem DAX mounts, NVMe model tier):
N worker threads, disjoint offset windows, 1 MiB os.pread chunks. PMem DAX
is never page-cached; for NVMe drop caches first (script notes it; needs
root). Output JSON wires into predict_decode_seconds via VLLM_VOLTA_TIER_BW.

COORDINATION NOTE: the benchmark session produced its own parallel PMem
measurements on branch evidence/pmem-parallel-read-20261006 — those measured
numbers take precedence; this tool exists to make the measurement repeatable
and to extend it to eds1 (NVMe).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time


def _worker(args) -> int:
    path, offset, nbytes, chunk = args
    fd = os.open(path, os.O_RDONLY)
    total = 0
    try:
        pos = offset
        end = offset + nbytes
        while pos < end:
            n = min(chunk, end - pos)
            data = os.pread(fd, n, pos)
            if not data:
                break
            total += len(data)
            pos += n
    finally:
        os.close(fd)
    return total


def probe_tier(tier_dir: str, jobs: int, size_gb: float, chunk_mib: int) -> dict:
    path = os.path.join(tier_dir, ".tier_probe.tmp")
    nbytes = int(size_gb * 1024**3)
    window = nbytes // jobs
    chunk = chunk_mib * 1024**2
    with open(path, "wb") as f:
        f.truncate(nbytes)
    os.sync()
    tasks = [(path, i * window, window, chunk) for i in range(jobs)]
    with mp.Pool(jobs) as pool:
        t0 = time.perf_counter()
        done = sum(pool.imap_unordered(_worker, tasks))
        dt = time.perf_counter() - t0
    os.unlink(path)
    return {"tier": tier_dir, "jobs": jobs, "bytes": done,
            "seconds": round(dt, 3),
            "aggregate_GBps": round(done / dt / 1e9, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tier", action="append", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--size-gb", type=float, default=4.0)
    ap.add_argument("--chunk-mib", type=int, default=1)
    ap.add_argument("--out", default="tier-bw.json")
    args = ap.parse_args()
    results = [probe_tier(t, args.jobs, args.size_gb, args.chunk_mib)
               for t in args.tier]
    with open(args.out, "w") as f:
        json.dump({"method": f"parallel pread, {args.jobs} jobs",
                   "date": time.strftime("%F"), "results": results}, f, indent=1)
    for r in results:
        print(f"[probe] {r['tier']}: {r['aggregate_GBps']} GB/s")
    return 0


if __name__ == "__main__":
    main()
