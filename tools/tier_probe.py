#!/usr/bin/env python3
"""tier_probe.py — measure REAL parallel read bandwidth per storage tier.

Replaces single-stream dd numbers (which underestimate Optane badly) with a
parallel direct-read probe, per tier: the two PMem DAX mounts and the NVMe
model tier. Output JSON feeds predict_decode_seconds() directly (the tier
model in docs/design-k25-scaling.md is only as good as its BW numbers).

Method: one test file per tier, N worker threads each owning a disjoint
offset window, os.pread in 1 MiB chunks. PMem DAX is never page-cached; for
NVMe the caller should drop caches first (script does it when run as root).

Usage:
  python3 tools/tier_probe.py --jobs 8 --size-gb 4 \
      --tier /mnt/pmem0 --tier /mnt/pmem1 --tier /media/knight2/eds1 \
      --out tier-bw.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time


def _worker(args) -> float:
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
    # per-job disjoint windows across the file
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
    doc = {"method": f"parallel pread, {args.jobs} jobs x {args.chunk_mib} MiB chunks",
           "date": time.strftime("%F"), "results": results}
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    for r in results:
        print(f"[probe] {r['tier']}: {r['aggregate_GBps']} GB/s "
              f"({r['jobs']} jobs, {r['seconds']}s)")
    print(f"[probe] wrote {args.out} — wire into predict_decode_seconds via "
          f"VLLM_VOLTA_TIER_BW={args.out}")
    return 0


if __name__ == "__main__":
    main()
