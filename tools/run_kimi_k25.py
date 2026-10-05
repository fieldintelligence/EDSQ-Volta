#!/usr/bin/env python3
"""Launcher for the Kimi K2.5 Tower TP2 route.

Mirrors 1Cat's tools/run_deepseek_v41.py pattern: an explicit launcher that
fixes topology, checks device locks and prepares the evidence directory
before the engine starts. Benchmark recording stays OUT of this script —
evidence/ is filled by the separate benchmark workflow.
"""

import argparse
import os
import subprocess
import sys
import time

GPU_LOCK_DIR = os.environ.get("VLLM_VOLTA_LOCK_DIR", "/tmp/vllm-volta-locks")


def acquire_device_locks(devices: list[str], lock_dir: str) -> list[str]:
    """Shared-device locks, mirroring the 1Cat run_deepseek_v41 launcher."""
    os.makedirs(lock_dir, exist_ok=True)
    taken = []
    for d in devices:
        path = os.path.join(lock_dir, f"cuda{d}.lock")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} t={time.time()}\n".encode())
            os.close(fd)
            taken.append(path)
        except FileExistsError:
            release(taken)
            sys.exit(f"[VOLTA] device cuda:{d} already locked at {path}")
    return taken


def release(locks: list[str]) -> None:
    for p in locks:
        try:
            os.unlink(p)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-dir", default=GPU_LOCK_DIR)
    ap.add_argument("model", help="path to Kimi-K2.5-Tower weights")
    ap.add_argument("--", dest="engine_args", nargs="*", default=[],
                    help="extra args passed through to the entrypoint")
    args = ap.parse_args()

    locks = acquire_device_locks(["0", "1"], args.lock_dir)
    try:
        cmd = [sys.executable, "-m", "vllm_volta.entrypoints.kimi_k25",
               args.model, *args.engine_args]
        print("[VOLTA]", " ".join(cmd))
        return subprocess.call(cmd)
    finally:
        release(locks)


if __name__ == "__main__":
    sys.exit(main())
