"""TP2 over NVLink: deferred reductions at the norm boundary.

Port of the 1Cat Qwen TP2 concept "fuse the reduction at the correct
boundary" (upstream PR #5,
https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/5): row-parallel projections
skip their per-layer AllReduce; the reduction is fused with residual-add +
RMSNorm into one boundary, so TP2 pays one NCCL round per boundary instead of
per projection. Upstream measured ~30% decode latency reduction for this
alone on Gaudi2; the Volta port targets the same counting argument over
NVLink (300 GB/s bidirectional per pair).

Rule inherited from upstream: collectives stay stock NCCL — no custom ring
games — until a port proves otherwise (AGENTS.md).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist

from vllm_volta.platform import env_flag


class DeferredReductionBuffer:
    """Accumulates partial sums across row-parallel projections; flushes once
    at the norm boundary via a single NCCL AllReduce."""

    def __init__(self, group) -> None:
        self.group = group
        self.pending: list[torch.Tensor] = []
        self.depth = 0

    def defer(self, partial: torch.Tensor) -> torch.Tensor:
        """Register a row-parallel output; returns the *unreduced* tensor.
        Consumers between here and the boundary must be reduction-agnostic
        (residual add is; per-token routing is not — see is_deferrable)."""
        self.pending.append(partial)
        self.depth += 1
        return partial

    def flush(self, residual: torch.Tensor) -> torch.Tensor:
        """AllReduce once, add residual, return the normalized-input tensor."""
        total = self.pending[0]
        for extra in self.pending[1:]:
            total = total + extra
        dist.all_reduce(total, group=self.group)
        self.pending.clear()
        self.depth = 0
        return total + residual


@contextmanager
def deferred_tp2_boundary(buffer: DeferredReductionBuffer) -> Iterator[DeferredReductionBuffer]:
    """Scope in which row-parallel outputs defer their AllReduce.

    Enabled via VLLM_VOLTA_TP2_DEFERRED_REDUCTIONS=1; default off mirrors the
    upstream opt-in contract until numerics are validated on Volta.
    """
    if not env_flag("VLLM_VOLTA_TP2_DEFERRED_REDUCTIONS"):
        yield buffer
        return
    try:
        yield buffer
    finally:
        if buffer.depth:  # safety flush: never leave a collective in flight
            buffer.flush(torch.zeros_like(buffer.pending[0]))


def is_deferrable(module: torch.nn.Module) -> bool:
    """Row-parallel linears are deferrable; anything consuming tokens
    non-linearly (routers, KV writers) is not — mirrors upstream's rule that
    only norm-adjacent consumers may see unreduced values."""
    return isinstance(module, torch.nn.Linear) and module.weight.shape[0] > module.weight.shape[1]
