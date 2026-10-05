"""MoE placement: expert offload + expert-grouped prefill, Volta edition.

Two ported 1Cat concepts:

1. Expert-grouped prefill (1Cat PR #26,
   https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/26): order input rows by
   expert before the grouped GEMM so each expert's weights serve all its rows
   in one pass. Ported as a pure tensor reorder — the Gaudi fused kernels are
   replaced by cuBLAS grouped GEMV/GEMM on fp16.

2. Rank-local expert placement (1Cat "Prepared weights" concept): experts are
   assigned a device residency at load time (GPU0 / GPU1 / CPU-PMem) and
   never reshuffled per step, so decode bandwidth is predictable. On this
   box: 2× V100-SXM2 take the hot layers, Optane PMem (App Direct, mounted
   DAX) backs the cold tail — capacity is not the constraint, bandwidth is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import torch


class ExpertResidency(str, Enum):
    GPU0 = "gpu0"
    GPU1 = "gpu1"
    PMEM_CPU = "pmem_cpu"   # mmap'd from the DAX mount; page-cache resident


@dataclass(frozen=True)
class ExpertPlacementPlan:
    """Immutable per-expert residency; mirrors 1Cat rank-local prepared weights."""

    residency: tuple[ExpertResidency, ...]          # indexed by expert id
    pmem_mount: str = "/mnt/pmem0"                  # fsdax mount with 4×512GB modules

    def device_for(self, expert_id: int) -> str:
        r = self.residency[expert_id]
        return {"gpu0": "cuda:0", "gpu1": "cuda:1"}.get(r, "cpu")


def plan_expert_placement(
    num_experts: int,
    expert_bytes_fp16: int,
    vram_budget_per_gpu: int = 30 * 1024**3,   # V100-SXM2-32GB minus KV headroom
    hot_fraction: float | None = None,
) -> ExpertPlacementPlan:
    """Split experts across NVLink GPUs and the PMem-backed CPU pool.

    GPU0/GPU1 alternate experts (balances TP2 expert GEMMs over NVLink);
    overflow lands on PMem. NVLink move (on miss) is 10-50x cheaper than a
    PCIe re-read, so residency is static by design.
    """
    if hot_fraction is None:
        hot_fraction = float(os.environ.get("VLLM_VOLTA_MOE_HOT_FRACTION", "0.35"))
    per_gpu = vram_budget_per_gpu // max(expert_bytes_fp16, 1)
    hot = min(num_experts, 2 * per_gpu)
    hot = min(hot, int(num_experts * hot_fraction) * 2)

    residency: list[ExpertResidency] = []
    for e in range(num_experts):
        if e < hot:
            residency.append(ExpertResidency.GPU0 if e % 2 == 0 else ExpertResidency.GPU1)
        else:
            residency.append(ExpertResidency.PMEM_CPU)
    return ExpertPlacementPlan(residency=tuple(residency))


def grouped_prefill_rows(
    router_probs: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert-grouped prefill, ported contract of 1Cat PR #26.

    Returns (sort_idx, group_offsets) so the caller can run each expert's GEMM
    over a contiguous row slice instead of scattering rows one by one.
    Preserves upstream invariants: clamped router logits, stable routing order
    and ordered accumulation (numerics match the unsorted path exactly).
    """
    top = torch.topk(router_probs, top_k, dim=-1)
    flat_expert = top.indices.reshape(-1)
    sort_idx = torch.argsort(flat_expert, stable=True)
    counts = torch.bincount(flat_expert, minlength=router_probs.shape[-1])
    group_offsets = torch.cumsum(counts, 0)
    return sort_idx, group_offsets


def restore_row_order(out_rows: torch.Tensor, sort_idx: torch.Tensor, top_k: int) -> torch.Tensor:
    """Undo grouped_prefill_rows ordering (weighted combine happens at caller)."""
    inv = torch.empty_like(sort_idx)
    inv[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
    return out_rows.index_select(0, inv).view(-1, top_k, out_rows.shape[-1])
