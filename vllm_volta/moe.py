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
    DDR4_NVME = "ddr4_nvme"  # shard on NVMe, served via the DDR4 page cache
    PMEM_DAX = "pmem_dax"    # DAX mount: every read is real Optane traffic
    PMEM_CPU = "pmem_cpu"    # legacy alias of PMEM_DAX (pre-storage-map name)


@dataclass(frozen=True)
class ExpertPlacementPlan:
    """Immutable per-expert residency; mirrors 1Cat rank-local prepared weights.

    pmem_mounts: one DAX mount per socket (2× ~1TB regions on the 4×512GB
    PMem100, see docs/storage.md). Shards assigned to PMem round-robin across
    the mounts so both NUMA nodes carry half the expert traffic; the loader
    is responsible for binding access to the mount's socket.
    """

    residency: tuple[ExpertResidency, ...]          # indexed by expert id
    pmem_mounts: tuple[str, ...] = ("/mnt/pmem0", "/mnt/pmem1")

    def device_for(self, expert_id: int) -> str:
        r = self.residency[expert_id]
        return {"gpu0": "cuda:0", "gpu1": "cuda:1"}.get(r, "cpu")

    def pmem_mount_for(self, expert_id: int) -> str:
        """Round-robin over DAX mounts for PMem-resident experts."""
        return self.pmem_mounts[expert_id % len(self.pmem_mounts)]


def plan_expert_placement(
    num_experts: int,
    expert_bytes_fp16: int,
    vram_budget_per_gpu: int = 30 * 1024**3,   # V100-SXM2-32GB minus KV headroom
    hot_fraction: float | None = None,
    pmem_mounts: tuple[str, ...] = ("/mnt/pmem0", "/mnt/pmem1"),
    freq: dict[int, float] | None = None,      # expert_id -> expected relative traffic
    ddr4_budget_bytes: int | None = None,      # page-cache tier capacity (NVMe shards)
) -> ExpertPlacementPlan:
    """Frequency-aware static placement (PowerInfer-style hot/cold split).

    Experts are ordered by expected traffic (calibration `freq`, uniform when
    absent) and greedily filled into tiers: GPU0/GPU1 (alternating, balances
    TP2 expert GEMMs over NVLink), then the DDR4 page-cache tier (NVMe shards
    when `ddr4_budget_bytes` is set), then the PMem DAX tail. Residency is
    static by design: routing is data-dependent, experts cannot migrate per
    step. If the whole model fits in VRAM, everything lands on the GPUs —
    callers should use stock tensor parallel instead (see
    docs/design-k25-scaling.md).
    """
    if hot_fraction is None:
        hot_fraction = float(os.environ.get("VLLM_VOLTA_MOE_HOT_FRACTION", "0.35"))
    per_gpu = vram_budget_per_gpu // max(expert_bytes_fp16, 1)

    weights = [1.0] * num_experts
    if freq:
        for e, w in freq.items():
            if 0 <= e < num_experts:
                weights[e] = float(w)
    order = sorted(range(num_experts), key=lambda e: weights[e], reverse=True)

    if ddr4_budget_bytes is None and freq is None:
        # legacy path: uniform hot fraction, tail straight to PMem
        hot = min(num_experts, 2 * per_gpu, int(num_experts * hot_fraction) * 2)
        residency = [ExpertResidency.PMEM_CPU] * num_experts
        for rank, e in enumerate(order[:hot]):
            residency[e] = ExpertResidency.GPU0 if rank % 2 == 0 else ExpertResidency.GPU1
        return ExpertPlacementPlan(residency=tuple(residency), pmem_mounts=pmem_mounts)

    gpu_cap = 2 * per_gpu
    ddr4_cap = (ddr4_budget_bytes or 0) // max(expert_bytes_fp16, 1)
    residency: list[ExpertResidency] = [ExpertResidency.PMEM_DAX] * num_experts
    gpu_left, ddr4_left, gpu_rank = gpu_cap, ddr4_cap, 0
    for e in order:
        if gpu_left > 0:
            residency[e] = ExpertResidency.GPU0 if gpu_rank % 2 == 0 else ExpertResidency.GPU1
            gpu_left -= 1
            gpu_rank += 1
        elif ddr4_left > 0:
            residency[e] = ExpertResidency.DDR4_NVME
            ddr4_left -= 1
    return ExpertPlacementPlan(residency=tuple(residency), pmem_mounts=pmem_mounts)


# Design-model bandwidth defaults (docs/design-k25-scaling.md): V100 HBM pair
# aggregated, dual-channel-ish DDR4-2666 effective, PMem parallel-read est.
DEFAULT_TIER_BANDWIDTH = {
    "gpu": 1800 * 1024**3,      # ~1.8 TB/s combined HBM2 (theoretical pair)
    "ddr4": 200 * 1024**3,      # ~200 GB/s effective DDR4-2666 8-channel
    "pmem": 10.5 * 1024**3,     # MEASURED 2026-10-06: 10.1-10.5 GB/s per mount,
                                # local socket, saturates at 4 readers
                                # (evidence/microbench/pmem_dax_parallel_read_20261006.json);
                                # remote-socket reads are 0.4-1.0 GB/s -> NUMA-local mandatory
}


def predict_decode_seconds(
    plan: ExpertPlacementPlan,
    active_bytes_per_token: float,
    bandwidth: dict[str, float] | None = None,
    traffic: list[float] | tuple[float, ...] | None = None,
) -> dict:
    """The design formula as code: T = max_tier(bytes*share / BW).

    `share` is the fraction of per-token expert TRAFFIC served by a tier. With
    `traffic` (one expected-traffic weight per residency entry, e.g. the
    per-layer-normalized calibration weights used to build the plan) the
    shares are traffic-weighted; without it every entry counts equally, which
    is only right for uniform routing — a frequency-aware plan puts the cold
    experts on PMem, so count shares overstate PMem traffic and hide the gain.

    Returns per-tier seconds, the binding (max) decode seconds and the
    implied tokens/s ceiling. GPU0+GPU1 count as one 'gpu' tier; the caller
    owns the KV-cache and attention allowance inside `active_bytes`.
    """
    bw = dict(DEFAULT_TIER_BANDWIDTH)
    if bandwidth:
        bw.update(bandwidth)
    n = len(plan.residency)
    if n == 0:
        raise ValueError("empty placement plan")
    if traffic is not None and len(traffic) != n:
        raise ValueError(f"traffic has {len(traffic)} weights for {n} residency entries")
    weights = [1.0] * n if traffic is None else [max(float(w), 0.0) for w in traffic]
    total = sum(weights)
    if total <= 0:
        weights, total = [1.0] * n, float(n)
    mass = {"gpu": 0.0, "ddr4": 0.0, "pmem": 0.0}
    for r, w in zip(plan.residency, weights):
        if r in (ExpertResidency.GPU0, ExpertResidency.GPU1):
            mass["gpu"] += w
        elif r == ExpertResidency.DDR4_NVME:
            mass["ddr4"] += w
        else:
            mass["pmem"] += w
    tiers = {}
    for tier, m in mass.items():
        if m:
            tiers[tier] = active_bytes_per_token * (m / total) / bw[tier]
    worst = max(tiers, key=tiers.get)
    t = tiers[worst]
    return {"tier_seconds": tiers, "binding_tier": worst,
            "traffic_share": {k: v / total for k, v in mass.items()},
            "decode_seconds": t, "tokens_per_second": 1.0 / t if t else float("inf")}


def plan_expert_instance_placement(
    num_layers: int,
    num_experts: int,
    expert_bytes: int,
    vram_budget_bytes: int,
    ddr4_budget_bytes: int,
    freq: dict[tuple[int, int], float] | None = None,
    pmem_mounts: tuple[str, ...] = ("/mnt/pmem0", "/mnt/pmem1"),
) -> ExpertPlacementPlan:
    """Per-(layer, expert) instance placement — the corrected unit.

    Why (evidence/calibration/README.md, 2026-10-06): expert ids are NOT
    correlated across layers (mean pairwise rho = -0.001 in the unsloth
    imatrix). Layer-collapsed placement throws that signal away; per-instance
    placement roughly HALVES PMem traffic at equal tier capacities
    (0.65 hot: 0.323 collapsed -> 0.198 per-instance).

    Instances are ordered by expected traffic (per-layer-normalized weights)
    and filled GPU0/GPU1 -> DDR4_NVME -> PMem_DAX. The returned plan indexes
    residency by instance idx = layer * num_experts + expert.
    """
    n = num_layers * num_experts
    weights = [0.0] * n
    if freq:
        for (l, e), w in freq.items():
            if 0 <= l < num_layers and 0 <= e < num_experts:
                weights[l * num_experts + e] = float(w)
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)

    residency: list[ExpertResidency] = [ExpertResidency.PMEM_DAX] * n
    per_gpu = vram_budget_bytes // max(expert_bytes, 1)
    ddr4_cap = ddr4_budget_bytes // max(expert_bytes, 1)
    gpu_left, ddr4_left, gpu_rank = 2 * per_gpu, ddr4_cap, 0
    for i in order:
        if gpu_left > 0:
            residency[i] = ExpertResidency.GPU0 if gpu_rank % 2 == 0 else ExpertResidency.GPU1
            gpu_left -= 1
            gpu_rank += 1
        elif ddr4_left > 0:
            residency[i] = ExpertResidency.DDR4_NVME
            ddr4_left -= 1
    return ExpertPlacementPlan(residency=tuple(residency), pmem_mounts=pmem_mounts)


def merge_calibration(events) -> dict[int, float]:
    """Aggregate router-activation events into per-expert traffic weights.

    `events` is an iterable of dicts, either {"layer": l, "expert": e} or
    {"layer": l, "counts": [n_0, .., n_{E-1}]}. Per-layer normalization keeps
    busy layers from dominating. Layer-collapsed by design: the placement
    plan is per expert-id across layers (per-instance placement is future
    work; see design doc).
    """
    per_layer: dict[int, dict[int, float]] = {}
    for ev in events:
        l = int(ev["layer"])
        d = per_layer.setdefault(l, {})
        if "counts" in ev:
            for e, c in enumerate(ev["counts"]):
                d[e] = d.get(e, 0.0) + float(c)
        else:
            d[int(ev["expert"])] = d.get(int(ev["expert"]), 0.0) + 1.0
    weights: dict[int, float] = {}
    for l, d in per_layer.items():
        total = sum(d.values()) or 1.0
        for e, c in d.items():
            weights[e] = weights.get(e, 0.0) + c / total
    return weights


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
