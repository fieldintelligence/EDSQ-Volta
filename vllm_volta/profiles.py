"""Hardware profiles — one place that maps a board class to placement args.

Named after what people actually own. The dual V100-SXM2-32GB NVLink board
(paired via PCIe bifurcation risers) is the flagship: a 64 GB HBM2 hot tier
at consumer-board prices, and common enough in the geek market that every
profile below assumes it as the execution center unless stated otherwise.

Each profile carries: GPU EXPERT-tier budget (after attention+KV; geek
boards assume attention at int8), DDR4 page-cache budget, tier bandwidths
(measured where available, estimated elsewhere). The entrypoint prints the predicted ceiling BEFORE running
— geeks deserve to see the wall before burning a night on it.
"""

from __future__ import annotations

from vllm_volta.moe import predict_decode_seconds

GIB = 1024**3

PROFILES: dict[str, dict] = {
    # our dev box: the reference measurement environment
    "v100x2-pmem": {
        "desc": "2×V100-SXM2 + 512G DDR4 + 2T Optane PMem (DAX) — node2",
        "vram_total_gib": 64, "gpu_budget_gib": 45, "ddr4_budget_gib": 384,
        "pmem_mounts": ("/mnt/pmem0", "/mnt/pmem1"),
        "bandwidth": {"gpu": 1800 * GIB, "ddr4": 200 * GIB, "pmem": 13 * GIB},
    },
    # the geek board: V100 NVLink pair on a bifurcation riser, big-RAM host
    "v100x2-geek-epyc": {
        "desc": "2×V100-SXM2 via bifurcation + EPYC (8ch DDR4/DDR5, 256G+)",
        "vram_total_gib": 64, "gpu_budget_gib": 45, "ddr4_budget_gib": 224,
        "pmem_mounts": (),
        "bandwidth": {"gpu": 1800 * GIB, "ddr4": 150 * GIB, "pmem": 4 * GIB},
        "pmem_is_nvme": True,   # no Optane: the 'pmem' tier IS an NVMe tier
    },
    "v100x2-geek-x99": {
        "desc": "2×V100-SXM2 via bifurcation + dual X99 (4ch DDR4, 96-128G)",
        "vram_total_gib": 64, "gpu_budget_gib": 45, "ddr4_budget_gib": 88,
        "pmem_mounts": (),
        "bandwidth": {"gpu": 1800 * GIB, "ddr4": 70 * GIB, "pmem": 3 * GIB},
        "pmem_is_nvme": True,
    },
    "h200x1": {
        "desc": "1× H200 141G + server DDR5 — hybrid still, but fast tiers",
        "vram_total_gib": 141, "gpu_budget_gib": 120, "ddr4_budget_gib": 384,
        "pmem_mounts": (),
        "bandwidth": {"gpu": 4800 * GIB, "ddr4": 300 * GIB, "pmem": 4 * GIB},
        "pmem_is_nvme": True,
    },
}


def resolve(name: str) -> dict:
    if name not in PROFILES:
        raise SystemExit(f"[VOLTA] unknown profile '{name}'; "
                         f"known: {', '.join(PROFILES)}")
    return PROFILES[name]


def estimate(name: str, model_total_bytes: int, active_bytes_per_token: int,
             num_experts: int, expert_bytes: int, layers: int = 1) -> dict:
    """Guard + predicted ceiling for a (profile, model, quant) combination.

    `expert_bytes` is per expert INSTANCE per layer (e.g. ~24 MiB int4 for
    K2-family shapes); placement is per expert-ID across all `layers`, so
    the planner sees expert_bytes * layers per ID. K2.5: 61 MoE layers.
    Returns {'fits_vram': bool, 'estimate': predict_decode_seconds-output}.
    When the whole model fits the GPU budget the estimate is degenerate by
    design — use stock tensor parallel instead (docs/design-k25-scaling.md).
    """
    p = resolve(name)
    fits = model_total_bytes <= p["vram_total_gib"] * GIB
    plan_args = dict(
        vram_budget_per_gpu=(p["gpu_budget_gib"] // 2) * GIB,
        ddr4_budget_bytes=p["ddr4_budget_gib"] * GIB,
        pmem_mounts=p["pmem_mounts"] or ("/mnt/pmem-nvme",),
    )
    # local import avoids a cycle at module load
    from vllm_volta.moe import plan_expert_placement
    plan = plan_expert_placement(num_experts=num_experts,
                                 expert_bytes_fp16=expert_bytes * layers,
                                 **plan_args)
    est = predict_decode_seconds(plan,
                                 active_bytes_per_token=active_bytes_per_token,
                                 bandwidth=p["bandwidth"])
    return {"profile": name, "fits_vram": fits, "plan": plan, "estimate": est}
