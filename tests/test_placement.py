"""Tests for frequency-aware placement and the tier-time model."""

import pytest

torch = pytest.importorskip("torch")

from vllm_volta.moe import (  # noqa: E402
    ExpertResidency,
    merge_calibration,
    plan_expert_placement,
    predict_decode_seconds,
)

GIB = 1024**3


def test_uniform_fallback_matches_legacy_behaviour():
    plan = plan_expert_placement(num_experts=32, expert_bytes_fp16=2 * GIB,
                                 vram_budget_per_gpu=8 * GIB)
    tiers = [r for r in plan.residency]
    assert tiers.count(ExpertResidency.GPU0) == 4
    assert tiers.count(ExpertResidency.GPU1) == 4
    assert tiers.count(ExpertResidency.PMEM_CPU) == 24


def test_high_frequency_experts_reach_gpu_tier():
    freq = {e: 0.01 * e for e in range(32)}          # expert 31 busiest
    plan = plan_expert_placement(num_experts=32, expert_bytes_fp16=2 * GIB,
                                 vram_budget_per_gpu=8 * GIB, freq=freq,
                                 ddr4_budget_bytes=0)
    on_gpu = {e for e, r in enumerate(plan.residency)
              if r in (ExpertResidency.GPU0, ExpertResidency.GPU1)}
    assert on_gpu == set(range(24, 32))              # top-8 by freq


def test_ddr4_tier_fills_before_pmem():
    freq = {e: 1.0 / (e + 1) for e in range(32)}     # expert 0 busiest
    plan = plan_expert_placement(num_experts=32, expert_bytes_fp16=1 * GIB,
                                 vram_budget_per_gpu=4 * GIB, freq=freq,
                                 ddr4_budget_bytes=12 * GIB)
    tiers = [r for r in plan.residency]
    assert tiers.count(ExpertResidency.GPU0) + tiers.count(ExpertResidency.GPU1) == 8
    assert tiers.count(ExpertResidency.DDR4_NVME) == 12
    assert tiers.count(ExpertResidency.PMEM_DAX) == 12
    # busiest expert on GPU, next 12 on DDR4, coldest on PMem
    assert plan.residency[0] in (ExpertResidency.GPU0, ExpertResidency.GPU1)
    assert plan.residency[31] == ExpertResidency.PMEM_DAX


def test_fits_in_vram_lands_everything_on_gpu():
    plan = plan_expert_placement(num_experts=8, expert_bytes_fp16=1 * GIB,
                                 vram_budget_per_gpu=8 * GIB, freq={0: 5.0},
                                 ddr4_budget_bytes=0)
    assert all(r in (ExpertResidency.GPU0, ExpertResidency.GPU1)
               for r in plan.residency)


def test_predict_decode_seconds_binding_tier_is_max():
    plan = plan_expert_placement(num_experts=32, expert_bytes_fp16=1 * GIB,
                                 vram_budget_per_gpu=4 * GIB,
                                 ddr4_budget_bytes=12 * GIB)
    out = predict_decode_seconds(plan, active_bytes_per_token=18 * GIB)
    assert out["binding_tier"] == "pmem"             # slowest tier dominates
    assert out["tokens_per_second"] == pytest.approx(1.0 / out["decode_seconds"])
    assert out["tier_seconds"]["gpu"] < out["tier_seconds"]["ddr4"] < out["tier_seconds"]["pmem"]


def test_merge_calibration_layer_normalization_and_counts_form():
    events = [
        {"layer": 0, "counts": [8, 2, 0, 0]},        # expert0 dominates layer0
        {"layer": 1, "expert": 3},
        {"layer": 1, "expert": 3},
    ]
    w = merge_calibration(events)
    assert w[0] == pytest.approx(8 / 10)             # layer0 normalized to 1
    assert w[3] == pytest.approx(2 / 2)              # layer1 normalized to 1
    assert w[1] == pytest.approx(2 / 10) and w[2] == 0
