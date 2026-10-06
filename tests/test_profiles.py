"""Profile honesty checks — the geek-facing numbers must not lie."""

import pytest

torch = pytest.importorskip("torch")

from vllm_volta.profiles import PROFILES, estimate, resolve  # noqa: E402

GIB = 1024**3
# Kimi K2.5 Tower, native INT4: ~595G total, ~18G active/token, 384 experts
# 24 MiB per expert instance per layer x 61 MoE layers (native INT4)
K25 = dict(model_total_bytes=595 * GIB, active_bytes_per_token=18 * GIB,
           num_experts=384, expert_bytes=24 * 1024**2, layers=61)


def test_all_profiles_declare_hardware_contract():
    for name, p in PROFILES.items():
        assert p["gpu_budget_gib"] > 0 and p["ddr4_budget_gib"] > 0
        assert set(p["bandwidth"]) >= {"gpu", "ddr4", "pmem"}


def test_k25_does_not_fit_any_vram_budget():
    for name in PROFILES:
        out = estimate(name, **K25)
        assert out["fits_vram"] is False, name


def test_node2_ceiling_is_pmem_bound_around_3_tok_s():
    out = estimate("v100x2-pmem", **K25)["estimate"]
    assert out["binding_tier"] == "pmem"
    assert 2.0 < out["tokens_per_second"] < 4.5


def test_node2_profile_is_the_fastest_k25_ceiling():
    ceilings = {n: estimate(n, **K25)["estimate"]["tokens_per_second"]
                for n in PROFILES}
    assert max(ceilings, key=ceilings.get) == "v100x2-pmem"


def test_geek_boards_honesty_is_below_1_tok_s():
    # NVMe tails: the board is fine, K2.5 is just too fat for the tiers
    for name in ("v100x2-geek-x99", "v100x2-geek-epyc"):
        out = estimate(name, **K25)["estimate"]
        assert out["tokens_per_second"] < 1.0, name
        assert out["binding_tier"] == "pmem", name


def test_fits_guard_uses_full_model_size_not_expert_budget():
    out = estimate("v100x2-pmem", **K25)
    assert out["fits_vram"] is False


def test_small_model_that_fits_vram_triggers_guard():
    tiny = dict(model_total_bytes=50 * GIB, active_bytes_per_token=2 * GIB,
                num_experts=64, expert_bytes=24 * 1024**2)
    out = estimate("v100x2-pmem", **tiny)
    assert out["fits_vram"] is True


def test_unknown_profile_fails_loud():
    with pytest.raises(SystemExit):
        resolve("rtx4090-dream")
