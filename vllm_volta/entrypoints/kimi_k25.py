"""Kimi K2.5 "Tower" serving entrypoint — TP2 over NVLink, 2× V100-SXM2-32GB.

Naming and contract mirror the 1Cat DeepSeek entrypoints
(`python -m vllm_gaudi.entrypoints.deepseek_v4`), see
https://github.com/1CatAI/1Cat-vLLM-Gaudi — fixed topology, fixed dtype,
single-request defaults until benchmarks prove a wider service envelope.

Model specs (verified 2026-10, public coverage): Kimi K2.5 ~1.04T total /
32B active MoE, 384 experts with 8 active per token, 61 layers (1 dense +
60 MoE), MLA attention, SwiGLU, 256k context. Consequence for this box:
fp16 weights ≈ 2.1 TB exceed even the 2 TB PMem pool — a sub-8-bit quant is
mandatory, after which the expert tail fits in DDR4 page cache and PMem acts
as the cold tier only. Decode is bounded by ~32B active × bits-per-weight
against ~150-200 GB/s effective CPU bandwidth.
"""

from __future__ import annotations

import argparse

from vllm_volta.moe import plan_expert_placement
from vllm_volta.platform import detect


def build_engine_args(args: argparse.Namespace) -> dict:
    platform = detect()
    pair = platform.tp2_group
    if pair is None:
        raise SystemExit("Kimi K2.5 entrypoint requires an NVLink sm_70 pair (TP2)")

    # Expert residency decided once at load, mirroring rank-local prepared
    # weights: hot experts on the two V100s, tail on the PMem pool.
    plan = plan_expert_placement(
        num_experts=args.num_experts,
        expert_bytes_fp16=args.expert_bytes_fp16,
    )

    return {
        "model": args.model,
        "served_model_name": "kimi-k2.5-tower",
        "tensor_parallel_size": 2,
        "device": [f"cuda:{i}" for i in pair],
        "dtype": platform.dtype,                       # fp16 — sm_70 contract
        "kv_cache_dtype": "fp8_e4m3" if False else "fp16",  # sm_70: fp8 KV unsupported
        "block_size": 16,
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1 if args.single_request else 8,
        "expert_placement": plan,
        "swap_space": 0,                                # PMem is the swap layer
        "enable_prefix_caching": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Serve Kimi K2.5 Tower on 2x V100-SXM2")
    p.add_argument("model", help="path to Kimi-K2.5-Tower weights (safetensors)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--single-request", action="store_true", default=True,
                   help="mirror 1Cat single-request service contract (default)")
    p.add_argument("--num-experts", type=int, default=384,
                   help="verified: Kimi K2.5 has 384 experts, 8 active per token")
    p.add_argument("--expert-bytes-fp16", type=int, default=88 * 1024**2,
                   help="per-expert fp16 bytes, est. 3*7168*2048*2B (K2-family "
                        "shapes); TODO-verify against the checkpoint config.json")
    args = p.parse_args()

    engine_args = build_engine_args(args)
    print("[VOLTA] engine args:", engine_args)
    # Engine wiring lands with the sm_70 vLLM pin (patches/README.md TODO):
    # from vllm.engine.arg_utils import EngineArgs ...
    raise SystemExit(
        "[VOLTA] engine pin not resolved yet — see patches/README.md; "
        "do not record any benchmark numbers until it is."
    )


if __name__ == "__main__":
    main()
