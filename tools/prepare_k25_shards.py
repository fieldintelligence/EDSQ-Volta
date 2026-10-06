#!/usr/bin/env python3
"""prepare_k25_shards.py — build EDSQ prepared-shard artifacts from a Kimi
K2.5 Tower checkpoint.

This tool makes the "own quant" claim concrete: it reads a safetensors
checkpoint, classifies tensors into expert vs non-expert weights, and (in
dry-run) emits the shard + placement plan per docs/storage.md. The full
write path lands with the loader (engine pin); classification and the
EDSQ numeric contract are already final and self-tested.

Modes:
  --dry-run   classify + print the shard/mount plan, no writes
  --selftest  synthetic checkpoint -> full pipeline check in a temp dir

Example:
  python tools/prepare_k25_shards.py \
      /media/knight2/eds1/Kimi-K2.5-Tower \
      /media/knight2/eds1/Kimi-K2.5-Tower-edsq \
      --expert-bits 4 --group-size 128 --shard-experts 64 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from vllm_volta.moe import plan_expert_placement  # noqa: E402
from vllm_volta.quant import (  # noqa: E402
    bits_for_residency, dequantize, format_header, fingerprint, quantize)

# Kimi K2-family layout: model.layers.<L>.mlp.experts.<E>.{w1,w2,w3}
EXPERT_RE = re.compile(
    r"^.*layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<suffix>.+)$"
)


def classify(tensors: dict) -> tuple[dict, dict]:
    """Split tensor names into (experts, other). Experts keyed by
    (layer, expert_id). An expert-ish key that does not match the known
    layout fails loudly — never silently treated as dense."""
    experts: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    other: dict[str, torch.Tensor] = {}
    for name, t in tensors.items():
        m = EXPERT_RE.match(name)
        if m:
            experts.setdefault((int(m["layer"]), int(m["expert"])), {})[m["suffix"]] = t
        else:
            if ".experts." in name:
                raise SystemExit(f"[edsq] unrecognised expert key layout: {name}")
            other[name] = t
    return experts, other


def expert_bytes_fp16(experts: dict) -> int:
    """fp16 bytes of one expert instance (first (layer, 0) seen)."""
    for (_, e), wt in sorted(experts.items()):
        if e == 0:
            return sum(t.numel() * t.element_size() for t in wt.values())
    return 0


def shard_plan(experts: dict, args) -> list[dict]:
    plan = plan_expert_placement(
        num_experts=args.num_experts,
        expert_bytes_fp16=expert_bytes_fp16(experts),
        pmem_mounts=tuple(args.pmem_mounts.split(",")),
    )
    keys = sorted(experts)
    shards = []
    for i in range(0, len(keys), args.shard_experts):
        group = keys[i:i + args.shard_experts]
        shards.append({
            "shard": f"shard-l{group[0][0]:02d}-{i:05d}.edq.safetensors",
            "experts": len(group),
            "layers": sorted({l for (l, _) in group}),
            "mount_hint": plan.pmem_mount_for(i),
            "bits": bits_for_residency(plan.residency[group[0]],
                                       hot_bits=args.hot_bits,
                                       tail_bits=args.tail_bits),
        })
    return shards


def selftest(args) -> int:
    from safetensors.torch import load_file, save_file
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "ckpt")
        os.makedirs(ckpt)
        tensors = {
            "model.embed_tokens.weight": torch.randn(32, 16).half(),
            "model.layers.0.mlp.experts.0.w1": torch.randn(16, 32).half(),
            "model.layers.0.mlp.experts.0.w2": torch.randn(32, 16).half(),
            "model.layers.0.mlp.experts.1.w1": torch.randn(16, 32).half(),
            "model.layers.0.mlp.experts.1.w2": torch.randn(32, 16).half(),
            "model.layers.0.input_layernorm.weight": torch.ones(16).half(),
        }
        save_file(tensors, os.path.join(ckpt, "model.safetensors"))
        loaded = load_file(os.path.join(ckpt, "model.safetensors"))

        experts, other = classify(loaded)
        assert len(experts) == 2 and len(other) == 3, (experts.keys(), other.keys())

        # numeric contract: quantize -> dequantize within RTN bound
        w = experts[(0, 0)]["w1"]
        rt = dequantize(quantize(w, bits=args.expert_bits, group_size=args.group_size))
        err = (rt.float() - w.float()).abs().max().item()
        bound = w.float().abs().max().item() / (7 if args.expert_bits == 4 else 3) / 2 + 1e-3
        assert err <= bound, (err, bound)

        # shard plan runs on classified experts
        shards = shard_plan(experts, args)
        manifest = {"header": format_header(
            {"bits": args.expert_bits, "group_size": args.group_size,
             "source": "selftest"}),
            "shards": shards}
        fp = fingerprint(manifest)
        print(f"[edsq] selftest OK: {len(experts)} experts, "
              f"{len(shards)} shard(s), round-trip err {err:.4f} <= {bound:.4f}, "
              f"manifest fp {fp}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("checkpoint", help="safetensors checkpoint dir on eds1")
    ap.add_argument("out_dir", nargs="?", help="EDSQ artifact dir (not yet written)")
    ap.add_argument("--expert-bits", type=int, default=4, choices=(3, 4))
    ap.add_argument("--hot-bits", type=int, default=4, choices=(3, 4),
                    help="bits for GPU/DDR4-tier experts")
    ap.add_argument("--tail-bits", type=int, default=3, choices=(2, 3),
                    help="bits for PMem-DAX-tail experts (~25%% fewer read bytes)")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--shard-experts", type=int, default=64)
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument("--pmem-mounts", default="/mnt/pmem0,/mnt/pmem1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args)

    from safetensors import safe_open
    tensors = {}
    index_path = os.path.join(args.checkpoint, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    for fn in files:
        with safe_open(os.path.join(args.checkpoint, fn), framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

    experts, other = classify(tensors)
    print(f"[edsq] classified: {len(experts)} expert instances "
          f"({expert_bytes_fp16(experts) / 1024**2:.1f} MiB fp16 per expert), "
          f"{len(other)} non-expert tensors")
    shards = shard_plan(experts, args)
    total = sum(s["experts"] for s in shards)
    for s in shards[:5]:
        print("  ", s)
    if len(shards) > 5:
        print(f"   ... +{len(shards) - 5} more")
    print(f"[edsq] shard plan: {len(shards)} shards covering {total} expert instances")
    if not args.dry_run:
        raise SystemExit("[edsq] write mode lands with the loader (engine pin); "
                         "rerun with --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
