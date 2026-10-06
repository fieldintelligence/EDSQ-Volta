# Kimi K2.7-Code serving test — tier-sharded llama.cpp (2026-10-06)

Artifact: `Kimi-K2.7-Code-UD-Q4_K_XL` (unsloth), 14 shards, 544 GB total.
Model: 1T MoE / ~32B active, **thinking-only** (vision, 256K ctx), agentic
coding specialist. Engines: officially vLLM/SGLang; llama.cpp supports the
K2 family; unsloth advises the `--special` flag for K2-thinking GGUFs.

## Tier architecture (symlink sharding — new pattern)

One model dir of symlinks spanning tiers, mmap'd as a single GGUF:
- v1: 456 G NVMe (page-cache tier) + 100 G PMem-DAX tail → **THRASHED**:
  idle proposer servers (46 G page cache) + OS pushed hot pages out of the
  503 G RAM (RES dropped 456→436 G) → TG collapsed to 0.41 t/s.
- v2: 380 G NVMe + 164 G PMem-DAX (shards 11-14 on pmem0) → fits: TG
  recovered to 0.48-0.68.

Lesson: **symlink-sharding works**, but the page-cache tier must be sized
`RAM − OS − other page-cache users − headroom`; idle members' caches are
page-cache competitors.

## Measured (build-v100, E-config flags, 32 threads, 6 GPUs for attention)

| probe | PP t/s | TG t/s |
|---|---|---|
| warmup 16 tok | 0.72 | 0.43 |
| 96 tok (cold-ish) | 1.89 | 0.48 |
| 96 tok (warm) | 0.47 | 0.56 |
| code task 633 tok (thinking-only) | 0.61 | **0.68** |

## Interpretation

- 2-3× slower than K2.5-UD-IQ3_XXS (TG 1.24) after tier rebalancing → the
  CPU expert path is **compute-bound** for K2.7-Code (heavier expert GEMMs),
  matching the earlier GGUF reality check. Bandwidth levers (int3 tail,
  placement) are exhausted for this engine; the lever now is expert compute
  (better kernels / more threads / GPU expert path) or the vLLM-TP2
  research route.
- Tier rebalance itself gained +37-65 % (0.41 → 0.56-0.68): the tier model
  predicted the thrash, and the fix followed from it.
- Thinking-only: max_tokens budgets MUST include thinking (96-token probes
  produced zero visible text). For judge use budget ≥ 2048.

## Role decision

- **MoM judge (interactive): stays K2.5** (1.24 t/s, 3-4 min/verdict
  thinking-on).
- **K2.7-Code = nightly/offline code judge**: ~16 min per ~600-token code
  verdict; quality sample excellent (implemented our tier_ceiling formula
  correctly, first try, greedy).
- Research lever for interactive K2.7: 1Cat-vLLM TP2 GPU path (their
  Qwen3.6-35B ran 111/980 t/s B1/B16 on this pair) — expert compute on GPU
  would break the CPU wall; untested for K2-family checkpoints.
