# Expert-frequency calibration for Kimi K2.5 (2026-10-06)

Source: `unsloth/Kimi-K2.5-GGUF` → `imatrix_unsloth.gguf_file` (llama.cpp imatrix, dataset
`unsloth_calibration_Kimi-K2.5.txt`, 50 chunks). The imatrix stores per-expert activation counts
(`blk.N.ffn_gate_exps.weight.counts`) for all 60 MoE layers × 384 experts. Third-party calibration text,
not our benchmark items, so it does not contaminate the holdout.

Files:
- `kimi_k25_unsloth_imatrix_expert_counts.jsonl` — `{"layer", "counts"}` per MoE layer (input format of
  `tools/calibrate_expert_freq.py`).
- `kimi_k25_unsloth_imatrix_calibration.json` — that tool's output (`edsq-calibration-1`, fp 7e03ba80d204e242).

## Findings

Skew per layer (mean over 60 layers): the hottest 10 / 25 / 50 % of experts carry 22 / 42 / 68 % of the
routed-expert traffic (top-25 % share ranges 0.38–0.50 across layers). No expert is never activated.

**Expert ids are not shared across layers:** mean pairwise correlation of the per-layer frequency vectors
is −0.001. `vllm_volta.moe.merge_calibration` collapses layers per expert id, which throws the signal away:

| hot tiers hold (share of expert bytes) | PMem traffic: uniform | layer-collapsed (current code) | per (layer, expert) |
|---|---|---|---|
| 0.65 | 0.350 | 0.323 | **0.198** |
| 0.76 | 0.240 | 0.219 | **0.120** |
| 0.85 | 0.150 | 0.136 | **0.066** |
| 0.90 | 0.100 | 0.088 | **0.038** |

Recommendation: make the placement unit the (layer, expert) instance — the shard prepper already writes
per-expert shards, so the plan key becomes `(layer, expert)` and the weights a per-layer-normalized dict of
that key. This roughly halves PMem traffic versus uniform at the same tier capacities, which matters more
than the int3 tail (−25 % bytes).

## Tier-model ceilings (bandwidth only; real engines land well below)

Routed experts int4: 507 GB; active routed-expert bytes per token: 10.6 GB (8 of 384 per layer, 3×7168×2048).
Assumptions: PMem 10.5 GB/s per socket (measured, `../microbench/pmem_dax_parallel_read_20261006.json`),
DDR4 ~150 GB/s effective, ~85 % of hot bytes from DDR4. Per-layer placement with the hot tiers holding 76 %
of expert bytes gives a bandwidth ceiling of ~8 t/s (int4, one socket's PMem bandwidth) to ~16 t/s (both
sockets fully parallel); the DDR4 term caps everything near ~17–18 t/s.

Reality check: the llama.cpp UD-IQ3_XXS GGUF with **all** experts in DDR4 page cache measured ~1.2 t/s
decode — far below its ~17 t/s bandwidth ceiling, so CPU expert compute, not bandwidth, bound that run.
The TP2/EDSQ engine has to show that its expert path is bandwidth-bound before these ceilings mean anything.
