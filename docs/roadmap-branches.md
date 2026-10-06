# Branch roadmap — prepared extensions (status 2026-10-06)

Unmerged feature branches, each a working scaffold for one extension from
docs/design-k25-scaling.md + docs/engine-colibri.md. Pick one, finish it,
gate it, PR it. Order = expected value.

| Branch | What | Why | Depends on |
|---|---|---|---|
| `feat/colibri-engine-track` | draft `kimi_k25` family descriptor + upstream PR plan for [JustVugg/colibri](https://github.com/JustVugg/colibri) | unblocks K2.5 Tower serving on the PRIMARY engine (no CUDA-arch restriction; MLA-rank values TODO-verify vs config.json) | colibri install (present), K2.5 checkpoint on eds1 |
| `feat/speculative-mtp` | MTP draft/verify contract + `--mtp` flag | on tier-bound hybrids, decode cost divides by acceptance length — biggest remaining speed lever | serving path (colibri or pinned vLLM) exposing MTP head |
| `feat/pmem-int3-tail` | per-tier bits (`bits_for_residency`, `--hot-bits/--tail-bits`) | DAX bypasses page cache: tail reads are real Optane traffic; int3 ≈ −25% bytes | shard-prepper write mode |
| `feat/engine-pin-sm70` | candidate prober + build recipes for V0-era vLLM | research path only (our CUDA kernels); serving no longer depends on it | nothing; run when there's a spare evening |
| `feat/tier-probe` | repeatable parallel tier probe | feeds `predict_decode_seconds` with real BW; extends to NVMe | — |

## Coordination with the benchmark session (parallel chat)

The benchmark session works in this repo on `evidence/pmem-parallel-read-20261006`:
its **measured PMem DAX parallel-read numbers take precedence** over our
estimates — wire them into `predict_decode_seconds` bandwidths via
`VLLM_VOLTA_TIER_BW` before re-forecasting. Its K2.5 expert-frequency
calibration (from the unsloth imatrix) is exactly the input
`tools/calibrate_expert_freq.py` + `plan_expert_placement(freq=...)`
consume — per-layer placement reportedly **halves PMem traffic**, which
moves our honesty ceilings up materially once wired and gated.

Their branch diverged before `feat/hardware-profiles` (PR #21): when
merging, expect profiles.py/test_profiles.py to appear as additions, not
conflicts.

## Standing rules (unchanged)

Kimi K2.5 stays Moonshot's model; colibri is JustVugg's engine — credit
both in every result row. Evidence before claims; holdout before publish.
