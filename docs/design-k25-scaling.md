# Design study: K2.5 Tower serving across hardware classes

Theoretical companion to `docs/storage.md` and the EDSQ contract. Question:
is our tier-balanced static placement the best solution for the 1Cat K2.5
branch, and should it extend to better or smaller systems?

## 1. The tier model (one formula)

Decode is bandwidth-bound. Per token, a MoE layer reads 8 of 384 expert
instances; expected bytes per token per tier:

    T_decode = max_tier( B_active * share_tier / BW_tier )

with B_active(K2.5, int4) ≈ 18 GB (32B active × ~0.56 B) + KV (MLA: small).
Static placement is forced, not chosen: routing is data-dependent per token,
so experts cannot migrate per step. Everything therefore hinges on
**share × quant-bytes vs tier bandwidth**, and the optimum equalizes
`bytes_tier / BW_tier` across tiers.

## 2. Our box (2×V100-SXM2 + 512G DDR4 + 2T PMem): near-optimal, two levers left

| Split (DDR4/PMem) | tail quant | per-token PMem read | PMem time | ≈ decode |
|---|---|---|---|---|
| 73/27 | int4 | 4.3 GB | ~330 ms (13 GB/s) | ~3.0 t/s |
| 73/27 | int3 | 3.2 GB | ~250 ms | ~3.9 t/s |
| 80/20 | int3 | 2.4 GB | ~185 ms | ~4.9 t/s |

- GPU tier (64G HBM) holds attention + KV + hottest experts; its share is
  capacity-capped, not bandwidth-capped — never the bottleneck.
- DDR4 at ~200 GB/s: even 100% share would cost ~90 ms/token (~11 t/s
  ceiling for the tier) — DDR4, not PMem, is the second wall.
- **Levers we have not pulled yet:**
  1. **Frequency-aware placement** (the real upgrade): our hot-fraction is
     uniform, but router activations are skewed. A calibration pass over
     holdout prompts builds a per-(layer, expert) histogram; placing the
     top-frequency experts into HBM/DDR4 shifts per-token reads toward fast
     tiers *without shrinking capacity*. Prior art: PowerInfer /
     PowerInfer-2 (hot/cold expert split by activation frequency) reports
     large wins from exactly this. One tool + one `weights` argument on
     `plan_expert_placement()` — cheap, measurable.
  2. **int3 PMem tail** (EDSQ contract already supports it).
- Not worth it: expert prefetching (token-lookahead too short to hide
  300 ms), per-step expert migration (routing is unpredictable), NVMe tier
  (strictly slower than PMem DAX).

Verdict for the 1Cat K2.5 branch: the architecture is right for this class;
add frequency-aware placement + int3 tail and we are at the model's floor
(~4–5 t/s) for this box. Below that floor there is no software to write.

## 3. Better systems: do NOT extend the placement machinery

| Class | Fit | Guidance |
|---|---|---|
| 8× H100/H200 (~640G HBM) | native INT4 fits fully | stock vLLM TP8/FP8; EDSQ is *misuse* there — no tier problem to solve |
| 1× H200 141G / MI300X 192G + EPYC DDR5 (12ch, ~460 GB/s) | hybrid, fast CPU tier | plan parameterizes as-is (`vram_budget`, `pmem_mounts`); expect 20–30 t/s CPU-expert decode |
| Apple Silicon ≥ 512G unified (~800 GB/s) | int3 (~450G) fits | use llama.cpp/GGUF — EDSQ adds nothing on Metal |

Extension = **a guard, not code**: `plan_expert_placement()` should refuse
(or degrade to a no-op) when `total_bytes ≤ vram_budget`, printing "model
fits in VRAM — use stock tensor parallel". Prevents cargo-culting the
hybrid path where it loses.

## 4. Smaller systems: a minimum spec, honestly stated

Tier capacity is the gate, bandwidth the speed:

- Minimum viable: **≥ ~500 GB addressable fast tier** (DDR4 or unified) +
  any GPU for attention. int3 (450G) is the smallest acceptable whole-model
  quant; below that int2 quality collapse is a *quality* problem, and NVMe
  tails put you back in the tier-bound trap at <1 t/s.
- 1×24G GPU + 128G DDR4 box: int4 needs 595G → ~470G spills to NVMe →
  tier-bound. **Not a supported profile.**
- CPU-only 12-channel EPYC: feasible (~20 t/s class) but attention on CPU
  makes prefill painful; not our target.

Extension = **preset profiles** (`--profile v100x2-pmem | epyc-cpu | h200x1`)
mapping to `plan_expert_placement` arguments, plus a printed tier-math
sanity line (predicted t/s ceiling) so users see the wall before benchmarking.

## 5. Decision

Extend, but narrowly — three items, all within the existing contract:

1. `tools/calibrate_expert_freq.py` + `weights` arg on the placement plan
   (frequency-aware hot ordering; PowerInfer-style).
2. `--profile` presets + the fits-in-VRAM guard + predicted-ceiling print.
3. EDSQ int3 tail default for PMem-resident shards.

Explicitly out of scope: platform kernels for other vendors, GGUF/Metal
paths, speculative expert prefetch. Those are solved better elsewhere
(llama.cpp, PowerInfer, stock vLLM) and would dilute the one thing this
branch contributes: **a verified tier-balanced serving stack for
over-VRAM MoE on constrained SM70 hardware.**
