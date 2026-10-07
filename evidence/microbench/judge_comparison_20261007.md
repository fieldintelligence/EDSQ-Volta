# Judge comparison — K2.5 / DSv4-Flash-0731 / GLM-5.3-Flash / K2.7-Code (2026-10-07)

Same portfolio question on all judges (60/40, E[r]=8%/3%, vol 12%/7%, ρ=0.2;
correct answer RETURN 6.0 % / VOL **8.2 %**), same drafts (all three wrong:
8.9 / 9.2 / 8.9 % — each misapplying the covariance term). Same box, llama.cpp
E-config flags, temperature 0.

| Judge | Quant | TG t/s | Budget | Wall | Verdict | Extra |
|---|---|---|---|---|---|---|
| Kimi K2.5 Tower | UD-IQ3_XXS (PMem-DAX) | 1.24 | 256 | 9m08s | 8.2 % (truncated mid-verification) | — |
| DeepSeek-V4-Flash-0731 | UD-IQ4_XS (RAM cache, 128 G) | 2.30 | 1024 | 7m36s | 8.2 % + per-draft corrections | 13B active, newest retrain |
| **GLM-5.3-Flash** | **UD-Q4_K_XL (RAM cache, 186 G)** | **2.41-6.20** | 2048 (incl. thinking) | 12m36s | **8.2 % + error diagnosis** | identified the drafts' exact mistake: "8.9 % is what you'd get with ρ = 0.5 instead of 0.2" |
| Kimi K2.7-Code (nightly) | UD-Q4_K_XL tier-sharded | 0.68 | — | — | not run (thinking-only, 16 min/class) | code specialist |

GLM probes: TG 5.1-6.2 t/s on short outputs, 2.41 t/s during a long thinking
block; PP 5.9-28.7. DSv4 probes: TG 1.4-2.3, PP 2.3-15.5.

## Findings

1. **All three judges correct the wrong majority** — the ensemble mechanism
   holds across model families and generations.
2. **GLM-5.3-Flash = fastest + sharpest**: highest TG (up to 6.2 t/s on the
   official unsloth UD-Q4_K_XL, 186 G RAM-resident; the REAP50 GGUF was a
   broken artifact, the official one loads fine on build-v100-new), and the
   only judge that diagnosed WHY the drafts were wrong (ρ misinterpretation).
3. **DSv4-Flash-0731 = latency-efficient judge**: complete verified verdict
   in 7m36s at 1024 tokens; stays the production MoM v3 aggregator.
4. **Budget rule refined**: llama.cpp separates thinking into
   `reasoning_content`; `max_tokens` counts BOTH, so thinking-only judges
   need budget ≥ thinking + visible answer or `content` comes back empty
   (GLM first pass: 1024 tokens all-thinking → empty content; 2048 → clean
   verdict).

## Roles

- MoM v3 production judge: DSv4-Flash-0731 (7-8 min/verdict, port 18021).
- Deep-verification judge: GLM-5.3-Flash (12-13 min, port 8027) — route
  reasoning-heavy or high-stakes questions here.
- Nightly code judge: K2.7-Code (18020). Legacy reference: K2.5 (18019,
  stopped by default — restart on demand).
