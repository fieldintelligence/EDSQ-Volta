# DeepSeek-V4-Flash-0731 judge test + MoM v3 swap (2026-10-06)

Question from the lab: does a DeepSeek-V4-Flash or GLM-5.3-Flash quant give
better-quality answers in a shorter time than the K2.5 judge?

## Artifact

`unsloth/DeepSeek-V4-Flash-0731-GGUF` **UD-IQ4_XS**, ~128 G (284B MoE,
13B active) — the 0731 retrain (Terminal Bench 83.9 vs 56.9 for original
V4-Flash). Fits the DDR4 page cache entirely (512 G): NO PMem tier needed.
Served on `build-v100-new` (port 18021), E-config flags, `--special`.

## Speed vs the K2.5 judge (same box, same flags)

| judge | TG t/s | tokens | wall |
|---|---|---|---|
| K2.5 UD-IQ3_XXS (PMem DAX) | 1.24 | 256 | 9m 08s (answer truncated mid-verification) |
| **DSv4-Flash-0731 UD-IQ4_XS (RAM page cache)** | **2.30** | **1024** | **7m 36s, complete** |

~1.9× TG at 4× the answer budget. Consistent with 13B vs 32B active
parameters on a CPU-compute-bound path.

## Judge quality (the portfolio question)

DSv4 verdict: RETURN 6.0 %, **VOL 8.2 %** — exact (var 0.0067744). It
verified and corrected every draft individually: devstral 8.9→8.2,
gemma4 9.2→8.2, qwen38 8.9→8.2. Same correction the K2.5 judge made, with
the verification fully inside the budget.

## MoM v3 swapped and validated

`mom-v3-proxy` (port 8031): aggregator `dsv4` (18021) + proposers
qwen38/devstral/gemma4. End-to-end streamed portfolio question:
**5m 21s** vs 19m 37s for MoM v2 (K2.5 judge) — **3.7× faster mixture**
at equal verdict quality class.

## Open

- GLM-5.3-Flash UD-Q4_K_XL downloading (queued after DSv4) — same test
  protocol when it lands (18B active: projected ~1.5-2 t/s, agentic leader).
- K2.7-Code (18020, 0.43-0.68 t/s) stays the nightly code judge: thinking-
  only, and CPU-compute-bound; its path to speed is the vLLM-TP2 research
  route, not tiers.
