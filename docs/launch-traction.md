# Launch & traction plan — K2.5 delivery (dual path)

Goal: turn the K2.5 Tower delivery into adoption, not just a repo. Two
paths ship together — the community speaks GGUF; researchers get EDSQ.

## Path 1 — GGUF recipe (everyone)

Artifact: **unsloth `UD-IQ3_XXS` (415 GB)** — their quant, our recipe. Never
re-upload their weights; ship the recipe + measured results and link out.

The recipe document must contain, verbatim from evidence:
- which quant + why (native INT4 source → Q4 requants pointless; Q3 is the
  sweet spot for tier capacity, see docs/design-k25-scaling.md);
- imatrix note (unsloth's `imatrix_unsloth.gguf`, dataset
  `unsloth_calibration_Kimi-K2.5.txt`, 50 chunks — third-party text, no
  holdout contamination);
- serving flags that BIT us: all-experts-in-DDR4 (`--cpu-moe`), attention on
  GPUs, `--n-cpu-moe` OOM math (50 GB > 20 G Ada), and the **thinking-mode
  token trap** (`chat_template_kwargs: {"thinking": false}` or reasoning
  eats fixed token budgets — cost us a whole battery run);
- measured results per box class, PP **and** TG, with the honesty ceilings
  from `python -m vllm_volta.profiles`.

## Path 2 — EDSQ + calibration (researchers)

Ship in-repo, already done or scaffolded:
- calibration JSONs: `evidence/calibration/` (unsloth imatrix →
  `edsq-calibration-1`, per-layer expert counts);
- placement tooling: `tools/calibrate_expert_freq.py` +
  `vllm_volta.moe.plan_expert_instance_placement`;
- **the ceiling calculator**: `python -m vllm_volta.profiles --profile X`
  — "werk het uit voor jouw bak". This is the reader→user converter: anyone
  with a different tier size gets their own number in one command, and
  learns the wall exists BEFORE benchmarking.

## Channels (in order)

1. **This repo** (fieldintelligence/EDSQ-Volta) — renamed off the 1Cat
   brand, crediting 1Cat (playbook), JustVugg/colibri (second engine),
   Moonshot (model), unsloth (GGUF artifact).
2. **HF model card** for the recipe (weights link-out to unsloth) — only
   after the holdout sweep lands.
3. **colibri upstream PR** (`kimi_k25` family descriptor) — discovery
   through their issue tracker/user base.
4. **1Cat thread** ([#987](https://github.com/1CatAI/1Cat-vLLM/issues/987))
   — already live; report the cross-engine results there.
5. **r/LocalLLaMA + ServeTheHome** launch post: the hook is the board —
   "dual V100-SXM2 NVLink via a bifurcation riser" is the cheapest 64 GB
   HBM2 a geek can own, and the honesty ceilings tell them exactly what
   runs and what doesn't. Numbers first, post after.

## Timing rule

Nothing public before the holdout sweep reports (contamination discipline).
The Q3 A/B/C tuning run in flight IS the launch content — TG and PP per
variant, plus the instant-vs-thinking battery, become the model card's
first results table.
