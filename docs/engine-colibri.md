# Engine study: Colibri (JustVugg/colibri) — second reference engine

Studied 2026-10-05 from the LOCAL install (v1.11.0,
`/media/knight2/EDS2/tools/colibri`) + public README
(https://github.com/JustVugg/colibri). Pure-C, zero-dependency engine for
frontier MoE (up to 2.8T) on disk/RAM/VRAM as ONE hierarchy.

## Verified facts (from the local install, not the README)

- `ldd colibri|coli`: **no CUDA linkage** — pure CPU binaries; the `--vram`
  GPU flags suggest runtime-dlopen'd CUDA at most. Verify in the benchmark
  session whether the V100 pair is actually usable from colibri.
- Family registry (`family_registry.py`): `qwen38` (our 42.4 t/s baseline!),
  `qwen36`, `deepseek_v4`, `deepseek_v41`, `glm53`, `kimi_k3`, `olmoe`,
  `inkling`. **No `kimi_k25`** — K2.5 needs a family descriptor (ours to
  contribute; the `kimi_k3` descriptor shows the exact pattern: MLA geometry
  keys q_lora/kv_lora/qk_nope/qk_rope/v_head + planner geometry fn).
- `resource_plan.py`: disk/RAM/VRAM placement planner — `auto-tier`,
  usage-ranked partial **mirror** (PowerInfer idea, built-in), `repin`,
  `dense_load_ratio` (families may re-quantize at load), analysis sidecar
  caching over ~116k tensor names on a 372 GB model (DeepSeek-V4-class
  already served from tiers TODAY).
- CLI: `build|info|plan|mirror|doctor|tune|run|chat|serve|cluster|stop|web|
  bench|convert` — OpenAI-compatible `serve`, works with
  `well_known_suite.py` (`coli serve --model-id x --no-think`).

## Strategic impact

1. **The vLLM sm_70 engine pin is demoted** from "the blocker" to "the
   research path". Colibri serves Qwen3.8 (baseline replicable) and
   DeepSeek-V4-Flash today; K2.5 Tower joins via a family descriptor.
2. **Two-engine benchmark protocol** (mirrors 1Cat's two-entrypoint style):
   colibri = reference/production path; `vllm_volta` = SM70 kernel research
   path (attention kernel, deferred TP2 reductions). Cross-check results on
   identical holdout prompts.
3. **Prior-art position of our work**: colibri's dynamic auto-tier vs our
   static frequency-aware plan are complements — their planner is mature and
   dynamic; our contribution that survives is (a) the SM70 CUDA path
   (colibri has none), (b) EDSQ as a portable prepared-shard format, (c) the
   tier honesty model (`predict_decode_seconds`, profiles).

## Contribution plan (upstream: JustVugg/colibri)

Branch `feat/colibri-engine-track`:
1. Draft `kimi_k25` family descriptor (MLA geometry from config.json;
   values TODO-verify: hidden 7168, 61 layers, 384 experts, q_lora 1536,
   kv_lora 512, qk_nope 128, qk_rope 64, v_head 128 — DeepSeek-V3-MLA family
   shape) + upstream PR with our V100/PMem test evidence.
2. `coli convert` on the native INT4 checkpoint → verify tier plan output
   (`coli plan --auto-tier`) against OUR `predict_decode_seconds` numbers.
3. If colibri dlopens CUDA and honors `--vram`: V100-pair-first profile as a
   colibri tune profile; else CPU/pmem-only and the CUDA path stays ours.

## Rules carried over

Kimi K2.5 remains Moonshot's model ("Kimi K2.5 Tower (EDSQ-1)"); colibri is
JustVugg's engine — credit in every result row; benchmarks via
`well_known_suite.py` for reproducibility, our holdout set on top.
