# AGENTS.md

Ported contribution policy, mirroring [1CatAI/1Cat-vLLM-Gaudi
AGENTS.md](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/main/AGENTS.md)
adapted to this Volta port.

- Commit subjects use the `[VOLTA]` prefix (upstream uses `[HPU]`), e.g.
  `[VOLTA] Defer TP2 reductions to the norm boundary (#4)`.
- Reference the upstream 1Cat PR a port derives from, e.g.
  `Port of 1Cat #26 (expert-grouped prefill)`.
- Changes land via feature branch + Pull Request; merges are `--no-ff` so the
  history stays auditable like upstream's PR-numbered history.
- Performance claims require: parent config, reproducible command, and a
  results file under `evidence/`. No numbers without artifacts — same
  discipline as upstream's "Correctness & Quality" section.
- Volta constraints that must never regress:
  - compute capability 7.0 only; **no bf16, no fp8, no sm_80+ kernels**
  - attention paths must keep FP32 accumulation for softmax
  - TP2 collectives go through stock NCCL unless a port explicitly proves
    otherwise (mirrors upstream's "stock HCCL" rule for Qwen TP2)
- `Signed-off-by` required on commits (`git commit -s`).
