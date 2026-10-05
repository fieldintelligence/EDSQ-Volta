# Upstream engagement (1CatAI)

Records of our interaction with the upstream projects, so the local history
stays the source of truth even before this repo is published.

## 2026-10-05 — 1CatAI/1Cat-vLLM (engine fork, V100/SM70-focused)

- **Fork**: https://github.com/febuz/1Cat-vLLM
- **Branch**: `feat/volta-edSQ-k25` (created at upstream main `7f1d25f`).
  Placeholder working branch — engine-pin patches and the eventual
  Kimi K2.5 Tower adapter land here first, upstreamed via PR.
- **Issue**: [#987 — "V100/SM70] Kimi K2.5 Tower native-INT4 on 2×V100-SXM2:
  EDSQ subquant + vllm_volta plugin port"](https://github.com/1CatAI/1Cat-vLLM/issues/987)

Issue content: hardware contract, EDSQ v1 format rationale (native INT4
source, Q4-GGUF rejection), the two documented SM70 pitfalls from the
attention kernel work, asks for (1) engine pin guidance, (2) EDSQ loader
contract disposition, (3) evidence exchange after our holdout sweep.
No performance numbers claimed — evidence ships per the 1Cat
"Correctness & Quality" discipline after the benchmark workflow runs.

## Positioning notes

- This repo is a *port of the playbook* from
  [1CatAI/1Cat-vLLM-Gaudi](https://github.com/1CatAI/1Cat-vLLM-Gaudi), not a
  competitor artifact; public naming must credit 1Cat and never present the
  quantized K2.5 weights as "our model" — it is Kimi K2.5 Tower (EDSQ-1).
- Publication order (contamination discipline): holdout sweep first, then
  GitHub publish (repo renamed to avoid 1Cat brand confusion, e.g.
  `EDSQ-Volta`), then Hugging Face artifact upload under own namespace.
