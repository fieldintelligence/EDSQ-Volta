# CLAUDE.md

Short context for AI agents working in this repo (see AGENTS.md for the full
policy; upstream equivalent: 1Cat-vLLM-Gaudi `CLAUDE.md`).

- This is a **port** of https://github.com/1CatAI/1Cat-vLLM-Gaudi to Volta
  V100-SXM2. When unsure how something should work, find the upstream Gaudi
  implementation and port the *contract*, not the kernel.
- Package name is `vllm_volta`; env vars are `VLLM_VOLTA_*`; commit prefix
  `[VOLTA]`. Never introduce `vllm_gaudi`/`VLLM_HPU_` names here.
- sm_70 only: no bf16/fp8 anywhere, no `torch.compile` assumptions that need
  Triton fp8. FP16 weights/activations, FP32 accumulation.
- Do not add benchmark harnesses or result claims; benchmarks are initialized
  in a separate workflow once code is ready (`evidence/` stays empty until
  then).
