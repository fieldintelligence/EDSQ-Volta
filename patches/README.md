# patches/

Engine pin and backport notes. Mirrors the upstream 1Cat `patches/` concept:
the plugin depends on a *pinned engine* it does not own.

## Engine pin (TODO before first run)

vLLM dropped Volta (compute capability 7.0) support when the V1 engine became
mandatory. This repo therefore pins the **last V0-engine release that still
builds for sm_70**. Action item before the first benchmark:

1. Identify the exact tag (candidates: v0.6.x series) by building
   `vllm-project/vllm` with `TORCH_CUDA_ARCH_LIST=7.0`.
2. Record the tag + git SHA here and in `requirements.txt`.
3. Any engine-side fixes go in this directory as `.patch` files with a
   `FIX_FOR_VLLM_CUSTOM=<sha>` header, mirroring the 1Cat convention (see
   upstream `.pre-commit-config.yaml` custom-engine marker).

## Fork points

- `vllm_volta.platform` replaces the HPU platform plugin registration path.
- `vllm_volta.moe` replaces Gaudi expert offload with a CUDA/PMem placement
  policy; contract documented in `docs/port-plan.md`.
