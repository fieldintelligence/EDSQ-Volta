"""vllm_volta — 1Cat-style vLLM hardware plugin for NVIDIA Volta V100-SXM2.

Port of the 1Cat-vLLM-Gaudi plugin concept (https://github.com/1CatAI/1Cat-vLLM-Gaudi)
to compute capability 7.0. Naming contract: `vllm_volta` / `[VOLTA]` /
`VLLM_VOLTA_*` everywhere; upstream uses `vllm_gaudi` / `[HPU]` / `VLLM_HPU_*`.
"""

__version__ = "0.1.0"
