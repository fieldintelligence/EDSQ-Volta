# 1Cat vLLM for NVIDIA Volta

**1Cat-vLLM-Volta** — a hardware-plugin workspace that ports the optimization
playbook of [1CatAI/1Cat-vLLM-Gaudi](https://github.com/1CatAI/1Cat-vLLM-Gaudi)
from Intel® Gaudi® to the NVIDIA Volta V100-SXM2-32GB (NVLink baseboard).

> Naming follows the upstream project: upstream `vllm_gaudi` / `[HPU]` /
> `VLLM_HPU_*` becomes `vllm_volta` / `[VOLTA]` / `VLLM_VOLTA_*` here.
> Where this repo *forks* upstream structure, the fork point is noted in
> [patches/README.md](patches/README.md).

## What ports, and what does not

The Gaudi kernels (TPC / MME, HCCL, Synapse recipes, Native Replay) have no
Volta equivalent and are **not** translated line-by-line. What this repo ports
is the *engineering method* documented by 1Cat — see
[docs/port-plan.md](docs/port-plan.md) for the full mapping:

| Upstream concept (1Cat) | Volta counterpart (this repo) |
|---|---|
| TPC custom kernels | CUDA kernels, `compute_70,sm_70` only (`csrc/`) |
| MME matrix engine | FP16 tensor cores (WMMA) + cuBLAS |
| HCCL collectives | NCCL over NVLink (2× V100-SXM2) |
| Native Replay (fixed-address decode) | CUDA Graphs capture/replay |
| Prepared weights (rank-local shards) | Same idea, safetensors per-rank layout |
| Deferred TP2 reductions at norm boundary | Same idea, NCCL (`vllm_volta/tp2.py`) |
| Expert-grouped prefill ([1Cat PR #26](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/26)) | Same idea, token sort by expert (`vllm_volta/moe.py`) |
| Unified accelerated paged attention ([1Cat PR #57](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/57)) | sm_70 paged-attention path (`vllm_volta/attention.py`) |

## Target hardware

- 2× Tesla V100-SXM2-32GB on NVLink baseboard (TP2 target)
- 4× NVIDIA RTX 4000 Ada 20GB (aux / offload targets, PCIe)
- 2× Xeon Platinum 8259CL, 832 GB DDR4-2666 + 4× 512 GB Optane PMem
- "6.4TB" NVMe pair (model tier, `eds1`) + 1.2 TB Intel 750 (staging only) — see [docs/storage.md](docs/storage.md)

## Status

Scaffold + ported modules with explicit contracts. **Not yet runnable
end-to-end.** Benchmarks will be initialized separately once the code is
declared ready; raw results will live in `evidence/` (currently empty by
design — upstream keeps the same discipline:
"[HPU] Promote unified V4.1 acceleration profile ([1Cat PR #36](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/36))").

## References

- Upstream being ported: [1CatAI/1Cat-vLLM-Gaudi](https://github.com/1CatAI/1Cat-vLLM-Gaudi) (Apache-2.0)
- Engine lineage: [vllm-project/vllm](https://github.com/vllm-project/vllm) · [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)
- Kernel ideas: [flashinfer](https://github.com/flashinfer-ai/flashinfer) (Gaudi sibling: `flashinfer_gaudi` in the 1Cat tree)
- Model: Kimi K2.5 "Tower" (Moonshot AI) — specs to be verified in `docs/port-plan.md`

## License

Apache-2.0, matching upstream. Model weights and third-party dependencies
carry their own licenses.
