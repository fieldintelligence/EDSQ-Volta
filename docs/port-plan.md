# Port plan: 1Cat-vLLM-Gaudi → vllm_volta (V100-SXM2)

Source of truth being ported: [1CatAI/1Cat-vLLM-Gaudi](https://github.com/1CatAI/1Cat-vLLM-Gaudi)
(engine lineage: [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) ←
[vllm-project/vllm](https://github.com/vllm-project/vllm)).

## 1. Concept mapping

| 1Cat (Gaudi) | vllm_volta (Volta) | Status |
|---|---|---|
| HPU platform plugin (`vllm_gaudi`) | `vllm_volta/platform.py`, sm_70-only guard | ✅ PR #1 |
| Unified accelerated paged attention ([#57](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/57)) | `vllm_volta/attention.py` + `csrc/volta_paged_attention.cu` | 🟡 V1 kernel skeleton; WMMA QK^T TODO |
| TPC custom kernels | CUDA `compute_70,sm_70` | per-kernel ports, opt-in build |
| MME matrix engine | cuBLAS + WMMA fp16 (tensor cores exist on sm_70) | delegated |
| HCCL collectives | stock NCCL over NVLink | ✅ contract in `tp2.py` |
| TP2 fused reduction boundary ([#5](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/5)) | `vllm_volta/tp2.py` deferred reductions | 🟡 logic ported; numerics validation TODO |
| Expert-grouped prefill ([#26](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/26)) | `moe.grouped_prefill_rows` (stable sort + offsets) | ✅ pure-tensor port |
| Prepared weights / rank-local shards | `moe.plan_expert_placement` (static residency) | 🟡 policy; loader TODO |
| Engram device-resume | **n/a** — DeepSeek-V4.1-specific | not ported |
| GDN / DFlash2 (Qwen linear attention) | **deferred** — separate port decision | not ported |
| Native Replay (fixed-address decode) | CUDA Graphs capture/replay | 🔲 TODO |
| V4.1 V2 async + paging ([#30](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/30), [#32](https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/32)) | V0-engine scheduler + chunked prefill config | 🔲 TODO after engine pin |

## 2. Hardware contract (this box)

- TP2 group: **2× V100-SXM2-32GB, NVLink baseboard** — the only sm_70 devices.
  Everything in this repo assumes that pair; the 4× RTX 4000 Ada (sm_89,
  PCIe) are *out of scope* for the plugin but usable as quantization /
  staging workers.
- CPUs: 2× Xeon Platinum 8259CL (Cascade Lake, AVX-512, **no AMX** — do not
  port any 1Cat CPU path that assumes AMX).
- Memory tier: 832 GB DDR4-2666 + **4× 512 GB Optane PMem** (App Direct,
  fsdax) + **6.4TB-class NVMe pair**. Storage rules live in
  [docs/storage.md](storage.md): models ALWAYS on `eds1` (the 6.4TB pair),
  the 1.2TB Intel 750 (EDS2) is staging/scratch only, PMem hosts prepared
  expert shards. Kimi K2.5 ships natively INT4 (~595 GB) — it fits the DDR4
  page cache entirely, so PMem-resident experts bound decode t/s only for
  shards we explicitly demote (see `moe.plan_expert_placement`).

## 3. Order of work

1. **Engine pin** — resolve the sm_70-capable vLLM tag (`patches/README.md`),
   gate: `import vllm` + a 1-token generation on GPU3/4 sanity run.
2. **Attention** — validate `paged_attention_v1` against the math fallback
   (bit tolerance FP32 softmax), then WMMA QK^T for long prefixes.
3. **TP2 numerics** — golden-logprob A/B with deferred reductions on/off
   (mirrors upstream's rule: no perf claim without quality gate).
4. **MoE placement** — verify grouped prefill row-equivalence, then measure
   PMem-resident expert decode ceiling (expect single-digit t/s; that is a
   *bandwidth* result, not a defect).
5. **CUDA Graphs replay** for fixed-shape decode, porting the Native Replay
   contract (fixed addresses, per-step input update, invalidation rules).
6. Only then: benchmark initialization (separate workflow, `evidence/`).

## 4. Future profile: 6-GPU heterogeneous mode (post-V1)

Benchmark later via PCIe bifurcation: **4× RTX 4000 Ada (sm_89) + 2×
V100-SXM2** in one engine view. Out of scope for the plugin until the TP2
V100 route is validated, because current vLLM-style TP requires homogeneous
groups. Planned approach mirrors the llama.cpp placement model instead:
static per-layer/per-tensor residency (Ada cards take attention+KV given
their compute advantage, V100 HBM takes expert tensors), no cross-vendor
tensor parallel. Placement policy extension of
`vllm_volta/moe.plan_expert_placement` (e.g. `plan_heterogeneous_6gpu`),
driven by the same correctness gates.

## 5. Explicit non-goals

- No bf16/fp8 anywhere (sm_70).
- No multimodal (matches upstream flash-tier scope decision).
- No claims that any 1Cat *number* transfers — Gaudi2 and V100 share nothing
  at the kernel level; only the counting arguments port.
