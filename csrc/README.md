# csrc/

Custom CUDA kernels, built only for `compute_70,sm_70` (see `../setup.py`,
opt-in via `VLLM_VOLTA_BUILD_OPS=1`). Port target of the upstream 1Cat TPC
kernel role; there is deliberately no MME-equivalent abstraction here —
matrix work delegates to cuBLAS/WMMA as in upstream's TPC↔MME split.

- `volta_paged_attention.cu` — V1 paged-attention decode kernel; port target
  of the unified long-prefix attention concept (1Cat PR #57).
