# Correctness record — volta_paged_attention.cu (V1) on sm_70

Date: 2026-10-05 · Host: node2 · Device: Tesla V100-SXM2-32GB (torch index 4/5)
Driver 580.178.04 (CUDA 13.0) · torch 2.6.0+cu124 · nvcc 12.0 (JIT build via
`torch.utils.cpp_extension.load`, `-gencode=arch=compute_70,code=sm_70`)

## Result

- `tests/test_paged_attention_kernel.py`: **3/3 passed** (seeds 0/1/2, 8 seqs ×
  4 heads × 128 dim, random lens ≤ 384, permuted block tables) vs FP32 math
  reference, tolerance: max abs diff < 2e-2, mean < 2e-3 (fp16 out).
- NaN soak: **50 identical reruns → 0 NaN** (previous design failed this).

## Bug history (why V1 is single-warp-per-(seq, head))

The first design used 8 warps per block with a shared-memory online-softmax
merge (m_red/l_red + syncthreads). On real sm_70 hardware it produced
sporadic NaN outputs:

1. **Volta ITS hazard (real, fixed):** the merge let all 32 lanes of warp 0
   read-modify-write `l_red[w]`; on Volta, Independent Thread Scheduling does
   not guarantee lockstep, so lanes raced their own shared writes.
   `compute-sanitizer --tool racecheck` flagged it (32 hazards).
2. **Residual sporadic NaN (unfixed by 1):** after the ITS fix, racecheck and
   memcheck were clean yet NaNs persisted — deterministic positions
   (head_dim indices ≡ 2,3 mod 4), bimodal presence across identical reruns,
   `torch.zeros_like` output buffer masking it entirely (i.e. elements
   sometimes unwritten). Attributed to a barrier/early-break codegen
   interaction under nvcc 12.0 `-O3 --use_fast_math` on sm_70; root cause not
   conclusively identified and deliberately not carried.

V1 removes the entire class: one warp per (sequence, head), no cross-warp
merge, no block barriers. MHA-only (guarded); GQA + WMMA QK^T split-K return
as the next optimization pass with their own correctness gates.

## Tooling notes

- `compute-sanitizer` at `/usr/local/cuda-12/bin/compute-sanitizer` works on
  this box; `-lineinfo` did not yield source-line mapping for racecheck.
- JIT extension caching lives in `~/.cache/torch_extensions` — wipe the
  `volta_ops_test` dir when changing `extra_cuda_cflags` (hash covers flags,
  but stale dirs have bitten us).
