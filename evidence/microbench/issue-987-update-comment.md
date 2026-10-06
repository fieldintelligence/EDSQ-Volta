**Update on #987** — small, clearly-scoped additions; no end-to-end numbers (those wait for the engine pin + our holdout sweep, per the evidence discipline).

**1. Kernel-level validation record** (from the issue's "two SM70 pitfalls"):
- correctness gate: `pytest` vs FP32 math reference **3/3** (max abs diff < 2e-2, fp16 out, FP32 softmax accumulation)
- NaN soak: **50/50 clean** on the final single-warp-per-(seq,head) V1
- the multi-warp merge version was retired (racecheck: 32 hazards; residual unwritten outputs under nvcc 12.0 `-O3 --use_fast_math`) — full history in our repo's `evidence/correctness/paged_attention_v1_sm70.md`

**2. Kernel-level microbenchmark** (single kernel, decode, MHA 32 heads × 128 dim, fp16, block 16, shared-KV pages across batch; V100-SXM2-32GB, driver 580.178.04, torch 2.6.0+cu124, sm_70 build):

| batch | seq_len | ms/step | tok/s | eff. KV BW |
|---|---|---|---|---|
| batch | seq_len | ms/step | tok/s | eff. KV BW |
|---|---|---|---|---|
| 1 | 512 | 0.795 | 1257.8 | 10.6 GB/s |
| 1 | 2048 | 3.025 | 330.5 | 11.1 GB/s |
| 1 | 8192 | 11.964 | 83.6 | 11.2 GB/s |
| 8 | 512 | 0.771 | 10373.0 | 87.0 GB/s |
| 8 | 2048 | 3.118 | 2565.8 | 86.1 GB/s |
| 8 | 8192 | 12.52 | 639.0 | 85.8 GB/s |
| 16 | 512 | 0.896 | 17861.6 | 149.8 GB/s |
| 16 | 2048 | 3.274 | 4887.0 | 164.0 GB/s |
| 16 | 8192 | 13.004 | 1230.3 | 165.1 GB/s |

Effective KV bandwidth reaches ~150–165 GB/s at batch 16 (~17–18% of HBM2 peak) and is latency-bound at batch 1 (~11 GB/s, only 32 warps resident): expected for V1 — one warp per (sequence, head) is the *correctness-first* design after the pitfalls above. (Run-to-run variance on this shared box is ~15%; raw JSON, not averages, ships in our evidence dir.) The next pass (multi-warp split-K + WMMA QK^T, GQA support) has its own correctness gate before any number ships.

**3. Hardware tier status:** the Optane PMem goal is provisioned (2× ~502 GiB AppDirect per socket, pending power cycle) — the Kimi K2.5 Tower native-INT4 (~595 GB) expert tier is taking shape: hot experts on DDR4 page cache, tail on the two DAX mounts, model home on the 6.4TB NVMe pair.

Questions from the original post still stand (engine pin, K2.5 adapter disposition, EDSQ loader contract). Ping @yangzhuxinyzx or anyone on the SM70 side if there's a preferred base branch to rebase against.
