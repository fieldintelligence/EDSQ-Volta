// volta_paged_attention.cu — sm_70 paged-attention decode kernel.
//
// Port target for the 1Cat "unified accelerated paged attention for long
// prefixes" concept (1Cat-vLLM-Gaudi PR #57). V1 design: ONE warp per
// (sequence, head), walking the full block table — no cross-warp merge, no
// block barriers. Rationale: the earlier multi-warp split-K version showed
// sporadic unwritten outputs on sm_70 under nvcc 12.0 -O3 --use_fast_math
// (deterministic positions, bimodal across identical reruns; both racecheck
// and memcheck stayed clean), i.e. a barrier/codegen interaction we will not
// carry. Correctness first; the WMMA split-K pass comes back as the
// documented next step (see tests + docs/port-plan.md).
//
// Correctness contract: matches the math fallback in
// vllm_volta/attention.py within FP32-softmax fp16 tolerance.

#include <torch/extension.h>
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// One warp per (seq, head). All lanes track identical (m, l) scalars (scores
// are warp-reduced); lane `d % 32` owns output element d.
__global__ void paged_attention_v1_kernel(
    half* __restrict__ out,            // [num_seqs, num_heads, head_dim]
    const half* __restrict__ q,        // [num_seqs, num_heads, head_dim]
    const half* __restrict__ k_cache,  // [num_pages, block_size, num_kv_heads, head_dim]
    const half* __restrict__ v_cache,
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks) {
    const int seq = blockIdx.x;
    const int head = blockIdx.y;                 // V1: MHA, num_heads == num_kv_heads
    const int kv_head = head;
    const int lane = threadIdx.x;

    const int seq_len = seq_lens[seq];
    if (seq_len <= 0) return;

    const long q_base = (long)seq * num_heads * head_dim + (long)head * head_dim;

    // Online softmax, FP32 accumulation throughout (sm_70 contract).
    float m = -INFINITY, l = 0.f;
    float acc[ /* head_dim / 32 */ 8 ];
    const int dmax = head_dim < 256 ? head_dim : 256;
    for (int d = lane; d < dmax; d += 32) acc[d / 32] = 0.f;

    for (int start = 0; start < seq_len; start += block_size) {
        const int page = start / block_size;
        const int len = min(block_size, seq_len - start);
        const int phys = block_tables[seq * max_blocks + page];
        const long kv_off = (long)phys * block_size * num_kv_heads * head_dim
                          + (long)kv_head * head_dim;

        for (int t = 0; t < len; ++t) {
            const half* k = k_cache + kv_off + (long)t * num_kv_heads * head_dim;
            float score = 0.f;
            for (int d = lane; d < dmax; d += 32)
                score += __half2float(q[q_base + d]) * __half2float(k[d]);
            score = warp_reduce_sum(score) / sqrtf((float)head_dim);

            const float m_new = fmaxf(m, score);
            const float p = __expf(score - m_new);
            const float corr = __expf(m - m_new);
            const half* v = v_cache + kv_off + (long)t * num_kv_heads * head_dim;
            for (int d = lane; d < dmax; d += 32) {
                float& a = acc[d / 32];
                a = a * corr + p * __half2float(v[d]);
            }
            l = l * corr + p;
            m = m_new;
        }
    }

    const float inv_l = 1.f / l;
    for (int d = lane; d < dmax; d += 32)
        out[q_base + d] = __float2half(acc[d / 32] * inv_l);
}

}  // namespace

torch::Tensor paged_attention_v1(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_tables, torch::Tensor seq_lens,
    int64_t block_size) {
    TORCH_CHECK(q.dtype() == torch::kHalf, "sm_70 contract: fp16 only");
    const int num_seqs = q.size(0);
    const int num_heads = q.size(1);
    const int head_dim = q.size(2);
    const int num_kv_heads = k_cache.size(2);
    const int max_blocks = block_tables.size(1);
    TORCH_CHECK(head_dim <= 256, "V1 kernel supports head_dim <= 256");
    TORCH_CHECK(num_heads == num_kv_heads, "V1 kernel is MHA (GQA lands with the WMMA pass)");

    auto out = torch::empty_like(q);
    const dim3 grid(num_seqs, num_heads);
    paged_attention_v1_kernel<<<grid, 32>>>(
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
        block_tables.data_ptr<int>(),
        seq_lens.data_ptr<int>(), num_heads, num_kv_heads, head_dim,
        (int)block_size, max_blocks);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_v1", &paged_attention_v1, "Volta sm_70 paged attention v1");
}
