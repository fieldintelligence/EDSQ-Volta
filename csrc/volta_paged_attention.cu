// volta_paged_attention.cu — sm_70 paged-attention decode kernel.
//
// Port target for the 1Cat "unified accelerated paged attention for long
// prefixes" concept (1Cat-vLLM-Gaudi PR #57). This is the V1 skeleton:
// one thread block per (sequence, KV head) group, FP32 accumulation,
// block-table walk with no per-shape special casing. Correctness contract:
// bit-comparable to the math fallback in vllm_volta/attention.py within
// FP32 softmax tolerance.
//
// TODO(port): tensor-core (WMMA fp16) QK^T for the 4096+ prefix buckets,
// mirroring the upstream MME delegation; keep softmax accumulation FP32.

#include <torch/extension.h>
#include <cuda_fp16.h>

namespace {

constexpr int kMaxBlocksPerSeq = 512;  // enough for 32k ctx @ block 16; long-prefix path tiles over this
constexpr int kWarps = 8;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

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
    const int kv_head = blockIdx.y;
    const int head = kv_head + threadIdx.z;      // GQA: heads share kv_head
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const int seq_len = seq_lens[seq];
    if (seq_len <= 0) return;

    // FP32 online softmax (flash-decoding style), per warp a KV chunk.
    extern __shared__ float smem[];            // [kWarps][head_dim] partials
    float* warp_out = smem + warp * head_dim;
    __shared__ float m_red[kWarps], l_red[kWarps];

    float m = -INFINITY, l = 0.f;
    for (int i = lane; i < head_dim; i += 32) warp_out[i] = 0.f;
    __syncthreads();

    for (int page = warp; page < max_blocks; page += kWarps) {
        const int start = page * block_size;
        if (start >= seq_len) break;
        const int len = min(block_size, seq_len - start);
        const int phys = block_tables[seq * max_blocks + page];

        for (int t = 0; t < len; ++t) {
            const half* k = k_cache + ((long)phys * block_size + t) * num_kv_heads * head_dim
                                  + (long)kv_head * head_dim;
            float score = 0.f;
            for (int d = lane; d < head_dim; d += 32)
                score += __half2float(q[(long)seq * num_heads * head_dim + head * head_dim + d])
                       * __half2float(k[d]);
            score = warp_reduce_sum(score) / sqrtf((float)head_dim);

            const float m_new = fmaxf(m, score);
            const float p = __expf(score - m_new);
            const float corr = __expf(m - m_new);
            const half* v = v_cache + ((long)phys * block_size + t) * num_kv_heads * head_dim
                                  + (long)kv_head * head_dim;
            for (int d = lane; d < head_dim; d += 32)
                warp_out[d] = warp_out[d] * corr + p * __half2float(v[d]);
            l = l * corr + p;
            m = m_new;
        }
    }

    m_red[warp] = m; l_red[warp] = l;
    __syncthreads();
    // Single-warp final merge of the kWarps partial softmax states.
    if (warp == 0) {
        float gm = -INFINITY, gl = 0.f;
        for (int w = 0; w < kWarps; ++w) {
            gm = fmaxf(gm, m_red[w]);
        }
        for (int w = 0; w < kWarps; ++w) {
            const float c = __expf(m_red[w] - gm);
            l_red[w] *= c;
            gl += l_red[w];
        }
        for (int d = lane; d < head_dim; d += 32) {
            float acc = 0.f;
            for (int w = 0; w < kWarps; ++w)
                acc += smem[w * head_dim + d] * __expf(m_red[w] - gm);
            out[((long)seq * num_heads + head) * head_dim + d] =
                __float2half(acc / gl);
        }
    }
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

    auto out = torch::empty_like(q);
    const dim3 grid(num_seqs, num_kv_heads);
    const dim3 block(32, 8, 1);  // x=lane, y=warp; GQA group resolved per block
    const size_t smem = 8 * head_dim * sizeof(float);
    paged_attention_v1_kernel<<<grid, block, smem>>>(
        out.data_ptr<at::Half>(),
        q.data_ptr<at::Half>(), k_cache.data_ptr<at::Half>(),
        v_cache.data_ptr<at::Half>(), block_tables.data_ptr<int>(),
        seq_lens.data_ptr<int>(), num_heads, num_kv_heads, head_dim,
        (int)block_size, max_blocks);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_v1", &paged_attention_v1, "Volta sm_70 paged attention v1");
}
