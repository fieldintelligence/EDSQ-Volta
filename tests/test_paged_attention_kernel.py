"""Correctness gate for csrc/volta_paged_attention.cu on real sm_70 hardware.

Acceptance: kernel output matches the math reference within fp16 tolerance
(FP32 softmax accumulation contract). Mirrors upstream 1Cat discipline:
kernel-level checks come before any performance claim (upstream README,
"Correctness & Quality").
"""

import os

import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs an sm_70 GPU"
)


def load_ops():
    from torch.utils.cpp_extension import load

    return load(
        name="volta_ops_test",
        sources=[os.path.join(REPO, "csrc", "volta_paged_attention.cu")],
        extra_cuda_cflags=[
            "-gencode=arch=compute_70,code=sm_70", "-O3", "--use_fast_math",
        ],
        verbose=False,
    )


def math_reference(q, k_cache, v_cache, block_tables, seq_lens, block_size):
    """FP32 reference: full softmax over the whole prefix, per (sequence, head)."""
    n, heads, dim = q.shape
    out = torch.empty_like(q, dtype=torch.float32)
    for b in range(n):
        length = int(seq_lens[b])
        n_blocks = (length + block_size - 1) // block_size
        pages = [int(block_tables[b, i]) for i in range(n_blocks)]
        for h in range(heads):
            ks, vs = [], []
            for i, p in enumerate(pages):
                take = min(block_size, length - i * block_size)
                ks.append(k_cache[p, :take, h])   # MHA: kv-head == q-head
                vs.append(v_cache[p, :take, h])
            k = torch.cat(ks).float()
            v = torch.cat(vs).float()
            scores = q[b, h].float() @ k.t() / (dim ** 0.5)
            out[b, h] = torch.softmax(scores, dim=-1) @ v
    return out


def sm70_device():
    """First compute-capability-7.0 device (torch enumeration order differs
    from nvidia-smi order; on node2 the Adas come first)."""
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(i) == (7, 0):
            return i
    return None


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_paged_attention_v1_matches_reference(seed):
    dev = sm70_device()
    if dev is None:
        pytest.skip("no sm_70 device visible")
    with torch.cuda.device(dev):
        torch.manual_seed(seed)
        ops = load_ops()

        n_seqs, heads, dim, block_size = 8, 4, 128, 16
        max_pages = 24
        seq_lens = torch.randint(1, max_pages * block_size + 1, (n_seqs,),
                                 device="cuda", dtype=torch.int32)

        q = torch.randn(n_seqs, heads, dim, device="cuda", dtype=torch.float16)
        n_pages_total = int(max_pages) * n_seqs  # worst case unique pages
        # V1 kernel is MHA: kv-head count equals q-head count
        k_cache = torch.randn(n_pages_total, block_size, heads, dim,
                              device="cuda", dtype=torch.float16)
        v_cache = torch.randn_like(k_cache)

        perm = torch.randperm(n_pages_total, device="cuda")
        block_tables = torch.stack([
            perm[b * max_pages:(b + 1) * max_pages] for b in range(n_seqs)
        ]).int()

        got = ops.paged_attention_v1(q, k_cache, v_cache, block_tables,
                                     seq_lens, block_size)
        want = math_reference(q, k_cache, v_cache, block_tables,
                              seq_lens, block_size)

        torch.cuda.synchronize()
        assert got.dtype == torch.float16
        diff = (got.float() - want).abs()
        # fp16 out with --use_fast_math: allow ~fp16 ulp on |v|<=~8
        assert diff.max() < 2e-2, f"max abs diff {diff.max().item():.4f}"
        assert diff.mean() < 2e-3, f"mean abs diff {diff.mean().item():.4f}"
