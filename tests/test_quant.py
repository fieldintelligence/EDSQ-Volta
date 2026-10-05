"""Contract tests for vllm_volta.quant (CPU-only, no GPU needed)."""

import pytest

torch = pytest.importorskip("torch")

from vllm_volta.quant import dequantize, fingerprint, quantize  # noqa: E402


@pytest.mark.parametrize("bits", [4, 3])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("shape", [(256,), (128, 256), (4, 32, 128)])
def test_round_trip_fidelity(bits, group_size, shape):
    torch.manual_seed(0)
    numel = 1
    for s in shape:
        numel *= s
    w = (torch.randn(*shape) * 0.02).half()  # LLM-like weight magnitudes
    p = quantize(w, bits=bits, group_size=group_size)
    d = dequantize(p)
    assert d.shape == w.shape
    assert d.dtype == torch.float16
    # RTN symmetric per-group bounds: |w - d| <= scale/2, scale <= amax/levels
    amax = w.abs().max().item()
    bound = amax / (2 * (7 if bits == 4 else 3)) + 1e-3
    assert (d.float() - w.float()).abs().max().item() <= bound


def test_quantization_error_shrinks_with_bits():
    torch.manual_seed(1)
    w = (torch.randn(64, 512) * 0.02).half()
    err = {}
    for bits in (3, 4):
        d = dequantize(quantize(w, bits=bits, group_size=128))
        err[bits] = (d.float() - w.float()).pow(2).mean().item()
    assert err[4] < err[3] * 0.5


def test_nondivisible_shapes_are_padded_and_truncated():
    torch.manual_seed(2)
    w = torch.randn(10, 100).half()  # not divisible by group_size 128
    d = dequantize(quantize(w, bits=4, group_size=128))
    assert d.shape == (10, 100)


def test_packed_size_matches_bits_per_weight():
    torch.manual_seed(3)
    w = torch.randn(8, 1024).half()
    for bits, bytes_per_val in ((4, 0.5), (3, 0.375)):
        p = quantize(w, bits=bits, group_size=128)
        expected = 8 * 1024 * bytes_per_val + (8 * 1024 // 128) * 2
        assert p.nbytes() == expected


def test_fingerprint_is_stable_and_sensitive():
    a = {"bits": 4, "files": ["shard-000.safetensors"]}
    b = {"files": ["shard-000.safetensors"], "bits": 4}  # same, other order
    c = {"bits": 3, "files": ["shard-000.safetensors"]}
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c)
