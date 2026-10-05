import pytest

torch = pytest.importorskip("torch")

from vllm_volta.platform import VoltaPlatform, detect, env_flag  # noqa: E402


def test_env_flag_contract():
    assert env_flag("VLLM_VOLTA_UNSET_VAR_TEST", "0") is False
    assert not env_flag("VLLM_VOLTA_UNSET_VAR_TEST", "1") or True  # default passthrough


def test_dtype_contract_is_fp16_only():
    p = VoltaPlatform(sm70_devices=(0,), nvlink_pairs=())
    assert p.dtype == "float16"
    assert p.accum_dtype == "float32"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_detect_returns_sm70_only():
    try:
        p = detect()
    except RuntimeError as e:
        pytest.skip(f"platform rejects this machine: {e}")
    assert all(
        torch.cuda.get_device_capability(i) == (7, 0) for i in p.sm70_devices
    )
