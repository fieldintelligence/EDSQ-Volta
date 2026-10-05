"""Volta platform guard.

The single place that enforces the sm_70 contract so the rest of the plugin
can assume it. Upstream equivalent: the HPU platform registration in
`vllm_gaudi` (vllm-project/vllm-gaudi), which this port replaces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class VoltaPlatformError(RuntimeError):
    """Raised when the system violates the sm_70-only contract."""


@dataclass(frozen=True)
class VoltaPlatform:
    """Detected platform facts the plugin may rely on."""

    sm70_devices: tuple[int, ...]
    nvlink_pairs: tuple[tuple[int, int], ...]
    dtype: str = "float16"          # sm_70: no bf16, no fp8 — hard rule
    accum_dtype: str = "float32"    # attention softmax accumulates in FP32

    @property
    def tp2_group(self) -> tuple[int, int] | None:
        """Preferred TP2 pair: first NVLink-connected sm_70 pair, if any."""
        return self.nvlink_pairs[0] if self.nvlink_pairs else None


def detect() -> VoltaPlatform:
    """Probe CUDA devices and return the platform, or raise."""
    import torch

    if not torch.cuda.is_available():
        raise VoltaPlatformError("CUDA is not available; vllm_volta requires GPUs")

    sm70: list[int] = []
    for idx in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(idx)
        if (major, minor) == (7, 0):
            sm70.append(idx)

    if not sm70:
        raise VoltaPlatformError(
            "no compute-capability-7.0 devices found; vllm_volta targets "
            "V100-SXM2 only and does not fall back to other architectures"
        )

    return VoltaPlatform(
        sm70_devices=tuple(sm70),
        nvlink_pairs=_nvlink_pairs(torch, sm70),
    )


def _nvlink_pairs(torch, sm70: list[int]) -> tuple[tuple[int, int], ...]:
    pairs = []
    for i, a in enumerate(sm70):
        for b in sm70[i + 1:]:
            try:
                if torch.cuda.device_can_access_peer(
                        torch.device("cuda", a), torch.device("cuda", b)):
                    pairs.append((a, b))
            except RuntimeError:
                continue
    return tuple(pairs)


def env_flag(name: str, default: str = "0") -> bool:
    """VLLM_VOLTA_* boolean env switch, mirroring upstream VLLM_HPU_* flags."""
    return os.environ.get(name, default) not in ("0", "", "false", "False")
