"""Reference group-wise quantization contract — the "subquant" schema.

Why this exists: Kimi K2.5 ships natively INT4 (~595 GB), so a Q4 requant is
pointless (Q4_K_M GGUF is *larger* than the source). The only reasons to
quantize below native are (a) a Q3-class tail for PMem-resident experts and
(b) mixed-precision prepared shards. Both must share ONE numeric contract so
placement decisions and kernel ports stay comparable — this module is it.

Scheme: symmetric, per-group scale, RTN (round-to-nearest). No GPTQ/AWQ —
RTN is reproducible without calibration data, which matters for the
contamination-audit trail; if a calibrated pass is added later it MUST land
next to this contract, not replace it silently.

Packing (little-endian nibbles/bit-groups):
  4-bit: two values per byte, low nibble first.
  3-bit: eight values in three bytes, LSB-first.
Levels: 4-bit q in [-7, 7]  (scale = max|w| / 7)
        3-bit q in [-3, 3]  (scale = max|w| / 3)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

_LEVELS = {4: 7, 3: 3}


@dataclass(frozen=True)
class PackedTensor:
    """A group-wise quantized tensor plus everything needed to dequantize."""

    packed: torch.Tensor      # uint8
    scales: torch.Tensor      # fp16, shape (..., num_groups)
    bits: int
    group_size: int
    orig_shape: tuple[int, ...]

    def nbytes(self) -> int:
        return self.packed.numel() + self.scales.numel() * 2


def quantize(w: torch.Tensor, bits: int = 4, group_size: int = 128) -> PackedTensor:
    """RTN-symmetric group-wise quantize an fp16/32 tensor.

    The last dimension is reshaped into groups of `group_size`; scales are
    stored fp16 per group. Shapes not divisible by the group size are
    zero-padded on the last group and the original shape is restored on
    dequantize.
    """
    if bits not in _LEVELS:
        raise ValueError(f"bits must be one of {sorted(_LEVELS)}, got {bits}")
    if w.dim() < 1 or w.shape[-1] < 1:
        raise ValueError("need a non-empty tensor")

    orig_shape = tuple(w.shape)
    flat = w.float().reshape(-1)
    pad = (-flat.numel()) % group_size
    wf = torch.nn.functional.pad(flat, (0, pad)).view(-1, group_size)
    scale = wf.abs().amax(dim=1, keepdim=True) / _LEVELS[bits]
    scale = torch.clamp(scale, min=1e-12)
    q = torch.clamp(torch.round(wf / scale), -_LEVELS[bits], _LEVELS[bits]).to(torch.int16)

    packed = _pack(q.reshape(-1), bits)
    return PackedTensor(packed=packed, scales=scale.reshape(-1).half(), bits=bits,
                        group_size=group_size, orig_shape=orig_shape,
    )


def dequantize(p: PackedTensor) -> torch.Tensor:
    """Inverse of `quantize` — returns an fp16 tensor of the original shape."""
    q = _unpack(p.packed, p.bits)
    numel = 1
    for s in p.orig_shape:
        numel *= s
    # De-scale on the PADDED grouping first (scales align with padded groups),
    # then truncate to the real numel so non-divisible shapes round-trip.
    groups = q.view(-1, p.group_size).float() * p.scales.reshape(-1, 1).float()
    return groups.reshape(-1)[:numel].reshape(p.orig_shape).half()


def _pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    q = q.to(torch.int64)
    if bits == 4:
        # two nibbles per byte, low nibble first; bias 8 maps [-7,7] -> [1,15]
        lo = q[0::2] + 8
        hi = (q[1::2] + 8) if q.numel() % 2 == 0 else torch.cat(
            [q[1::2], torch.zeros(1, dtype=torch.int64)]) + 8
        return (lo | (hi << 4)).to(torch.uint8)
    if bits == 3:
        # eight biased values (bias 3 -> [0,6], pad 7) in three bytes, LSB-first
        n = q.numel()
        padded = torch.cat([q + 3, torch.full(((-n) % 8,), 7, dtype=torch.int64)])
        g = padded.view(-1, 8)
        b0 = g[:, 0] | (g[:, 1] << 3) | ((g[:, 2] & 0x3) << 6)
        b1 = ((g[:, 2] >> 2) & 0x1) | (g[:, 3] << 1) | (g[:, 4] << 4) | ((g[:, 5] & 0x3) << 7)
        b2 = ((g[:, 5] >> 1) & 0x3) | (g[:, 6] << 2) | (g[:, 7] << 5)
        return torch.stack([b0, b1, b2], dim=1).reshape(-1).to(torch.uint8)
    raise ValueError(bits)


def _unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    p = packed.to(torch.int64)
    if bits == 4:
        return torch.stack([(p & 0xF) - 8, (p >> 4) - 8], dim=1).reshape(-1).to(torch.int16)
    if bits == 3:
        g = p.view(-1, 3).to(torch.int64)
        v = torch.empty(g.shape[0], 8, dtype=torch.int64)
        v[:, 0] = g[:, 0] & 0x7
        v[:, 1] = (g[:, 0] >> 3) & 0x7
        v[:, 2] = ((g[:, 0] >> 6) & 0x3) | ((g[:, 1] & 0x1) << 2)
        v[:, 3] = (g[:, 1] >> 1) & 0x7
        v[:, 4] = (g[:, 1] >> 4) & 0x7
        v[:, 5] = ((g[:, 1] >> 7) & 0x1) | ((g[:, 2] & 0x3) << 1)
        v[:, 6] = (g[:, 2] >> 2) & 0x7
        v[:, 7] = (g[:, 2] >> 5) & 0x7
        return (v - 3).reshape(-1).to(torch.int16)
    raise ValueError(bits)


def fingerprint(manifest: dict) -> str:
    """Stable manifest fingerprint (1Cat manifest discipline)."""
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
