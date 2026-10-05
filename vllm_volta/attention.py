"""Unified accelerated paged attention for long prefixes — sm_70 path.

Port of the 1Cat concept "Unify accelerated paged attention for long
prefixes" (upstream PR #57,
https://github.com/1CatAI/1Cat-vLLM-Gaudi/pull/57): one paged-attention entry
serving short decode, long-prefix decode and chunked prefill, so long-context
workloads stop paying per-shape kernel-selection costs.

Volta specifics: FlashAttention (the sm_80+ path) is unavailable, so decode
goes through the custom `csrc/volta_paged_attention.cu` kernel (FP32 softmax
accumulation), with a math-attention fallback that is correct but slow and is
only meant for contract tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vllm_volta.platform import env_flag


class AttentionBackend(str, Enum):
    VOLTA_PAGED = "volta_paged"   # custom csrc kernel (requires VLLM_VOLTA_BUILD_OPS=1)
    MATH = "math"                 # reference fallback, tests only


@dataclass(frozen=True)
class PagedAttentionConfig:
    block_size: int = 16           # KV block size; 16 keeps sm_70 shared-mem happy
    chunked_prefill_max_tokens: int = 8192  # mirrors 1Cat V4.1 prefill chunking
    long_prefix_threshold: int = 4096

    def __post_init__(self) -> None:
        if self.block_size not in (8, 16, 32):
            raise ValueError("block_size must be 8/16/32 for the sm_70 kernel")


def select_backend(prefer_custom: bool | None = None) -> AttentionBackend:
    """Single selection point, mirroring upstream's unified entry idea.

    Precedence: explicit argument > VLLM_VOLTA_ATTENTION_BACKEND > build
    availability > math fallback.
    """
    if prefer_custom is not None:
        return AttentionBackend.VOLTA_PAGED if prefer_custom else AttentionBackend.MATH
    forced = __import__("os").environ.get("VLLM_VOLTA_ATTENTION_BACKEND", "")
    if forced:
        return AttentionBackend(forced)
    try:
        import vllm_volta._volta_ops  # noqa: F401  (built with VLLM_VOLTA_BUILD_OPS=1)
        return AttentionBackend.VOLTA_PAGED
    except ImportError:
        return AttentionBackend.MATH


class VoltaPagedAttention:
    """Contract wrapper; kernel lives in csrc/volta_paged_attention.cu."""

    def __init__(self, config: PagedAttentionConfig | None = None):
        self.config = config or PagedAttentionConfig()
        self.backend = select_backend()
        if env_flag("VLLM_VOLTA_STRICT") and self.backend is AttentionBackend.MATH:
            raise RuntimeError(
                "VLLM_VOLTA_STRICT=1 refuses the math fallback; build the "
                "sm_70 kernel with VLLM_VOLTA_BUILD_OPS=1"
            )

    def forward_decode(self, q, k_cache, v_cache, block_tables, seq_lens):
        """Decode over paged KV. Long prefixes are the ported 1Cat #57 case:
        one kernel walks the full block table instead of splitting paths."""
        if self.backend is AttentionBackend.MATH:
            return self._math_decode(q, k_cache, v_cache, block_tables, seq_lens)
        import vllm_volta._volta_ops as ops
        return ops.paged_attention_v1(
            q, k_cache, v_cache, block_tables, seq_lens,
            self.config.block_size,
        )

    def _math_decode(self, q, k_cache, v_cache, block_tables, seq_lens):
        import torch
        import torch.nn.functional as F
        # Reference path: gather blocks, single fused softmax, FP32 accum.
        outs = []
        for b, length in enumerate(seq_lens.tolist()):
            n_blocks = (length + self.config.block_size - 1) // self.config.block_size
            pages = [block_tables[b, i].item() for i in range(n_blocks)]
            k = torch.cat([k_cache[p, :min(self.config.block_size,
                                           length - i * self.config.block_size)]
                           for i, p in enumerate(pages)]).to(q.dtype)
            v = torch.cat([v_cache[p, :min(self.config.block_size,
                                           length - i * self.config.block_size)]
                           for i, p in enumerate(pages)]).to(q.dtype)
            attn = torch.matmul(q[b:b + 1].float(), k.t().float()) / q.shape[-1] ** 0.5
            outs.append(torch.matmul(F.softmax(attn, dim=-1), v.float()).to(q.dtype))
        return torch.cat(outs)
