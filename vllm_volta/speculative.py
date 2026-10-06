"""MTP speculative decoding contract (design only — engine pin pending).

K2.5 Tower ships multi-token-prediction (MTP) modules; DeepSeek's serving
notes and 1Cat's DSpark/DFlash2 show the pattern: the MTP head drafts k
tokens, the target verifies a block, accepted prefixes continue and rejected
ones roll back. On tier-bound hybrid setups MTP is disproportionately
valuable: one target pass amortizes over (1+accept) tokens, so the PMem
tier's per-token cost divides by the acceptance length.

Contract the engine must satisfy (mirrors 1Cat Native Replay rules):
- draft from the MTP head only; never sample the target during drafting
- verify the whole block in ONE target forward (paged attention V1 path)
- acceptance prefix → reuse; rollback restores KV slots + recurrent state
- per-iteration inputs (token ids, positions, KV slots) are REAL inputs —
  replay must update them, never reuse stale ones (1Cat replay contract)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MTPConfig:
    draft_tokens: int = 3          # k drafted per step
    verify_block: int = 4          # 1 + k verified per target pass
    acceptance_report: bool = True # log accept lengths into evidence/
    greedy_only: bool = True       # mirrors 1Cat single-request greedy contract

    def __post_init__(self):
        if self.verify_block != self.draft_tokens + 1:
            raise ValueError("verify_block must equal draft_tokens + 1")


def expected_speedup(accept_len: float) -> float:
    """Ideal-case speedup for a measured mean accepted length."""
    if accept_len < 1.0:
        raise ValueError("accept_len >= 1.0 (greedy always accepts >= 1)")
    return accept_len
