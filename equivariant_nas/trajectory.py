"""Learning-rate-phase-aware optimization response fingerprints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class OptimizationResponseFingerprint:
    pre_transition_mae: float
    post_transition_mae: float
    warmup_complete_mae: float
    lr_shock_ratio: float
    warmup_recovery_ratio: float
    early_to_warmup_rank_risk: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def response_fingerprint(
    fidelity_mae: Mapping[int, float],
    pre_step: int = 300,
    transition_step: int = 1000,
    warmup_step: int = 5000,
) -> OptimizationResponseFingerprint:
    """Summarize how an architecture responds to the LR phase transition.

    A high shock ratio means a candidate that looked promising under the tiny
    warmup LR became unstable after the first LR increase. This is useful
    negative evidence for promotion, rather than treating every short curve as
    a noisy estimate of the same latent accuracy.
    """

    pre = float(fidelity_mae[pre_step])
    post = float(fidelity_mae[transition_step])
    warm = float(fidelity_mae[warmup_step])
    eps = 1.0e-12
    shock = post / max(pre, eps)
    recovery = warm / max(post, eps)
    # Product highlights candidates whose early ranking is most misleading.
    risk = max(0.0, shock - 1.0) * max(0.0, pre / max(warm, eps) - 1.0)
    return OptimizationResponseFingerprint(
        pre_transition_mae=pre,
        post_transition_mae=post,
        warmup_complete_mae=warm,
        lr_shock_ratio=shock,
        warmup_recovery_ratio=recovery,
        early_to_warmup_rank_risk=risk,
    )

