"""Self-calibrating trust gate for low-fidelity NAS rankings.

A cheap proxy is not assumed to be useful. It earns promotion authority only
after its ranking agrees with a phase-complete reference cohort.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence


@dataclass(frozen=True)
class FidelityTrustReport:
    cohort_size: int
    spearman: float
    kendall_tau: float
    top_k: int
    top_k_recall: float
    selection_regret: float
    normalized_selection_regret: float
    trustworthy: bool
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _ranks(values: Mapping[str, float]) -> Dict[str, float]:
    """Average ranks for ties; lower metric values receive better ranks."""

    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    output: Dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for key, _ in ordered[index:end]:
            output[key] = average_rank
        index = end
    return output


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_norm = sum((y - right_mean) ** 2 for y in right) ** 0.5
    denominator = left_norm * right_norm
    return numerator / denominator if denominator else 0.0


def _kendall(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(left)
    concordant = 0
    discordant = 0
    for i, first in enumerate(keys):
        for second in keys[i + 1 :]:
            product = (left[first] - left[second]) * (right[first] - right[second])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    pairs = concordant + discordant
    return (concordant - discordant) / pairs if pairs else 0.0


def assess_fidelity_trust(
    proxy_mae: Mapping[str, float],
    reference_mae: Mapping[str, float],
    top_k: int = 2,
    minimum_cohort: int = 8,
    minimum_spearman: float = 0.5,
    minimum_top_k_recall: float = 0.5,
    maximum_normalized_regret: float = 0.1,
) -> FidelityTrustReport:
    common = sorted(set(proxy_mae) & set(reference_mae))
    if len(common) < 2:
        raise ValueError("at least two shared candidates are required")
    proxy = {key: float(proxy_mae[key]) for key in common}
    reference = {key: float(reference_mae[key]) for key in common}
    proxy_ranks = _ranks(proxy)
    reference_ranks = _ranks(reference)
    spearman = _pearson(
        [proxy_ranks[key] for key in common],
        [reference_ranks[key] for key in common],
    )
    kendall = _kendall(proxy, reference)
    effective_k = max(1, min(int(top_k), len(common)))
    proxy_top = set(sorted(common, key=lambda key: proxy[key])[:effective_k])
    reference_top = set(sorted(common, key=lambda key: reference[key])[:effective_k])
    recall = len(proxy_top & reference_top) / effective_k
    proxy_winner = min(common, key=lambda key: proxy[key])
    reference_best = min(reference.values())
    regret = reference[proxy_winner] - reference_best
    scale = max(reference_best, 1.0e-12)
    normalized_regret = regret / scale

    checks = {
        "cohort": len(common) >= int(minimum_cohort),
        "spearman": spearman >= float(minimum_spearman),
        "top_k_recall": recall >= float(minimum_top_k_recall),
        "selection_regret": normalized_regret <= float(maximum_normalized_regret),
    }
    trustworthy = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    reason = "passed all trust checks" if trustworthy else "failed: " + ", ".join(failed)
    return FidelityTrustReport(
        cohort_size=len(common),
        spearman=spearman,
        kendall_tau=kendall,
        top_k=effective_k,
        top_k_recall=recall,
        selection_regret=regret,
        normalized_selection_regret=normalized_regret,
        trustworthy=trustworthy,
        reason=reason,
    )
