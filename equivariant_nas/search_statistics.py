"""Pre-registered statistics for comparing architecture-search trajectories."""

from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SearchTrajectorySummary:
    evaluations: int
    final_best_mae: float
    normalized_best_so_far_auc: float
    time_to_threshold: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def best_so_far(values: Iterable[float], initial: float) -> Tuple[float, ...]:
    best = float(initial)
    output = [best]
    for value in values:
        best = min(best, float(value))
        output.append(best)
    return tuple(output)


def summarize_trajectory(
    values: Sequence[float],
    baseline_mae: float,
    threshold_mae: float,
) -> SearchTrajectorySummary:
    curve = best_so_far(values, baseline_mae)
    normalized_auc = sum(value / baseline_mae for value in curve[1:]) / max(1, len(values))
    reached = next(
        (index for index, value in enumerate(curve[1:], 1) if value <= threshold_mae),
        None,
    )
    return SearchTrajectorySummary(
        evaluations=len(values),
        final_best_mae=curve[-1],
        normalized_best_so_far_auc=normalized_auc,
        time_to_threshold=reached,
    )


def exact_paired_sign_flip_pvalue(
    method_a: Sequence[float],
    method_b: Sequence[float],
) -> float:
    """Exact one-sided paired randomization test; lower values are better."""

    if len(method_a) != len(method_b) or not method_a:
        raise ValueError("paired non-empty samples are required")
    differences = [float(a) - float(b) for a, b in zip(method_a, method_b)]
    observed = sum(differences) / len(differences)
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistics.append(
            sum(sign * difference for sign, difference in zip(signs, differences))
            / len(differences)
        )
    return sum(value <= observed + 1.0e-15 for value in statistics) / len(statistics)


def bootstrap_mean_difference_ci(
    method_a: Sequence[float],
    method_b: Sequence[float],
    samples: int = 10000,
    seed: int = 20260721,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    if len(method_a) != len(method_b) or not method_a:
        raise ValueError("paired non-empty samples are required")
    rng = random.Random(seed)
    differences = [float(a) - float(b) for a, b in zip(method_a, method_b)]
    estimates: List[float] = []
    for _ in range(int(samples)):
        draw = [differences[rng.randrange(len(differences))] for _ in differences]
        estimates.append(sum(draw) / len(draw))
    estimates.sort()
    tail = (1.0 - float(confidence)) / 2.0
    lower = estimates[max(0, int(tail * len(estimates)))]
    upper = estimates[min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))]
    return lower, upper
