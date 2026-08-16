"""Auditable parent selection without mixing raw metrics across fidelities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ParentSelection:
    candidate: Mapping[str, Any]
    strategy: str
    eligible_architecture_ids: tuple[str, ...]
    normalized_weights: tuple[float, ...]


class FidelityAwareParentSampler:
    STRATEGIES = ("elite_archive", "island_uniform", "island_rank_weighted", "novelty", "global_cross_island")

    def __init__(self, probabilities: Mapping[str, float], seed: int = 201):
        if set(probabilities) != set(self.STRATEGIES):
            raise ValueError("incomplete parent sampling probability table")
        if any(float(value) < 0 for value in probabilities.values()):
            raise ValueError("parent sampling probabilities cannot be negative")
        if abs(sum(map(float, probabilities.values())) - 1.0) > 1e-9:
            raise ValueError("parent sampling probabilities must sum to one")
        self.probabilities = {key: float(probabilities[key]) for key in self.STRATEGIES}
        self.rng = random.Random(seed)

    @staticmethod
    def rank_weights(candidates: Sequence[Mapping[str, Any]]) -> list[float]:
        """Return positive within-fidelity rank weights; negative MAE remains informative."""
        if not candidates:
            return []
        fidelities = {int(item["highest_completed_fidelity"]) for item in candidates}
        if len(fidelities) != 1:
            raise ValueError("rank weighting cannot mix candidates from different fidelities")
        ordered = sorted(range(len(candidates)), key=lambda index: float(candidates[index]["validation_alpha_mae"]))
        raw = [0.0] * len(candidates)
        for rank, index in enumerate(ordered):
            raw[index] = float(len(candidates) - rank)
        total = sum(raw)
        return [value / total for value in raw]

    def _choose_strategy(self) -> str:
        draw = self.rng.random()
        cumulative = 0.0
        for strategy in self.STRATEGIES:
            cumulative += self.probabilities[strategy]
            if draw < cumulative:
                return strategy
        return self.STRATEGIES[-1]

    def sample(self, candidates: Sequence[Mapping[str, Any]], *, current_island: int) -> ParentSelection:
        if not candidates:
            raise ValueError("cannot sample an empty population")
        strategy = self._choose_strategy()
        pool = list(candidates)
        weights: list[float]
        if strategy == "elite_archive":
            archived = [item for item in pool if item.get("is_elite") or item.get("in_archive")]
            pool = archived or pool
            weights = self.rank_weights(_same_highest_fidelity(pool))
            pool = _same_highest_fidelity(pool)
        elif strategy == "island_uniform":
            pool = [item for item in pool if int(item.get("island", -1)) == current_island] or pool
            weights = [1.0 / len(pool)] * len(pool)
        elif strategy == "island_rank_weighted":
            pool = [item for item in pool if int(item.get("island", -1)) == current_island] or pool
            pool = _same_highest_fidelity(pool)
            weights = self.rank_weights(pool)
        elif strategy == "novelty":
            raw = [max(float(item.get("novelty_score", 0.0)), 0.0) + 1e-12 for item in pool]
            total = sum(raw)
            weights = [value / total for value in raw]
        else:
            cross = [item for item in pool if int(item.get("island", -1)) != current_island]
            pool = cross or pool
            weights = [1.0 / len(pool)] * len(pool)
        selected = self.rng.choices(pool, weights=weights, k=1)[0]
        return ParentSelection(
            candidate=selected,
            strategy=strategy,
            eligible_architecture_ids=tuple(str(item["architecture_id"]) for item in pool),
            normalized_weights=tuple(weights),
        )


def _same_highest_fidelity(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    highest = max(int(item["highest_completed_fidelity"]) for item in candidates)
    return [item for item in candidates if int(item["highest_completed_fidelity"]) == highest]

