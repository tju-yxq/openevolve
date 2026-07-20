"""Cohort-safe successive halving for architecture evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FidelityLevel:
    name: str
    steps: int
    promote_fraction: float
    minimum_cohort: int


@dataclass
class FidelityObservation:
    architecture_id: str
    level: str
    validation_alpha_mae: float
    seed: int
    valid: bool = True
    promoted: bool = False


DEFAULT_TRIAL_LEVELS = (
    FidelityLevel("micro", 300, 0.25, 8),
    FidelityLevel("low", 1000, 0.25, 4),
    FidelityLevel("medium", 5000, 0.5, 2),
)


class SuccessiveHalvingScheduler:
    """Rank candidates only against peers evaluated at the same fidelity."""

    def __init__(self, levels: Iterable[FidelityLevel] = DEFAULT_TRIAL_LEVELS):
        self.levels = tuple(levels)
        if not self.levels:
            raise ValueError("at least one fidelity level is required")
        self.by_name = {item.name: item for item in self.levels}
        if len(self.by_name) != len(self.levels):
            raise ValueError("fidelity level names must be unique")
        self.observations: List[FidelityObservation] = []

    def record(self, observation: FidelityObservation) -> None:
        if observation.level not in self.by_name:
            raise ValueError("unknown fidelity level {}".format(observation.level))
        duplicate = any(
            item.architecture_id == observation.architecture_id
            and item.level == observation.level
            and item.seed == observation.seed
            for item in self.observations
        )
        if duplicate:
            raise ValueError("duplicate fidelity observation")
        self.observations.append(observation)

    def promotion_candidates(self, level_name: str) -> Tuple[str, ...]:
        level = self.by_name[level_name]
        cohort = [
            item
            for item in self.observations
            if item.level == level_name and item.valid and not item.promoted
        ]
        if len(cohort) < level.minimum_cohort:
            return ()
        count = max(1, int(math.floor(len(cohort) * level.promote_fraction)))
        ranked = sorted(cohort, key=lambda item: item.validation_alpha_mae)
        return tuple(item.architecture_id for item in ranked[:count])

    def mark_promoted(self, architecture_id: str, level_name: str) -> None:
        matches = [
            item
            for item in self.observations
            if item.architecture_id == architecture_id and item.level == level_name
        ]
        if not matches:
            raise ValueError("observation not found")
        for item in matches:
            item.promoted = True

    def next_level(self, level_name: str) -> Optional[FidelityLevel]:
        index = [item.name for item in self.levels].index(level_name)
        return self.levels[index + 1] if index + 1 < len(self.levels) else None

    def save(self, path: str) -> None:
        payload = {
            "levels": [asdict(item) for item in self.levels],
            "observations": [asdict(item) for item in self.observations],
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "SuccessiveHalvingScheduler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scheduler = cls(FidelityLevel(**item) for item in payload["levels"])
        scheduler.observations = [
            FidelityObservation(**item) for item in payload["observations"]
        ]
        return scheduler

