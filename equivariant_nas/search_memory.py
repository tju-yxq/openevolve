"""Deterministic search-memory summaries for SPARK-style reflection.

SPARK motivates separating structural reflection from architecture editing and
conditioning the reflection on evolution history.  This module makes that
history a trusted, compact data product rather than asking an LLM to invent a
summary from an unbounded transcript.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Mapping, Optional


_PROMPT_METRIC_KEYS = (
    "valid",
    "fidelity_steps",
    "validation_alpha_mae",
    "best_validation_alpha_mae",
    "best_step",
    "parameter_count",
    "parameter_ratio",
    "step_time_ms",
    "lmax",
    "higher_order_fraction",
    "num_layers",
    "max_symmetry_error",
    "max_layerwise_symmetry_error",
    "failure_stage",
    "error_type",
    "error",
)


def compact_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    """Keep measured prompt evidence bounded and semantically explicit."""

    return {key: metrics[key] for key in _PROMPT_METRIC_KEYS if key in metrics}


def compact_layerwise_symmetry(metrics: Mapping[str, object]) -> Dict[str, float]:
    report = metrics.get("symmetry_report", {})
    layerwise = report.get("layerwise_rotation_equivariance", {}) if isinstance(report, dict) else {}
    output = {}
    for name, value in layerwise.items():
        if isinstance(value, dict) and isinstance(value.get("maximum"), (int, float)):
            output[str(name)] = float(value["maximum"])
    return output


@dataclass(frozen=True)
class SearchMemory:
    lineage_depth: int
    valid_observations: int
    validation_mae_history: List[float]
    factor_history: List[str]
    best_validation_mae: Optional[float]
    recent_best_improvement: Optional[float]
    plateau_status: str
    mutation_regime: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def summarize_lineage(
    parent,
    get_program: Callable[[str], object],
    window: int = 4,
    minimum_points: int = 3,
    plateau_absolute_gain: float = 0.02,
) -> SearchMemory:
    """Summarize measured lineage evidence without touching the test split.

    Plateau detection uses validation MAE only when enough same-fidelity values
    exist.  Zero-step feasibility records therefore produce ``unknown`` rather
    than a fabricated convergence conclusion.
    """

    programs = []
    current = parent
    while current is not None and len(programs) < max(1, int(window)):
        programs.append(current)
        parent_id = getattr(current, "parent_id", None)
        current = get_program(parent_id) if parent_id else None
    programs.reverse()
    maes = []
    factors = []
    for program in programs:
        metrics = getattr(program, "metrics", {}) or {}
        value = metrics.get("validation_alpha_mae")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            maes.append(float(value))
        factor = (getattr(program, "metadata", {}) or {}).get("selected_factor")
        if factor:
            factors.append(str(factor))

    best = min(maes) if maes else None
    recent_gain = None
    plateau = "unknown"
    regime = "conservative"
    if len(maes) >= int(minimum_points):
        previous_best = min(maes[:-1])
        recent_gain = previous_best - min(previous_best, maes[-1])
        plateau = (
            "yes" if recent_gain < float(plateau_absolute_gain) else "no"
        )
        regime = "exploratory_within_factor" if plateau == "yes" else "conservative"

    return SearchMemory(
        lineage_depth=len(programs) - 1,
        valid_observations=len(maes),
        validation_mae_history=maes,
        factor_history=factors,
        best_validation_mae=best,
        recent_best_improvement=recent_gain,
        plateau_status=plateau,
        mutation_regime=regime,
    )
