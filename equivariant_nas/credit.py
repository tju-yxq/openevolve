"""Parent-child credit signals for factor-conditioned evolution."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def factor_credit(parent: Mapping[str, Any], child: Mapping[str, Any]) -> Dict[str, float]:
    """Return signed deltas with consistent 'larger is better' semantics."""

    def value(metrics, key, default=0.0):
        raw = metrics.get(key, default)
        return float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else default

    return {
        "mae_gain": value(parent, "validation_alpha_mae")
        - value(child, "validation_alpha_mae"),
        "speed_gain": value(parent, "step_time_ms") - value(child, "step_time_ms"),
        "parameter_reduction": value(parent, "parameter_count")
        - value(child, "parameter_count"),
        "symmetry_gain": value(parent, "max_symmetry_error")
        - value(child, "max_symmetry_error"),
    }

