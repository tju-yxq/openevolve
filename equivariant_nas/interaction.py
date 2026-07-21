"""Counterfactual credit for interacting architecture factors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Set


@dataclass(frozen=True)
class InteractionContrast:
    baseline_mae: float
    factor_a_only_mae: float
    factor_b_only_mae: float
    joint_mae: float
    factor_a_gain: float
    factor_b_gain_without_a: float
    factor_b_gain_with_a: float
    epistasis_mae: float

    def to_dict(self):
        return asdict(self)


def interaction_contrast(
    baseline_mae: float,
    factor_a_only_mae: float,
    factor_b_only_mae: float,
    joint_mae: float,
) -> InteractionContrast:
    """Compute a 2x2 loss contrast; negative epistasis improves MAE synergistically."""

    f00 = float(baseline_mae)
    f10 = float(factor_a_only_mae)
    f01 = float(factor_b_only_mae)
    f11 = float(joint_mae)
    return InteractionContrast(
        baseline_mae=f00,
        factor_a_only_mae=f10,
        factor_b_only_mae=f01,
        joint_mae=f11,
        factor_a_gain=f00 - f10,
        factor_b_gain_without_a=f00 - f01,
        factor_b_gain_with_a=f10 - f11,
        epistasis_mae=f11 - f10 - f01 + f00,
    )


def rescue_requires_counterfactual(
    ancestor_mae: float,
    parent_mae: float,
    child_mae: float,
    minimum_rescue_gain: float = 0.1,
) -> bool:
    """Trigger a sibling evaluation when a child rescues a degraded parent."""

    parent_degraded = float(parent_mae) > float(ancestor_mae)
    child_rescue = float(parent_mae) - float(child_mae) >= float(minimum_rescue_gain)
    return parent_degraded and child_rescue


def resolved_counterfactual_credits(path: str, applied: Set[str]) -> List[Dict[str, object]]:
    """Read newly resolved IACC main effects for router feedback.

    The immediate rescue-chain gain is intentionally not returned.  Credit is
    the counterfactual main effect without the earlier factor, preventing an
    interaction-specific rescue from being learned as a generally useful edit.
    """

    source = Path(path)
    if not path or not source.exists():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("resolved counterfactual file must contain a JSON list")
    output = []
    for record in payload:
        if record.get("status") != "resolved":
            continue
        key = "{}:{}".format(
            record.get("child_architecture_id", ""),
            record.get("counterfactual_architecture_id", ""),
        )
        if not key.strip(":") or key in applied:
            continue
        contrast = record.get("interaction_contrast", {})
        output.append(
            {
                "key": key,
                "selected_factor": str(record["selected_factor"]),
                "mae_gain": float(contrast["factor_b_gain_without_a"]),
                "epistasis_mae": float(contrast["epistasis_mae"]),
                "source": str(source),
            }
        )
    return output
