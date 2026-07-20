"""Trusted feasibility evaluator for architecture candidates."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .builder import build_equiformer, count_trainable_parameters
from .candidate import extract_literal_spec
from .spec import SpecValidationError, baseline_spec


BASELINE_PARAMETERS = 3_531_715


def static_evaluate(
    program_path: str,
    equiformer_root: str,
    parameter_ratio_limit: float = 1.2,
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        spec = extract_literal_spec(program_path)
        model = build_equiformer(spec, equiformer_root)
        parameter_count = count_trainable_parameters(model)
        parameter_ratio = parameter_count / BASELINE_PARAMETERS
        if parameter_ratio > parameter_ratio_limit:
            raise SpecValidationError(
                "parameter ratio {:.4f} exceeds {:.4f}".format(
                    parameter_ratio, parameter_ratio_limit
                )
            )
        capacity = spec.representation.capacity_profile()
        return {
            "valid": True,
            "architecture_id": spec.architecture_id(),
            "parameter_count": parameter_count,
            "parameter_ratio": parameter_ratio,
            "lmax": spec.representation.lmax,
            "num_layers": spec.macro.num_layers,
            "higher_order_fraction": capacity["higher_order_fraction"],
            "static_eval_seconds": time.perf_counter() - started,
            "failure_stage": "",
            # Static fitness is deliberately neutral. Training MAE is the real
            # objective; feasibility must not masquerade as model quality.
            "combined_score": 0.0,
        }
    except Exception as exc:
        return {
            "valid": False,
            "combined_score": -1.0e9,
            "failure_stage": "static",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "static_eval_seconds": time.perf_counter() - started,
        }


def cached_static_evaluate(
    program_path: str,
    equiformer_root: str,
    cache_dir: str,
    parameter_ratio_limit: float = 1.2,
) -> Dict[str, Any]:
    try:
        spec = extract_literal_spec(program_path)
        cache_path = Path(cache_dir) / (spec.architecture_id() + ".static.json")
    except Exception:
        return static_evaluate(program_path, equiformer_root, parameter_ratio_limit)
    if cache_path.exists():
        result = json.loads(cache_path.read_text(encoding="utf-8"))
        result["cache_hit"] = True
        return result
    result = static_evaluate(program_path, equiformer_root, parameter_ratio_limit)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["cache_hit"] = False
    return result

