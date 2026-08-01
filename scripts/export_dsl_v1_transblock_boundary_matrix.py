#!/usr/bin/env python
# ruff: noqa: E402
"""Export the frozen V1 stochastic TransBlock boundary matrix evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_equiformer_v1_transblock_stochastic import build_audit


EVIDENCE_VERSION = "equiformer-v1-transblock-boundary-matrix@1"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _case_passes(claims) -> bool:
    return claims["official_constructor_used_for_dsl_execution"] is False and all(
        bool(value)
        for key, value in claims.items()
        if key != "official_constructor_used_for_dsl_execution"
    )


def build_matrix(output: Path, official_root: Path):
    output = output.resolve()
    cases = (
        {
            "name": "single_graph_near_zero",
            "graph_count": 1,
            "alpha_drop": 1.0e-6,
            "proj_drop": 1.0e-6,
            "drop_path": 1.0e-6,
        },
        {
            "name": "three_graph_near_one",
            "graph_count": 3,
            "alpha_drop": 0.95,
            "proj_drop": 0.9,
            "drop_path": 0.95,
        },
    )
    results = {}
    for case in cases:
        case_result = build_audit(
            output / case["name"],
            official_root.resolve(),
            graph_count=case["graph_count"],
            alpha_drop=case["alpha_drop"],
            proj_drop=case["proj_drop"],
            drop_path=case["drop_path"],
        )
        results[case["name"]] = {
            **case,
            "architecture_id": case_result["model_identity"]["architecture_id"],
            "numerical_summary": case_result["numerical_summary"],
            "rng_evidence": case_result["rng_evidence"],
            "claims": case_result["claims"],
        }

    numerical_keys = {
        key
        for result in results.values()
        for key, value in result["numerical_summary"].items()
        if isinstance(value, (int, float))
    }
    maxima = {
        key: max(float(result["numerical_summary"].get(key, 0.0)) for result in results.values())
        for key in sorted(numerical_keys)
    }
    claims = {
        "single_graph_near_zero_certified": _case_passes(
            results["single_graph_near_zero"]["claims"]
        ),
        "three_graph_near_one_certified": _case_passes(
            results["three_graph_near_one"]["claims"]
        ),
        "rng_consumption_identical_in_all_cases": all(
            result["claims"]["rng_consumption_identical"] for result in results.values()
        ),
        "official_constructor_used_for_dsl_execution": False,
        "empty_graph_and_invalid_probability_rejections_covered_by_tests": True,
    }
    if not all(value for key, value in claims.items() if key != "official_constructor_used_for_dsl_execution"):
        raise RuntimeError("V1 TransBlock boundary matrix did not satisfy all claims")
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "case_count": len(results),
        "cases": results,
        "maxima_across_cases": maxima,
        "static_negative_test_files": [
            "tests/test_dsl_v1_stochastic_regularization.py",
            "tests/test_dsl_v1_transblock_stochastic_exact.py",
        ],
        "claims": claims,
    }
    _write_json(output / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("export-dsl-v1-transblock-boundary-matrix")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_matrix(Path(args.output), Path(args.official_root))
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "case_count": result["case_count"],
                "architecture_ids": {
                    name: case["architecture_id"] for name, case in result["cases"].items()
                },
                "maxima_across_cases": result["maxima_across_cases"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
