#!/usr/bin/env python
"""Build a compact, atomic status index for a continuous evolution run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def write_atomic(path: Path, text: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    evolution = read_jsonl(run / "evolution.jsonl")
    heartbeat = read_json(run / "heartbeat.json") or {}
    current = read_json(run / "current_candidate.json")
    candidates = []
    for record in evolution:
        metrics = record.get("metrics") or {}
        if record.get("kind") == "initial" or record.get("iteration", 0) > 0:
            candidates.append(
                {
                    "iteration": record.get("iteration"),
                    "kind": record.get("kind", "child"),
                    "architecture_id": metrics.get("architecture_id")
                    or record.get("architecture_id"),
                    "parent_id": record.get("parent_id"),
                    "selected_factor": record.get("selected_factor"),
                    "valid": bool(metrics.get("valid")),
                    "validation_alpha_mae": metrics.get("validation_alpha_mae"),
                    "best_validation_alpha_mae": metrics.get(
                        "best_validation_alpha_mae"
                    ),
                    "parameter_count": metrics.get("parameter_count"),
                    "parameter_ratio": metrics.get("parameter_ratio"),
                    "step_time_ms": metrics.get("step_time_ms"),
                    "training_wall_time_sec": metrics.get(
                        "training_wall_time_sec"
                    ),
                    "failure_stage": metrics.get("failure_stage"),
                    "error": metrics.get("error") or record.get("error"),
                    "credit_status": record.get("credit_status"),
                }
            )
    partial = None
    if current:
        architecture_id = current.get("architecture_id")
        matches = list(
            Path("/home/20262202788/equivariant-nas/runs/candidates").glob(
                "{}/seed0_steps5000_*/training/progress.json".format(architecture_id)
            )
        )
        progress = read_json(matches[-1]) if matches else None
        checkpoint = (
            matches[-1].with_name("checkpoint_last.pth") if matches else None
        )
        partial = dict(current)
        partial["progress"] = progress
        partial["checkpoint"] = str(checkpoint) if checkpoint and checkpoint.exists() else None
    valid = [
        item
        for item in candidates
        if item["valid"] and item["validation_alpha_mae"] is not None
    ]
    best = min(valid, key=lambda item: item["validation_alpha_mae"]) if valid else None
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "heartbeat": heartbeat,
        "completed_candidates": candidates,
        "completed_count": len(candidates),
        "valid_trained_count": len(valid),
        "best": best,
        "current_candidate": partial,
        "test_split_used": any(
            bool((record.get("metrics") or {}).get("test_evaluated"))
            for record in evolution
        ),
    }
    write_atomic(
        run / "candidate_index.json",
        json.dumps(payload, indent=2, sort_keys=True),
    )
    lines = [
        "# Continuous Equiformer NAS Status",
        "",
        "Updated: `{}`".format(payload["updated_at"]),
        "",
        "- Completed records: {}".format(len(candidates)),
        "- Trained-valid candidates: {}".format(len(valid)),
        "- Test split used: {}".format(payload["test_split_used"]),
    ]
    if best:
        lines.append(
            "- Best validation MAE: `{:.9f} a0^3` (iteration {}, `{}`)".format(
                best["validation_alpha_mae"],
                best["iteration"],
                best["architecture_id"],
            )
        )
    if partial:
        progress = partial.get("progress") or {}
        lines.extend(
            [
                "",
                "## Current Candidate",
                "",
                "- Iteration: `{}`".format(partial.get("iteration")),
                "- Architecture: `{}`".format(partial.get("architecture_id")),
                "- Factor: `{}`".format(partial.get("selected_factor")),
                "- Global step: `{}`".format(progress.get("global_step")),
                "- Checkpoint: `{}`".format(partial.get("checkpoint")),
            ]
        )
    lines.extend(
        [
            "",
            "## Completed Candidates",
            "",
            "| Iteration | Architecture | Factor | Valid | Val MAE | Params ratio | Step ms |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for item in candidates:
        lines.append(
            "| {} | `{}` | {} | {} | {} | {} | {} |".format(
                item.get("iteration"),
                item.get("architecture_id") or "",
                item.get("selected_factor") or item.get("kind"),
                item.get("valid"),
                "{:.9f}".format(item["validation_alpha_mae"])
                if item.get("validation_alpha_mae") is not None
                else "",
                "{:.4f}".format(item["parameter_ratio"])
                if item.get("parameter_ratio") is not None
                else "",
                "{:.1f}".format(item["step_time_ms"])
                if item.get("step_time_ms") is not None
                else "",
            )
        )
    write_atomic(run / "STATUS.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
