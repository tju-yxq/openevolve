#!/usr/bin/env python
"""Deterministically generate one certified structural V3 child for the online controller."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from equivariant_nas.dsl import architecture_id, core_registry
from equivariant_nas.dsl.serialization import dumps_program, load_program
from equivariant_nas.dsl.v3_structural_evolution import (
    apply_v3_structural_patch, build_v3_structural_patch,
    choose_deterministic_v3_structural_action, v3_structural_novelty_report,
)


def main():
    request = json.loads(os.environ["EQUINAS_REQUEST_JSON"])
    attempt = int(request["attempt"]); parent_record = request.get("parent")
    parent_path = str(parent_record.get("program", "")) if parent_record else os.environ.get("EQUINAS_INITIAL_PROGRAM", "")
    if not parent_path:
        raise RuntimeError("set EQUINAS_INITIAL_PROGRAM for the first online generation")
    parent = load_program(parent_path); registry = core_registry()
    seen = [str(item["architecture_id"]) for item in request.get("population", [])]
    action = choose_deterministic_v3_structural_action(parent, round_index=attempt, seen_states=seen, registry=registry)
    patch = build_v3_structural_patch(parent, action.action_id, registry)
    child, _ = apply_v3_structural_patch(parent, patch, registry)
    child_id = architecture_id(child, registry); parent_id = architecture_id(parent, registry)
    output = Path(os.environ.get("EQUINAS_RUN_ROOT", ".")) / "candidates" / f"attempt_{attempt:03d}_{child_id}"
    output.mkdir(parents=True, exist_ok=False)
    program_path = output / "candidate.dsl.json"; program_path.write_text(dumps_program(child), encoding="utf-8")
    novelty = v3_structural_novelty_report(parent, child, registry)
    (output / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    (output / "novelty.json").write_text(json.dumps(novelty, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "architecture_id": child_id, "parent_architecture_id": parent_id,
        "program": str(program_path.resolve()), "patch_path": str((output / "patch.json").resolve()),
        "novelty_score": float(novelty["graph_edit_count"]) + (0.25 if novelty["attribute_changes"] else 0.0),
        "structural_novelty": novelty,
    }, sort_keys=True))


if __name__ == "__main__": main()

