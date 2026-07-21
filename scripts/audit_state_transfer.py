#!/usr/bin/env python
"""Audit a parent checkpoint against a child state schema without training."""

import argparse
import json
from pathlib import Path

import torch

from equivariant_nas.inheritance import analyze_transfer
from equivariant_nas.spec import ArchitectureSpec


def model_state(path):
    payload = torch.load(path, map_location="cpu")
    return payload["model"] if "model" in payload else payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-spec", required=True)
    parser.add_argument("--child-spec", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument(
        "--child-state-checkpoint",
        required=True,
        help="Used only for child tensor names/shapes; its values are never scored.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parent = ArchitectureSpec.from_json(Path(args.parent_spec).read_text(encoding="utf-8"))
    child = ArchitectureSpec.from_json(Path(args.child_spec).read_text(encoding="utf-8"))
    report = analyze_transfer(
        parent,
        child,
        model_state(args.parent_checkpoint),
        model_state(args.child_state_checkpoint),
    ).to_dict()
    report["child_checkpoint_values_used_for_scoring"] = False
    report["optimizer_state_inherited"] = False
    report["purpose"] = "calibration-only acceleration proxy"
    report["test_split_allowed"] = False
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
