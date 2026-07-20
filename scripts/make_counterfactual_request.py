#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from equivariant_nas.candidate import extract_literal_spec
from equivariant_nas.spec import EvolutionFactor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ancestor-program", required=True)
    parser.add_argument("--child-program", required=True)
    parser.add_argument("--selected-factor", choices=[item.value for item in EvolutionFactor], required=True)
    parser.add_argument("--ancestor-mae", type=float, required=True)
    parser.add_argument("--parent-mae", type=float, required=True)
    parser.add_argument("--child-mae", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ancestor = extract_literal_spec(args.ancestor_program)
    child = extract_literal_spec(args.child_program)
    factor = EvolutionFactor(args.selected_factor)
    request = {
        "iteration": 2,
        "selected_factor": factor.value,
        "ancestor_architecture_id": ancestor.architecture_id(),
        "child_architecture_id": child.architecture_id(),
        "ancestor_mae": args.ancestor_mae,
        "parent_mae": args.parent_mae,
        "child_mae": args.child_mae,
        "ancestor_spec": ancestor.to_dict(),
        "factor_replacement": child.to_dict()[factor.value.lower()],
        "status": "required_before_mae_credit",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(request, sort_keys=True))


if __name__ == "__main__":
    main()
