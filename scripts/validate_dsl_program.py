#!/usr/bin/env python
"""Validate and summarize an EvoEquiLang architecture program."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from equivariant_nas.dsl import Compiler, core_registry, estimate_static_cost, reference_motif_registry
from equivariant_nas.dsl.serialization import load_program


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("program")
    args = parser.parse_args()
    program = load_program(args.program)
    compiler = Compiler(core_registry(), reference_motif_registry())
    artifact = compiler.analyze(program)
    cost = estimate_static_cost(artifact.expanded_program, artifact.inference)
    print(json.dumps({
        "valid": True,
        "architecture_id": artifact.architecture_id,
        "language_version": program.language_version,
        "source_nodes": len(program.nodes),
        "expanded_nodes": len(artifact.expanded_program.nodes),
        "open_obligations": [item.to_dict() for item in artifact.inference.open_obligations],
        "cost": {
            "parameter_estimate": cost.parameter_estimate,
            "activation_elements_per_entity": cost.activation_elements_per_entity,
            "flops_per_entity_estimate": cost.flops_per_entity_estimate,
            "unknown_nodes": list(cost.unknown_nodes),
            "certified_upper_bound": cost.certified_upper_bound,
            "model_version": cost.model_version,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
