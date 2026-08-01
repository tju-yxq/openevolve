#!/usr/bin/env python
"""Export the reference V1 DSL program and machine-readable JSON Schema."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from equivariant_nas.dsl import import_equiformer_v1
from equivariant_nas.dsl import ResourceContract, TaskContract
from equivariant_nas.dsl.schema import architecture_program_schema
from equivariant_nas.dsl.serialization import save_program, save_task_contract
from equivariant_nas.dsl.backends import baseline_spec
from equivariant_nas.dsl.constants import BASELINE_PARAMETERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    program = import_equiformer_v1(baseline_spec())
    save_program(program, str(output / "equiformer_v1_reference.json"))
    save_task_contract(
        TaskContract(
            "qm9_alpha",
            program.outputs[0].expected_type.group,
            program.outputs[0].expected_type,
            "qm9-alpha-fixed-step-batch64-v1",
            ResourceContract(int(BASELINE_PARAMETERS * 1.2), 80 * 1024 ** 3, 2.0),
            metadata={"dataset": "QM9", "target": "alpha", "test_policy": "final-audit-only"},
        ),
        str(output / "qm9_alpha_task_contract.json"),
    )
    (output / "evoequilang_architecture_schema.json").write_text(
        json.dumps(architecture_program_schema(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
