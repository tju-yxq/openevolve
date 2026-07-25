import json
import subprocess
import sys

import pytest

from equivariant_nas.dsl import import_equiformer_v1
from equivariant_nas.dsl.schema import architecture_program_schema
from equivariant_nas.dsl.serialization import save_program
from equivariant_nas.spec import baseline_spec


def test_exported_json_schema_accepts_reference_program():
    jsonschema = pytest.importorskip("jsonschema")
    program = import_equiformer_v1(baseline_spec())
    jsonschema.validate(program.to_dict(), architecture_program_schema())


def test_validator_cli_reports_semantic_identity_and_no_open_obligations(tmp_path):
    path = tmp_path / "program.json"
    save_program(import_equiformer_v1(baseline_spec()), str(path))
    completed = subprocess.run(
        [sys.executable, "scripts/validate_dsl_program.py", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["valid"] is True
    assert result["architecture_id"]
    assert result["open_obligations"] == []
    assert result["expanded_nodes"] > result["source_nodes"]
