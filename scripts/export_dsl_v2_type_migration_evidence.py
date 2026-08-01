#!/usr/bin/env python
"""Export deterministic v1-to-v2 type migration evidence for current reference flows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    Compiler,
    EquivariantTensorType,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    VALUE_TYPE_SCHEMA_VERSION,
    core_registry,
    import_equiformer_v1,
    import_equiformer_v3,
    migrate_program_v1_to_v2,
    reference_motif_registry,
)
from equivariant_nas.dsl.backends.equiformer_v1_spec import ArchitectureSpec
from equivariant_nas.dsl.backends.equiformer_v3_spec import baseline_v3_spec
from equivariant_nas.dsl.canonicalize import COMPILER_SEMANTICS_VERSION
from equivariant_nas.dsl.serialization import dumps_program


EVIDENCE_VERSION = "evoequilang-m2-type-migration-evidence@1"


def _v2_mechanism_program() -> ArchitectureProgram:
    group = GroupSpec.so3()
    irreps = Irreps.parse("2x0+2x1+2x2", group.family)
    value_type = EquivariantType(group, Carrier.NODE, irreps)
    return ArchitectureProgram(
        "1.0.0",
        "v2-mechanism",
        (InputPort("x", value_type),),
        (
            Node(
                "v2",
                "motif.v2_so2_residual_message",
                {"x": ("input:x",)},
                {"hidden_irreps": str(irreps), "frame_id": "v2-edge"},
            ),
        ),
        (OutputPort("prediction", "v2", value_type),),
    )


def _reference_programs() -> Tuple[Tuple[str, ArchitectureProgram], ...]:
    return (
        ("equiformer_v1_representation_flow", import_equiformer_v1(ArchitectureSpec())),
        ("equiformer_v2_mechanism_witness", _v2_mechanism_program()),
        ("equiformer_v3_compositional_flow", import_equiformer_v3(baseline_v3_spec())),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_evidence(output: Path, *, pytest_summary: str = "") -> Dict[str, Any]:
    output = output.resolve()
    primitives = core_registry()
    motifs = reference_motif_registry()
    compiler = Compiler(primitives, motifs)
    records = []

    for name, source in _reference_programs():
        migrated = migrate_program_v1_to_v2(source, primitives, motifs=motifs)
        artifact = compiler.analyze(migrated.program)
        all_v2 = all(
            isinstance(value, EquivariantTensorType)
            for value in artifact.inference.value_types.values()
        )
        if artifact.architecture_id != migrated.manifest.target_architecture_id or not all_v2:
            raise RuntimeError("{} failed the v2 type migration closure".format(name))

        item_dir = output / name
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "program_v2.json").write_text(dumps_program(migrated.program), encoding="utf-8")
        _write_json(item_dir / "migration_manifest.json", migrated.manifest.to_dict())
        record = {
            "name": name,
            "source_program_id": source.program_id,
            "source_node_count": len(source.nodes),
            "expanded_node_count": len(artifact.expanded_program.nodes),
            "source_architecture_id": migrated.manifest.source_architecture_id,
            "target_architecture_id": migrated.manifest.target_architecture_id,
            "migration_manifest_hash": migrated.manifest.content_hash(),
            "migrated_type_paths": list(migrated.manifest.migrated_type_paths),
            "all_inferred_values_use_v2_equivariant_tensor_type": all_v2,
            "claims": {
                "type_schema_migration": True,
                "typecheck_after_motif_expansion": True,
                "generic_lowering_revalidated_here": False,
                "official_parameter_identity": False,
                "official_full_model_reproduction": False,
            },
        }
        _write_json(item_dir / "compile_evidence.json", record)
        records.append(record)

    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "value_type_schema_version": VALUE_TYPE_SCHEMA_VERSION,
        "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
        "pytest_summary": pytest_summary,
        "reference_flow_count": len(records),
        "records": records,
        "scope": (
            "This artifact proves deterministic type-schema migration and post-expansion type checking "
            "for the current approximate/compositional V1/V2/V3 flows. It does not prove official model reproduction."
        ),
    }
    _write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("export-dsl-v2-type-migration-evidence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pytest-summary", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_evidence(Path(args.output), pytest_summary=str(args.pytest_summary))
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "reference_flow_count": summary["reference_flow_count"],
        "value_type_schema_version": summary["value_type_schema_version"],
        "target_architecture_ids": [item["target_architecture_id"] for item in summary["records"]],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
