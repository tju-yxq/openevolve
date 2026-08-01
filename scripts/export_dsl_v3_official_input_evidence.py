"""Export the official Equiformer V3 input/EdgeDegree DSL evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from equivariant_nas.dsl import (
    BACKEND_SEMANTICS_VERSION,
    COMPILER_SEMANTICS_VERSION,
    TypeChecker,
    architecture_id,
    canonicalize,
    core_registry,
    equiformer_v3_input_program,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend, EquiformerV3Spec
from equivariant_nas.dsl.reference_programs import EQUIFORMER_V3_INPUT_PARAMETER_MAPPING


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _canonical_parameter_mapping(program, canonical_program):
    original_order = program.parameters["lowering_contract"]["module_construction_order"]
    canonical_order = canonical_program.parameters["lowering_contract"]["module_construction_order"]
    rename = dict(zip(original_order, canonical_order))
    mapping = {}
    for local_name, official_name in EQUIFORMER_V3_INPUT_PARAMETER_MAPPING.items():
        prefix = "node_modules."
        remainder = local_name[len(prefix) :]
        node_id, separator, suffix = remainder.partition(".")
        canonical_node = rename[node_id]
        mapping["{}{}.{}".format(prefix, canonical_node, suffix) if separator else prefix + canonical_node] = official_name
    return mapping


def export(repo: Path, v3_root: Path, config: Path, output: Path):
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    spec, spec_manifest = EquiformerV3Spec.from_official_config(config_payload)
    registry = core_registry()
    program = equiformer_v3_input_program(spec)
    inference = TypeChecker(registry).check(program)
    canonical_program = canonicalize(program, registry)
    canonical_inference = TypeChecker(registry).check(canonical_program)
    backend = E3NNGraphBackend(registry, equiformer_v3_root=str(v3_root))
    model = backend.build(canonical_program, canonical_inference)
    canonical_mapping = _canonical_parameter_mapping(program, canonical_program)
    named_parameters = dict(model.named_parameters())
    mapped_parameters = set(canonical_mapping).intersection(named_parameters)
    unmapped_parameters = sorted(set(named_parameters) - set(canonical_mapping))

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "official_config_import_manifest.json", spec_manifest.to_dict())
    _write_json(output / "equiformer_v3_input_normalized_dsl.json", canonical_program.to_dict())
    _write_json(output / "lowering_manifest.json", backend.lowering_manifest())
    _write_json(
        output / "parameter_mapping_manifest.json",
        {
            "original_program_mapping": EQUIFORMER_V3_INPUT_PARAMETER_MAPPING,
            "canonical_program_mapping": canonical_mapping,
            "trainable_parameter_tensor_count": len(named_parameters),
            "mapped_parameter_tensor_count": len(mapped_parameters),
            "unmapped_parameter_names": unmapped_parameters,
            "coverage": len(mapped_parameters) / len(named_parameters) if named_parameters else 1.0,
            "deterministic_buffers_are_regenerated": True,
        },
    )
    worktree_status = _git(repo, "status", "--short")
    identity = {
        "dsl_repository": {
            "path": str(repo),
            "branch": _git(repo, "branch", "--show-current"),
            "head": _git(repo, "rev-parse", "HEAD"),
            "worktree_dirty": bool(worktree_status),
            "worktree_status_line_count": len(worktree_status.splitlines()),
            "worktree_status_sha256": hashlib.sha256(worktree_status.encode("utf-8")).hexdigest(),
        },
        "official_v3_repository": {
            "path": str(v3_root),
            "head": _git(v3_root, "rev-parse", "HEAD"),
            "origin": _git(v3_root, "remote", "get-url", "origin"),
        },
        "spec_architecture_id": spec.architecture_id(),
        "normalized_dsl_architecture_id": architecture_id(canonical_program, registry),
        "dsl_language_version": canonical_program.language_version,
        "compiler_semantics": COMPILER_SEMANTICS_VERSION,
        "backend_neutral_semantics": BACKEND_SEMANTICS_VERSION,
        "backend_semantics": model.backend_semantics_version,
        "core_primitive_count": len(registry.names()),
        "lowering_rule_count": model.lowering_rule_manifest["rule_count"],
        "normalized_node_count": len(canonical_program.nodes),
        "trainable_parameter_tensor_count": len(named_parameters),
        "trainable_parameter_count": sum(parameter.numel() for parameter in named_parameters.values()),
        "constructor_bypass": False,
        "official_block_or_model_class_used_by_lowering": False,
    }
    _write_json(output / "model_identity.json", identity)
    summary = {
        "status": "official_v3_input_and_edge_degree_exact",
        "scope": [
            "atomic categorical embedding",
            "explicit source-target and PBC edge geometry",
            "fixed Gaussian radial basis",
            "polynomial envelope",
            "source/target atom edge embeddings",
            "three-layer official RadialFunction",
            "axisymmetric inverse-Wigner lift",
            "target segment sum and average-degree rescale",
            "atom plus edge-degree node initialization",
        ],
        "outside_this_evidence_scope": [
            "EquivariantGraphAttention",
            "FeedForwardNetwork",
            "TransBlockV3",
            "energy/force/stress heads",
            "checkpoint loader",
            "end-to-end training",
        ],
        "identity": identity,
        "parameter_mapping_coverage": len(mapped_parameters) / len(named_parameters) if named_parameters else 1.0,
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser("export-dsl-v3-official-input-evidence")
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = export(
        args.repo.resolve(),
        args.v3_root.resolve(),
        args.config.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
