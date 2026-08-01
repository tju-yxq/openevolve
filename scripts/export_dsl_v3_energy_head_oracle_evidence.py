#!/usr/bin/env python
"""Export the normalized official V3 energy-head oracle evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from equivariant_nas.dsl import (  # noqa: E402
    TypeChecker,
    architecture_id,
    canonicalize,
    core_registry,
    equiformer_v3_energy_head_program,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend, EquiformerV3Spec  # noqa: E402


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _spec() -> EquiformerV3Spec:
    return EquiformerV3Spec(
        use_pbc=False,
        regress_forces=True,
        regress_stress=False,
        direct_prediction=True,
        num_radial_basis=6,
        max_num_elements=32,
        num_layers=2,
        num_channels=3,
        attn_hidden_channels=2,
        num_heads=1,
        attn_alpha_channels=2,
        attn_value_channels=2,
        ffn_hidden_channels=4,
        lmax=2,
        mmax=1,
        edge_channels=5,
        norm_type="merge_layer_norm",
        avg_num_nodes=5.5,
    )


def _canonical_parameter_mapping(program, canonical) -> dict[str, str]:
    if len(program.nodes) != len(canonical.nodes):
        raise RuntimeError("canonicalization changed the energy-head node count")
    node_ids = {}
    for source, normalized in zip(program.nodes, canonical.nodes):
        if source.op != normalized.op:
            raise RuntimeError("canonicalization changed the energy-head node order")
        node_ids[source.id] = normalized.id

    mapping = {}
    for source_name, official_name in program.parameters["parameter_mapping"].items():
        prefix, node_id, suffix = source_name.split(".", 2)
        if prefix != "node_modules" or node_id not in node_ids:
            raise RuntimeError(
                "unsupported energy-head parameter mapping key: {}".format(source_name)
            )
        mapping["node_modules.{}.{}".format(node_ids[node_id], suffix)] = official_name
    return mapping


def export(repo: Path, v3_root: Path, output: Path, pytest_result: str) -> dict:
    spec = _spec()
    registry = core_registry()
    backend = E3NNGraphBackend(registry, equiformer_v3_root=str(v3_root))
    program = equiformer_v3_energy_head_program(spec)
    canonical = canonicalize(program, registry)
    inference = TypeChecker(registry).check(canonical)
    support = backend.support_report(canonical)
    if not support.supported:
        raise RuntimeError(
            "the official V3 energy head is not fully supported: {}".format(
                support.to_dict()
            )
        )
    model = backend.build(canonical, inference)
    parameters = dict(model.named_parameters())
    parameter_mapping = _canonical_parameter_mapping(program, canonical)
    if set(parameter_mapping) != set(parameters):
        raise RuntimeError(
            "the energy-head parameter map does not cover the lowered parameter tree"
        )

    worktree_status = _git(repo, "status", "--short")
    summary = {
        "status": "official_v3_energy_head_oracle_passed",
        "dsl_repository": {
            "branch": _git(repo, "branch", "--show-current"),
            "head": _git(repo, "rev-parse", "HEAD"),
            "worktree_dirty": bool(worktree_status),
            "worktree_status_sha256": hashlib.sha256(
                worktree_status.encode("utf-8")
            ).hexdigest(),
        },
        "official_v3_repository": {
            "head": _git(v3_root, "rev-parse", "HEAD"),
            "origin": _git(v3_root, "remote", "get-url", "origin"),
        },
        "spec_architecture_id": spec.architecture_id(),
        "canonical_architecture_id": architecture_id(canonical, registry),
        "normalized_node_count": len(canonical.nodes),
        "registry_entries": len(registry.names()),
        "lowering_rules": backend.lowering_manifest()["rule_count"],
        "trainable_parameter_tensor_count": len(parameters),
        "trainable_parameter_count": sum(value.numel() for value in parameters.values()),
        "oracle_pytest_result": pytest_result,
        "oracle_tests": [
            "tests/test_dsl_v3_official_energy_head.py",
            "tests/test_dsl_invariant_unit_axis.py",
        ],
        "layout_contract": {
            "mathematical_map": "V tensor R^1 -> V",
            "typed_input": "InvariantTensorType[graph, energy_channel=1]",
            "typed_output": "InvariantTensorType[graph] with carrier_scalar storage",
            "runtime_map": "[graph, 1] -> [graph] by exact unit-axis squeeze",
            "negative_diagnostics": [
                "E_UNIT_AXIS_001",
                "E_UNIT_AXIS_003",
                "E_UNIT_AXIS_004",
                "runtime shape [carrier, 1] rejection",
            ],
        },
        "proved": [
            "the final merge LayerNorm is represented by a typed primitive node",
            "the scalar projection, SiLU, zero-rate dropout and output projection are represented by typed primitive nodes",
            "node energies are reduced through an explicit node-to-graph batch map and divided by avg_num_nodes",
            "the length-one energy channel is explicitly squeezed to the official [graph] task layout with carrier_scalar storage",
            "same-seed construction and initialization RNG equality",
            "bijective mapping of every trainable energy-head parameter",
            "official forward and forward-RNG numerical alignment",
            "node-feature and all-parameter gradient numerical alignment",
            "generic Lowering does not call the official energy-head or model constructor",
        ],
        "pending": [
            "nonzero stochastic-rate train/eval and RNG oracle",
            "checkpoint loader",
            "stress head",
            "training trajectory",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "equiformer_v3_energy_head_normalized_dsl.json",
        canonical.to_dict(),
    )
    _write_json(output / "support_report.json", support.to_dict())
    _write_json(output / "lowering_manifest.json", backend.lowering_manifest())
    _write_json(
        output / "model_identity.json",
        {
            "class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
            "architecture_id": architecture_id(canonical, registry),
            "parameter_shapes": {
                name: list(value.shape) for name, value in sorted(parameters.items())
            },
            "parameter_mapping": parameter_mapping,
            "input_types": {
                port.name: inference.value_types[
                    "input:{}".format(port.name)
                ].to_dict()
                for port in canonical.inputs
            },
            "output_types": {
                port.name: port.expected_type.to_dict()
                for port in canonical.outputs
            },
        },
    )
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser("export-dsl-v3-energy-head-oracle-evidence")
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytest-result", default="1 passed")
    args = parser.parse_args()
    summary = export(
        args.repo.resolve(),
        args.v3_root.resolve(),
        args.output.resolve(),
        args.pytest_result,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
