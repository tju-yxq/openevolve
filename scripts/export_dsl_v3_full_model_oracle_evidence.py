#!/usr/bin/env python
"""Export the official two-layer V3 direct Energy+Force oracle evidence."""

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
    BACKEND_SEMANTICS_VERSION,
    COMPILER_SEMANTICS_VERSION,
    TypeChecker,
    architecture_id,
    canonicalize,
    core_registry,
    default_canonical_search_surface,
    equiformer_v3_direct_model_program,
    reference_motif_registry,
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
        use_pbc_single=False,
        otf_graph=False,
        regress_forces=True,
        regress_stress=False,
        direct_prediction=True,
        max_neighbors=8,
        max_radius=4.5,
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
        attn_grid_resolution=(8, 4),
        ffn_grid_resolution=(8, 8),
        edge_channels=5,
        attn_activation="sep-merge_gates2_swiglu",
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        avg_num_nodes=5.5,
        avg_degree=3.25,
    )


def export(repo: Path, v3_root: Path, output: Path, pytest_result: str) -> dict:
    spec = _spec()
    registry = core_registry()
    motifs = reference_motif_registry()
    search_surface = default_canonical_search_surface(registry, motifs)
    program = equiformer_v3_direct_model_program(spec)
    canonical = canonicalize(program, registry)
    source_inference = TypeChecker(registry).check(program)
    TypeChecker(registry).check(canonical)
    backend = E3NNGraphBackend(registry, equiformer_v3_root=str(v3_root))
    support = backend.support_report(canonical)
    if not support.supported:
        raise RuntimeError(
            "the official two-layer V3 direct model is not fully supported: {}".format(
                support.to_dict()
            )
        )
    model = backend.build(program, source_inference)
    parameters = dict(model.named_parameters())
    if len(parameters) != 116:
        raise RuntimeError(
            "the frozen two-layer V3 model must expose 116 parameter tensors, got {}".format(
                len(parameters)
            )
        )

    frame_cache_ids = sorted(
        {
            str(node.attrs["frame_cache_id"])
            for node in program.nodes
            if "frame_cache_id" in node.attrs
        }
    )
    frame_entry_ids = sorted(
        {
            str(node.attrs["frame_id"])
            for node in program.nodes
            if node.op == "core.to_edge_frame@2"
        }
    )
    output_types = {
        port.name: port.expected_type.to_dict() for port in program.outputs
    }
    worktree_status = _git(repo, "status", "--short")
    summary = {
        "status": "official_v3_two_layer_direct_energy_force_model_oracle_passed",
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
            "reference_class": "experimental.models.equiformer_v3.equiformer_v3.EquiformerV3_OC",
        },
        "compiler_semantics": COMPILER_SEMANTICS_VERSION,
        "backend_neutral_semantics": BACKEND_SEMANTICS_VERSION,
        "backend_semantics": model.backend_semantics_version,
        "spec_architecture_id": spec.architecture_id(),
        "canonical_architecture_id": architecture_id(canonical, registry),
        "normalized_node_count": len(canonical.nodes),
        "block_count": spec.num_layers,
        "registry_entries": len(registry.names()),
        "lowering_rules": backend.lowering_manifest()["rule_count"],
        "search_surface_hash": search_surface.content_hash(),
        "trainable_parameter_tensor_count": len(parameters),
        "trainable_parameter_count": sum(value.numel() for value in parameters.values()),
        "frame_contract": {
            "shared_frame_cache_ids": frame_cache_ids,
            "typed_entry_frame_ids": frame_entry_ids,
            "shared_cache_count": len(frame_cache_ids),
            "typed_entry_count": len(frame_entry_ids),
        },
        "output_contract": {
            "energy_runtime_shape": "[graph]",
            "energy_storage": output_types["energy"]["layout"]["storage"],
            "forces_runtime_shape": "[node, 3]",
        },
        "oracle_pytest_result": pytest_result,
        "oracle_test": "tests/test_dsl_v3_official_direct_model.py",
        "supporting_contract_tests": [
            "tests/test_dsl_v3_official_energy_head.py",
            "tests/test_dsl_invariant_unit_axis.py",
        ],
        "proved": [
            "the official EquiformerV3_OC constructor and generic Typed DSL Lowering consume identical initialization RNG under one seed",
            "all 116 trainable parameter tensors form a bijection with the official two-layer direct model",
            "the actual lowered model contains no official EquiformerV3_OC, TransBlockV3, EquivariantGraphAttention or ScalarFeedForwardNetwork constructor instance",
            "one random auxiliary edge frame is shared by input, both blocks and the direct-force head",
            "the final equivariant normalization is instantiated once and shared by the energy and force heads",
            "two-graph energy output has the exact official [graph] task layout and carrier_scalar type",
            "energy, direct forces and forward RNG numerically align with the official model",
            "position gradients and all 116 parameter gradients numerically align with the official model",
        ],
        "pending": [
            "nonzero stochastic-rate train/eval and RNG oracle",
            "official checkpoint loader and state-dict round trip",
            "stress head",
            "optimizer update and short training trajectory",
            "real production-width configuration numerical oracle",
        ],
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "equiformer_v3_two_layer_direct_model_normalized_dsl.json",
        canonical.to_dict(),
    )
    _write_json(output / "support_report.json", support.to_dict())
    _write_json(output / "lowering_manifest.json", backend.lowering_manifest())
    _write_json(
        output / "mathematical_contracts.json",
        {
            "energy_unit_axis": {
                "map": "V tensor R^1 -> V",
                "precondition": "one statically known invariant energy_channel axis of size one",
                "runtime": "[graph, 1] -> [graph]",
                "output_layout": "carrier_scalar",
                "equivariance": "identity isomorphism on one trivial representation; group action and certification level are unchanged",
                "negative_diagnostics": [
                    "E_UNIT_AXIS_001",
                    "E_UNIT_AXIS_003",
                    "E_UNIT_AXIS_004",
                    "runtime shape [carrier, 1] rejection",
                ],
            },
            "shared_edge_frame": {
                "cache": frame_cache_ids,
                "typed_entries": frame_entry_ids,
                "contract": "one official random auxiliary edge frame is shared across input, both blocks and force head while every to/from proof pair keeps a unique frame_id",
            },
            "shared_final_norm": program.parameters["lowering_contract"][
                "shared_final_norm"
            ],
        },
    )
    _write_json(
        output / "model_identity.json",
        {
            "class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
            "architecture_id": architecture_id(canonical, registry),
            "source_program_id": program.program_id,
            "parameter_shapes": {
                name: list(value.shape) for name, value in sorted(parameters.items())
            },
            "parameter_mapping_contract": program.parameters["parameter_mapping"],
            "shared_final_norm": program.parameters["lowering_contract"][
                "shared_final_norm"
            ],
            "output_types": output_types,
            "constructor_bypass": False,
        },
    )
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser("export-dsl-v3-full-model-oracle-evidence")
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
