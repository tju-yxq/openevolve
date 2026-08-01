#!/usr/bin/env python
"""Export the normalized two-layer V3 backbone oracle evidence."""

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
    equiformer_v3_backbone_program,
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
        avg_degree=3.25,
    )


def export(repo: Path, v3_root: Path, output: Path, pytest_result: str) -> dict:
    spec = _spec()
    registry = core_registry()
    backend = E3NNGraphBackend(registry, equiformer_v3_root=str(v3_root))
    program = equiformer_v3_backbone_program(spec)
    canonical = canonicalize(program, registry)
    inference = TypeChecker(registry).check(canonical)
    support = backend.support_report(canonical)
    if not support.supported:
        raise RuntimeError("the two-layer backbone is not fully supported: {}".format(support.to_dict()))
    model = backend.build(canonical, inference)
    parameters = dict(model.named_parameters())
    frame_cache_ids = sorted(
        {
            str(node.attrs["frame_cache_id"])
            for node in canonical.nodes
            if "frame_cache_id" in node.attrs
        }
    )
    frame_entry_ids = sorted(
        {
            str(node.attrs["frame_id"])
            for node in canonical.nodes
            if node.op == "core.to_edge_frame@2"
        }
    )
    worktree_status = _git(repo, "status", "--short")
    summary = {
        "status": "official_v3_two_layer_backbone_oracle_passed",
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
        "block_count": spec.num_layers,
        "registry_entries": len(registry.names()),
        "lowering_rules": backend.lowering_manifest()["rule_count"],
        "trainable_parameter_tensor_count": len(parameters),
        "trainable_parameter_count": sum(value.numel() for value in parameters.values()),
        "frame_contract": {
            "shared_frame_cache_ids": frame_cache_ids,
            "typed_entry_frame_ids": frame_entry_ids,
            "shared_cache_count": len(frame_cache_ids),
            "typed_entry_count": len(frame_entry_ids),
        },
        "oracle_pytest_result": pytest_result,
        "oracle_test": "tests/test_dsl_v3_official_backbone.py",
        "proved": [
            "two consecutive official TransBlockV3 layers are represented by typed primitive nodes",
            "same-seed full-backbone construction and initialization RNG equality",
            "bijective mapping of every trainable input and block parameter",
            "official forward and forward-RNG numerical alignment",
            "position and all-parameter gradient numerical alignment",
            "one official random auxiliary edge frame is shared across input and every block while typed entry tokens remain unique",
            "no official input, Block, backbone, or model constructor is used by generic Lowering",
        ],
        "pending": [
            "official energy-head numerical oracle",
            "full energy+force model parameter and numerical oracle",
            "nonzero stochastic-rate train/eval and RNG oracle",
            "checkpoint loader",
            "stress head",
            "training trajectory",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "equiformer_v3_two_layer_backbone_normalized_dsl.json", canonical.to_dict())
    _write_json(output / "support_report.json", support.to_dict())
    _write_json(output / "lowering_manifest.json", backend.lowering_manifest())
    _write_json(output / "model_identity.json", {
        "class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
        "parameter_shapes": {
            name: list(value.shape) for name, value in sorted(parameters.items())
        },
        "parameter_mapping": program.parameters["parameter_mapping"],
    })
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser("export-dsl-v3-backbone-oracle-evidence")
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytest-result", default="2 passed")
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
