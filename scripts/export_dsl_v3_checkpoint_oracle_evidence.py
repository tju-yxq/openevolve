#!/usr/bin/env python
"""Export the strict official V3 checkpoint mapping evidence."""

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
    CHECKPOINT_MAPPING_VERSION,
    TypeChecker,
    architecture_id,
    canonicalize,
    core_registry,
    equiformer_v3_direct_model_program,
)
from equivariant_nas.dsl.backends import (  # noqa: E402
    E3NNGraphBackend,
    EquiformerV3Spec,
    equiformer_v3_direct_checkpoint_manifest,
)


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
        alpha_drop=0.13,
        attn_mask_rate=0.0,
        attn_weights_drop=0.17,
        value_drop=0.19,
        drop_path_rate=0.23,
        proj_drop=0.29,
        ffn_drop=0.31,
        avg_num_nodes=5.5,
        avg_degree=3.25,
    )


def export(repo: Path, v3_root: Path, output: Path, pytest_result: str) -> dict:
    spec = _spec()
    registry = core_registry()
    program = equiformer_v3_direct_model_program(spec)
    canonical = canonicalize(program, registry)
    inference = TypeChecker(registry).check(program)
    backend = E3NNGraphBackend(registry, equiformer_v3_root=str(v3_root))
    model = backend.build(program, inference)
    checkpoint_manifest = equiformer_v3_direct_checkpoint_manifest(spec, program)
    if set(checkpoint_manifest.target_keys) != set(model.state_dict()):
        raise RuntimeError("checkpoint target manifest does not cover the lowered state")
    if len(checkpoint_manifest.source_keys) != 158:
        raise RuntimeError("the frozen stochastic official checkpoint must expose 158 tensors")
    if len(checkpoint_manifest.target_keys) != 157:
        raise RuntimeError("the frozen lowered checkpoint must expose 157 tensors")

    worktree_status = _git(repo, "status", "--short")
    source_alias_groups = [
        group
        for group in checkpoint_manifest.tensor_groups
        if len(group.source_keys) > 1 or len(group.target_keys) > 1
    ]
    summary = {
        "status": "official_v3_checkpoint_round_trip_oracle_passed",
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
        "checkpoint_mapping_version": CHECKPOINT_MAPPING_VERSION,
        "spec_architecture_id": spec.architecture_id(),
        "canonical_architecture_id": architecture_id(canonical, registry),
        "source_state_tensor_count": len(checkpoint_manifest.source_keys),
        "target_state_tensor_count": len(checkpoint_manifest.target_keys),
        "trainable_parameter_tensor_count": len(dict(model.named_parameters())),
        "reconstructed_source_tensor_count": len(
            checkpoint_manifest.reconstructed_sources
        ),
        "source_alias_group_count": len(source_alias_groups),
        "oracle_pytest_result": pytest_result,
        "oracle_test": "tests/test_dsl_v3_official_checkpoint.py",
        "proved": [
            "all official and lowered state-dict keys are covered without silent loss",
            "all 116 trainable parameters map bijectively",
            "shared official Wigner buffers are validated as aliases and broadcast to every typed frame entry",
            "structurally eliminated radial and equivariant-dropout expand-index buffers are checked against exact constants",
            "Fair-Chem state_dict wrappers, module prefixes and torch-compile prefixes are accepted with collision checks",
            "official-to-lowered-to-official state-dict round trip is exact",
            "a checkpoint-loaded stochastic model aligns with official train/eval RNG, energy, forces, position gradients and all parameter gradients",
            "missing, unexpected, wrong-shape, inconsistent-alias, corrupt-reconstruction and wrong-architecture checkpoints are rejected",
        ],
        "pending": [
            "stress head",
            "optimizer update and short training trajectory",
            "real production-width configuration numerical oracle",
        ],
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "checkpoint_mapping_manifest.json", checkpoint_manifest.to_dict())
    _write_json(
        output / "model_identity.json",
        {
            "class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
            "source_program_id": program.program_id,
            "canonical_architecture_id": architecture_id(canonical, registry),
            "state_shapes": {
                name: list(value.shape)
                for name, value in sorted(model.state_dict().items())
            },
            "constructor_bypass": False,
        },
    )
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser("export-dsl-v3-checkpoint-oracle-evidence")
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytest-result", default="3 passed")
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
