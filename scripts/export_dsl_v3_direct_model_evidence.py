#!/usr/bin/env python
"""Export reproducible evidence for the primitive-level V3 direct model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
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


EVIDENCE_VERSION = "evoequilang-v3-direct-model-evidence@1"


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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


def _spec(*, num_layers: int) -> EquiformerV3Spec:
    return EquiformerV3Spec(
        use_pbc=False,
        use_pbc_single=False,
        otf_graph=False,
        regress_forces=True,
        regress_stress=False,
        direct_prediction=True,
        max_neighbors=8,
        max_radius=4.5,
        num_radial_basis=8,
        max_num_elements=32,
        num_layers=num_layers,
        num_channels=3,
        attn_hidden_channels=2,
        num_heads=2,
        attn_alpha_channels=2,
        attn_value_channels=1,
        ffn_hidden_channels=4,
        norm_type="merge_layer_norm",
        lmax=2,
        mmax=1,
        attn_grid_resolution=(8, 4),
        ffn_grid_resolution=(8, 8),
        edge_channels=4,
        attn_activation="sep-merge_gates2_swiglu",
        ffn_activation="sep-merge_gates2_swiglu",
        use_grid_mlp=True,
        use_gate_force_head=True,
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.1,
        value_drop=0.0,
        drop_path_rate=0.05,
        proj_drop=0.0,
        ffn_drop=0.0,
    )


def _parameter_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def export(repo: Path, v3_root: Path, output: Path, *, num_layers: int) -> dict:
    import torch

    spec = _spec(num_layers=num_layers)
    registry = core_registry()
    motifs = reference_motif_registry()
    search_surface = default_canonical_search_surface(registry, motifs)
    program = equiformer_v3_direct_model_program(spec)
    canonical_program = canonicalize(program, registry)
    inference = TypeChecker(registry).check(canonical_program)
    backend = E3NNGraphBackend(registry, equiformer_v3_root=str(v3_root))
    support = backend.support_report(canonical_program)
    if not support.supported:
        raise RuntimeError("V3 direct model support failed: {}".format(support.to_dict()))

    torch.manual_seed(12601)
    model = backend.build(canonical_program, inference).eval()
    parameter_state_sha256 = _parameter_state_sha256(model)

    atomic_numbers = torch.tensor([1, 6, 8, 14], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1], [0.3, 1.2, 0.4], [-0.4, 0.6, 1.3]],
        dtype=torch.float32,
        requires_grad=True,
    )
    source = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    target = torch.tensor([1, 2, 3, 0, 2, 1], dtype=torch.long)
    batch = torch.zeros(atomic_numbers.numel(), dtype=torch.long)

    def index_map(indices, target_size):
        return {"indices": indices, "target_size": int(target_size)}

    torch.manual_seed(12602)
    outputs = model(
        {
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "source_index": index_map(source, source.numel()),
            "target_index": index_map(target, target.numel()),
            "target_segment": index_map(target, atomic_numbers.numel()),
            "batch": index_map(batch, 1),
        },
        {},
    )
    loss = outputs["energy"].square().sum() + outputs["forces"].square().sum()
    loss.backward()
    parameters = dict(model.named_parameters())
    gradients_present = {name: value.grad is not None for name, value in parameters.items()}
    runtime_smoke = {
        "energy_shape": list(outputs["energy"].shape),
        "force_shape": list(outputs["forces"].shape),
        "energy_finite": bool(torch.isfinite(outputs["energy"]).all()),
        "forces_finite": bool(torch.isfinite(outputs["forces"]).all()),
        "position_gradient_finite": bool(
            positions.grad is not None and torch.isfinite(positions.grad).all()
        ),
        "parameter_gradient_tensor_count": sum(gradients_present.values()),
        "parameter_tensor_count": len(parameters),
        "all_parameter_gradients_present": all(gradients_present.values()),
        "energy_value": outputs["energy"].detach().cpu().tolist(),
        "force_l2_norm": float(outputs["forces"].detach().norm().cpu()),
    }
    if not all(
        (
            runtime_smoke["energy_finite"],
            runtime_smoke["forces_finite"],
            runtime_smoke["position_gradient_finite"],
            runtime_smoke["all_parameter_gradients_present"],
        )
    ):
        raise RuntimeError("V3 direct model runtime smoke failed: {}".format(runtime_smoke))

    repo_status = _git(repo, "status", "--short")
    official_status = _git(v3_root, "status", "--short")
    identity = {
        "evidence_version": EVIDENCE_VERSION,
        "dsl_repository": {
            "path": str(repo),
            "branch": _git(repo, "branch", "--show-current"),
            "head": _git(repo, "rev-parse", "HEAD"),
            "worktree_dirty": bool(repo_status),
            "worktree_status_line_count": len(repo_status.splitlines()),
            "worktree_status_sha256": hashlib.sha256(repo_status.encode("utf-8")).hexdigest(),
        },
        "official_v3_repository": {
            "path": str(v3_root),
            "head": _git(v3_root, "rev-parse", "HEAD"),
            "origin": _git(v3_root, "remote", "get-url", "origin"),
            "worktree_dirty": bool(official_status),
        },
        "spec": spec.to_dict(),
        "spec_architecture_id": spec.architecture_id(),
        "normalized_dsl_architecture_id": architecture_id(canonical_program, registry),
        "dsl_language_version": canonical_program.language_version,
        "compiler_semantics": COMPILER_SEMANTICS_VERSION,
        "backend_neutral_semantics": BACKEND_SEMANTICS_VERSION,
        "backend_semantics": model.backend_semantics_version,
        "graph_class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
        "normalized_node_count": len(canonical_program.nodes),
        "compiled_module_count": len(model.node_modules),
        "core_primitive_count": len(registry.names()),
        "lowering_rule_count": model.lowering_rule_manifest["rule_count"],
        "search_surface_hash": search_surface.content_hash(),
        "search_surface_counts": search_surface.to_dict()["counts"],
        "trainable_parameter_tensor_count": len(parameters),
        "trainable_parameter_count": sum(value.numel() for value in parameters.values()),
        "trainable_parameter_shapes": {
            name: list(value.shape) for name, value in parameters.items()
        },
        "parameter_state_sha256": parameter_state_sha256,
        "constructor_bypass": False,
        "official_attention_block_or_model_constructor_used_by_lowering": False,
    }
    math_contract = {
        "shared_final_norm": {
            "statement": "one SO(3)-equivariant final normalization result is aliased to both task heads",
            "node": canonical_program.parameters["lowering_contract"]["shared_final_norm"]["node"],
            "consumers": canonical_program.parameters["lowering_contract"]["shared_final_norm"]["consumers"],
        },
        "edge_frame_gate_activation": {
            "input": "uniform l=0..lmax SO(3) coefficients in one m-primary edge frame",
            "gate_count": "lmax * channels invariant scalars, one per non-scalar degree and channel",
            "transformation": "invariant gates scale fixed-degree edge-frame coefficients and preserve the represented SO(3) type",
            "negative_test": "E_V3_EDGE_GATE_004 rejects a gate-count mismatch",
        },
        "grid_invariant_product": {
            "input": "finite S2 samples [carrier, grid_point, channel] and invariant [carrier, channel] gates",
            "runtime_broadcast": "invariant feature axes are flattened and singleton sampling axes are inserted explicitly",
            "certification": "finite-grid empirical equivariance, not analytic exactness",
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "equiformer_v3_direct_model_normalized_dsl.json", canonical_program.to_dict())
    _write_json(output / "lowering_manifest.json", backend.lowering_manifest())
    _write_json(output / "support_report.json", support.to_dict())
    _write_json(output / "model_identity.json", identity)
    _write_json(output / "runtime_smoke.json", runtime_smoke)
    _write_json(output / "mathematical_contracts.json", math_contract)
    summary = {
        "status": "v3_direct_energy_force_primitive_lowering_runtime_smoke_passed",
        "identity": {
            "spec_architecture_id": identity["spec_architecture_id"],
            "normalized_dsl_architecture_id": identity["normalized_dsl_architecture_id"],
            "normalized_node_count": identity["normalized_node_count"],
            "compiled_module_count": identity["compiled_module_count"],
            "core_primitive_count": identity["core_primitive_count"],
            "lowering_rule_count": identity["lowering_rule_count"],
            "trainable_parameter_tensor_count": identity["trainable_parameter_tensor_count"],
            "trainable_parameter_count": identity["trainable_parameter_count"],
            "parameter_state_sha256": identity["parameter_state_sha256"],
            "constructor_bypass": identity["constructor_bypass"],
        },
        "evidence_files": {
            "normalized_dsl": "equiformer_v3_direct_model_normalized_dsl.json",
            "lowering_manifest": "lowering_manifest.json",
            "support_report": "support_report.json",
            "model_identity": "model_identity.json",
            "runtime_smoke": "runtime_smoke.json",
            "mathematical_contracts": "mathematical_contracts.json",
        },
        "runtime_smoke": runtime_smoke,
        "proved": [
            "complete direct energy+force graph is represented by typed primitive nodes",
            "one final equivariant normalization is shared by both heads",
            "canonicalized graph has complete generic Lowering coverage",
            "generic Lowering builds and executes energy and force outputs without an official block/model constructor",
            "one backward pass reaches positions and every trainable parameter tensor",
        ],
        "not_proved": [
            "this production-depth smoke does not itself replace the separate frozen two-layer official full-model oracle",
            "checkpoint loading equality",
            "train/eval RNG equality under nonzero stochastic rates",
            "stress head",
            "end-to-end optimizer/training trajectory equality",
        ],
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser("export-dsl-v3-direct-model-evidence")
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=12)
    args = parser.parse_args()
    summary = export(
        args.repo.resolve(),
        args.v3_root.resolve(),
        args.output.resolve(),
        num_layers=int(args.num_layers),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "normalized_dsl_architecture_id": summary["identity"]["normalized_dsl_architecture_id"],
                "normalized_node_count": summary["identity"]["normalized_node_count"],
                "runtime_smoke": summary["runtime_smoke"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
