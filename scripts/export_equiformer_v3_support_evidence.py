#!/usr/bin/env python
"""Export canonical Equiformer V3 and F6.3 compositional-lowering evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from equivariant_nas.dsl import (  # noqa: E402
    ArchitectureProgram,
    Carrier,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    TypeChecker,
    architecture_id,
    canonicalize,
    core_registry,
    expand_motifs,
    import_equiformer_v1,
    import_equiformer_v3,
    reference_motif_registry,
)
from equivariant_nas.dsl.backends import (  # noqa: E402
    E3NNGraphBackend,
    EquiformerV3Spec,
    V3_REFERENCE_COMMIT,
    baseline_spec,
    baseline_v3_spec,
    resolve_equiformer_v3_package_path,
)
from equivariant_nas.dsl.backends.e3nn_backend import to_e3nn_irreps  # noqa: E402


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(args, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd)] + list(args),
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def _git_optional(args, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(cwd)] + list(args),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _repository_identity(path: Path):
    status = _git(["status", "--porcelain=v1", "--untracked-files=all"], path)
    tracked_status = _git(["status", "--porcelain=v1", "--untracked-files=no"], path)
    return {
        "path": str(path.resolve()),
        "head": _git(["rev-parse", "HEAD"], path),
        "branch": _git(["branch", "--show-current"], path),
        "origin": _git_optional(["remote", "get-url", "origin"], path),
        "dirty": bool(status),
        "tracked_dirty": bool(tracked_status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_status_sha256": hashlib.sha256(tracked_status.encode("utf-8")).hexdigest(),
        "status_line_count": len(status.splitlines()) if status else 0,
    }


def _implementation_identity():
    relative_paths = (
        "equivariant_nas/dsl/__init__.py",
        "equivariant_nas/dsl/registry.py",
        "equivariant_nas/dsl/reference_motifs.py",
        "equivariant_nas/dsl/reference_programs.py",
        "equivariant_nas/dsl/backends/__init__.py",
        "equivariant_nas/dsl/backends/e3nn_backend.py",
        "equivariant_nas/dsl/backends/v2_runtime.py",
        "equivariant_nas/dsl/backends/v3_runtime.py",
        "equivariant_nas/dsl/backends/equiformer_v3_spec.py",
        "tests/test_equiformer_v3_compositional_support.py",
        "scripts/export_equiformer_v3_support_evidence.py",
    )
    files = {}
    aggregate = hashlib.sha256()
    for relative in relative_paths:
        path = REPOSITORY_ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = digest
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
    return {"aggregate_sha256": aggregate.hexdigest(), "files": files}


def _small_v3_spec():
    return EquiformerV3Spec(
        num_layers=1,
        num_channels=2,
        attn_hidden_channels=2,
        ffn_hidden_channels=3,
        num_heads=1,
        lmax=2,
        mmax=1,
        attn_grid_resolution=(8, 6),
        ffn_grid_resolution=(8, 8),
        regress_forces=False,
        regress_stress=False,
    )


def _f63_program():
    parent = import_equiformer_v1(baseline_spec())
    graph_scalar = parent.outputs[0].expected_type
    nodes = tuple(
        Node(
            "graph_pool",
            "motif.v1_multilevel_readout",
            {"terminal": ("scalar_readout",), "aux": ("block2",)},
            declared_types={"out": graph_scalar},
        )
        if node.id == "graph_pool"
        else node
        for node in parent.nodes
    )
    return replace(
        parent,
        nodes=nodes,
        program_id=parent.program_id + "_f63_block2_core",
        annotations=dict(
            parent.annotations,
            factor_id="F6.3",
            auxiliary_tap="block2",
            generic_lowering_scope="full simplified representation graph; not official V1 numerical reconstruction",
        ),
    )


def _s2_error_curve(v3_root: str):
    import torch

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    torch.manual_seed(123)
    group = GroupSpec.so3()
    input_irreps = Irreps.parse("4x0+4x1+4x2", "SO3")
    output_irreps = Irreps.parse("2x0+2x1+2x2", "SO3")
    input_type = EquivariantType(group, Carrier.NODE, input_irreps, dtype="float64")
    output_type = EquivariantType(group, Carrier.NODE, output_irreps, dtype="float64")
    features = torch.randn(6, input_irreps.dimension, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    input_action = o3.Irreps(to_e3nn_irreps(input_irreps)).D_from_matrix(rotation)
    output_action = o3.Irreps(to_e3nn_irreps(output_irreps)).D_from_matrix(rotation)
    rows = []
    for resolution in (8, 10, 14, 18, 24):
        program = ArchitectureProgram(
            "1.0.0",
            "v3_s2_resolution_probe",
            (InputPort("x", input_type),),
            (
                Node(
                    "op",
                    "core.s2_swiglu",
                    {"x": ("input:x",)},
                    {
                        "out_irreps": str(output_irreps),
                        "mmax": 2,
                        "grid_resolution": [resolution, resolution],
                    },
                ),
            ),
            (OutputPort("out", "op", output_type),),
        )
        registry = core_registry()
        inference = TypeChecker(registry).check(program)
        model = E3NNGraphBackend(registry, equiformer_v3_root=v3_root).build(program, inference).double().eval()
        reference = model({"x": features}, {})["out"]
        rotated = model({"x": features @ input_action.transpose(0, 1)}, {})["out"]
        expected = reference @ output_action.transpose(0, 1)
        relative_error = (rotated - expected).norm() / expected.norm().clamp_min(1.0e-12)
        rows.append({"resolution": [resolution, resolution], "relative_rotation_error": float(relative_error)})
    return {
        "seed": 123,
        "dtype": "float64",
        "input_irreps": str(input_irreps),
        "output_irreps": str(output_irreps),
        "interpretation": "finite-grid empirical equivariance; error should decrease with resolution",
        "measurements": rows,
    }


def export(output: Path, v3_root: Path):
    output.mkdir(parents=True, exist_ok=True)
    package_path = resolve_equiformer_v3_package_path(str(v3_root))
    if package_path is None:
        raise RuntimeError("official Equiformer V3 operator source is unavailable")

    registry = core_registry()
    motifs = reference_motif_registry()
    baseline_program = expand_motifs(import_equiformer_v3(baseline_v3_spec()), motifs)
    smoke_program = expand_motifs(import_equiformer_v3(_small_v3_spec()), motifs)
    f63_program = expand_motifs(_f63_program(), motifs)
    baseline_inference = TypeChecker(registry).check(baseline_program)
    smoke_inference = TypeChecker(registry).check(smoke_program)
    f63_inference = TypeChecker(registry).check(f63_program)

    backend = E3NNGraphBackend(
        registry,
        equiformer_v2_root=str(v3_root),
        equiformer_v3_root=str(v3_root),
    )
    smoke_model = backend.build(smoke_program, smoke_inference).eval()
    f63_backend = E3NNGraphBackend(registry)
    f63_model = f63_backend.build(f63_program, f63_inference).eval()

    _write_json(output / "equiformer_v3_baseline_normalized_dsl.json", canonicalize(baseline_program, registry).to_dict())
    _write_json(output / "equiformer_v3_smoke_normalized_dsl.json", canonicalize(smoke_program, registry).to_dict())
    _write_json(output / "f63_block2_normalized_dsl.json", canonicalize(f63_program, registry).to_dict())

    lowering_evidence = {
        "registry_manifest": backend.lowering_manifest(),
        "baseline_support": backend.support_report(baseline_program).to_dict(),
        "smoke_support": backend.support_report(smoke_program).to_dict(),
        "f63_support": f63_backend.support_report(f63_program).to_dict(),
        "baseline_node_count": len(baseline_program.nodes),
        "smoke_node_count": len(smoke_program.nodes),
        "f63_node_count": len(f63_program.nodes),
        "baseline_open_obligations": [item.to_dict() for item in baseline_inference.open_obligations],
        "smoke_open_obligations": [item.to_dict() for item in smoke_inference.open_obligations],
        "f63_open_obligations": [item.to_dict() for item in f63_inference.open_obligations],
    }
    _write_json(output / "lowering_manifest.json", lowering_evidence)

    source_identity = _repository_identity(v3_root)
    source_identity.update(
        {
            "expected_official_commit": V3_REFERENCE_COMMIT,
            "commit_matches_expected": source_identity["head"] == V3_REFERENCE_COMMIT,
            "operator_package_path": str(package_path),
        }
    )
    dsl_identity = _repository_identity(REPOSITORY_ROOT)
    model_identity = {
        "official_source": source_identity,
        "dsl_worktree": dsl_identity,
        "v3_support_implementation": _implementation_identity(),
        "compiled_smoke_model": {
            "class": type(smoke_model).__name__,
            "backend_semantics_version": smoke_model.backend_semantics_version,
            "program_id": smoke_program.program_id,
            "architecture_id": architecture_id(smoke_program, registry),
            "parameter_count": sum(parameter.numel() for parameter in smoke_model.parameters()),
            "node_module_count": len(smoke_model.node_modules),
            "fused_subgraph_count": len(smoke_model.fused_subgraphs),
            "constructor_bypass": False,
            "numerical_identity": "generic primitive composition, not official Equiformer V3 checkpoint identity",
        },
        "compiled_f63_model": {
            "class": type(f63_model).__name__,
            "backend_semantics_version": f63_model.backend_semantics_version,
            "program_id": f63_program.program_id,
            "architecture_id": architecture_id(f63_program, registry),
            "parameter_count": sum(parameter.numel() for parameter in f63_model.parameters()),
            "node_module_count": len(f63_model.node_modules),
            "fused_subgraph_count": len(f63_model.fused_subgraphs),
            "readout_subgraph_fully_core_lowered": True,
            "official_v1_backbone_exact": False,
        },
    }
    _write_json(output / "model_identity.json", model_identity)
    _write_json(output / "s2_swiglu_equivariance_curve.json", _s2_error_curve(str(v3_root)))
    return {
        "output": str(output),
        "baseline_program": baseline_program.program_id,
        "baseline_nodes": len(baseline_program.nodes),
        "rules": len(registry.names()),
        "official_commit": source_identity["head"],
        "official_tracked_clean": not source_identity["tracked_dirty"],
        "smoke_model_parameters": model_identity["compiled_smoke_model"]["parameter_count"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output.resolve(), args.v3_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
