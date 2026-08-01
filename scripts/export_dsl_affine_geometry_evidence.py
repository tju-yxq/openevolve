#!/usr/bin/env python
"""Export auditable evidence for affine coordinates and periodic displacement.

This exporter exercises only typed primitive programs and the generic e3nn
node-graph Lowering path.  It deliberately does not call an official model
constructor, so the evidence cannot be confused with official Equiformer
V1/V2/V3 reproduction.
"""

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
    AffinePointType,
    ArchitectureProgram,
    Carrier,
    Compiler,
    EquivariantTensorType,
    GroupSpec,
    IndexMapType,
    InputPort,
    Irreps,
    LatticeShiftType,
    LatticeType,
    Node,
    OutputPort,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.serialization import dumps_program


EVIDENCE_VERSION = "evoequilang-affine-geometry-evidence@1"
SEED = 20260731


def _endpoint_types(group: GroupSpec, edge_count: int) -> Tuple[IndexMapType, IndexMapType]:
    return (
        IndexMapType(group, Carrier.NODE, Carrier.EDGE, "source", target_size=edge_count),
        IndexMapType(group, Carrier.NODE, Carrier.EDGE, "target", target_size=edge_count),
    )


def _nonperiodic_program() -> ArchitectureProgram:
    group = GroupSpec.o3()
    positions = AffinePointType(group, dtype="float64", measure="angstrom")
    source, target = _endpoint_types(group, 3)
    vectors = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x1o", group.family),
        dtype="float64",
        measure="angstrom",
    )
    return ArchitectureProgram(
        "2.1.0",
        "affine-geometry-evidence-nonperiodic",
        (
            InputPort("positions", positions),
            InputPort("source_index", source),
            InputPort("target_index", target),
        ),
        (
            Node(
                "displacement",
                "core.relative_displacement@2",
                {
                    "positions": ("input:positions",),
                    "source_index": ("input:source_index",),
                    "target_index": ("input:target_index",),
                },
            ),
        ),
        (OutputPort("vectors", "displacement", vectors),),
        program_id="affine-geometry-nonperiodic@1",
    )


def _periodic_program() -> ArchitectureProgram:
    group = GroupSpec("SO3", 3, periodicity="lattice")
    positions = AffinePointType(group, dtype="float64", measure="angstrom")
    source, target = _endpoint_types(group, 2)
    lattice = LatticeType(group, "cell-0", dtype="float64", measure="angstrom")
    shift = LatticeShiftType(group, "cell-0")
    vectors = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x1", group.family),
        dtype="float64",
        measure="angstrom",
    )
    return ArchitectureProgram(
        "2.1.0",
        "affine-geometry-evidence-periodic",
        (
            InputPort("positions", positions),
            InputPort("source_index", source),
            InputPort("target_index", target),
            InputPort("lattice", lattice),
            InputPort("lattice_shift", shift),
        ),
        (
            Node(
                "displacement",
                "core.periodic_displacement@1",
                {
                    "positions": ("input:positions",),
                    "source_index": ("input:source_index",),
                    "target_index": ("input:target_index",),
                    "lattice": ("input:lattice",),
                    "lattice_shift": ("input:lattice_shift",),
                },
            ),
        ),
        (OutputPort("vectors", "displacement", vectors),),
        program_id="affine-geometry-periodic@1",
    )


def _index_payload(torch, indices, target_size: int) -> Dict[str, Any]:
    return {"indices": torch.as_tensor(indices, dtype=torch.long), "target_size": target_size}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _max_abs(left, right) -> float:
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def _model_identity(model, artifact, program: ArchitectureProgram) -> Dict[str, Any]:
    return {
        "python_class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
        "architecture_id": artifact.architecture_id,
        "program_id": program.program_id,
        "node_ops": [node.op for node in program.nodes],
        "backend_semantics_version": model.backend_semantics_version,
        "lowering_rule_count": model.lowering_rule_manifest["rule_count"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "constructor_bypass_used": False,
    }


def _obligations(artifact) -> list[Dict[str, Any]]:
    return [dict(item.to_dict()) for item in artifact.inference.obligations]


def _nonperiodic_evidence(torch, backend, registry):
    program = _nonperiodic_program()
    artifact = Compiler(registry).analyze(program)
    support = backend.support_report(program)
    if not support.supported:
        raise RuntimeError("nonperiodic program is unsupported: {}".format(support.to_dict()))
    model = backend.build(program, artifact.inference).eval()

    positions = torch.tensor(
        [[0.2, -0.1, 0.4], [1.1, 0.3, -0.2], [-0.4, 0.8, 0.5], [0.6, -0.7, 1.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    source = _index_payload(torch, [0, 1, 3], 3)
    target = _index_payload(torch, [1, 2, 0], 3)
    inputs = {"positions": positions, "source_index": source, "target_index": target}
    reference = model(inputs, {})["vectors"]
    expected = positions.index_select(0, target["indices"]) - positions.index_select(0, source["indices"])

    translation = torch.tensor([3.0, -2.0, 5.0], dtype=torch.float64)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    translated = model({**inputs, "positions": positions + translation}, {})["vectors"]
    rotated = model({**inputs, "positions": positions @ rotation.T}, {})["vectors"]
    reflected = model({**inputs, "positions": positions @ reflection.T}, {})["vectors"]

    reference.square().sum().backward()
    gradient = positions.grad
    metrics = {
        "formula_max_abs_error": _max_abs(reference, expected),
        "translation_max_abs_error": _max_abs(translated, reference),
        "so3_rotation_max_abs_error": _max_abs(rotated, reference @ rotation.T),
        "o3_reflection_max_abs_error": _max_abs(reflected, reference @ reflection.T),
        "position_gradient_finite": bool(torch.isfinite(gradient).all().item()),
        "translation_gradient_balance_max_abs_error": float(gradient.sum(dim=0).abs().max().item()),
        "position_gradient_norm": float(gradient.norm().item()),
    }
    return program, artifact, support, model, metrics


def _periodic_evidence(torch, backend, registry):
    program = _periodic_program()
    artifact = Compiler(registry).analyze(program)
    support = backend.support_report(program)
    if not support.supported:
        raise RuntimeError("periodic program is unsupported: {}".format(support.to_dict()))
    model = backend.build(program, artifact.inference).eval()

    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.2, -0.4, 0.8], [-0.2, 1.1, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    lattice = torch.tensor(
        [[2.0, 0.0, 0.0], [0.3, 1.8, 0.0], [0.1, 0.2, 2.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    source = _index_payload(torch, [0, 1], 2)
    target = _index_payload(torch, [1, 2], 2)
    shifts = torch.tensor([[1, 0, 0], [0, -1, 1]], dtype=torch.long)
    inputs = {
        "positions": positions,
        "source_index": source,
        "target_index": target,
        "lattice": lattice,
        "lattice_shift": shifts,
    }
    reference = model(inputs, {})["vectors"]
    expected = (
        positions.index_select(0, target["indices"])
        + shifts.to(torch.float64) @ lattice
        - positions.index_select(0, source["indices"])
    )
    translation = torch.tensor([7.0, -3.0, 2.0], dtype=torch.float64)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    translated = model({**inputs, "positions": positions + translation}, {})["vectors"]
    rotated = model(
        {**inputs, "positions": positions @ rotation.T, "lattice": lattice @ rotation.T},
        {},
    )["vectors"]

    image_labels = torch.tensor([[1, 0, -1], [-1, 1, 0], [0, -1, 1]], dtype=torch.long)
    relabeled_positions = positions + image_labels.to(torch.float64) @ lattice
    adjusted_shifts = (
        shifts
        - image_labels.index_select(0, target["indices"])
        + image_labels.index_select(0, source["indices"])
    )
    relabeled = model(
        {**inputs, "positions": relabeled_positions, "lattice_shift": adjusted_shifts},
        {},
    )["vectors"]

    loss = reference.square().sum()
    loss.backward()
    position_gradient = positions.grad
    lattice_gradient = lattice.grad

    epsilon = 1.0e-6

    def loss_value(position_value, lattice_value):
        return model(
            {**inputs, "positions": position_value, "lattice": lattice_value},
            {},
        )["vectors"].square().sum()

    position_plus = positions.detach().clone()
    position_minus = positions.detach().clone()
    position_plus[0, 0] += epsilon
    position_minus[0, 0] -= epsilon
    position_fd = float(
        ((loss_value(position_plus, lattice.detach()) - loss_value(position_minus, lattice.detach())) / (2.0 * epsilon)).item()
    )
    lattice_plus = lattice.detach().clone()
    lattice_minus = lattice.detach().clone()
    lattice_plus[0, 0] += epsilon
    lattice_minus[0, 0] -= epsilon
    lattice_fd = float(
        ((loss_value(positions.detach(), lattice_plus) - loss_value(positions.detach(), lattice_minus)) / (2.0 * epsilon)).item()
    )

    metrics = {
        "formula_max_abs_error": _max_abs(reference, expected),
        "translation_max_abs_error": _max_abs(translated, reference),
        "so3_rotation_max_abs_error": _max_abs(rotated, reference @ rotation.T),
        "periodic_image_relabel_max_abs_error": _max_abs(relabeled, reference),
        "position_gradient_finite": bool(torch.isfinite(position_gradient).all().item()),
        "lattice_gradient_finite": bool(torch.isfinite(lattice_gradient).all().item()),
        "translation_gradient_balance_max_abs_error": float(position_gradient.sum(dim=0).abs().max().item()),
        "position_gradient_norm": float(position_gradient.norm().item()),
        "lattice_gradient_norm": float(lattice_gradient.norm().item()),
        "position_gradient_finite_difference_abs_error": abs(float(position_gradient[0, 0].item()) - position_fd),
        "lattice_gradient_finite_difference_abs_error": abs(float(lattice_gradient[0, 0].item()) - lattice_fd),
    }
    return program, artifact, support, model, metrics


def build_evidence(output: Path, *, pytest_summary: str = "") -> Dict[str, Any]:
    import torch

    torch.manual_seed(SEED)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = core_registry()
    backend = E3NNGraphBackend(registry)

    nonperiodic = _nonperiodic_evidence(torch, backend, registry)
    periodic = _periodic_evidence(torch, backend, registry)
    np_program, np_artifact, np_support, np_model, np_metrics = nonperiodic
    p_program, p_artifact, p_support, p_model, p_metrics = periodic

    all_error_values = [
        value
        for metrics in (np_metrics, p_metrics)
        for key, value in metrics.items()
        if key.endswith("_error") and isinstance(value, float)
    ]
    threshold = 1.0e-8
    if max(all_error_values, default=0.0) > threshold:
        raise RuntimeError("affine geometry evidence exceeded the {} threshold".format(threshold))
    if not all(
        metrics[key]
        for metrics in (np_metrics, p_metrics)
        for key in metrics
        if key.endswith("_finite")
    ):
        raise RuntimeError("affine geometry evidence found a non-finite gradient")

    support_payload = {
        "nonperiodic": np_support.to_dict(),
        "periodic": p_support.to_dict(),
    }
    evidence_payload = {
        "nonperiodic": np_metrics,
        "periodic": p_metrics,
        "threshold": threshold,
    }
    identities = {
        "nonperiodic": _model_identity(np_model, np_artifact, np_program),
        "periodic": _model_identity(p_model, p_artifact, p_program),
    }
    obligations = {
        "nonperiodic": _obligations(np_artifact),
        "periodic": _obligations(p_artifact),
    }
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "seed": SEED,
        "pytest_summary": pytest_summary,
        "primitive_count": len(registry.names()),
        "lowering_rule_count": np_model.lowering_rule_manifest["rule_count"],
        "model_identity": identities,
        "symmetry_and_gradient_evidence": evidence_payload,
        "proof_obligations": obligations,
        "claims": {
            "affine_points_are_distinct_from_translation_free_vectors": True,
            "periodic_context_is_explicit_in_the_typed_program": True,
            "relative_displacement_uses_target_minus_source": True,
            "periodic_displacement_uses_target_plus_shift_times_lattice_minus_source": True,
            "generic_node_level_lowering_used": True,
            "constructor_bypass_used": False,
            "official_v1_v2_v3_reproduction": False,
        },
    }

    (output / "nonperiodic_program.json").write_text(dumps_program(np_program), encoding="utf-8")
    (output / "periodic_program.json").write_text(dumps_program(p_program), encoding="utf-8")
    _write_json(output / "support_report.json", support_payload)
    _write_json(output / "lowering_manifest.json", np_model.lowering_rule_manifest)
    _write_json(output / "model_identity.json", identities)
    _write_json(output / "proof_obligations.json", obligations)
    _write_json(output / "symmetry_and_gradient_evidence.json", evidence_payload)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("export-dsl-affine-geometry-evidence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pytest-summary", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_evidence(Path(args.output), pytest_summary=str(args.pytest_summary))
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "primitive_count": result["primitive_count"],
        "lowering_rule_count": result["lowering_rule_count"],
        "nonperiodic_architecture_id": result["model_identity"]["nonperiodic"]["architecture_id"],
        "periodic_architecture_id": result["model_identity"]["periodic"]["architecture_id"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
