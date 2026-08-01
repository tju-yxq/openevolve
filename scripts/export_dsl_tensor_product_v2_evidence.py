#!/usr/bin/env python
"""Export external-weight tensor_product@2 evidence."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from equivariant_nas.dsl import (
    ArchitectureProgram,
    AxisSpec,
    Carrier,
    Compiler,
    DSLValidationError,
    EquivariantTensorType,
    FeatureRole,
    GroupSpec,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.serialization import dumps_program


EVIDENCE_VERSION = "evoequilang-tensor-product-v2-evidence@1"
SEED = 20260731


def _types(path_count=3):
    group = GroupSpec.o3()
    left = EquivariantTensorType(
        group, Carrier.EDGE, Irreps.parse("1x1o", group.family), dtype="float64", measure="angstrom"
    )
    right = EquivariantTensorType(
        group, Carrier.EDGE, Irreps.parse("1x1o", group.family), dtype="float64", measure="dimensionless"
    )
    output = replace(left, irreps=Irreps.parse("1x0e+1x1e+1x2e", group.family), measure="angstrom")
    weight = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("{}x0e".format(path_count), group.family),
        axes=("tp_path",),
        axis_specs=(AxisSpec("tp_path", path_count, FeatureRole.TP_PATH, "independent", 0),),
        dtype="float64",
        measure="dimensionless",
        feature_role=FeatureRole.RADIAL_WEIGHT,
    )
    return left, right, weight, output


def _program(path_count=3):
    left, right, weight, output = _types(path_count)
    return ArchitectureProgram(
        "2.4.0",
        "external-tensor-product-evidence",
        (InputPort("left", left), InputPort("right", right), InputPort("weight", weight)),
        (
            Node(
                "tp",
                "core.tensor_product@2",
                {"left": ("input:left",), "right": ("input:right",), "weight": ("input:weight",)},
                {"out_irreps": str(output.irreps)},
            ),
        ),
        (OutputPort("out", "tp", output),),
        program_id="external-uvw-tensor-product@1",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _max_abs(left, right):
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def _invalid_weight_negative(registry):
    program = _program(path_count=2)
    try:
        Compiler(registry).analyze(program)
    except DSLValidationError as error:
        return {
            "rejected": True,
            "expected_code_present": any(item.code == "E_TP_V2_010" for item in error.diagnostics),
            "diagnostics": [item.to_dict() for item in error.diagnostics],
        }
    return {"rejected": False, "expected_code_present": False, "diagnostics": []}


def build_evidence(output: Path, *, pytest_summary="") -> Dict[str, Any]:
    import torch
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    torch.manual_seed(SEED)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = core_registry()
    program = _program()
    artifact = Compiler(registry).analyze(program)
    backend = E3NNGraphBackend(registry)
    support = backend.support_report(program)
    if not support.supported:
        raise RuntimeError("tensor_product@2 support report failed: {}".format(support.to_dict()))
    model = backend.build(program, artifact.inference).double().eval()
    module = model.node_modules["tp"]

    left = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    reference = model({"left": left, "right": right, "weight": weight}, {})["out"]
    direct = module(left, right, weight)

    rotation = o3.rand_matrix(dtype=torch.float64)
    input_action = o3.Irreps("1x1o").D_from_matrix(rotation)
    output_action = o3.Irreps("1x0e+1x1e+1x2e").D_from_matrix(rotation)
    rotated = model(
        {
            "left": left @ input_action.transpose(0, 1),
            "right": right @ input_action.transpose(0, 1),
            "weight": weight,
        },
        {},
    )["out"]
    expected_rotated = reference @ output_action.transpose(0, 1)
    rotation_relative_error = float(
        ((rotated - expected_rotated).norm() / expected_rotated.norm().clamp_min(1.0e-12)).detach().item()
    )
    permutation = torch.tensor([7, 2, 0, 5, 3, 1, 6, 4], dtype=torch.long)
    permuted = model(
        {
            "left": left.index_select(0, permutation),
            "right": right.index_select(0, permutation),
            "weight": weight.index_select(0, permutation),
        },
        {},
    )["out"]

    reference.square().sum().backward()
    metrics = {
        "direct_module_forward_max_abs_error": _max_abs(reference, direct),
        "rotation_relative_error": rotation_relative_error,
        "edge_permutation_max_abs_error": _max_abs(permuted, reference.index_select(0, permutation)),
        "left_gradient_finite": bool(torch.isfinite(left.grad).all()),
        "right_gradient_finite": bool(torch.isfinite(right.grad).all()),
        "external_weight_gradient_finite": bool(torch.isfinite(weight.grad).all()),
        "left_gradient_norm": float(left.grad.norm().item()),
        "right_gradient_norm": float(right.grad.norm().item()),
        "external_weight_gradient_norm": float(weight.grad.norm().item()),
    }
    if max(
        metrics["direct_module_forward_max_abs_error"],
        metrics["rotation_relative_error"],
        metrics["edge_permutation_max_abs_error"],
    ) > 1.0e-7:
        raise RuntimeError("tensor_product@2 evidence exceeded 1e-7")
    if not all(metrics[key] for key in metrics if key.endswith("_finite")):
        raise RuntimeError("tensor_product@2 produced non-finite gradients")
    negative = _invalid_weight_negative(registry)
    if not negative["rejected"] or not negative["expected_code_present"]:
        raise RuntimeError("tensor_product@2 invalid weight negative was not enforced")

    contracts = artifact.inference.parameter_contracts["tp"]
    obligations = [item.to_dict() for item in artifact.inference.obligations]
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "seed": SEED,
        "pytest_summary": pytest_summary,
        "primitive_count": len(registry.names()),
        "lowering_rule_count": model.lowering_rule_manifest["rule_count"],
        "architecture_id": artifact.architecture_id,
        "parameter_contracts": [contract.to_dict() for contract in contracts],
        "proof_obligations": obligations,
        "model_identity": {
            "backend_semantics_version": model.backend_semantics_version,
            "python_class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
            "node_op": "core.tensor_product@2",
            "weight_numel": int(module.weight_numel),
            "internal_parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        },
        "numerical_and_gradient_evidence": metrics,
        "invalid_weight_negative": negative,
        "claims": {
            "external_weight_input_is_explicit": True,
            "external_weight_parameter_contract_inferred": True,
            "backend_internal_weights_disabled": True,
            "fully_connected_uvw_unshared_weights_supported": True,
            "custom_depthwise_instructions_supported": False,
            "bias_and_rescale_supported": False,
            "official_v1_depthwise_tp_reproduced": False,
            "official_v1_v2_v3_reproduction": False,
            "constructor_bypass_used": False,
        },
    }

    (output / "tensor_product_v2_program.json").write_text(dumps_program(program), encoding="utf-8")
    _write_json(output / "support_report.json", support.to_dict())
    _write_json(output / "parameter_contracts.json", {"tp": [contract.to_dict() for contract in contracts]})
    _write_json(output / "proof_obligations.json", {"tp": obligations})
    _write_json(output / "lowering_manifest.json", model.lowering_rule_manifest)
    _write_json(output / "numerical_and_gradient_evidence.json", metrics)
    _write_json(output / "invalid_weight_negative.json", negative)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("export-dsl-tensor-product-v2-evidence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pytest-summary", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_evidence(Path(args.output), pytest_summary=str(args.pytest_summary))
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "architecture_id": result["architecture_id"],
        "primitive_count": result["primitive_count"],
        "lowering_rule_count": result["lowering_rule_count"],
        "weight_numel": result["model_identity"]["weight_numel"],
        "internal_parameter_count": result["model_identity"]["internal_parameter_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
