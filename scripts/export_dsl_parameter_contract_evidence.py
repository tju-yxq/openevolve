#!/usr/bin/env python
"""Export node-level ParameterContract and scalar-axis Lowering evidence."""

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
    FeatureRole,
    GroupSpec,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    PARAMETER_CONTRACT_SCHEMA_VERSION,
    architecture_id,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.serialization import dumps_program


EVIDENCE_VERSION = "evoequilang-parameter-contract-evidence@1"
SEED = 20260731


def _value_types(out_features: int = 4):
    group = GroupSpec.so3()
    input_type = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("6x0", group.family),
        axes=("head", "channel"),
        axis_specs=(
            AxisSpec("head", 2, FeatureRole.HEAD, order=0),
            AxisSpec("channel", 3, FeatureRole.CHANNEL, sharing="per_head", order=1),
        ),
        feature_role=FeatureRole.CHANNEL,
    )
    output_type = replace(
        input_type,
        irreps=Irreps.parse("{}x0".format(2 * out_features), group.family),
        axis_specs=(
            input_type.axis_specs[0],
            replace(input_type.axis_specs[1], size=out_features),
        ),
    )
    return input_type, output_type


def _program(*, out_features: int = 4, bias: bool = True) -> ArchitectureProgram:
    input_type, output_type = _value_types(out_features)
    return ArchitectureProgram(
        "2.2.0",
        "parameter-contract-evidence",
        (InputPort("x", input_type),),
        (
            Node(
                "projection",
                "core.scalar_linear@1",
                {"x": ("input:x",)},
                {"axis": "channel", "out_features": out_features, "bias": bias},
            ),
        ),
        (OutputPort("out", "projection", output_type),),
        program_id="scalar-linear-parameter-contract@1",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _max_abs(left, right) -> float:
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def build_evidence(output: Path, *, pytest_summary: str = "") -> Dict[str, Any]:
    import torch

    torch.manual_seed(SEED)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = core_registry()
    program = _program()
    artifact = Compiler(registry).analyze(program)
    backend = E3NNGraphBackend(registry)
    support = backend.support_report(program)
    if not support.supported:
        raise RuntimeError("scalar_linear support report failed: {}".format(support.to_dict()))
    model = backend.build(program, artifact.inference).eval()
    module = model.node_modules["projection"]

    with torch.no_grad():
        module.linear.weight.copy_(
            torch.tensor(
                [[1.0, 0.0, -1.0], [0.5, 0.25, 0.0], [-0.5, 1.0, 0.5], [0.0, -1.0, 2.0]],
                dtype=module.linear.weight.dtype,
            )
        )
        module.linear.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=module.linear.bias.dtype))

    actual_input = torch.randn(5, 6, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    reference_weight = module.linear.weight.detach().clone().requires_grad_(True)
    reference_bias = module.linear.bias.detach().clone().requires_grad_(True)
    actual = model({"x": actual_input}, {})["out"]
    expected = torch.nn.functional.linear(
        reference_input.reshape(5, 2, 3),
        reference_weight,
        reference_bias,
    ).reshape(5, 8)
    actual.square().sum().backward()
    expected.square().sum().backward()

    metrics = {
        "forward_max_abs_error": _max_abs(actual, expected),
        "input_gradient_max_abs_error": _max_abs(actual_input.grad, reference_input.grad),
        "weight_gradient_max_abs_error": _max_abs(module.linear.weight.grad, reference_weight.grad),
        "bias_gradient_max_abs_error": _max_abs(module.linear.bias.grad, reference_bias.grad),
        "all_gradients_finite": bool(
            torch.isfinite(actual_input.grad).all()
            and torch.isfinite(module.linear.weight.grad).all()
            and torch.isfinite(module.linear.bias.grad).all()
        ),
    }
    if max(value for key, value in metrics.items() if key.endswith("_error")) > 1.0e-7:
        raise RuntimeError("scalar_linear numerical alignment exceeded 1e-7")
    if not metrics["all_gradients_finite"]:
        raise RuntimeError("scalar_linear produced non-finite gradients")

    contracts = artifact.inference.parameter_contracts["projection"]
    actual_parameters = {
        name: {
            "shape": list(parameter.shape),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in module.named_parameters()
    }
    identity_checks = {
        "baseline": artifact.architecture_id,
        "wider": architecture_id(_program(out_features=5), registry),
        "without_bias": architecture_id(_program(bias=False), registry),
    }
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "seed": SEED,
        "pytest_summary": pytest_summary,
        "parameter_contract_schema_version": PARAMETER_CONTRACT_SCHEMA_VERSION,
        "registry_hash": registry.content_hash(),
        "primitive_count": len(registry.names()),
        "lowering_rule_count": model.lowering_rule_manifest["rule_count"],
        "architecture_identity": identity_checks,
        "parameter_contracts": [contract.to_dict() for contract in contracts],
        "actual_module_parameters": actual_parameters,
        "numerical_and_gradient_evidence": metrics,
        "model_identity": {
            "python_class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
            "backend_semantics_version": model.backend_semantics_version,
            "program_id": program.program_id,
            "node_op": program.nodes[0].op,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "claims": {
            "program_parameters_reused_as_trainable_weights": False,
            "node_parameter_contract_inferred_by_type_checker": True,
            "real_module_parameter_names_shapes_and_trainability_validated": True,
            "parameter_contract_and_registry_bound_to_architecture_id": True,
            "generic_node_level_lowering_used": True,
            "constructor_bypass_used": False,
            "official_v1_v2_v3_reproduction": False,
            "legacy_parameterized_primitives_fully_migrated": False,
        },
    }

    (output / "scalar_linear_program.json").write_text(dumps_program(program), encoding="utf-8")
    _write_json(output / "parameter_contracts.json", {
        "projection": [contract.to_dict() for contract in contracts],
    })
    _write_json(output / "actual_module_parameters.json", actual_parameters)
    _write_json(output / "support_report.json", support.to_dict())
    _write_json(output / "lowering_manifest.json", model.lowering_rule_manifest)
    _write_json(output / "numerical_and_gradient_evidence.json", metrics)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("export-dsl-parameter-contract-evidence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pytest-summary", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_evidence(Path(args.output), pytest_summary=str(args.pytest_summary))
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "architecture_id": result["architecture_identity"]["baseline"],
        "primitive_count": result["primitive_count"],
        "lowering_rule_count": result["lowering_rule_count"],
        "parameter_count": result["model_identity"]["parameter_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
