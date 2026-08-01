#!/usr/bin/env python
"""Export invariant head-axis and per-head scalar contraction evidence."""

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


EVIDENCE_VERSION = "evoequilang-invariant-multihead-evidence@1"
SEED = 20260731


def _source_type():
    group = GroupSpec.so3()
    return InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("12x0", group.family),
        axes=("channel",),
        axis_specs=(AxisSpec("channel", 12, FeatureRole.CHANNEL, "independent", 0),),
        feature_role=FeatureRole.CHANNEL,
    )


def _alpha_type(source):
    return replace(
        source,
        irreps=Irreps.parse("3x0", source.group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", 3, FeatureRole.HEAD, "independent", 0),),
        feature_role=FeatureRole.ALPHA,
    )


def _split_node():
    return Node(
        "split",
        "core.head_split@1",
        {"x": ("input:x",)},
        {"axis": "channel", "head_axis": "head", "channel_axis": "per_head_channel", "num_heads": 3},
    )


def _split_merge_program():
    source = _source_type()
    return ArchitectureProgram(
        "2.3.0",
        "invariant-multihead-view-evidence",
        (InputPort("x", source),),
        (
            _split_node(),
            Node(
                "merge",
                "core.head_merge@1",
                {"x": ("split",)},
                {"head_axis": "head", "channel_axis": "per_head_channel", "out_axis": "channel"},
            ),
        ),
        (OutputPort("out", "merge", source),),
        program_id="invariant-head-split-merge@1",
    )


def _alpha_program(*, bias=True):
    source = _source_type()
    return ArchitectureProgram(
        "2.3.0",
        "invariant-multihead-alpha-evidence",
        (InputPort("x", source),),
        (
            _split_node(),
            Node(
                "alpha",
                "core.headwise_scalar_contraction@1",
                {"x": ("split",)},
                {"head_axis": "head", "channel_axis": "per_head_channel", "bias": bias},
            ),
        ),
        (OutputPort("alpha", "alpha", _alpha_type(source)),),
        program_id="invariant-headwise-alpha@1",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _max_abs(left, right):
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def _equivariant_value_negative(registry):
    group = GroupSpec.so3()
    value = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("4x0+4x1", group.family),
        axes=("channel",),
        axis_specs=(AxisSpec("channel", 8, FeatureRole.CHANNEL, "independent", 0),),
    )
    program = ArchitectureProgram(
        "2.3.0",
        "equivariant-value-head-negative",
        (InputPort("x", value),),
        (_split_node(),),
        (OutputPort("out", "split", value),),
    )
    try:
        Compiler(registry).analyze(program)
    except DSLValidationError as error:
        return {
            "rejected": True,
            "diagnostics": [item.to_dict() for item in error.diagnostics],
            "expected_code_present": any(item.code == "E_HEAD_001" for item in error.diagnostics),
        }
    return {"rejected": False, "diagnostics": [], "expected_code_present": False}


def build_evidence(output: Path, *, pytest_summary="") -> Dict[str, Any]:
    import torch

    torch.manual_seed(SEED)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = core_registry()
    backend = E3NNGraphBackend(registry)

    view_program = _split_merge_program()
    view_artifact = Compiler(registry).analyze(view_program)
    view_support = backend.support_report(view_program)
    view_model = backend.build(view_program, view_artifact.inference).eval()

    alpha_program = _alpha_program()
    alpha_artifact = Compiler(registry).analyze(alpha_program)
    alpha_support = backend.support_report(alpha_program)
    alpha_model = backend.build(alpha_program, alpha_artifact.inference).eval()
    alpha_module = alpha_model.node_modules["alpha"]

    with torch.no_grad():
        alpha_module.weight.copy_(
            torch.tensor(
                [[1.0, 0.0, -1.0, 0.5], [0.5, 0.25, 0.0, -0.5], [-0.5, 1.0, 0.5, 2.0]],
                dtype=alpha_module.weight.dtype,
            )
        )
        alpha_module.bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=alpha_module.bias.dtype))

    view_input = torch.randn(6, 12, requires_grad=True)
    view_output = view_model({"x": view_input}, {})["out"]

    actual_input = torch.randn(6, 12, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    reference_weight = alpha_module.weight.detach().clone().requires_grad_(True)
    reference_bias = alpha_module.bias.detach().clone().requires_grad_(True)
    actual = alpha_model({"x": actual_input}, {})["alpha"]
    expected = (
        reference_input.reshape(6, 3, 4) * reference_weight
    ).sum(dim=-1) + reference_bias
    permutation = torch.tensor([5, 1, 3, 0, 4, 2], dtype=torch.long)
    permuted = alpha_model({"x": actual_input.index_select(0, permutation)}, {})["alpha"]

    actual.square().sum().backward()
    expected.square().sum().backward()
    metrics = {
        "split_merge_forward_max_abs_error": _max_abs(view_output, view_input),
        "alpha_forward_max_abs_error": _max_abs(actual, expected),
        "edge_permutation_max_abs_error": _max_abs(permuted, actual.index_select(0, permutation)),
        "input_gradient_max_abs_error": _max_abs(actual_input.grad, reference_input.grad),
        "weight_gradient_max_abs_error": _max_abs(alpha_module.weight.grad, reference_weight.grad),
        "bias_gradient_max_abs_error": _max_abs(alpha_module.bias.grad, reference_bias.grad),
        "all_gradients_finite": bool(
            torch.isfinite(actual_input.grad).all()
            and torch.isfinite(alpha_module.weight.grad).all()
            and torch.isfinite(alpha_module.bias.grad).all()
        ),
    }
    if max(value for key, value in metrics.items() if key.endswith("_error")) > 1.0e-7:
        raise RuntimeError("invariant multihead evidence exceeded 1e-7")
    if not metrics["all_gradients_finite"]:
        raise RuntimeError("invariant multihead evidence produced non-finite gradients")
    negative = _equivariant_value_negative(registry)
    if not negative["rejected"] or not negative["expected_code_present"]:
        raise RuntimeError("equivariant value negative boundary was not enforced")

    contracts = alpha_artifact.inference.parameter_contracts["alpha"]
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "seed": SEED,
        "pytest_summary": pytest_summary,
        "primitive_count": len(registry.names()),
        "lowering_rule_count": alpha_model.lowering_rule_manifest["rule_count"],
        "architecture_identity": {
            "split_merge": view_artifact.architecture_id,
            "headwise_alpha": alpha_artifact.architecture_id,
            "headwise_alpha_without_bias": Compiler(registry).analyze(_alpha_program(bias=False)).architecture_id,
        },
        "parameter_contracts": [contract.to_dict() for contract in contracts],
        "actual_module_parameters": {
            name: {"shape": list(parameter.shape), "requires_grad": bool(parameter.requires_grad)}
            for name, parameter in alpha_module.named_parameters()
        },
        "numerical_and_gradient_evidence": metrics,
        "equivariant_value_negative": negative,
        "model_identity": {
            "backend_semantics_version": alpha_model.backend_semantics_version,
            "node_ops": [node.op for node in alpha_program.nodes],
            "parameter_count": sum(parameter.numel() for parameter in alpha_model.parameters()),
        },
        "claims": {
            "invariant_head_split_merge_supported": True,
            "headwise_scalar_contraction_supported": True,
            "equivariant_value_head_split_supported": False,
            "full_v1_v2_v3_multihead_attention_supported": False,
            "generic_node_level_lowering_used": True,
            "constructor_bypass_used": False,
            "official_v1_v2_v3_reproduction": False,
        },
    }

    (output / "split_merge_program.json").write_text(dumps_program(view_program), encoding="utf-8")
    (output / "headwise_alpha_program.json").write_text(dumps_program(alpha_program), encoding="utf-8")
    _write_json(output / "support_report.json", {"split_merge": view_support.to_dict(), "headwise_alpha": alpha_support.to_dict()})
    _write_json(output / "parameter_contracts.json", {"alpha": [contract.to_dict() for contract in contracts]})
    _write_json(output / "lowering_manifest.json", alpha_model.lowering_rule_manifest)
    _write_json(output / "numerical_and_gradient_evidence.json", metrics)
    _write_json(output / "equivariant_value_negative.json", negative)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("export-dsl-invariant-multihead-evidence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pytest-summary", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_evidence(Path(args.output), pytest_summary=str(args.pytest_summary))
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "architecture_id": result["architecture_identity"]["headwise_alpha"],
        "primitive_count": result["primitive_count"],
        "lowering_rule_count": result["lowering_rule_count"],
        "parameter_count": result["model_identity"]["parameter_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
