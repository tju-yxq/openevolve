#!/usr/bin/env python
"""Export primitive-level evidence against the official V1 depthwise TP oracle."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_equiformer_v1_graph_attention_contract import _load_official_module
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
    RepresentationLayout,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.canonicalize import canonicalize
from equivariant_nas.dsl.serialization import dumps_program


EVIDENCE_VERSION = "evoequilang-tensor-product-v3-official-v1-evidence@1"
SEED = 20260731

PATH_BLOCKS = (
    {"multiplicity": 4, "irrep": "0e"},
    {"multiplicity": 2, "irrep": "0e"},
    {"multiplicity": 1, "irrep": "0e"},
    {"multiplicity": 4, "irrep": "1e"},
    {"multiplicity": 2, "irrep": "1e"},
    {"multiplicity": 2, "irrep": "1e"},
    {"multiplicity": 2, "irrep": "1e"},
    {"multiplicity": 1, "irrep": "1e"},
    {"multiplicity": 1, "irrep": "1e"},
    {"multiplicity": 4, "irrep": "2e"},
    {"multiplicity": 2, "irrep": "2e"},
    {"multiplicity": 2, "irrep": "2e"},
    {"multiplicity": 1, "irrep": "2e"},
    {"multiplicity": 1, "irrep": "2e"},
    {"multiplicity": 1, "irrep": "2e"},
)
INSTRUCTION_INDICES = (
    (0, 0, 0), (0, 1, 3), (0, 2, 9),
    (1, 0, 4), (1, 1, 1), (1, 1, 5), (1, 1, 10), (1, 2, 6), (1, 2, 11),
    (2, 0, 12), (2, 1, 7), (2, 1, 13), (2, 2, 2), (2, 2, 8), (2, 2, 14),
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attrs():
    return {
        "path_blocks": list(deepcopy(PATH_BLOCKS)),
        "instructions": [
            {
                "left": left,
                "right": right,
                "out": output,
                "mode": "uvu",
                "has_weight": True,
                "path_weight": 1.0,
            }
            for left, right, output in INSTRUCTION_INDICES
        ],
    }


def _types(*, weight_numel=30, left_layout=None):
    group = GroupSpec.o3()
    left = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("4x0e+2x1e+1x2e", group.family),
        dtype="float64",
        measure="dimensionless",
        layout=left_layout or RepresentationLayout(),
    )
    right = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x0e+1x1e+1x2e", group.family),
        dtype="float64",
        measure="dimensionless",
    )
    weight = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("{}x0e".format(weight_numel), group.family),
        axes=("tp_path",),
        axis_specs=(AxisSpec("tp_path", weight_numel, FeatureRole.TP_PATH, "independent", 0),),
        dtype="float64",
        measure="dimensionless",
        feature_role=FeatureRole.RADIAL_WEIGHT,
    )
    output = replace(left, irreps=Irreps.parse("7x0e+12x1e+11x2e", group.family))
    return left, right, weight, output


def _program(*, attrs=None, weight_numel=30, left_layout=None):
    left, right, weight, output = _types(weight_numel=weight_numel, left_layout=left_layout)
    return ArchitectureProgram(
        "2.6.0",
        "official-v1-depthwise-tensor-product-evidence",
        (InputPort("left", left), InputPort("right", right), InputPort("weight", weight)),
        (
            Node(
                "tp",
                "core.tensor_product@3",
                {"left": ("input:left",), "right": ("input:right",), "weight": ("input:weight",)},
                attrs or _attrs(),
            ),
        ),
        (OutputPort("out", "tp", output),),
        program_id="official-v1-depthwise-uvu-tp@1",
    )


def _instruction_payload(module):
    return [
        {
            "left": int(item.i_in1),
            "right": int(item.i_in2),
            "out": int(item.i_out),
            "mode": str(item.connection_mode),
            "has_weight": bool(item.has_weight),
            "path_weight": float(item.path_weight),
            "path_shape": list(item.path_shape),
        }
        for item in module.instructions
    ]


def _max_abs(left, right):
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def _relative_error(left, right):
    return float(((left - right).norm() / right.norm().clamp_min(1.0e-12)).item())


def _negative_contracts(registry):
    cases = {}

    def run(name, expected_code, *, attrs=None, weight_numel=30, left_layout=None):
        try:
            Compiler(registry).analyze(
                _program(attrs=attrs, weight_numel=weight_numel, left_layout=left_layout)
            )
            cases[name] = {"rejected": False, "expected_code": expected_code, "diagnostics": []}
        except DSLValidationError as error:
            cases[name] = {
                "rejected": True,
                "expected_code": expected_code,
                "expected_code_present": any(item.code == expected_code for item in error.diagnostics),
                "diagnostics": [item.to_dict() for item in error.diagnostics],
            }

    attrs = _attrs()
    attrs["instructions"][0]["left"] = 9
    run("invalid_instruction_index", "E_TP_V3_010", attrs=attrs)

    attrs = _attrs()
    attrs["instructions"][0]["out"] = 3
    run("illegal_clebsch_gordan_path", "E_TP_V3_014", attrs=attrs)

    attrs = _attrs()
    attrs["path_blocks"][0]["multiplicity"] = 3
    run("uvu_output_multiplicity_mismatch", "E_TP_V3_015", attrs=attrs)

    attrs = _attrs()
    attrs["instructions"][0]["mode"] = "uvw"
    run("unsupported_connection_mode", "E_TP_V3_011", attrs=attrs)

    run("external_weight_length_mismatch", "E_TP_V3_023", attrs=_attrs(), weight_numel=29)

    attrs = _attrs()
    del attrs["instructions"][-1]
    run("unreferenced_path_block", "E_TP_V3_016", attrs=attrs)

    attrs = _attrs()
    attrs["path_blocks"][2], attrs["path_blocks"][3] = attrs["path_blocks"][3], attrs["path_blocks"][2]
    run("noncanonical_path_order", "E_TP_V3_006", attrs=attrs)

    run(
        "non_irrep_major_layout",
        "E_TP_V3_019",
        attrs=_attrs(),
        left_layout=RepresentationLayout(storage="grid", truncation_state="grid_projected"),
    )
    return cases


def build_evidence(output: Path, *, official_v1_root: Path, pytest_summary: str = "") -> Dict[str, Any]:
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
        raise RuntimeError("tensor_product@3 support report failed: {}".format(support.to_dict()))
    dsl_model = backend.build(program, artifact.inference).double().eval()
    dsl_tp = dsl_model.node_modules["tp"]

    official_module, source_path = _load_official_module(official_v1_root.resolve(), torch)
    config = {
        "irreps_node_input": o3.Irreps("4x0e+2x1e+1x2e"),
        "irreps_node_attr": o3.Irreps("1x0e"),
        "irreps_edge_attr": o3.Irreps("1x0e+1x1e+1x2e"),
        "irreps_node_output": o3.Irreps("4x0e+2x1e+1x2e"),
        "fc_neurons": [6, 8],
        "irreps_head": o3.Irreps("2x0e+1x1e+1x2e"),
        "num_heads": 2,
        "irreps_pre_attn": None,
        "rescale_degree": False,
        "nonlinear_message": False,
        "alpha_drop": 0.0,
        "proj_drop": 0.0,
    }
    official_graph_attention = official_module.GraphAttention(**config).double().eval()
    official_tp = official_graph_attention.sep.dtp.tp

    if official_tp.irreps_out != dsl_tp.irreps_out:
        raise RuntimeError("official and DSL unsimplified path-block irreps differ")
    if _instruction_payload(official_tp) != _instruction_payload(dsl_tp):
        raise RuntimeError("official and DSL normalized e3nn instructions differ")

    edge_count = 8
    left = torch.randn(edge_count, 15, dtype=torch.float64, requires_grad=True)
    right = torch.randn(edge_count, 9, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(edge_count, 30, dtype=torch.float64, requires_grad=True)
    official_left = left.detach().clone().requires_grad_(True)
    official_right = right.detach().clone().requires_grad_(True)
    official_weight = weight.detach().clone().requires_grad_(True)
    dsl_output = dsl_model({"left": left, "right": right, "weight": weight}, {})["out"]
    official_output = official_tp(official_left, official_right, official_weight)
    probe = torch.randn_like(dsl_output)
    dsl_gradients = torch.autograd.grad((dsl_output * probe).sum(), (left, right, weight))
    official_gradients = torch.autograd.grad(
        (official_output * probe).sum(),
        (official_left, official_right, official_weight),
    )

    transformations = {
        "rotation": o3.rand_matrix(dtype=torch.float64),
        "inversion": -torch.eye(3, dtype=torch.float64),
    }
    equivariance = {}
    for name, matrix in transformations.items():
        left_action = o3.Irreps("4x0e+2x1e+1x2e").D_from_matrix(matrix)
        right_action = o3.Irreps("1x0e+1x1e+1x2e").D_from_matrix(matrix)
        output_action = o3.Irreps("7x0e+12x1e+11x2e").D_from_matrix(matrix)
        transformed = dsl_model(
            {
                "left": left.detach() @ left_action.transpose(0, 1),
                "right": right.detach() @ right_action.transpose(0, 1),
                "weight": weight.detach(),
            },
            {},
        )["out"]
        equivariance[name] = _relative_error(
            transformed,
            dsl_output.detach() @ output_action.transpose(0, 1),
        )

    permutation = torch.tensor([7, 2, 0, 5, 3, 1, 6, 4], dtype=torch.long)
    permuted = dsl_model(
        {
            "left": left.detach().index_select(0, permutation),
            "right": right.detach().index_select(0, permutation),
            "weight": weight.detach().index_select(0, permutation),
        },
        {},
    )["out"]
    metrics = {
        "official_forward_max_abs_error": _max_abs(dsl_output.detach(), official_output.detach()),
        "official_left_gradient_max_abs_error": _max_abs(dsl_gradients[0], official_gradients[0]),
        "official_right_gradient_max_abs_error": _max_abs(dsl_gradients[1], official_gradients[1]),
        "official_weight_gradient_max_abs_error": _max_abs(dsl_gradients[2], official_gradients[2]),
        "so3_rotation_relative_error": equivariance["rotation"],
        "o3_inversion_relative_error": equivariance["inversion"],
        "edge_permutation_max_abs_error": _max_abs(
            permuted,
            dsl_output.detach().index_select(0, permutation),
        ),
        "left_gradient_finite": bool(torch.isfinite(dsl_gradients[0]).all()),
        "right_gradient_finite": bool(torch.isfinite(dsl_gradients[1]).all()),
        "weight_gradient_finite": bool(torch.isfinite(dsl_gradients[2]).all()),
    }
    error_keys = [name for name in metrics if name.endswith("error")]
    if max(float(metrics[name]) for name in error_keys) > 1.0e-7:
        raise RuntimeError("tensor_product@3 evidence exceeded 1e-7: {}".format(metrics))
    if not all(metrics[name] for name in metrics if name.endswith("_finite")):
        raise RuntimeError("tensor_product@3 produced non-finite gradients")

    negative = _negative_contracts(registry)
    if not all(item.get("rejected") and item.get("expected_code_present") for item in negative.values()):
        raise RuntimeError("one or more tensor_product@3 negative contracts were not enforced")

    source_text = source_path.read_text(encoding="utf-8")
    source_identity = {
        "repository_root": str(official_v1_root.resolve()),
        "source_file": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "oracle_object": "GraphAttention.sep.dtp.tp",
        "official_constructor_role": "oracle construction only; never used by DSL execution",
    }
    model_identity = {
        "backend_semantics_version": dsl_model.backend_semantics_version,
        "graph_class": "{}.{}".format(type(dsl_model).__module__, type(dsl_model).__qualname__),
        "lowered_module_class": "{}.{}".format(type(dsl_tp).__module__, type(dsl_tp).__qualname__),
        "official_oracle_module_class": "{}.{}".format(type(official_tp).__module__, type(official_tp).__qualname__),
        "node_op": "core.tensor_product@3",
        "irreps_in1": str(dsl_tp.irreps_in1),
        "irreps_in2": str(dsl_tp.irreps_in2),
        "unsimplified_path_irreps_out": str(dsl_tp.irreps_out),
        "simplified_value_irreps_out": str(dsl_tp.irreps_out.simplify()),
        "instruction_count": len(dsl_tp.instructions),
        "weight_numel": int(dsl_tp.weight_numel),
        "internal_weights": bool(dsl_tp.internal_weights),
        "shared_weights": bool(dsl_tp.shared_weights),
        "internal_parameter_count": sum(parameter.numel() for parameter in dsl_tp.parameters()),
        "official_network_block_or_operator_used_for_dsl_execution": False,
    }
    parameter_contracts = {
        node_id: [contract.to_dict() for contract in contracts]
        for node_id, contracts in artifact.inference.parameter_contracts.items()
        if contracts
    }
    math_contract = {
        "transformation_law": "TP(D_g x, D_g a; w) = D_g TP(x, a; w) for every g in O(3); radial weights w are invariant scalars",
        "selection_rule": "an instruction is legal iff |l1-l2| <= l_out <= l1+l2 and p_out = p1*p2",
        "uvu_channel_contract": "each output path block preserves the selected left multiplicity u; one external weight is supplied for every (u,v) pair",
        "coefficient_contract": "Clebsch-Gordan contraction is delegated only to low-level e3nn.o3.TensorProduct; path topology, multiplicities, ordering, weights, carriers, frames and layouts are fixed by Typed DSL",
        "path_block_contract": "instruction output indices address the unsimplified 15-block irreps_out; only the public result type is adjacent-block simplified",
        "permutation_contract": "the primitive is pointwise over the leading edge axis, so a common edge permutation commutes with execution",
    }
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "seed": SEED,
        "pytest_summary": pytest_summary,
        "primitive_count": len(registry.names()),
        "lowering_rule_count": dsl_model.lowering_rule_manifest["rule_count"],
        "architecture_id": artifact.architecture_id,
        "official_v1_source_identity": source_identity,
        "model_identity": model_identity,
        "parameter_contracts": parameter_contracts,
        "math_contract": math_contract,
        "numerical_and_gradient_evidence": metrics,
        "negative_contracts": negative,
        "claims": {
            "official_v1_depthwise_tp_path_graph_reproduced": True,
            "official_v1_depthwise_tp_forward_reproduced": True,
            "official_v1_depthwise_tp_three_input_gradients_reproduced": True,
            "primitive_level_generic_lowering_used": True,
            "official_constructor_bypass_used": False,
            "official_v1_complete_graph_attention_reproduced": False,
            "official_v1_v2_v3_complete_networks_reproduced": False,
            "mixed_parity_even_first_layout_supported": False,
        },
    }

    (output / "canonical_tensor_product_v3_program.json").write_text(
        dumps_program(canonicalize(program, registry)),
        encoding="utf-8",
    )
    _write_json(output / "support_report.json", support.to_dict())
    _write_json(output / "lowering_manifest.json", dsl_model.lowering_rule_manifest)
    _write_json(output / "parameter_contracts.json", parameter_contracts)
    _write_json(output / "model_identity.json", model_identity)
    _write_json(output / "official_v1_source_identity.json", source_identity)
    _write_json(output / "normalized_instruction_contract.json", {"instructions": _instruction_payload(dsl_tp)})
    _write_json(output / "math_contract.json", math_contract)
    _write_json(output / "numerical_and_gradient_evidence.json", metrics)
    _write_json(output / "negative_contracts.json", negative)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("export-dsl-tensor-product-v3-official-evidence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--official-v1-root", default=str(PROJECT_ROOT.parent / "equiformer"))
    parser.add_argument("--pytest-summary", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_evidence(
        Path(args.output),
        official_v1_root=Path(args.official_v1_root),
        pytest_summary=str(args.pytest_summary),
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "architecture_id": result["architecture_id"],
        "primitive_count": result["primitive_count"],
        "lowering_rule_count": result["lowering_rule_count"],
        "official_forward_max_abs_error": result["numerical_and_gradient_evidence"]["official_forward_max_abs_error"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
