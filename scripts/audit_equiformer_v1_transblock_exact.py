#!/usr/bin/env python
"""Audit the deterministic Equiformer V1 TransBlock against primitive lowering.

The official implementation is loaded only as a numerical oracle.  The DSL
model is compiled from ``equiformer_v1_transblock_program`` and executed by the
generic e3nn graph backend without calling an official block or operator
constructor.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping


SEED = 20260731
EVIDENCE_VERSION = "equiformer-v1-transblock-exact-audit@1"
HIDDEN_IRREPS = "4x0e+2x1e+1x2e"
EDGE_IRREPS = "1x0e+1x1e+1x2e"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _max_abs(left, right) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _relative_error(actual, expected) -> float:
    numerator = (actual.detach() - expected.detach()).norm()
    denominator = expected.detach().norm().clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def _official_config(
    o3,
    *,
    node_output_irreps: str = HIDDEN_IRREPS,
    rescale_degree: bool = False,
    nonlinear_message: bool = False,
) -> Dict[str, Any]:
    return {
        "irreps_node_input": o3.Irreps(HIDDEN_IRREPS),
        "irreps_node_attr": o3.Irreps("1x0e"),
        "irreps_edge_attr": o3.Irreps(EDGE_IRREPS),
        "irreps_node_output": o3.Irreps(str(node_output_irreps)),
        "fc_neurons": [6, 8],
        "irreps_head": o3.Irreps("2x0e+1x1e+1x2e"),
        "num_heads": 2,
        "irreps_pre_attn": None,
        "rescale_degree": bool(rescale_degree),
        "nonlinear_message": bool(nonlinear_message),
        "alpha_drop": 0.0,
        "proj_drop": 0.0,
        "drop_path_rate": 0.0,
        "irreps_mlp_mid": o3.Irreps(HIDDEN_IRREPS),
        "norm_layer": "layer",
    }


def _serializable_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in config.items():
        if key.startswith("irreps_") and value is not None:
            result[key] = str(value)
        else:
            result[key] = value
    return result


def _radial_output_scales(official, torch):
    scales = torch.ones(30, dtype=torch.float64)
    depthwise = official.ga.sep_act.dtp if official.ga.nonlinear_message else official.ga.sep.dtp
    for output_slice, scale in depthwise.slices_sqrt_k.values():
        scales[output_slice] *= float(scale)
    return scales.tolist()


def _official_intermediates(
    official,
    node_input,
    node_attr,
    edge_src,
    edge_dst,
    edge_attr,
    edge_scalars,
    batch,
):
    norm_1 = official.norm_1(node_input, batch=batch)
    attention = official.ga(
        node_input=norm_1,
        node_attr=node_attr,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_attr=edge_attr,
        edge_scalars=edge_scalars,
        batch=batch,
    )
    attention_residual = node_input + attention
    norm_2 = official.norm_2(attention_residual, batch=batch)
    ffn = official.ffn(norm_2, node_attr)
    residual = (
        official.ffn_shortcut(attention_residual, node_attr)
        if official.ffn_shortcut is not None
        else attention_residual
    )
    result = {
        "norm_1": norm_1,
        "attention": attention,
        "attention_residual": attention_residual,
        "norm_2": norm_2,
        "ffn": ffn,
        "out": residual + ffn,
    }
    if official.ffn_shortcut is not None:
        result["ffn_shortcut"] = residual
    return result


def _dsl_inputs(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars):
    return {
        "node_input": node_input,
        "node_attr": node_attr,
        "edge_attr": edge_attr,
        "edge_scalars": edge_scalars,
        "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
        "target_index": {"indices": edge_dst, "target_size": edge_dst.numel()},
        "segment_index": {"indices": edge_dst, "target_size": node_input.shape[0]},
    }


def _diagnostic_codes(compiler, program):
    from equivariant_nas.dsl import DSLValidationError

    try:
        compiler.analyze(program)
    except DSLValidationError as error:
        return sorted({item.code for item in error.diagnostics})
    return []


def build_audit(
    output: Path,
    official_root: Path,
    *,
    node_output_irreps: str = HIDDEN_IRREPS,
    rescale_degree: bool = False,
    nonlinear_message: bool = False,
) -> Dict[str, Any]:
    import torch

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    from scripts.audit_equiformer_v1_graph_attention_contract import _load_official_module
    from equivariant_nas.dsl import Compiler, core_registry, reference_motif_registry
    from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
    from equivariant_nas.dsl.canonicalize import (
        BACKEND_SEMANTICS_VERSION,
        COMPILER_SEMANTICS_VERSION,
    )
    from equivariant_nas.dsl.reference_programs import (
        EQUIFORMER_V1_TRANSBLOCK_NONLINEAR_SHORTCUT_PARAMETER_MAPPING,
        EQUIFORMER_V1_TRANSBLOCK_NONLINEAR_PARAMETER_MAPPING,
        EQUIFORMER_V1_TRANSBLOCK_PARAMETER_MAPPING,
        EQUIFORMER_V1_TRANSBLOCK_SHORTCUT_PARAMETER_MAPPING,
        equiformer_v1_transblock_program,
    )

    official_module, source_path = _load_official_module(official_root, torch)
    config = _official_config(
        o3,
        node_output_irreps=node_output_irreps,
        rescale_degree=rescale_degree,
        nonlinear_message=nonlinear_message,
    )
    torch.manual_seed(SEED)
    official = official_module.TransBlock(**config).double().eval()

    program = equiformer_v1_transblock_program(
        _radial_output_scales(official, torch),
        node_output_irreps=node_output_irreps,
        rescale_degree=rescale_degree,
        nonlinear_message=nonlinear_message,
    )
    registry = core_registry()
    motifs = reference_motif_registry()
    compiler = Compiler(registry, motifs)
    artifact = compiler.analyze(program)
    backend = E3NNGraphBackend(registry)
    torch.manual_seed(SEED)
    model = backend.build(artifact.expanded_program, artifact.inference).double().eval()

    uses_shortcut = config["irreps_node_input"] != config["irreps_node_output"]
    if nonlinear_message and uses_shortcut:
        mapping_source = EQUIFORMER_V1_TRANSBLOCK_NONLINEAR_SHORTCUT_PARAMETER_MAPPING
    elif nonlinear_message:
        mapping_source = EQUIFORMER_V1_TRANSBLOCK_NONLINEAR_PARAMETER_MAPPING
    elif uses_shortcut:
        mapping_source = EQUIFORMER_V1_TRANSBLOCK_SHORTCUT_PARAMETER_MAPPING
    else:
        mapping_source = EQUIFORMER_V1_TRANSBLOCK_PARAMETER_MAPPING
    mapping = dict(mapping_source)
    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(model.named_parameters())
    initialization_errors = {
        dsl_name: _max_abs(dsl_parameters[dsl_name], official_parameters[official_name])
        for dsl_name, official_name in mapping.items()
    }
    parameter_shapes = {
        dsl_name: list(dsl_parameters[dsl_name].shape)
        for dsl_name in sorted(dsl_parameters)
    }

    torch.manual_seed(SEED + 1)
    node_count = 5
    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    edge_dst = torch.tensor([1, 1, 3, 3, 0, 4, 4, 2], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    node_input = torch.randn(node_count, 15, dtype=torch.float64, requires_grad=True)
    node_attr = torch.randn(node_count, 1, dtype=torch.float64, requires_grad=True)
    edge_attr = torch.randn(edge_src.numel(), 9, dtype=torch.float64, requires_grad=True)
    edge_scalars = torch.randn(edge_src.numel(), 6, dtype=torch.float64, requires_grad=True)
    official_node = node_input.detach().clone().requires_grad_(True)
    official_node_attr = node_attr.detach().clone().requires_grad_(True)
    official_edge = edge_attr.detach().clone().requires_grad_(True)
    official_radial = edge_scalars.detach().clone().requires_grad_(True)

    actual = model(
        _dsl_inputs(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars),
        {},
    )
    expected = _official_intermediates(
        official,
        official_node,
        official_node_attr,
        edge_src,
        edge_dst,
        official_edge,
        official_radial,
        batch,
    )
    forward_errors = {name: _max_abs(actual[name], expected[name]) for name in expected}

    torch.manual_seed(SEED + 2)
    probe = torch.randn_like(actual["out"])
    (actual["out"] * probe).sum().backward()
    (expected["out"] * probe).sum().backward()
    input_gradient_errors = {
        "node_input": _max_abs(node_input.grad, official_node.grad),
        "node_attr": _max_abs(node_attr.grad, official_node_attr.grad),
        "edge_attr": _max_abs(edge_attr.grad, official_edge.grad),
        "edge_scalars": _max_abs(edge_scalars.grad, official_radial.grad),
    }
    parameter_gradient_errors = {
        dsl_name: _max_abs(dsl_parameters[dsl_name].grad, official_parameters[official_name].grad)
        for dsl_name, official_name in mapping.items()
    }

    with torch.no_grad():
        hidden_irreps = o3.Irreps(HIDDEN_IRREPS)
        output_irreps = o3.Irreps(str(node_output_irreps))
        edge_irreps = o3.Irreps(EDGE_IRREPS)
        symmetry = {}
        torch.manual_seed(SEED + 3)
        for label, matrix in (
            ("so3_rotation", o3.rand_matrix(dtype=torch.float64)),
            ("o3_inversion", -torch.eye(3, dtype=torch.float64)),
        ):
            hidden_action = hidden_irreps.D_from_matrix(matrix)
            output_action = output_irreps.D_from_matrix(matrix)
            edge_action = edge_irreps.D_from_matrix(matrix)
            transformed = model(
                _dsl_inputs(
                    node_input.detach() @ hidden_action.transpose(0, 1),
                    node_attr.detach(),
                    edge_src,
                    edge_dst,
                    edge_attr.detach() @ edge_action.transpose(0, 1),
                    edge_scalars.detach(),
                ),
                {},
            )
            symmetry[label] = {}
            for name in actual:
                action = (
                    output_action
                    if name in {"ffn", "ffn_shortcut", "out"}
                    else hidden_action
                )
                symmetry[label][name] = _relative_error(
                    transformed[name],
                    actual[name] @ action.transpose(0, 1),
                )

        edge_permutation = torch.tensor([5, 0, 7, 3, 1, 6, 2, 4], dtype=torch.long)
        edge_permuted = model(
            _dsl_inputs(
                node_input.detach(),
                node_attr.detach(),
                edge_src.index_select(0, edge_permutation),
                edge_dst.index_select(0, edge_permutation),
                edge_attr.detach().index_select(0, edge_permutation),
                edge_scalars.detach().index_select(0, edge_permutation),
            ),
            {},
        )
        symmetry["edge_permutation_out_max_abs_error"] = _max_abs(
            edge_permuted["out"], actual["out"]
        )

        node_permutation = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
        inverse = torch.empty_like(node_permutation)
        inverse[node_permutation] = torch.arange(node_count)
        node_permuted = model(
            _dsl_inputs(
                node_input.detach().index_select(0, node_permutation),
                node_attr.detach().index_select(0, node_permutation),
                inverse.index_select(0, edge_src),
                inverse.index_select(0, edge_dst),
                edge_attr.detach(),
                edge_scalars.detach(),
            ),
            {},
        )
        symmetry["node_permutation_out_max_abs_error"] = _max_abs(
            node_permuted["out"], actual["out"].index_select(0, node_permutation)
        )

        model.train()
        official.train()
        dsl_train = model(
            _dsl_inputs(
                node_input.detach(),
                node_attr.detach(),
                edge_src,
                edge_dst,
                edge_attr.detach(),
                edge_scalars.detach(),
            ),
            {},
        )["out"]
        official_train = official(
            node_input.detach(),
            node_attr.detach(),
            edge_src,
            edge_dst,
            edge_attr.detach(),
            edge_scalars.detach(),
            batch,
        )
        train_mode = {
            "dsl_vs_official_max_abs_error": _max_abs(dsl_train, official_train),
            "dsl_train_vs_eval_max_abs_error": _max_abs(dsl_train, actual["out"]),
            "stochastic_operators_enabled": False,
        }
        model.eval()
        official.eval()

    bad_affine_nodes = tuple(
        replace(node, attrs={**node.attrs, "affine": False}) if node.id == "norm_1" else node
        for node in program.nodes
    )
    bad_norm_nodes = tuple(
        replace(node, attrs={**node.attrs, "normalization": "batch"})
        if node.id == "norm_1"
        else node
        for node in program.nodes
    )
    negative_contracts = {
        "affine_false": _diagnostic_codes(compiler, replace(program, nodes=bad_affine_nodes)),
        "invalid_norm_mode": _diagnostic_codes(compiler, replace(program, nodes=bad_norm_nodes)),
    }
    try:
        equiformer_v1_transblock_program([1.0] * 29)
    except ValueError as error:
        negative_contracts["wrong_radial_scale_count"] = {
            "exception": type(error).__name__,
            "message": str(error),
        }

    support = backend.support_report(artifact.expanded_program).to_dict()
    parameter_contracts = {
        node_id: [contract.to_dict() for contract in contracts]
        for node_id, contracts in sorted(artifact.inference.parameter_contracts.items())
        if contracts
    }
    model_identity = {
        "architecture_id": artifact.architecture_id,
        "program_id": program.program_id,
        "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
        "backend_neutral_semantics_version": BACKEND_SEMANTICS_VERSION,
        "backend_semantics_version": model.backend_semantics_version,
        "graph_class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
        "source_node_count": len(program.nodes),
        "expanded_node_count": len(artifact.expanded_program.nodes),
        "expanded_node_ids": [node.id for node in artifact.expanded_program.nodes],
        "expanded_node_ops": [node.op for node in artifact.expanded_program.nodes],
        "trainable_parameter_tensor_count": len(dsl_parameters),
        "trainable_parameter_count": sum(parameter.numel() for parameter in dsl_parameters.values()),
        "trainable_parameter_shapes": parameter_shapes,
        "official_parameter_mapping": mapping,
        "official_network_block_or_operator_used_for_dsl_execution": False,
    }
    math_contract = {
        "group": "O(3) with SO(3) rotation and inversion checks",
        "node_representation": HIDDEN_IRREPS,
        "edge_representation": EDGE_IRREPS,
        "pre_norm_residual_equations": [
            "h1 = LayerNorm_irrep(h)",
            "r1 = h + GraphAttention(h1, edge_attr, radial)",
            "h2 = LayerNorm_irrep(r1)",
            (
                "out = FCTP_shortcut(r1, node_attr) + FCTP2(Gate(FCTP1(h2, node_attr)), node_attr)"
                if uses_shortcut
                else "out = r1 + FCTP2(Gate(FCTP1(h2, node_attr)), node_attr)"
            ),
        ],
        "layer_norm": "scalar multiplicities are centered; every irrep field is normalized by its component second moment; affine weights are per irrep instance and affine bias is limited to even scalars",
        "attention": (
            "endpoint LinearRS merge, explicit gather/add, radial external-weight uvu tensor product, LinearRS plus second-moment SiLU/sigmoid Gate, separate alpha LinearRS, internal/shared uvu value tensor product, value LinearRS, head split, second-moment SmoothLeakyReLU, alpha_dot, target-segment softmax, invariant scaling, segment sum, head merge and LinearRS projection"
            if nonlinear_message
            else "endpoint LinearRS merge, explicit gather/add, radial external-weight uvu tensor product, head split, second-moment SmoothLeakyReLU, alpha_dot, target-segment softmax, invariant scaling, segment sum, head merge and LinearRS projection"
        ),
        "feed_forward": "internal/shared uvw fully-connected tensor products with even-scalar bias, second-moment SiLU/sigmoid gates, irrep-wise gating and canonical concatenation",
        "parameter_initialization": "canonical program order controls global-RNG module construction; alpha_dot preserves official randn allocation followed by Glorot overwrite",
        "drop_contract": "alpha_drop=proj_drop=drop_path=0; this evidence certifies the deterministic TransBlock branch only",
        "degree_rescale": "segment_reduce normalization=target_cardinality multiplies each target-node attention sum by its explicit segment cardinality before head merge; this commutes with head merge and matches the official post-merge degree multiplication",
    }
    source_text = source_path.read_text(encoding="utf-8")
    source_identity = {
        "repository_root": str(official_root.resolve()),
        "source_file": str(source_path.resolve()),
        "source_sha256": _sha256_text(source_text),
        "official_class": "{}.{}".format(type(official).__module__, type(official).__qualname__),
        "oracle_only": True,
    }
    numerical_summary = {
        "max_initialization_error": max(initialization_errors.values()),
        "max_forward_error": max(forward_errors.values()),
        "max_input_gradient_error": max(input_gradient_errors.values()),
        "max_parameter_gradient_error": max(parameter_gradient_errors.values()),
        "max_so3_relative_error": max(symmetry["so3_rotation"].values()),
        "max_o3_inversion_relative_error": max(symmetry["o3_inversion"].values()),
        "edge_permutation_out_max_abs_error": symmetry["edge_permutation_out_max_abs_error"],
        "node_permutation_out_max_abs_error": symmetry["node_permutation_out_max_abs_error"],
        "train_mode_dsl_vs_official_max_abs_error": train_mode["dsl_vs_official_max_abs_error"],
    }
    claims = {
        "deterministic_v1_transblock_expressed_by_typed_dsl": True,
        "generic_e3nn_graph_lowering_executed": True,
        "official_constructor_used_for_dsl_execution": False,
        "all_parameter_tensors_seed_identical": max(initialization_errors.values()) == 0.0,
        "all_six_exposed_intermediates_exact_within_float64_tolerance": max(forward_errors.values()) < 1.0e-12,
        "four_input_gradients_exact_within_float64_tolerance": max(input_gradient_errors.values()) < 1.0e-11,
        "all_parameter_gradients_exact_within_float64_tolerance": max(parameter_gradient_errors.values()) < 1.0e-11,
        "stochastic_transblock_branches_certified": False,
        "degree_rescale_branch_certified": bool(rescale_degree),
        "nonlinear_message_branch_certified": bool(nonlinear_message),
        "different_output_irreps_shortcut_certified": bool(uses_shortcut),
    }
    summary = {
        "evidence_version": (
            "equiformer-v1-transblock-shortcut-audit@1"
            if uses_shortcut
            else (
                "equiformer-v1-transblock-nonlinear-message-audit@1"
                if nonlinear_message
                else (
                    "equiformer-v1-transblock-degree-rescale-audit@1"
                    if rescale_degree
                    else EVIDENCE_VERSION
                )
            )
        ),
        "seed": SEED,
        "oracle_config": _serializable_config(config),
        "model_identity": model_identity,
        "numerical_summary": numerical_summary,
        "negative_contracts": negative_contracts,
        "claims": claims,
    }

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "canonical_transblock_program.json", program.to_dict())
    _write_json(output / "expanded_transblock_program.json", artifact.expanded_program.to_dict())
    _write_json(output / "lowering_manifest.json", backend.lowering_manifest())
    _write_json(output / "support_report.json", support)
    _write_json(output / "parameter_contracts.json", parameter_contracts)
    _write_json(output / "model_identity.json", model_identity)
    _write_json(output / "math_contract.json", math_contract)
    _write_json(output / "official_v1_source_identity.json", source_identity)
    _write_json(output / "initialization_errors.json", initialization_errors)
    _write_json(output / "intermediate_forward_errors.json", forward_errors)
    _write_json(output / "input_gradient_errors.json", input_gradient_errors)
    _write_json(output / "parameter_gradient_errors.json", parameter_gradient_errors)
    _write_json(output / "symmetry_evidence.json", symmetry)
    _write_json(output / "train_mode_evidence.json", train_mode)
    _write_json(output / "negative_contracts.json", negative_contracts)
    _write_json(output / "numerical_summary.json", numerical_summary)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("audit-equiformer-v1-transblock-exact")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rescale-degree", action="store_true")
    parser.add_argument("--nonlinear-message", action="store_true")
    parser.add_argument("--node-output-irreps", default=HIDDEN_IRREPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_audit(
        Path(args.output),
        Path(args.official_root),
        node_output_irreps=str(args.node_output_irreps),
        rescale_degree=bool(args.rescale_degree),
        nonlinear_message=bool(args.nonlinear_message),
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "architecture_id": result["model_identity"]["architecture_id"],
                "parameter_tensors": result["model_identity"]["trainable_parameter_tensor_count"],
                "parameter_count": result["model_identity"]["trainable_parameter_count"],
                **result["numerical_summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
