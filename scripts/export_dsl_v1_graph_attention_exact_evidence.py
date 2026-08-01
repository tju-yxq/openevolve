#!/usr/bin/env python
"""Export exact primitive-level evidence for frozen Equiformer V1 GraphAttention."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_equiformer_v1_graph_attention_contract import (
    _load_official_module,
    _segment_reduce,
    _segment_softmax,
)
from equivariant_nas.dsl import Compiler, DSLValidationError, core_registry, reference_motif_registry
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.canonicalize import canonicalize
from equivariant_nas.dsl.reference_programs import (
    EQUIFORMER_V1_GRAPH_ATTENTION_PARAMETER_MAPPING,
    equiformer_v1_graph_attention_program,
)
from equivariant_nas.dsl.serialization import dumps_program


EVIDENCE_VERSION = "evoequilang-v1-graph-attention-exact-evidence@1"
SEED = 20260731


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _max_abs(left, right) -> float:
    return float((left - right).detach().abs().max().item()) if left.numel() else 0.0


def _relative_error(left, right) -> float:
    return float(((left - right).norm() / right.norm().clamp_min(1.0e-12)).detach().item())


def _official_config(o3):
    return {
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


def _official_forward(model, node_input, edge_src, edge_dst, edge_attr, edge_scalars, torch):
    merge_src = model.merge_src(node_input)
    merge_dst = model.merge_dst(node_input)
    message = merge_src.index_select(0, edge_src) + merge_dst.index_select(0, edge_dst)
    radial = model.sep.dtp_rad(edge_scalars)
    tp = model.sep.dtp(message, edge_attr, radial)
    post_tp = model.sep.lin(tp)
    heads = model.vec2heads(post_tp)
    alpha_channels = heads.narrow(2, 0, model.mul_alpha_head)
    value = heads.narrow(2, model.mul_alpha_head, heads.shape[-1] - model.mul_alpha_head)
    alpha_activated = model.alpha_act(alpha_channels)
    logits = torch.einsum("bik,aik->bi", alpha_activated, model.alpha_dot)
    softmax = _segment_softmax(logits, edge_dst, torch)
    weighted = value * softmax.unsqueeze(-1)
    aggregate = _segment_reduce(weighted, edge_dst, node_input.shape[0], torch, reduce="sum")
    merged = model.heads2vec(aggregate)
    out = model.proj(merged)
    return {
        "merge_src": merge_src,
        "merge_dst": merge_dst,
        "message": message,
        "radial": radial,
        "tp": tp,
        "post_tp": post_tp,
        "heads": heads,
        "alpha_channels": alpha_channels,
        "value": value,
        "alpha_activated": alpha_activated,
        "logits": logits,
        "softmax": softmax,
        "weighted": weighted,
        "aggregate": aggregate,
        "merged": merged,
        "out": out,
    }


def _copy_official_parameters(dsl_model, official_model):
    mapping = dict(EQUIFORMER_V1_GRAPH_ATTENTION_PARAMETER_MAPPING)
    dsl_parameters = dict(dsl_model.named_parameters())
    official_parameters = dict(official_model.named_parameters())
    if set(dsl_parameters) != set(mapping):
        raise RuntimeError("DSL parameter set no longer equals the frozen 14-tensor mapping")
    for dsl_name, official_name in mapping.items():
        if tuple(dsl_parameters[dsl_name].shape) != tuple(official_parameters[official_name].shape):
            raise RuntimeError("parameter shape mismatch: {} -> {}".format(dsl_name, official_name))
        dsl_parameters[dsl_name].data.copy_(official_parameters[official_name].data)
    return mapping, dsl_parameters, official_parameters


def _replace_activation_normalization(program, normalization: str):
    nodes = []
    for node in program.nodes:
        if node.id == "alpha_activated":
            attrs = dict(node.attrs)
            attrs["normalization"] = normalization
            node = replace(node, attrs=attrs)
        nodes.append(node)
    return replace(program, nodes=tuple(nodes), program_id="{}-normalization-{}".format(program.program_id, normalization))


def build_evidence(output: Path, *, official_v1_root: Path, pytest_summary: str = "") -> Dict[str, Any]:
    import torch
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    torch.manual_seed(SEED)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    official_module, source_path = _load_official_module(official_v1_root.resolve(), torch)
    official = official_module.GraphAttention(**_official_config(o3)).double().eval()
    scales = torch.ones(30, dtype=torch.float64)
    for output_slice, scale in official.sep.dtp.slices_sqrt_k.values():
        scales[output_slice] *= float(scale)

    registry = core_registry()
    motifs = reference_motif_registry()
    program = equiformer_v1_graph_attention_program(scales.tolist())
    artifact = Compiler(registry, motifs).analyze(program)
    backend = E3NNGraphBackend(registry)
    support = backend.support_report(artifact.expanded_program)
    if not support.supported:
        raise RuntimeError("complete V1 GraphAttention support report failed: {}".format(support.to_dict()))
    model = backend.build(artifact.expanded_program, artifact.inference).double().eval()
    mapping, dsl_parameters, official_parameters = _copy_official_parameters(model, official)

    node_count = 5
    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    edge_dst = torch.tensor([1, 1, 3, 3, 0, 4, 4, 2], dtype=torch.long)
    node_input = torch.randn(node_count, 15, dtype=torch.float64, requires_grad=True)
    edge_attr = torch.randn(edge_src.numel(), 9, dtype=torch.float64, requires_grad=True)
    edge_scalars = torch.randn(edge_src.numel(), 6, dtype=torch.float64, requires_grad=True)
    official_node = node_input.detach().clone().requires_grad_(True)
    official_edge = edge_attr.detach().clone().requires_grad_(True)
    official_radial = edge_scalars.detach().clone().requires_grad_(True)
    runtime_inputs = {
        "node_input": node_input,
        "edge_attr": edge_attr,
        "edge_scalars": edge_scalars,
        "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
        "target_index": {"indices": edge_dst, "target_size": edge_dst.numel()},
        "segment_index": {"indices": edge_dst, "target_size": node_count},
    }
    actual = model(runtime_inputs, {})
    expected = _official_forward(
        official,
        official_node,
        edge_src,
        edge_dst,
        official_edge,
        official_radial,
        torch,
    )
    intermediate_errors = {name: _max_abs(actual[name], expected[name]) for name in expected}

    probe = torch.randn_like(actual["out"])
    (actual["out"] * probe).sum().backward()
    (expected["out"] * probe).sum().backward()
    input_gradient_errors = {
        "node_input": _max_abs(node_input.grad, official_node.grad),
        "edge_attr": _max_abs(edge_attr.grad, official_edge.grad),
        "edge_scalars": _max_abs(edge_scalars.grad, official_radial.grad),
    }
    parameter_gradient_errors = {
        dsl_name: _max_abs(dsl_parameters[dsl_name].grad, official_parameters[official_name].grad)
        for dsl_name, official_name in mapping.items()
    }

    symmetry = {}
    node_irreps = o3.Irreps("4x0e+2x1e+1x2e")
    edge_irreps = o3.Irreps("1x0e+1x1e+1x2e")
    with torch.no_grad():
        for name, matrix in {
            "so3_rotation": o3.rand_matrix(dtype=torch.float64),
            "o3_inversion": -torch.eye(3, dtype=torch.float64),
        }.items():
            node_action = node_irreps.D_from_matrix(matrix)
            edge_action = edge_irreps.D_from_matrix(matrix)
            transformed = model(
                {
                    "node_input": node_input.detach() @ node_action.transpose(0, 1),
                    "edge_attr": edge_attr.detach() @ edge_action.transpose(0, 1),
                    "edge_scalars": edge_scalars.detach(),
                    "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
                    "target_index": {"indices": edge_dst, "target_size": edge_dst.numel()},
                    "segment_index": {"indices": edge_dst, "target_size": node_count},
                },
                {},
            )
            symmetry[name + "_output_relative_error"] = _relative_error(
                transformed["out"],
                actual["out"].detach() @ node_action.transpose(0, 1),
            )
            symmetry[name + "_attention_max_abs_error"] = _max_abs(
                transformed["softmax"], actual["softmax"].detach()
            )

        permutation = torch.tensor([5, 0, 7, 3, 1, 6, 2, 4], dtype=torch.long)
        permuted_src = edge_src.index_select(0, permutation)
        permuted_dst = edge_dst.index_select(0, permutation)
        permuted = model(
            {
                "node_input": node_input.detach(),
                "edge_attr": edge_attr.detach().index_select(0, permutation),
                "edge_scalars": edge_scalars.detach().index_select(0, permutation),
                "source_index": {"indices": permuted_src, "target_size": edge_src.numel()},
                "target_index": {"indices": permuted_dst, "target_size": edge_dst.numel()},
                "segment_index": {"indices": permuted_dst, "target_size": node_count},
            },
            {},
        )
        symmetry["edge_permutation_output_max_abs_error"] = _max_abs(
            permuted["out"], actual["out"].detach()
        )
        symmetry["edge_permutation_attention_max_abs_error"] = _max_abs(
            permuted["softmax"], actual["softmax"].detach().index_select(0, permutation)
        )

    raw_program = _replace_activation_normalization(program, "none")
    raw_artifact = Compiler(registry, motifs).analyze(raw_program)
    raw_model = backend.build(raw_artifact.expanded_program, raw_artifact.inference).double().eval()
    _copy_official_parameters(raw_model, official)
    with torch.no_grad():
        raw_output = raw_model(
            {
                "node_input": node_input.detach(),
                "edge_attr": edge_attr.detach(),
                "edge_scalars": edge_scalars.detach(),
                "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
                "target_index": {"indices": edge_dst, "target_size": edge_dst.numel()},
                "segment_index": {"indices": edge_dst, "target_size": node_count},
            },
            {},
        )["out"]
    negative_contracts = {
        "missing_second_moment_normalization": {
            "official_forward_max_abs_error": _max_abs(raw_output, expected["out"].detach()),
            "mismatch_detected": _max_abs(raw_output, expected["out"].detach()) > 1.0e-6,
        }
    }
    invalid_program = _replace_activation_normalization(program, "batch")
    invalid_artifact = Compiler(registry, motifs).analyze(invalid_program)
    try:
        backend.build(invalid_artifact.expanded_program, invalid_artifact.inference)
        invalid_result = {"rejected": False, "expected_code": "E_BACKEND_020", "diagnostics": []}
    except DSLValidationError as error:
        invalid_result = {
            "rejected": True,
            "expected_code": "E_BACKEND_020",
            "expected_code_present": any(item.code == "E_BACKEND_020" for item in error.diagnostics),
            "diagnostics": [item.to_dict() for item in error.diagnostics],
        }
    negative_contracts["unknown_activation_normalization"] = invalid_result

    numerical = {
        "max_intermediate_forward_error": max(intermediate_errors.values()),
        "max_input_gradient_error": max(input_gradient_errors.values()),
        "max_parameter_gradient_error": max(parameter_gradient_errors.values()),
        "all_dsl_gradients_finite": all(
            tensor.grad is not None and bool(torch.isfinite(tensor.grad).all())
            for tensor in (node_input, edge_attr, edge_scalars)
        ) and all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in dsl_parameters.values()
        ),
    }
    if max(numerical[key] for key in numerical if key.endswith("error")) > 1.0e-10:
        raise RuntimeError("V1 GraphAttention exact evidence exceeded 1e-10: {}".format(numerical))
    if not numerical["all_dsl_gradients_finite"]:
        raise RuntimeError("V1 GraphAttention produced non-finite gradients")
    if max(value for key, value in symmetry.items() if key.endswith("relative_error")) > 1.0e-7:
        raise RuntimeError("V1 GraphAttention equivariance exceeded 1e-7: {}".format(symmetry))
    if max(value for key, value in symmetry.items() if key.endswith("max_abs_error")) > 1.0e-7:
        raise RuntimeError("V1 GraphAttention invariance/permutation exceeded 1e-7: {}".format(symmetry))
    if not negative_contracts["missing_second_moment_normalization"]["mismatch_detected"]:
        raise RuntimeError("second-moment negative control did not change the official output")
    if not invalid_result.get("rejected") or not invalid_result.get("expected_code_present"):
        raise RuntimeError("unknown activation normalization was not rejected")

    source_identity = {
        "repository_root": str(official_v1_root.resolve()),
        "source_file": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "oracle_object": "GraphAttention(nonlinear_message=False, dropout=0, rescale_degree=False)",
        "official_constructor_role": "oracle construction only; no official module executes inside the DSL graph",
    }
    parameter_contracts = {
        node_id: [contract.to_dict() for contract in contracts]
        for node_id, contracts in artifact.inference.parameter_contracts.items()
        if contracts
    }
    model_identity = {
        "backend_semantics_version": model.backend_semantics_version,
        "graph_class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
        "expanded_node_ids": [node.id for node in artifact.expanded_program.nodes],
        "expanded_node_ops": [node.op for node in artifact.expanded_program.nodes],
        "expanded_node_count": len(artifact.expanded_program.nodes),
        "trainable_parameter_tensor_count": len(dsl_parameters),
        "trainable_parameter_count": sum(parameter.numel() for parameter in dsl_parameters.values()),
        "trainable_parameter_shapes": {
            name: list(parameter.shape) for name, parameter in dsl_parameters.items()
        },
        "official_parameter_mapping": mapping,
        "official_network_block_or_operator_used_for_dsl_execution": False,
    }
    math_contract = {
        "message": "m_ij = LinearRS_src(x_i) + LinearRS_dst(x_j)",
        "radial": "w_ij = offset + W_2 SiLU(LayerNorm(W_1 r_ij + b_1)); W_2 and offset use slice-sqrt-k initialization scales",
        "tensor_product": "t_ij = TP_uvu(m_ij, Y_ij; w_ij) over 15 explicit legal O(3) paths",
        "attention": "alpha_ijh = softmax_j(<normalize2mom(SmoothLeakyReLU(a_ijh)), q_h>)",
        "aggregation": "z_ih = sum_{j->i} alpha_ijh v_ijh; output = LinearRS(head_merge(z_i))",
        "equivariance": "for every g in O(3), F(D_g x, D_g edge_attr, r, graph) = D_g F(x, edge_attr, r, graph)",
        "permutation": "a common permutation of edge rows and all edge index maps leaves node output unchanged and permutes edge attention rows",
    }
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "seed": SEED,
        "pytest_summary": pytest_summary,
        "primitive_count": len(registry.names()),
        "motif_count": len(motifs.names()),
        "lowering_rule_count": model.lowering_rule_manifest["rule_count"],
        "architecture_id": artifact.architecture_id,
        "official_v1_source_identity": source_identity,
        "model_identity": model_identity,
        "parameter_contracts": parameter_contracts,
        "math_contract": math_contract,
        "intermediate_forward_errors": intermediate_errors,
        "input_gradient_errors": input_gradient_errors,
        "parameter_gradient_errors": parameter_gradient_errors,
        "numerical_summary": numerical,
        "symmetry_evidence": symmetry,
        "negative_contracts": negative_contracts,
        "claims": {
            "official_v1_graph_attention_exactly_reproduced": True,
            "all_16_exported_intermediate_tensors_reproduced": True,
            "all_14_parameter_tensors_mapped": True,
            "three_input_gradients_reproduced": True,
            "all_parameter_gradients_reproduced": True,
            "so3_rotation_and_o3_inversion_verified": True,
            "edge_permutation_verified": True,
            "generic_primitive_lowering_used": True,
            "official_constructor_bypass_used": False,
            "official_v1_complete_transformer_block_reproduced": False,
            "official_v1_complete_network_reproduced": False,
        },
    }

    (output / "canonical_graph_attention_program.json").write_text(
        dumps_program(canonicalize(artifact.expanded_program, registry)),
        encoding="utf-8",
    )
    _write_json(output / "support_report.json", support.to_dict())
    _write_json(output / "lowering_manifest.json", model.lowering_rule_manifest)
    _write_json(output / "parameter_contracts.json", parameter_contracts)
    _write_json(output / "model_identity.json", model_identity)
    _write_json(output / "official_v1_source_identity.json", source_identity)
    _write_json(output / "math_contract.json", math_contract)
    _write_json(output / "intermediate_forward_errors.json", intermediate_errors)
    _write_json(output / "input_gradient_errors.json", input_gradient_errors)
    _write_json(output / "parameter_gradient_errors.json", parameter_gradient_errors)
    _write_json(output / "numerical_summary.json", numerical)
    _write_json(output / "symmetry_evidence.json", symmetry)
    _write_json(output / "negative_contracts.json", negative_contracts)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("export-dsl-v1-graph-attention-exact-evidence")
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
        "motif_count": result["motif_count"],
        "lowering_rule_count": result["lowering_rule_count"],
        "max_forward_error": result["numerical_summary"]["max_intermediate_forward_error"],
        "max_parameter_gradient_error": result["numerical_summary"]["max_parameter_gradient_error"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
