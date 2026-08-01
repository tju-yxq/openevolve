#!/usr/bin/env python
"""Export official-oracle evidence for blockwise equivariant multihead DSL v2."""

from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
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
    IndexMapType,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend, to_e3nn_irreps
from equivariant_nas.dsl.serialization import dumps_program


EVIDENCE_VERSION = "evoequilang-equivariant-multihead-v2-evidence@1"
SEED = 20260731


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _max_abs(left, right) -> float:
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def _relative_error(left, right) -> float:
    return float(((left - right).norm() / right.norm().clamp_min(1.0e-12)).item())


def _load_official_head_classes(source_path: Path, torch, o3):
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"Vec2AttnHeads", "AttnHeads2Vec"}
    ]
    if len(selected) != 2:
        raise RuntimeError("official V1 source does not contain the expected head conversion classes")
    from e3nn.util.jit import compile_mode

    namespace = {"torch": torch, "o3": o3, "compile_mode": compile_mode}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["Vec2AttnHeads"], namespace["AttnHeads2Vec"], source


def _types():
    group = GroupSpec.o3()
    message = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("6x0e+2x1o", group.family),
        dtype="float64",
    )
    segment = IndexMapType(group, Carrier.EDGE, Carrier.NODE, "segment", target_size=3)
    head = replace(
        message,
        irreps=Irreps.parse("3x0e+1x1o", group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", 2, FeatureRole.HEAD, "independent", 0),),
    )
    alpha_head = replace(head, irreps=Irreps.parse("2x0e", group.family))
    value_head = replace(head, irreps=Irreps.parse("1x0e+1x1o", group.family))
    alpha = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("2x0e", group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", 2, FeatureRole.HEAD, "independent", 0),),
        dtype="float64",
        feature_role=FeatureRole.ALPHA,
    )
    output = replace(
        message,
        carrier=Carrier.NODE,
        irreps=Irreps.parse("2x0e+2x1o", group.family),
    )
    return message, segment, head, alpha_head, value_head, alpha, output


def _program():
    message, segment, _head, _alpha_head, _value_head, alpha, output = _types()
    return ArchitectureProgram(
        "2.5.0",
        "v1-attention-core-equivariant-multihead-evidence",
        (InputPort("message", message), InputPort("segment_index", segment)),
        (
            Node("split", "core.head_split@2", {"x": ("input:message",)}, {"head_axis": "head", "num_heads": 2}),
            Node(
                "alpha_value",
                "core.irrep_select@2",
                {"x": ("split",)},
                {"selections": [{"irrep": "0e", "start": 0, "multiplicity": 2}]},
            ),
            Node(
                "value",
                "core.irrep_select@2",
                {"x": ("split",)},
                {
                    "selections": [
                        {"irrep": "0e", "start": 2, "multiplicity": 1},
                        {"irrep": "1o", "start": 0, "multiplicity": 1},
                    ]
                },
            ),
            Node(
                "logits",
                "core.headwise_scalar_contraction@2",
                {"x": ("alpha_value",)},
                {"head_axis": "head", "bias": False},
            ),
            Node(
                "activated",
                "core.scalar_activation@1",
                {"x": ("logits",)},
                {"activation": "smooth_leaky_relu", "negative_slope": 0.2},
            ),
            Node(
                "softmax",
                "core.segment_softmax@2",
                {"logits": ("activated",), "index": ("input:segment_index",)},
                {},
            ),
            Node("weighted", "core.invariant_scale@1", {"weight": ("softmax",), "value": ("value",)}, {}),
            Node(
                "aggregate",
                "core.segment_reduce@1",
                {"x": ("weighted",), "index": ("input:segment_index",)},
                {"reduce": "sum", "normalization": "none"},
            ),
            Node("merge", "core.head_merge@2", {"x": ("aggregate",)}, {"head_axis": "head"}),
        ),
        (OutputPort("out", "merge", output), OutputPort("alpha", "softmax", alpha)),
        program_id="official-v1-attention-core-equivariant-heads@1",
    )


def _segment_softmax(logits, index, torch):
    output = torch.empty_like(logits)
    for target in torch.unique(index):
        mask = index == target
        output[mask] = torch.softmax(logits[mask], dim=0)
    return output


def _smooth_leaky_relu(value, torch, slope=0.2):
    return ((1.0 + slope) / 2.0) * value + ((1.0 - slope) / 2.0) * value * (2.0 * torch.sigmoid(value) - 1.0)


def _negative_contracts(registry, message, alpha, segment):
    results = {}

    invalid_split = replace(message, irreps=Irreps.parse("3x0e+2x1o", message.group.family))
    program = ArchitectureProgram(
        "2.5.0",
        "negative-head-divisibility",
        (InputPort("x", invalid_split),),
        (Node("split", "core.head_split@2", {"x": ("input:x",)}, {"head_axis": "head", "num_heads": 2}),),
        (OutputPort("out", "split", invalid_split),),
    )
    try:
        Compiler(registry).analyze(program)
        results["nondivisible_multiplicity"] = {"rejected": False, "diagnostics": []}
    except DSLValidationError as error:
        results["nondivisible_multiplicity"] = {
            "rejected": True,
            "expected_code_present": any(item.code == "E_HEAD_V2_006" for item in error.diagnostics),
            "diagnostics": [item.to_dict() for item in error.diagnostics],
        }

    wrong_index = IndexMapType(message.group, Carrier.NODE, Carrier.EDGE, "target", target_size=8)
    program = ArchitectureProgram(
        "2.5.0",
        "negative-softmax-index",
        (InputPort("logits", alpha), InputPort("index", wrong_index)),
        (Node("softmax", "core.segment_softmax@2", {"logits": ("input:logits",), "index": ("input:index",)}, {}),),
        (OutputPort("out", "softmax", alpha),),
    )
    try:
        Compiler(registry).analyze(program)
        results["wrong_segment_index"] = {"rejected": False, "diagnostics": []}
    except DSLValidationError as error:
        results["wrong_segment_index"] = {
            "rejected": True,
            "expected_code_present": any(item.code == "E_SOFTMAX_V2_004" for item in error.diagnostics),
            "diagnostics": [item.to_dict() for item in error.diagnostics],
        }

    return results


def build_evidence(output: Path, *, official_v1_root: Path, pytest_summary: str = "") -> Dict[str, Any]:
    import torch
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    torch.manual_seed(SEED)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    official_source = official_v1_root.resolve() / "nets" / "graph_attention_transformer.py"
    if not official_source.exists():
        raise FileNotFoundError(official_source)
    Vec2AttnHeads, AttnHeads2Vec, official_text = _load_official_head_classes(official_source, torch, o3)

    registry = core_registry()
    program = _program()
    message_type, segment_type, head_type, _alpha_head, value_head, alpha_type, output_type = _types()
    artifact = Compiler(registry).analyze(program)
    backend = E3NNGraphBackend(registry)
    support = backend.support_report(program)
    if not support.supported:
        raise RuntimeError("equivariant multihead support report failed: {}".format(support.to_dict()))
    model = backend.build(program, artifact.inference).double().eval()
    contraction = model.node_modules["logits"]
    with torch.no_grad():
        contraction.weight.copy_(torch.tensor([[[0.7, -0.2], [-0.4, 0.9]]], dtype=torch.float64))

    official_split = Vec2AttnHeads(o3.Irreps(to_e3nn_irreps(head_type.irreps)), 2).double()
    official_merge = AttnHeads2Vec(o3.Irreps(to_e3nn_irreps(value_head.irreps))).double()
    edge_dst = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long)
    segment_payload = {"indices": edge_dst, "target_size": int(segment_type.target_size)}
    message = torch.randn(edge_dst.numel(), message_type.irreps.dimension, dtype=torch.float64, requires_grad=True)
    oracle_message = message.detach().clone().requires_grad_(True)
    oracle_weight = contraction.weight.detach().clone().requires_grad_(True)

    actual = model({"message": message, "segment_index": segment_payload}, {})
    official_heads = official_split(oracle_message)
    alpha_value = official_heads[..., :2]
    value = torch.cat((official_heads[..., 2:3], official_heads[..., 3:6]), dim=-1)
    raw_logits = (alpha_value * oracle_weight).sum(dim=-1)
    oracle_logits = _smooth_leaky_relu(raw_logits, torch)
    oracle_alpha = _segment_softmax(oracle_logits, edge_dst, torch)
    weighted = value * oracle_alpha.unsqueeze(-1)
    aggregated = weighted.new_zeros((3,) + weighted.shape[1:]).index_add_(0, edge_dst, weighted)
    oracle_output = official_merge(aggregated)

    cotangent = torch.randn_like(actual["out"])
    (actual["out"] * cotangent).sum().backward()
    (oracle_output * cotangent).sum().backward()

    rotation = o3.rand_matrix(dtype=torch.float64)
    inversion = -o3.rand_matrix(dtype=torch.float64)
    input_irreps = o3.Irreps(to_e3nn_irreps(message_type.irreps))
    output_irreps = o3.Irreps(to_e3nn_irreps(output_type.irreps))

    def symmetry_error(matrix):
        input_action = input_irreps.D_from_matrix(matrix)
        output_action = output_irreps.D_from_matrix(matrix)
        transformed = model(
            {
                "message": message.detach() @ input_action.transpose(0, 1),
                "segment_index": segment_payload,
            },
            {},
        )
        expected = actual["out"].detach() @ output_action.transpose(0, 1)
        return _relative_error(transformed["out"], expected), _max_abs(transformed["alpha"], actual["alpha"].detach())

    rotation_error, rotation_alpha_error = symmetry_error(rotation)
    inversion_error, inversion_alpha_error = symmetry_error(inversion)
    permutation = torch.tensor([5, 0, 7, 3, 1, 6, 2, 4], dtype=torch.long)
    permuted_index = edge_dst.index_select(0, permutation)
    permuted = model(
        {
            "message": message.detach().index_select(0, permutation),
            "segment_index": {"indices": permuted_index, "target_size": 3},
        },
        {},
    )

    roundtrip_source = torch.randn(7, message_type.irreps.dimension, dtype=torch.float64, requires_grad=True)
    split_program = ArchitectureProgram(
        "2.5.0",
        "split-oracle-only",
        (InputPort("x", message_type),),
        (Node("split", "core.head_split@2", {"x": ("input:x",)}, {"head_axis": "head", "num_heads": 2}),),
        (OutputPort("out", "split", head_type),),
    )
    split_artifact = Compiler(registry).analyze(split_program)
    split_model = backend.build(split_program, split_artifact.inference).double().eval()
    split_actual = split_model({"x": roundtrip_source}, {})["out"]
    split_oracle_input = roundtrip_source.detach().clone().requires_grad_(True)
    split_oracle = official_split(split_oracle_input)
    split_cotangent = torch.randn_like(split_actual)
    (split_actual * split_cotangent).sum().backward()
    (split_oracle * split_cotangent).sum().backward()

    metrics = {
        "official_attention_core_forward_max_abs_error": _max_abs(actual["out"].detach(), oracle_output.detach()),
        "official_attention_alpha_max_abs_error": _max_abs(actual["alpha"].detach(), oracle_alpha.detach()),
        "official_attention_input_gradient_max_abs_error": _max_abs(message.grad, oracle_message.grad),
        "official_attention_parameter_gradient_max_abs_error": _max_abs(contraction.weight.grad, oracle_weight.grad),
        "official_vec2heads_forward_max_abs_error": _max_abs(split_actual.detach(), split_oracle.detach()),
        "official_vec2heads_input_gradient_max_abs_error": _max_abs(roundtrip_source.grad, split_oracle_input.grad),
        "rotation_relative_error": rotation_error,
        "rotation_alpha_invariance_max_abs_error": rotation_alpha_error,
        "inversion_relative_error": inversion_error,
        "inversion_alpha_invariance_max_abs_error": inversion_alpha_error,
        "edge_permutation_output_max_abs_error": _max_abs(permuted["out"], actual["out"].detach()),
        "edge_permutation_alpha_max_abs_error": _max_abs(permuted["alpha"], actual["alpha"].detach().index_select(0, permutation)),
        "input_gradient_finite": bool(torch.isfinite(message.grad).all()),
        "parameter_gradient_finite": bool(torch.isfinite(contraction.weight.grad).all()),
    }
    numerical_keys = [key for key in metrics if key.endswith("error")]
    if max(float(metrics[key]) for key in numerical_keys) > 1.0e-8:
        raise RuntimeError("equivariant multihead evidence exceeded 1e-8: {}".format(metrics))
    if not metrics["input_gradient_finite"] or not metrics["parameter_gradient_finite"]:
        raise RuntimeError("equivariant multihead evidence produced non-finite gradients")

    negative = _negative_contracts(registry, message_type, alpha_type, segment_type)
    if not all(item.get("rejected") and item.get("expected_code_present") for item in negative.values()):
        raise RuntimeError("one or more equivariant multihead negative contracts were not enforced")

    runtime_kinds = dict(support.runtime_kinds)
    source_identity = {
        "repository_root": str(official_v1_root.resolve()),
        "source_file": str(official_source),
        "source_sha256": hashlib.sha256(official_text.encode("utf-8")).hexdigest(),
        "oracle_loading": "AST-extracted exact official Vec2AttnHeads and AttnHeads2Vec class definitions",
        "official_constructor_bypass_used_for_dsl_execution": False,
    }
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "seed": SEED,
        "pytest_summary": pytest_summary,
        "primitive_count": len(registry.names()),
        "lowering_rule_count": model.lowering_rule_manifest["rule_count"],
        "architecture_id": artifact.architecture_id,
        "official_v1_source_identity": source_identity,
        "model_identity": {
            "backend_semantics_version": model.backend_semantics_version,
            "python_class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
            "node_ops": [node.op for node in program.nodes],
            "trainable_parameter_shapes": {
                name: list(parameter.shape) for name, parameter in model.named_parameters()
            },
            "official_network_or_block_constructor_used": False,
        },
        "parameter_contracts": {
            node_id: [contract.to_dict() for contract in contracts]
            for node_id, contracts in artifact.inference.parameter_contracts.items()
            if contracts
        },
        "runtime_kinds": runtime_kinds,
        "numerical_and_gradient_evidence": metrics,
        "negative_contracts": negative,
        "claims": {
            "official_vec2attnheads_forward_and_input_gradient_matched": True,
            "official_attnheads2vec_order_matched": True,
            "alpha_value_same_irrep_range_partition_supported": True,
            "explicit_segment_index_softmax_supported": True,
            "head_packed_explicit_segment_sum_supported": True,
            "smooth_leaky_relu_and_alpha_dot_parameter_shape_supported": True,
            "minimal_v1_attention_alpha_value_core_reproduced": True,
            "official_v1_complete_graph_attention_reproduced": False,
            "official_v1_block_or_network_reproduced": False,
            "official_v2_v3_reproduced": False,
            "constructor_bypass_used": False,
        },
    }

    (output / "equivariant_multihead_program.json").write_text(dumps_program(program), encoding="utf-8")
    _write_json(output / "support_report.json", support.to_dict())
    _write_json(output / "lowering_manifest.json", model.lowering_rule_manifest)
    _write_json(output / "parameter_contracts.json", summary["parameter_contracts"])
    _write_json(output / "runtime_kinds.json", runtime_kinds)
    _write_json(output / "official_v1_source_identity.json", source_identity)
    _write_json(output / "numerical_and_gradient_evidence.json", metrics)
    _write_json(output / "negative_contracts.json", negative)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("export-dsl-equivariant-multihead-v2-evidence")
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
        "official_source_sha256": result["official_v1_source_identity"]["source_sha256"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
