#!/usr/bin/env python
"""Export official-oracle evidence for the expanded Equiformer V1 RadialProfile."""

from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import json
import math
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
    EquivariantTensorType,
    FeatureRole,
    GroupSpec,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    core_registry,
    reference_motif_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.canonicalize import canonicalize
from equivariant_nas.dsl.serialization import dumps_program


EVIDENCE_VERSION = "evoequilang-v1-radial-profile-evidence@1"
SEED = 20260731


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _max_abs(left, right):
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def _types():
    group = GroupSpec.o3()
    radial = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("6x0e", group.family),
        axes=("radial_channel",),
        axis_specs=(AxisSpec("radial_channel", 6, FeatureRole.CHANNEL, "independent", 0),),
        dtype="float64",
        measure="dimensionless",
        feature_role=FeatureRole.CHANNEL,
    )
    weight = replace(
        radial,
        irreps=Irreps.parse("30x0e", group.family),
        axes=("tp_path",),
        axis_specs=(AxisSpec("tp_path", 30, FeatureRole.TP_PATH, "independent", 0),),
        feature_role=FeatureRole.RADIAL_WEIGHT,
    )
    left = EquivariantTensorType(
        group, Carrier.EDGE, Irreps.parse("4x0e+2x1e+1x2e", group.family), dtype="float64"
    )
    right = EquivariantTensorType(
        group, Carrier.EDGE, Irreps.parse("1x0e+1x1e+1x2e", group.family), dtype="float64"
    )
    output = replace(left, irreps=Irreps.parse("7x0e+12x1e+11x2e", group.family))
    return radial, weight, left, right, output


def _tp_attrs():
    blocks = (
        (4, "0e"), (2, "0e"), (1, "0e"),
        (4, "1e"), (2, "1e"), (2, "1e"), (2, "1e"), (1, "1e"), (1, "1e"),
        (4, "2e"), (2, "2e"), (2, "2e"), (1, "2e"), (1, "2e"), (1, "2e"),
    )
    indices = (
        (0, 0, 0), (0, 1, 3), (0, 2, 9),
        (1, 0, 4), (1, 1, 1), (1, 1, 5), (1, 1, 10), (1, 2, 6), (1, 2, 11),
        (2, 0, 12), (2, 1, 7), (2, 1, 13), (2, 2, 2), (2, 2, 8), (2, 2, 14),
    )
    return {
        "path_blocks": [{"multiplicity": multiplicity, "irrep": irrep} for multiplicity, irrep in blocks],
        "instructions": [
            {"left": left, "right": right, "out": output, "mode": "uvu", "has_weight": True, "path_weight": 1.0}
            for left, right, output in indices
        ],
    }


def _program(scales):
    radial, _weight, left, right, output = _types()
    return ArchitectureProgram(
        "2.7.0",
        "official-v1-radial-profile-depthwise-chain",
        (InputPort("radial", radial), InputPort("left", left), InputPort("right", right)),
        (
            Node(
                "profile",
                "motif.v1_radial_profile@1",
                {"x": ("input:radial",)},
                {
                    "axis": "radial_channel",
                    "out_axis": "tp_path",
                    "hidden_features": 8,
                    "out_features": 30,
                    "output_scales": list(scales),
                },
            ),
            Node(
                "tp",
                "core.tensor_product@3",
                {"left": ("input:left",), "right": ("input:right",), "weight": ("profile",)},
                _tp_attrs(),
            ),
        ),
        (OutputPort("out", "tp", output),),
        program_id="official-v1-radial-profile-depthwise-chain@1",
    )


def _parameter_mapping(prefix="node_modules.profile__"):
    return {
        prefix + "linear_in.linear.weight": "net.0.weight",
        prefix + "linear_in.linear.bias": "net.0.bias",
        prefix + "norm.layer_norm.weight": "net.1.weight",
        prefix + "norm.layer_norm.bias": "net.1.bias",
        prefix + "linear_out.linear.weight": "net.3.weight",
        prefix + "offset.offset": "offset",
    }


def _extract_radial_profile(source_path: Path, torch):
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RadialProfile"]
    if len(selected) != 1:
        raise RuntimeError("official radial_func.py no longer contains exactly one RadialProfile")
    namespace = {"torch": torch, "nn": torch.nn, "init": torch.nn.init, "math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["RadialProfile"], source


def _slice_scales(depthwise, output_size, torch):
    scales = torch.ones(output_size, dtype=torch.float64)
    for output_slice, scale in depthwise.slices_sqrt_k.values():
        scales[output_slice] *= float(scale)
    return scales


def build_evidence(output: Path, *, official_v1_root: Path, pytest_summary: str = "") -> Dict[str, Any]:
    import torch
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    official_module, graph_source = _load_official_module(official_v1_root.resolve(), torch)
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
    torch.manual_seed(SEED)
    official_graph = official_module.GraphAttention(**config).double().eval()
    official_profile = official_graph.sep.dtp_rad
    scales_tensor = _slice_scales(official_graph.sep.dtp, int(official_graph.sep.dtp.tp.weight_numel), torch)
    scales = [float(value) for value in scales_tensor.tolist()]

    registry = core_registry()
    motifs = reference_motif_registry()
    program = _program(scales)
    artifact = Compiler(registry, motifs).analyze(program)
    backend = E3NNGraphBackend(registry)
    support = backend.support_report(artifact.expanded_program)
    if not support.supported:
        raise RuntimeError("expanded radial profile support report failed: {}".format(support.to_dict()))
    dsl_model = backend.build(artifact.expanded_program, artifact.inference).double().eval()
    mapping = _parameter_mapping()
    official_parameters = dict(official_profile.named_parameters())
    dsl_parameters = dict(dsl_model.named_parameters())
    for dsl_name, official_name in mapping.items():
        dsl_parameters[dsl_name].data.copy_(official_parameters[official_name].data)

    edge_count = 8
    radial = torch.randn(edge_count, 6, dtype=torch.float64, requires_grad=True)
    left = torch.randn(edge_count, 15, dtype=torch.float64, requires_grad=True)
    right = torch.randn(edge_count, 9, dtype=torch.float64, requires_grad=True)
    official_radial = radial.detach().clone().requires_grad_(True)
    official_left = left.detach().clone().requires_grad_(True)
    official_right = right.detach().clone().requires_grad_(True)
    dsl_output = dsl_model({"radial": radial, "left": left, "right": right}, {})["out"]
    official_output = official_graph.sep.dtp(
        official_left,
        official_right,
        official_profile(official_radial),
    )
    probe = torch.randn_like(dsl_output)
    dsl_loss = (dsl_output * probe).sum()
    official_loss = (official_output * probe).sum()
    dsl_loss.backward()
    official_loss.backward()

    source_path = official_v1_root.resolve() / "nets" / "radial_func.py"
    RadialProfile, radial_source = _extract_radial_profile(source_path, torch)
    torch.manual_seed(SEED)
    standalone = RadialProfile([6, 8, 30]).double()
    with torch.no_grad():
        standalone.net[-1].weight.mul_(scales_tensor.reshape(-1, 1))
        standalone.offset.mul_(scales_tensor)
    torch.manual_seed(SEED)
    init_model = backend.build(artifact.expanded_program, artifact.inference).double()
    init_parameters = dict(init_model.named_parameters())
    standalone_parameters = dict(standalone.named_parameters())
    initialization_errors = {
        dsl_name: _max_abs(init_parameters[dsl_name], standalone_parameters[official_name])
        for dsl_name, official_name in mapping.items()
    }

    parameter_gradient_errors = {
        dsl_name: _max_abs(dsl_parameters[dsl_name].grad, official_parameters[official_name].grad)
        for dsl_name, official_name in mapping.items()
    }
    metrics = {
        "radial_profile_depthwise_chain_forward_max_abs_error": _max_abs(dsl_output.detach(), official_output.detach()),
        "radial_input_gradient_max_abs_error": _max_abs(radial.grad, official_radial.grad),
        "left_input_gradient_max_abs_error": _max_abs(left.grad, official_left.grad),
        "right_input_gradient_max_abs_error": _max_abs(right.grad, official_right.grad),
        "max_parameter_gradient_error": max(parameter_gradient_errors.values()),
        "max_standalone_initialization_error": max(initialization_errors.values()),
        "all_dsl_gradients_finite": all(
            value.grad is not None and bool(torch.isfinite(value.grad).all())
            for value in (radial, left, right)
        ) and all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in dsl_parameters.values()
        ),
    }
    if max(float(value) for key, value in metrics.items() if key.endswith("error")) > 1.0e-10:
        raise RuntimeError("V1 radial profile evidence exceeded 1e-10: {}".format(metrics))
    if not metrics["all_dsl_gradients_finite"]:
        raise RuntimeError("V1 radial profile produced non-finite gradients")

    parameter_contracts = {
        node_id: [contract.to_dict() for contract in contracts]
        for node_id, contracts in artifact.inference.parameter_contracts.items()
        if contracts
    }
    source_identity = {
        "repository_root": str(official_v1_root.resolve()),
        "graph_attention_source": str(graph_source.resolve()),
        "graph_attention_sha256": hashlib.sha256(graph_source.read_bytes()).hexdigest(),
        "radial_profile_source": str(source_path.resolve()),
        "radial_profile_sha256": hashlib.sha256(radial_source.encode("utf-8")).hexdigest(),
        "oracle_objects": ["GraphAttention.sep.dtp_rad", "AST-extracted RadialProfile"],
        "official_constructor_role": "oracle construction only; no official operator executes inside the DSL model",
    }
    model_identity = {
        "backend_semantics_version": dsl_model.backend_semantics_version,
        "expanded_node_ops": [node.op for node in artifact.expanded_program.nodes],
        "trainable_parameter_shapes": {
            name: list(parameter.shape) for name, parameter in dsl_parameters.items()
        },
        "official_parameter_mapping": mapping,
        "radial_output_size": 30,
        "slice_sqrt_k_output_scales": scales,
        "downstream_tp_op": "core.tensor_product@3",
        "official_network_block_or_operator_used_for_dsl_execution": False,
    }
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "seed": SEED,
        "pytest_summary": pytest_summary,
        "primitive_count": len(registry.names()),
        "motif_count": len(motifs.names()),
        "lowering_rule_count": dsl_model.lowering_rule_manifest["rule_count"],
        "architecture_id": artifact.architecture_id,
        "official_v1_source_identity": source_identity,
        "model_identity": model_identity,
        "parameter_contracts": parameter_contracts,
        "initialization_errors": initialization_errors,
        "parameter_gradient_errors": parameter_gradient_errors,
        "numerical_and_gradient_evidence": metrics,
        "claims": {
            "official_radial_profile_expanded_to_five_core_nodes": True,
            "official_six_parameter_tensors_mapped": True,
            "standalone_parameter_initialization_reproduced": True,
            "slice_sqrt_k_applied_only_to_final_linear_and_offset_initialization": True,
            "radial_profile_plus_depthwise_tp_forward_and_gradients_reproduced": True,
            "official_constructor_bypass_used": False,
            "official_v1_complete_graph_attention_reproduced": False,
            "official_v1_complete_block_reproduced": False,
        },
    }

    (output / "canonical_radial_profile_tp_program.json").write_text(
        dumps_program(canonicalize(artifact.expanded_program, registry)), encoding="utf-8"
    )
    _write_json(output / "support_report.json", support.to_dict())
    _write_json(output / "lowering_manifest.json", dsl_model.lowering_rule_manifest)
    _write_json(output / "parameter_contracts.json", parameter_contracts)
    _write_json(output / "model_identity.json", model_identity)
    _write_json(output / "official_v1_source_identity.json", source_identity)
    _write_json(output / "initialization_errors.json", initialization_errors)
    _write_json(output / "parameter_gradient_errors.json", parameter_gradient_errors)
    _write_json(output / "numerical_and_gradient_evidence.json", metrics)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("export-dsl-v1-radial-profile-evidence")
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
        "forward_error": result["numerical_and_gradient_evidence"]["radial_profile_depthwise_chain_forward_max_abs_error"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
