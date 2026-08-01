#!/usr/bin/env python
"""Audit the stochastic Equiformer V1 TransBlock against primitive lowering.

The official implementation is an oracle only.  The executed DSL model is
compiled from typed core primitives and never calls an official block,
attention, feed-forward, dropout, or stochastic-depth constructor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping


SEED = 20260731
EVIDENCE_VERSION = "equiformer-v1-transblock-stochastic-audit@1"
HIDDEN_IRREPS = "4x0e+2x1e+1x2e"
EDGE_IRREPS = "1x0e+1x1e+1x2e"
ALPHA_DROP = 0.2
PROJ_DROP = 0.15
DROP_PATH = 0.25

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _max_abs(left, right) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _relative_error(actual, expected) -> float:
    numerator = (actual.detach() - expected.detach()).norm()
    denominator = expected.detach().norm().clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def _official_config(
    o3,
    *,
    alpha_drop: float = ALPHA_DROP,
    proj_drop: float = PROJ_DROP,
    drop_path: float = DROP_PATH,
) -> Dict[str, Any]:
    return {
        "irreps_node_input": o3.Irreps(HIDDEN_IRREPS),
        "irreps_node_attr": o3.Irreps("1x0e"),
        "irreps_edge_attr": o3.Irreps(EDGE_IRREPS),
        "irreps_node_output": o3.Irreps(HIDDEN_IRREPS),
        "fc_neurons": [6, 8],
        "irreps_head": o3.Irreps("2x0e+1x1e+1x2e"),
        "num_heads": 2,
        "irreps_pre_attn": None,
        "rescale_degree": False,
        "nonlinear_message": False,
        "alpha_drop": float(alpha_drop),
        "proj_drop": float(proj_drop),
        "drop_path_rate": float(drop_path),
        "irreps_mlp_mid": o3.Irreps(HIDDEN_IRREPS),
        "norm_layer": "layer",
    }


def _serializable_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: str(value) if key.startswith("irreps_") and value is not None else value
        for key, value in config.items()
    }


def _radial_output_scales(official, torch):
    scales = torch.ones(30, dtype=torch.float64)
    for output_slice, scale in official.ga.sep.dtp.slices_sqrt_k.values():
        scales[output_slice] *= float(scale)
    return scales.tolist()


def _dsl_inputs(
    node_input,
    node_attr,
    edge_src,
    edge_dst,
    edge_attr,
    edge_scalars,
    batch,
    *,
    graph_count: int = 2,
):
    return {
        "node_input": node_input,
        "node_attr": node_attr,
        "edge_attr": edge_attr,
        "edge_scalars": edge_scalars,
        "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
        "target_index": {"indices": edge_dst, "target_size": edge_dst.numel()},
        "segment_index": {"indices": edge_dst, "target_size": node_input.shape[0]},
        "batch_index": {"indices": batch, "target_size": int(graph_count)},
    }


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
    attention_drop_path = official.drop_path(attention, batch)
    attention_residual = node_input + attention_drop_path
    norm_2 = official.norm_2(attention_residual, batch=batch)
    ffn_pre_dropout = official.ffn.fctp_2(
        official.ffn.fctp_1(norm_2, node_attr),
        node_attr,
    )
    ffn = official.ffn.proj_drop(ffn_pre_dropout)
    ffn_drop_path = official.drop_path(ffn, batch)
    return {
        "norm_1": norm_1,
        "attention": attention,
        "attention_drop_path": attention_drop_path,
        "attention_residual": attention_residual,
        "norm_2": norm_2,
        "ffn_pre_dropout": ffn_pre_dropout,
        "ffn": ffn,
        "ffn_drop_path": ffn_drop_path,
        "out": attention_residual + ffn_drop_path,
    }


def build_audit(
    output: Path,
    official_root: Path,
    *,
    graph_count: int = 2,
    alpha_drop: float = ALPHA_DROP,
    proj_drop: float = PROJ_DROP,
    drop_path: float = DROP_PATH,
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
        EQUIFORMER_V1_TRANSBLOCK_PARAMETER_MAPPING,
        equiformer_v1_transblock_program,
    )

    official_module, source_path = _load_official_module(official_root, torch)
    graph_count = int(graph_count)
    if graph_count <= 0 or graph_count > 5:
        raise ValueError("graph_count must satisfy 1 <= graph_count <= 5 for the frozen five-node audit")
    for name, value in {
        "alpha_drop": alpha_drop,
        "proj_drop": proj_drop,
        "drop_path": drop_path,
    }.items():
        if not 0.0 < float(value) < 1.0:
            raise ValueError("{} must satisfy 0 < p < 1 for the stochastic audit".format(name))
    config = _official_config(
        o3,
        alpha_drop=alpha_drop,
        proj_drop=proj_drop,
        drop_path=drop_path,
    )
    torch.manual_seed(SEED)
    official = official_module.TransBlock(**config).double().train()
    program = equiformer_v1_transblock_program(
        _radial_output_scales(official, torch),
        graph_count=graph_count,
        alpha_drop=alpha_drop,
        proj_drop=proj_drop,
        drop_path=drop_path,
    )
    registry = core_registry()
    artifact = Compiler(registry, reference_motif_registry()).analyze(program)
    backend = E3NNGraphBackend(registry)
    torch.manual_seed(SEED)
    model = backend.build(artifact.expanded_program, artifact.inference).double().train()

    mapping = dict(EQUIFORMER_V1_TRANSBLOCK_PARAMETER_MAPPING)
    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(model.named_parameters())
    initialization_errors = {
        dsl_name: _max_abs(dsl_parameters[dsl_name], official_parameters[official_name])
        for dsl_name, official_name in mapping.items()
    }

    node_count = 5
    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    edge_dst = torch.tensor([1, 1, 3, 3, 0, 4, 4, 2], dtype=torch.long)
    batch = torch.arange(node_count, dtype=torch.long) % graph_count
    torch.manual_seed(SEED + 1)
    node_input = torch.randn(node_count, 15, dtype=torch.float64, requires_grad=True)
    node_attr = torch.randn(node_count, 1, dtype=torch.float64, requires_grad=True)
    edge_attr = torch.randn(edge_src.numel(), 9, dtype=torch.float64, requires_grad=True)
    edge_scalars = torch.randn(edge_src.numel(), 6, dtype=torch.float64, requires_grad=True)
    official_node = node_input.detach().clone().requires_grad_(True)
    official_node_attr = node_attr.detach().clone().requires_grad_(True)
    official_edge = edge_attr.detach().clone().requires_grad_(True)
    official_radial = edge_scalars.detach().clone().requires_grad_(True)

    torch.manual_seed(SEED + 2)
    actual = model(
        _dsl_inputs(
            node_input,
            node_attr,
            edge_src,
            edge_dst,
            edge_attr,
            edge_scalars,
            batch,
            graph_count=graph_count,
        ),
        {},
    )
    actual_rng = torch.get_rng_state()
    torch.manual_seed(SEED + 2)
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
    expected_rng = torch.get_rng_state()
    forward_errors = {name: _max_abs(actual[name], expected[name]) for name in expected}
    stochastic_outputs = ("attention", "attention_drop_path", "ffn", "ffn_drop_path")
    mask_evidence = {
        name: {
            "zero_mask_equal": bool(torch.equal(actual[name] == 0, expected[name] == 0)),
            "dsl_zero_count": int((actual[name] == 0).sum().item()),
            "official_zero_count": int((expected[name] == 0).sum().item()),
        }
        for name in stochastic_outputs
    }
    rng_evidence = {
        "state_equal_after_forward": bool(torch.equal(actual_rng, expected_rng)),
        "dsl_state_sha256": _sha256_bytes(actual_rng.cpu().numpy().tobytes()),
        "official_state_sha256": _sha256_bytes(expected_rng.cpu().numpy().tobytes()),
    }

    torch.manual_seed(SEED + 3)
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
        rotation = o3.rand_matrix(dtype=torch.float64)
        node_action = o3.Irreps(HIDDEN_IRREPS).D_from_matrix(rotation)
        edge_action = o3.Irreps(EDGE_IRREPS).D_from_matrix(rotation)
        torch.manual_seed(SEED + 4)
        reference = model(
            _dsl_inputs(
                node_input.detach(), node_attr.detach(), edge_src, edge_dst,
                edge_attr.detach(), edge_scalars.detach(), batch,
                graph_count=graph_count,
            ),
            {},
        )["out"]
        torch.manual_seed(SEED + 4)
        transformed = model(
            _dsl_inputs(
                node_input.detach() @ node_action.transpose(0, 1),
                node_attr.detach(),
                edge_src,
                edge_dst,
                edge_attr.detach() @ edge_action.transpose(0, 1),
                edge_scalars.detach(),
                batch,
                graph_count=graph_count,
            ),
            {},
        )["out"]
        so3_relative_error = _relative_error(
            transformed,
            reference @ node_action.transpose(0, 1),
        )

        official.eval()
        model.eval()
        actual_eval = model(
            _dsl_inputs(
                node_input.detach(), node_attr.detach(), edge_src, edge_dst,
                edge_attr.detach(), edge_scalars.detach(), batch,
                graph_count=graph_count,
            ),
            {},
        )["out"]
        expected_eval = official(
            node_input.detach(),
            node_attr.detach(),
            edge_src,
            edge_dst,
            edge_attr.detach(),
            edge_scalars.detach(),
            batch,
        )
        eval_error = _max_abs(actual_eval, expected_eval)

    expanded_ops = [node.op for node in artifact.expanded_program.nodes]
    stochastic_op_counts = {
        name: expanded_ops.count(name)
        for name in (
            "core.scalar_dropout@1",
            "core.equivariant_dropout@1",
            "core.graph_stochastic_depth@1",
        )
    }
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
        "source_node_count": len(program.nodes),
        "expanded_node_count": len(artifact.expanded_program.nodes),
        "expanded_node_ids": [node.id for node in artifact.expanded_program.nodes],
        "expanded_node_ops": expanded_ops,
        "stochastic_op_counts": stochastic_op_counts,
        "trainable_parameter_tensor_count": len(dsl_parameters),
        "trainable_parameter_count": sum(parameter.numel() for parameter in dsl_parameters.values()),
        "official_parameter_mapping": mapping,
        "official_network_block_or_operator_used_for_dsl_execution": False,
    }
    numerical_summary = {
        "max_initialization_error": max(initialization_errors.values()),
        "max_forward_error": max(forward_errors.values()),
        "max_input_gradient_error": max(input_gradient_errors.values()),
        "max_parameter_gradient_error": max(parameter_gradient_errors.values()),
        "so3_relative_error": so3_relative_error,
        "eval_dsl_vs_official_max_abs_error": eval_error,
    }
    math_contract = {
        "alpha_dropout": "independent inverted-dropout mask per invariant attention coefficient",
        "projection_dropout": "one inverted-dropout mask per carrier item and irrep instance; all m coordinates of that irrep instance share the mask",
        "graph_stochastic_depth": "one inverted-drop-path mask per graph shared by every node and feature coordinate in the complete residual branch",
        "rng_order": "alpha dropout, attention projection dropout, attention graph drop-path, FFN projection dropout, FFN graph drop-path",
        "runtime_layout": "headwise alpha contraction uses the official einsum-compatible strided layout because PyTorch dropout mask placement depends on tensor storage layout",
    }
    source_identity = {
        "repository_root": str(official_root.resolve()),
        "source_file": str(source_path.resolve()),
        "source_sha256": _sha256_text(source_path.read_text(encoding="utf-8")),
        "official_class": "{}.{}".format(type(official).__module__, type(official).__qualname__),
        "oracle_only": True,
    }
    claims = {
        "stochastic_v1_transblock_expressed_by_typed_dsl": True,
        "generic_e3nn_graph_lowering_executed": True,
        "official_constructor_used_for_dsl_execution": False,
        "all_parameter_tensors_seed_identical": max(initialization_errors.values()) == 0.0,
        "rng_consumption_identical": rng_evidence["state_equal_after_forward"],
        "all_exposed_random_zero_masks_identical": all(
            item["zero_mask_equal"] for item in mask_evidence.values()
        ),
        "forward_within_float64_tolerance": max(forward_errors.values()) < 1.0e-12,
        "input_gradients_within_float64_tolerance": max(input_gradient_errors.values()) < 1.0e-11,
        "parameter_gradients_within_float64_tolerance": max(parameter_gradient_errors.values()) < 1.0e-11,
        "so3_equivariant_with_fixed_random_masks": so3_relative_error < 1.0e-7,
        "eval_mode_matches_official": eval_error < 1.0e-12,
    }
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "seed": SEED,
        "graph_count": graph_count,
        "oracle_config": _serializable_config(config),
        "model_identity": model_identity,
        "numerical_summary": numerical_summary,
        "rng_evidence": rng_evidence,
        "mask_evidence": mask_evidence,
        "claims": claims,
    }

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "canonical_transblock_program.json", program.to_dict())
    _write_json(output / "expanded_transblock_program.json", artifact.expanded_program.to_dict())
    _write_json(output / "lowering_manifest.json", backend.lowering_manifest())
    _write_json(output / "support_report.json", backend.support_report(artifact.expanded_program).to_dict())
    _write_json(output / "parameter_contracts.json", parameter_contracts)
    _write_json(output / "model_identity.json", model_identity)
    _write_json(output / "math_contract.json", math_contract)
    _write_json(output / "official_v1_source_identity.json", source_identity)
    _write_json(output / "initialization_errors.json", initialization_errors)
    _write_json(output / "intermediate_forward_errors.json", forward_errors)
    _write_json(output / "input_gradient_errors.json", input_gradient_errors)
    _write_json(output / "parameter_gradient_errors.json", parameter_gradient_errors)
    _write_json(output / "rng_evidence.json", rng_evidence)
    _write_json(output / "mask_evidence.json", mask_evidence)
    _write_json(output / "numerical_summary.json", numerical_summary)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("audit-equiformer-v1-transblock-stochastic")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--graph-count", type=int, default=2)
    parser.add_argument("--alpha-drop", type=float, default=ALPHA_DROP)
    parser.add_argument("--proj-drop", type=float, default=PROJ_DROP)
    parser.add_argument("--drop-path", type=float, default=DROP_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_audit(
        Path(args.output),
        Path(args.official_root),
        graph_count=int(args.graph_count),
        alpha_drop=float(args.alpha_drop),
        proj_drop=float(args.proj_drop),
        drop_path=float(args.drop_path),
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "architecture_id": result["model_identity"]["architecture_id"],
                **result["numerical_summary"],
                "rng_consumption_identical": result["claims"]["rng_consumption_identical"],
                "random_zero_masks_identical": result["claims"]["all_exposed_random_zero_masks_identical"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
