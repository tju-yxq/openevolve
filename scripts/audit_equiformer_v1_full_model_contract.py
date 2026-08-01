#!/usr/bin/env python
"""Freeze a deterministic official Equiformer V1 whole-model oracle.

The audit imports the official source and executes its real modules.  Optional
PyG/OCP dependencies are replaced only at their public mathematical boundary:
``radius_graph`` uses a deterministic, stable, pure-PyTorch implementation and
``scatter`` uses the exact dim-0 sum/mean stub shared by the M5 oracle.  These
stubs never participate in DSL execution.

The resulting evidence is a construction/forward contract for M6.  It is not
itself an importer and must not be counted as primitive-by-primitive lowering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

try:
    from scripts.audit_equiformer_v1_graph_attention_contract import (
        _load_official_module,
        _module_tree,
    )
except ModuleNotFoundError:  # direct ``python scripts/<name>.py`` execution
    from audit_equiformer_v1_graph_attention_contract import (
        _load_official_module,
        _module_tree,
    )


EVIDENCE_VERSION = "equiformer-v1-full-model-contract-audit@1"
SEED = 20260731


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _rng_digest(torch) -> str:
    return _sha256_bytes(bytes(torch.random.get_rng_state().tolist()))


def _tensor_digest(value) -> Dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": _sha256_bytes(tensor.numpy().tobytes()),
    }


def _value_digest(value) -> Any:
    if hasattr(value, "detach"):
        return _tensor_digest(value)
    if isinstance(value, tuple):
        return [_value_digest(item) for item in value]
    if isinstance(value, list):
        return [_value_digest(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _value_digest(item) for key, item in value.items()}
    return {"type": type(value).__name__, "repr": repr(value)}


def _max_abs_error(left, right) -> float:
    if isinstance(left, (tuple, list)):
        if not isinstance(right, (tuple, list)) or len(left) != len(right):
            return float("inf")
        return max((_max_abs_error(a, b) for a, b in zip(left, right)), default=0.0)
    if left.shape != right.shape:
        return float("inf")
    if left.numel() == 0:
        return 0.0
    return float((left - right).abs().max().item())


def _stable_radius_graph(pos, radius: float, batch, max_num_neighbors: int, torch):
    """Return a stable target-major directed radius graph without self loops."""

    sources = []
    targets = []
    node_count = int(pos.shape[0])
    for target in range(node_count):
        candidates = []
        for source in range(node_count):
            if source == target or int(batch[source]) != int(batch[target]):
                continue
            distance = torch.linalg.vector_norm(pos[source] - pos[target])
            if bool(distance <= float(radius)):
                candidates.append(source)
        for source in candidates[: int(max_num_neighbors)]:
            sources.append(source)
            targets.append(target)
    device = pos.device
    return torch.tensor((sources, targets), dtype=torch.long, device=device)


def _constructor_config(o3) -> Dict[str, Any]:
    raw = {
        "irreps_in": "1x0e",
        "irreps_node_embedding": "4x0e+2x1e+1x2e",
        "num_layers": 2,
        "irreps_node_attr": "1x0e",
        "irreps_sh": "1x0e+1x1e+1x2e",
        "max_radius": 2.5,
        "number_of_basis": 6,
        "basis_type": "gaussian",
        "fc_neurons": [8],
        "irreps_feature": "5x0e",
        "irreps_head": "2x0e+1x1e+1x2e",
        "num_heads": 2,
        "irreps_pre_attn": None,
        "rescale_degree": False,
        "nonlinear_message": False,
        "irreps_mlp_mid": "6x0e+2x1e+1x2e",
        "norm_layer": "layer",
        "alpha_drop": 0.0,
        "proj_drop": 0.0,
        "out_drop": 0.0,
        "drop_path_rate": 0.0,
        "mean": None,
        "std": None,
        "scale": None,
        "atomref": None,
    }
    result = dict(raw)
    for key in (
        "irreps_in",
        "irreps_node_embedding",
        "irreps_node_attr",
        "irreps_sh",
        "irreps_feature",
        "irreps_head",
        "irreps_mlp_mid",
    ):
        result[key] = o3.Irreps(result[key])
    return result


def _json_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in config.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            result[key] = value
        elif type(value) in (list, tuple):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result


def _inputs(torch, *, requires_grad: bool) -> Dict[str, Any]:
    positions = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.90, 0.10, 0.20],
            [-0.40, 1.10, 0.30],
            [4.00, -0.20, 0.10],
            [4.80, 0.30, -0.40],
        ],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    return {
        "f_in": torch.zeros((5, 1), dtype=torch.float32),
        "pos": positions,
        "batch": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        "node_atom": torch.tensor([1, 6, 8, 7, 9], dtype=torch.long),
    }


def _explicit_forward(model, inputs, o3, torch) -> Tuple[Any, Dict[str, Any]]:
    trace = {}
    edge_index = _stable_radius_graph(
        inputs["pos"],
        model.max_radius,
        inputs["batch"],
        1000,
        torch,
    )
    edge_src, edge_dst = edge_index[0], edge_index[1]
    edge_vec = inputs["pos"].index_select(0, edge_src) - inputs["pos"].index_select(0, edge_dst)
    edge_sh = o3.spherical_harmonics(
        l=model.irreps_edge_attr,
        x=edge_vec,
        normalize=True,
        normalization="component",
    )
    atom_lookup = inputs["node_atom"].new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])
    remapped_atom = atom_lookup[inputs["node_atom"]]
    atom_embedding = model.atom_embed(remapped_atom)
    edge_length = edge_vec.norm(dim=1)
    edge_length_embedding = model.rbf(edge_length)
    edge_degree_embedding = model.edge_deg_embed(
        atom_embedding[0],
        edge_sh,
        edge_length_embedding,
        edge_src,
        edge_dst,
        inputs["batch"],
    )
    node_features = atom_embedding[0] + edge_degree_embedding
    node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

    trace["edge_index"] = edge_index
    trace["edge_vector"] = edge_vec
    trace["edge_spherical_harmonics"] = edge_sh
    trace["remapped_atom"] = remapped_atom
    trace["atom_embed"] = atom_embedding
    trace["edge_length"] = edge_length
    trace["rbf"] = edge_length_embedding
    trace["edge_degree_embedding"] = edge_degree_embedding
    trace["node_features_initial"] = node_features
    trace["node_attr"] = node_attr

    for index, block in enumerate(model.blocks):
        node_features = block(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_scalars=edge_length_embedding,
            batch=inputs["batch"],
        )
        trace["block_{}".format(index)] = node_features

    node_features = model.norm(node_features, batch=inputs["batch"])
    trace["norm"] = node_features
    if model.out_dropout is not None:
        node_features = model.out_dropout(node_features)
        trace["out_dropout"] = node_features
    outputs = model.head(node_features)
    trace["head"] = outputs
    outputs = model.scale_scatter(outputs, inputs["batch"], dim=0)
    trace["scale_scatter"] = outputs
    if model.scale is not None:
        outputs = model.scale * outputs
        trace["task_scale"] = outputs
    return outputs, trace


def _official_hook_trace(model, forward_call) -> Tuple[Any, Dict[str, Any]]:
    trace = {}
    handles = []

    def register(name, module):
        def hook(_module, _inputs, output):
            trace[name] = output

        handles.append(module.register_forward_hook(hook))

    register("atom_embed", model.atom_embed)
    register("rbf", model.rbf)
    register("edge_degree_embedding", model.edge_deg_embed)
    for index, block in enumerate(model.blocks):
        register("block_{}".format(index), block)
    register("norm", model.norm)
    register("head", model.head)
    register("scale_scatter", model.scale_scatter)
    try:
        output = forward_call()
    finally:
        for handle in handles:
            handle.remove()
    return output, trace


def _gradient_snapshot(model, output, position, torch) -> Tuple[Any, Dict[str, Any]]:
    named_parameters = tuple(model.named_parameters())
    targets = (position,) + tuple(parameter for _, parameter in named_parameters)
    gradients = torch.autograd.grad(
        output.square().sum() + output.sum(),
        targets,
        allow_unused=True,
    )
    position_gradient = gradients[0]
    parameter_gradients = {
        name: gradient
        for (name, _parameter), gradient in zip(named_parameters, gradients[1:])
    }
    return position_gradient, parameter_gradients


def _gradient_errors(left, right) -> Dict[str, Any]:
    names = sorted(set(left) | set(right))
    errors = {}
    for name in names:
        left_value = left.get(name)
        right_value = right.get(name)
        if left_value is None or right_value is None:
            errors[name] = {
                "max_abs_error": None,
                "official_is_none": left_value is None,
                "replay_is_none": right_value is None,
            }
        else:
            errors[name] = {
                "max_abs_error": _max_abs_error(left_value, right_value),
                "shape": list(left_value.shape),
            }
    return errors


def _source_identity(root: Path) -> Dict[str, Any]:
    relative_paths = (
        "nets/graph_attention_transformer.py",
        "nets/gaussian_rbf.py",
        "nets/radial_func.py",
        "nets/tensor_product_rescale.py",
        "nets/layer_norm.py",
        "nets/fast_activation.py",
        "nets/drop.py",
    )
    files = {}
    for relative in relative_paths:
        path = root / relative
        files[relative] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
    return {"root": str(root.resolve()), "files": files}


def build_audit(output: Path, official_root: Path) -> Dict[str, Any]:
    import torch

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    official_root = official_root.resolve()
    torch.manual_seed(SEED)
    rng_before = _rng_digest(torch)
    module, source_path = _load_official_module(official_root, torch)
    module.radius_graph = lambda pos, r, batch, max_num_neighbors=1000, **_kwargs: _stable_radius_graph(
        pos,
        r,
        batch,
        max_num_neighbors,
        torch,
    )
    config = _constructor_config(o3)
    # The official NodeEmbeddingNetwork calls one_hot(...).float() explicitly,
    # so the faithful whole-model contract is float32 even though isolated
    # Blocks can be audited in float64.
    model = module.GraphAttentionTransformer(**config).eval()
    rng_after = _rng_digest(torch)

    hook_inputs = _inputs(torch, requires_grad=False)
    official_output, official_trace = _official_hook_trace(
        model,
        lambda: model(**hook_inputs),
    )
    replay_output, replay_trace = _explicit_forward(model, hook_inputs, o3, torch)

    common_trace_names = sorted(set(official_trace) & set(replay_trace))
    intermediate_errors = {
        name: _max_abs_error(official_trace[name], replay_trace[name])
        for name in common_trace_names
    }

    official_grad_inputs = _inputs(torch, requires_grad=True)
    official_grad_output = model(**official_grad_inputs)
    official_position_gradient, official_parameter_gradients = _gradient_snapshot(
        model,
        official_grad_output,
        official_grad_inputs["pos"],
        torch,
    )

    replay_grad_inputs = _inputs(torch, requires_grad=True)
    replay_grad_output, _ = _explicit_forward(model, replay_grad_inputs, o3, torch)
    replay_position_gradient, replay_parameter_gradients = _gradient_snapshot(
        model,
        replay_grad_output,
        replay_grad_inputs["pos"],
        torch,
    )

    parameter_gradients = _gradient_errors(
        official_parameter_gradients,
        replay_parameter_gradients,
    )
    finite_parameters = {
        name: bool(torch.isfinite(parameter).all())
        for name, parameter in model.named_parameters()
    }
    parameter_shapes = {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters()
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    max_parameter_gradient_error = max(
        (
            item["max_abs_error"]
            for item in parameter_gradients.values()
            if item["max_abs_error"] is not None
        ),
        default=0.0,
    )
    missing_parameter_gradients = sorted(
        name
        for name, item in parameter_gradients.items()
        if item.get("official_is_none", False) != item.get("replay_is_none", False)
    )

    edge_index = replay_trace["edge_index"]
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "status": "pass",
        "seed": SEED,
        "official_source": str(source_path.resolve()),
        "constructor_config": _json_config(config),
        "module_count": len(tuple(model.named_modules())),
        "parameter_tensor_count": len(parameter_shapes),
        "parameter_count": parameter_count,
        "edge_count": int(edge_index.shape[1]),
        "graph_count": 2,
        "forward_max_abs_error": _max_abs_error(official_output, replay_output),
        "position_gradient_max_abs_error": _max_abs_error(
            official_position_gradient,
            replay_position_gradient,
        ),
        "parameter_gradient_max_abs_error": max_parameter_gradient_error,
        "missing_parameter_gradient_mismatches": missing_parameter_gradients,
        "intermediate_max_abs_error": max(intermediate_errors.values(), default=0.0),
        "construction_rng_before": rng_before,
        "construction_rng_after": rng_after,
        "radius_graph_contract": {
            "implementation": "stable_target_major_pure_torch",
            "directed": True,
            "self_loops": False,
            "cutoff_comparison": "distance <= radius",
            "max_num_neighbors": 1000,
            "official_dependency_boundary_stub_only": True,
            "dsl_execution_uses_stub": False,
        },
        "official_constructor_used_for_dsl_execution": False,
        "counts_as_full_model_lowering": False,
    }
    if (
        summary["forward_max_abs_error"] > 1.0e-6
        or summary["position_gradient_max_abs_error"] > 1.0e-5
        or summary["parameter_gradient_max_abs_error"] > 1.0e-5
        or summary["intermediate_max_abs_error"] > 1.0e-6
        or missing_parameter_gradients
        or not all(finite_parameters.values())
    ):
        summary["status"] = "fail"

    contract = {
        "summary": summary,
        "source_identity": _source_identity(official_root),
        "module_tree": _module_tree(model),
        "parameter_shapes": parameter_shapes,
        "finite_parameters": finite_parameters,
        "official_trace": {name: _value_digest(value) for name, value in official_trace.items()},
        "replay_trace": {name: _value_digest(value) for name, value in replay_trace.items()},
        "intermediate_errors": intermediate_errors,
        "official_output": _tensor_digest(official_output),
        "replay_output": _tensor_digest(replay_output),
        "official_position_gradient": _tensor_digest(official_position_gradient),
        "replay_position_gradient": _tensor_digest(replay_position_gradient),
        "parameter_gradient_errors": parameter_gradients,
        "construction_order": [name or "<root>" for name, _module in model.named_modules()],
        "forward_dataflow": [
            "radius_graph",
            "relative_displacement(source-target)",
            "spherical_harmonics",
            "atomic_number_remap",
            "one_hot_node_embedding",
            "edge_distance",
            "gaussian_radial_basis",
            "edge_degree_embedding",
            "atom_plus_edge_degree",
            "transblock_0",
            "transblock_1_to_feature_irreps",
            "final_irrep_layer_norm",
            "two_layer_scalar_head",
            "scaled_graph_scatter_sum",
        ],
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "official_full_model_contract.json", contract)
    if summary["status"] != "pass":
        raise RuntimeError("official full-model explicit replay did not match")
    return contract


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence = build_audit(args.output, args.official_root)
    print(json.dumps(evidence["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
