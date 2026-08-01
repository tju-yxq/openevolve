#!/usr/bin/env python
"""Freeze an official Equiformer V1 GraphAttention oracle contract.

The official repository imports optional PyG/OCP packages at module import
time.  This audit installs process-local mathematical stubs only for the
functions exercised by GraphAttention, then loads the official source file
without importing the broad ``nets.__init__`` package.  No stub participates
in DSL execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any, Dict, Mapping


EVIDENCE_VERSION = "equiformer-v1-graph-attention-contract-audit@1"
SEED = 20260731


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _segment_reduce(values, index, dim_size, torch, *, reduce="sum"):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    output = values.new_zeros((int(dim_size),) + values.shape[1:])
    output.index_add_(0, index, values)
    if reduce == "mean":
        counts = values.new_zeros(int(dim_size))
        counts.index_add_(0, index, values.new_ones(index.shape[0]))
        output = output / counts.clamp_min(1.0).reshape((int(dim_size),) + (1,) * (values.dim() - 1))
    return output


def _segment_softmax(values, index, torch):
    if values.shape[0] == 0:
        return values
    output = torch.empty_like(values)
    for target in torch.unique(index):
        mask = index == target
        output[mask] = torch.softmax(values[mask], dim=0)
    return output


def _install_optional_dependency_stubs(torch) -> Dict[str, Any]:
    """Install minimal process-local modules required to import the oracle."""

    installed = {}

    torch_cluster = types.ModuleType("torch_cluster")

    def radius_graph(*_args, **_kwargs):
        raise RuntimeError("radius_graph is outside the GraphAttention oracle audit")

    torch_cluster.radius_graph = radius_graph
    sys.modules["torch_cluster"] = torch_cluster
    installed["torch_cluster.radius_graph"] = "disabled_import_stub"

    torch_scatter = types.ModuleType("torch_scatter")

    def scatter(src, index, dim=0, dim_size=None, reduce="sum", **_kwargs):
        if dim != 0:
            raise RuntimeError("audit scatter stub only supports dim=0")
        return _segment_reduce(src, index, dim_size, torch, reduce=reduce)

    torch_scatter.scatter = scatter
    sys.modules["torch_scatter"] = torch_scatter
    installed["torch_scatter.scatter"] = "exact_dim0_sum_or_mean_stub"

    torch_geometric = types.ModuleType("torch_geometric")
    torch_geometric_nn = types.ModuleType("torch_geometric.nn")
    torch_geometric_utils = types.ModuleType("torch_geometric.utils")
    torch_geometric_inits = types.SimpleNamespace()

    def glorot(tensor):
        if tensor is None:
            return
        bound = (6.0 / float(tensor.size(-2) + tensor.size(-1))) ** 0.5
        with torch.no_grad():
            tensor.uniform_(-bound, bound)

    def global_mean_pool(x, batch, size=None):
        return _segment_reduce(x, batch, size, torch, reduce="mean")

    def global_max_pool(x, batch, size=None):
        if size is None:
            size = int(batch.max().item()) + 1 if batch.numel() else 0
        output = x.new_full((int(size),) + x.shape[1:], float("-inf"))
        for target in torch.unique(batch):
            output[int(target.item())] = x[batch == target].amax(dim=0)
        return output

    def degree(index, num_nodes=None, dtype=None):
        if num_nodes is None:
            num_nodes = int(index.max().item()) + 1 if index.numel() else 0
        output = torch.zeros(int(num_nodes), dtype=dtype or torch.float32, device=index.device)
        output.index_add_(0, index, output.new_ones(index.shape[0]))
        return output

    torch_geometric_inits.glorot = glorot
    torch_geometric_nn.inits = torch_geometric_inits
    torch_geometric_nn.global_mean_pool = global_mean_pool
    torch_geometric_nn.global_max_pool = global_max_pool
    torch_geometric_utils.softmax = lambda values, index: _segment_softmax(values, index, torch)
    torch_geometric_utils.degree = degree
    torch_geometric.nn = torch_geometric_nn
    torch_geometric.utils = torch_geometric_utils
    sys.modules["torch_geometric"] = torch_geometric
    sys.modules["torch_geometric.nn"] = torch_geometric_nn
    sys.modules["torch_geometric.utils"] = torch_geometric_utils
    installed["torch_geometric.nn.inits.glorot"] = "exact_formula_stub"
    installed["torch_geometric.utils.softmax"] = "exact_segment_softmax_stub"
    installed["torch_geometric.utils.degree"] = "exact_count_stub"
    installed["torch_geometric.nn.global_mean_pool"] = "exact_segment_mean_stub"
    installed["torch_geometric.nn.global_max_pool"] = "exact_segment_max_stub"

    ocpmodels = types.ModuleType("ocpmodels")
    ocpmodels_models = types.ModuleType("ocpmodels.models")
    ocpmodels_gemnet = types.ModuleType("ocpmodels.models.gemnet")
    ocpmodels_layers = types.ModuleType("ocpmodels.models.gemnet.layers")
    ocpmodels_radial = types.ModuleType("ocpmodels.models.gemnet.layers.radial_basis")

    class RadialBasis(torch.nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            raise RuntimeError("OCP RadialBasis is outside the GraphAttention oracle audit")

    ocpmodels_radial.RadialBasis = RadialBasis
    sys.modules["ocpmodels"] = ocpmodels
    sys.modules["ocpmodels.models"] = ocpmodels_models
    sys.modules["ocpmodels.models.gemnet"] = ocpmodels_gemnet
    sys.modules["ocpmodels.models.gemnet.layers"] = ocpmodels_layers
    sys.modules["ocpmodels.models.gemnet.layers.radial_basis"] = ocpmodels_radial
    installed["ocpmodels.RadialBasis"] = "disabled_import_stub"
    return installed


def _load_official_module(root: Path, torch):
    source_path = root / "nets" / "graph_attention_transformer.py"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    _install_optional_dependency_stubs(torch)

    package = types.ModuleType("nets")
    package.__path__ = [str(root / "nets")]
    sys.modules["nets"] = package
    spec = importlib.util.spec_from_file_location("nets.graph_attention_transformer", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to build import spec for {}".format(source_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source_path


def _shape(value):
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [_shape(item) for item in value]
    return type(value).__name__


def _module_tree(model) -> Dict[str, Any]:
    result = {}
    for name, module in model.named_modules():
        parameters = {
            parameter_name: list(parameter.shape)
            for parameter_name, parameter in module.named_parameters(recurse=False)
        }
        result[name or "<root>"] = {
            "class": "{}.{}".format(type(module).__module__, type(module).__qualname__),
            "direct_parameters": parameters,
            "direct_parameter_numel": sum(parameter.numel() for parameter in module.parameters(recurse=False)),
        }
    return result


def _instruction_payload(instruction) -> Dict[str, Any]:
    fields = {}
    for name in (
        "i_in1", "i_in2", "i_out", "connection_mode", "has_weight", "path_weight", "path_shape"
    ):
        if hasattr(instruction, name):
            value = getattr(instruction, name)
            if isinstance(value, tuple):
                value = list(value)
            fields[name] = value
    fields["repr"] = repr(instruction)
    return fields


def _explicit_forward(model, inputs, torch):
    trace = {}
    node_input = inputs["node_input"]
    edge_src = inputs["edge_src"]
    edge_dst = inputs["edge_dst"]
    edge_attr = inputs["edge_attr"]
    edge_scalars = inputs["edge_scalars"]

    message_src = model.merge_src(node_input)
    message_dst = model.merge_dst(node_input)
    trace["merge_src"] = _shape(message_src)
    trace["merge_dst"] = _shape(message_dst)
    message = message_src.index_select(0, edge_src) + message_dst.index_select(0, edge_dst)
    trace["endpoint_gather_add"] = _shape(message)

    if model.nonlinear_message:
        raise RuntimeError("first M5 oracle freezes nonlinear_message=False")
    radial_weight = model.sep.dtp_rad(edge_scalars)
    trace["radial_profile"] = _shape(radial_weight)
    tp_output = model.sep.dtp(message, edge_attr, radial_weight)
    trace["depthwise_tensor_product"] = _shape(tp_output)
    message = model.sep.lin(tp_output)
    trace["post_tp_linear"] = _shape(message)
    message = model.vec2heads(message)
    trace["vec2heads"] = _shape(message)
    alpha = message.narrow(2, 0, model.mul_alpha_head)
    value = message.narrow(2, model.mul_alpha_head, message.shape[-1] - model.mul_alpha_head)
    trace["alpha_head_channels"] = _shape(alpha)
    trace["value_head_channels"] = _shape(value)

    alpha = model.alpha_act(alpha)
    trace["alpha_activation"] = _shape(alpha)
    alpha = torch.einsum("bik,aik->bi", alpha, model.alpha_dot)
    trace["alpha_dot"] = _shape(alpha)
    alpha = _segment_softmax(alpha, edge_dst, torch)
    trace["segment_softmax"] = _shape(alpha)
    alpha = alpha.unsqueeze(-1)
    if model.alpha_dropout is not None:
        alpha = model.alpha_dropout(alpha)
    trace["alpha_after_dropout"] = _shape(alpha)
    attention = value * alpha
    trace["weighted_value"] = _shape(attention)
    attention = _segment_reduce(attention, edge_dst, node_input.shape[0], torch, reduce="sum")
    trace["segment_sum_heads"] = _shape(attention)
    attention = model.heads2vec(attention)
    trace["heads2vec"] = _shape(attention)
    if model.rescale_degree:
        degree = torch.bincount(edge_dst, minlength=node_input.shape[0]).to(dtype=node_input.dtype).view(-1, 1)
        attention = attention * degree
        trace["degree_rescale"] = _shape(attention)
    node_output = model.proj(attention)
    trace["projection"] = _shape(node_output)
    if model.proj_drop is not None:
        node_output = model.proj_drop(node_output)
        trace["projection_dropout"] = _shape(node_output)
    return node_output, trace


def build_audit(output: Path, official_root: Path) -> Dict[str, Any]:
    import torch
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    torch.manual_seed(SEED)
    module, source_path = _load_official_module(official_root.resolve(), torch)
    config = {
        "irreps_node_input": "4x0e+2x1e+1x2e",
        "irreps_node_attr": "1x0e",
        "irreps_edge_attr": "1x0e+1x1e+1x2e",
        "irreps_node_output": "4x0e+2x1e+1x2e",
        "fc_neurons": [6, 8],
        "irreps_head": "2x0e+1x1e+1x2e",
        "num_heads": 2,
        "irreps_pre_attn": None,
        "rescale_degree": False,
        "nonlinear_message": False,
        "alpha_drop": 0.0,
        "proj_drop": 0.0,
    }
    constructor_config = dict(config)
    for key in (
        "irreps_node_input", "irreps_node_attr", "irreps_edge_attr",
        "irreps_node_output", "irreps_head",
    ):
        constructor_config[key] = o3.Irreps(constructor_config[key])
    model = module.GraphAttention(**constructor_config).double().eval()
    node_count = 5
    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    edge_dst = torch.tensor([1, 1, 3, 3, 0, 4, 4, 2], dtype=torch.long)
    inputs = {
        "node_input": torch.randn(node_count, model.irreps_node_input.dim, dtype=torch.float64, requires_grad=True),
        "node_attr": torch.ones(node_count, model.irreps_node_attr.dim, dtype=torch.float64),
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_attr": torch.randn(edge_src.numel(), model.irreps_edge_attr.dim, dtype=torch.float64, requires_grad=True),
        "edge_scalars": torch.randn(edge_src.numel(), config["fc_neurons"][0], dtype=torch.float64, requires_grad=True),
        "batch": torch.zeros(node_count, dtype=torch.long),
    }
    official = model(**inputs)
    explicit, tensor_trace = _explicit_forward(model, inputs, torch)
    forward_error = float((official - explicit).abs().max().item())
    official.square().sum().backward()

    sep_tp = model.sep.dtp.tp
    parameter_manifest = {
        name: {
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    }
    tp_contract = {
        "irreps_in1": str(sep_tp.irreps_in1),
        "irreps_in2": str(sep_tp.irreps_in2),
        "irreps_out": str(sep_tp.irreps_out),
        "weight_numel": int(sep_tp.weight_numel),
        "internal_weights": bool(getattr(sep_tp, "internal_weights", False)),
        "shared_weights": bool(getattr(sep_tp, "shared_weights", False)),
        "instructions": [_instruction_payload(item) for item in sep_tp.instructions],
        "rescale_slices": {
            str(key): {
                "slice": [value[0].start, value[0].stop, value[0].step],
                "slice_sqrt_k": float(value[1]),
            }
            for key, value in model.sep.dtp.slices_sqrt_k.items()
        },
    }
    source_text = source_path.read_text(encoding="utf-8")
    metrics = {
        "official_vs_explicit_forward_max_abs_error": forward_error,
        "node_input_gradient_finite": bool(torch.isfinite(inputs["node_input"].grad).all()),
        "edge_attr_gradient_finite": bool(torch.isfinite(inputs["edge_attr"].grad).all()),
        "edge_scalars_gradient_finite": bool(torch.isfinite(inputs["edge_scalars"].grad).all()),
        "all_parameter_gradients_present": all(parameter.grad is not None for parameter in model.parameters()),
        "all_parameter_gradients_finite": all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ),
    }
    if forward_error != 0.0 or not all(value for key, value in metrics.items() if key.endswith("_finite") or key.endswith("_present")):
        raise RuntimeError("official GraphAttention contract audit failed: {}".format(metrics))

    source_identity = {
        "repository_root": str(official_root.resolve()),
        "source_file": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "graph_attention_class": "{}.{}".format(type(model).__module__, type(model).__qualname__),
        "optional_dependency_policy": "process-local mathematical import stubs; no package installation",
    }
    summary = {
        "evidence_version": EVIDENCE_VERSION,
        "seed": SEED,
        "source_identity": source_identity,
        "oracle_config": config,
        "implicit_official_call_contracts": {
            "irreps_head_argument": "must be an e3nn.o3.Irreps object because the official constructor multiplies the original argument before consistently using self.irreps_head"
        },
        "module_count": len(tuple(model.named_modules())),
        "parameter_tensor_count": len(parameter_manifest),
        "parameter_numel": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_manifest": parameter_manifest,
        "tensor_trace": tensor_trace,
        "depthwise_tensor_product_contract": tp_contract,
        "metrics": metrics,
        "claims": {
            "official_graph_attention_constructed": True,
            "official_graph_attention_forward_executed": True,
            "explicit_python_replay_matches_official_forward": True,
            "dsl_graph_attention_reproduced": False,
            "official_constructor_used_for_dsl_execution": False,
            "depthwise_tp_dsl_lowering_completed": False,
        },
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summary)
    _write_json(output / "source_identity.json", source_identity)
    _write_json(output / "oracle_config.json", config)
    _write_json(output / "module_tree.json", _module_tree(model))
    _write_json(output / "parameter_manifest.json", parameter_manifest)
    _write_json(output / "tensor_trace.json", tensor_trace)
    _write_json(output / "depthwise_tensor_product_contract.json", tp_contract)
    _write_json(output / "metrics.json", metrics)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("audit-equiformer-v1-graph-attention-contract")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_audit(Path(args.output), Path(args.official_root))
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "module_count": result["module_count"],
        "parameter_tensor_count": result["parameter_tensor_count"],
        "parameter_numel": result["parameter_numel"],
        "forward_error": result["metrics"]["official_vs_explicit_forward_max_abs_error"],
        "tp_weight_numel": result["depthwise_tensor_product_contract"]["weight_numel"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
