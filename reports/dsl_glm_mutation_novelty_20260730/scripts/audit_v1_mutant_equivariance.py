#!/usr/bin/env python
"""Low-cost numerical equivariance audit for one compositional V1 mutant.

The audit uses random weights and synthetic graphs.  It performs forward and
one backward pass only; no optimizer, dataset, checkpoint, or test split is
loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def parse_args():
    parser = argparse.ArgumentParser("audit-v1-mutant-equivariance")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--task-contract", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument(
        "--parent-program",
        default="",
        help="Optional parent DSL program used to identify added expanded-graph nodes.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--rotations", type=int, default=3)
    parser.add_argument("--translations", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=3)
    parser.add_argument("--joint", type=int, default=2)
    parser.add_argument("--seed", type=int, default=201)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values):
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "p95": float(np.quantile(array, 0.95)) if array.size else 0.0,
        "p99": float(np.quantile(array, 0.99)) if array.size else 0.0,
        "maximum": float(array.max()) if array.size else 0.0,
    }


def portable_dict(value):
    method = getattr(value, "to_dict", None)
    if callable(method):
        return method()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("value has no portable dictionary representation")


def complete_edges(pos, batch, radius):
    import torch

    sources = []
    destinations = []
    for source in range(int(pos.shape[0])):
        for destination in range(int(pos.shape[0])):
            if source == destination or int(batch[source]) != int(batch[destination]):
                continue
            if float(torch.linalg.vector_norm(pos[destination] - pos[source])) <= float(radius):
                sources.append(source)
                destinations.append(destination)
    return torch.tensor([sources, destinations], dtype=torch.long, device=pos.device)


def random_rotation(generator, *, dtype, device):
    import torch

    matrix = torch.randn((3, 3), generator=generator, dtype=dtype).to(device)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q @ torch.diag(signs)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def build_probe(seed, dtype, device):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    features = torch.randn((7, 5), generator=generator, dtype=dtype).to(device)
    positions = torch.tensor(
        [
            [-0.7, 0.1, 0.2],
            [0.5, -0.4, 0.3],
            [0.2, 0.8, -0.6],
            [-0.3, -0.5, 0.9],
            [0.1, 0.2, -0.2],
            [0.9, -0.1, 0.4],
            [-0.6, 0.7, 0.5],
        ],
        dtype=dtype,
        device=device,
    )
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long, device=device)
    return generator, features, positions, batch


def model_forward(model, input_types, features, positions, batch):
    from e3nn import o3
    from equivariant_nas.dsl.backends.e3nn_backend import to_e3nn_irreps

    edge_index = complete_edges(positions, batch, radius=5.0)
    edge_src, edge_dst = edge_index[0], edge_index[1]
    edge_vectors = positions.index_select(0, edge_dst) - positions.index_select(0, edge_src)
    values = {"node_features": features}
    if "edge_sh" in input_types:
        values["edge_sh"] = o3.spherical_harmonics(
            to_e3nn_irreps(input_types["edge_sh"].irreps),
            edge_vectors,
            normalize=True,
            normalization="component",
        )
    return model(
        values,
        {
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_vectors": edge_vectors,
            "num_nodes": int(positions.shape[0]),
            "batch": batch,
            "num_graphs": int(batch.max().item()) + 1,
        },
    )["prediction"]


def errors(reference, transformed):
    import torch

    delta = transformed - reference
    absolute = float(delta.detach().abs().max().cpu())
    relative = float(
        (delta.detach().norm() / reference.detach().norm().clamp_min(torch.finfo(reference.dtype).eps)).cpu()
    )
    return relative, absolute


def node_payload(node):
    """Return the semantic node payload used for parent/candidate comparison."""

    payload = node.to_dict()
    payload.pop("annotations", None)
    return payload


def compare_expanded_nodes(parent_program, candidate_program):
    """Compare expanded graphs without relying on mutation-specific ID prefixes."""

    parent_by_id = {node.id: node for node in parent_program.nodes}
    candidate_by_id = {node.id: node for node in candidate_program.nodes}
    parent_ids = set(parent_by_id)
    candidate_ids = set(candidate_by_id)
    added = tuple(sorted(candidate_ids - parent_ids))
    removed = tuple(sorted(parent_ids - candidate_ids))
    modified = tuple(
        sorted(
            node_id
            for node_id in parent_ids.intersection(candidate_ids)
            if node_payload(parent_by_id[node_id]) != node_payload(candidate_by_id[node_id])
        )
    )
    unchanged = tuple(sorted(parent_ids.intersection(candidate_ids) - set(modified)))
    return {
        "added_node_ids": added,
        "removed_node_ids": removed,
        "modified_node_ids": modified,
        "unchanged_node_ids": unchanged,
    }


def iter_output_tensors(value, path="output"):
    """Yield semantic tensor leaves from dense and structured backend values."""

    import torch

    if torch.is_tensor(value):
        yield path, value
        return
    embedding = getattr(value, "embedding", None)
    if torch.is_tensor(embedding) and callable(getattr(value, "replace_embedding", None)):
        yield path + ".embedding", embedding
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from iter_output_tensors(item, "{}[{}]".format(path, key))
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from iter_output_tensors(item, "{}[{}]".format(path, index))


def zero_output_tensors(value):
    """Return a structure-preserving zero ablation and its tensor-leaf count."""

    import torch

    if torch.is_tensor(value):
        return torch.zeros_like(value), 1
    embedding = getattr(value, "embedding", None)
    replace_embedding = getattr(value, "replace_embedding", None)
    if torch.is_tensor(embedding) and callable(replace_embedding):
        return replace_embedding(torch.zeros_like(embedding)), 1
    if isinstance(value, Mapping):
        result = {}
        count = 0
        for key, item in value.items():
            replacement, item_count = zero_output_tensors(item)
            result[key] = replacement
            count += item_count
        return result, count
    if isinstance(value, tuple):
        items = []
        count = 0
        for item in value:
            replacement, item_count = zero_output_tensors(item)
            items.append(replacement)
            count += item_count
        return tuple(items), count
    if isinstance(value, list):
        items = []
        count = 0
        for item in value:
            replacement, item_count = zero_output_tensors(item)
            items.append(replacement)
            count += item_count
        return items, count
    return value, 0


def instrument_lowering_executors(backend, node_ids):
    """Wrap lowering executors so functional and module-backed nodes are observable."""

    target_ids = frozenset(node_ids)
    state = {
        "mode": "lowering_rule_registry",
        "capture": False,
        "captured": {},
        "ablate_node_id": None,
        "ablation_hits": 0,
        "ablation_zeroed_tensor_count": 0,
        "handles": [],
    }
    try:
        from equivariant_nas.dsl.backends.lowering import LoweringRuleRegistry
    except ImportError:
        # The frozen server commit predates the generic LoweringRuleRegistry.
        # Module-backed nodes can still be hooked after model construction;
        # pure functional nodes are covered by the shared-weight parent
        # counterfactual implemented below.
        state["mode"] = "module_hooks_plus_parent_counterfactual"
        return state
    source = backend.lowering_rules
    instrumented = LoweringRuleRegistry(source.backend, source.base_dependencies)

    for rule in source.rules():
        original_executor = rule.executor

        def wrapped_executor(context, _executor=original_executor):
            output = _executor(context)
            node_id = context.node.id
            if node_id not in target_ids:
                return output
            if state["capture"]:
                seen = set()
                captured = []
                for path, tensor in iter_output_tensors(output):
                    identity = id(tensor)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    if tensor.requires_grad:
                        tensor.retain_grad()
                    captured.append((path, tensor))
                state["captured"][node_id] = captured
            if state["ablate_node_id"] == node_id:
                state["ablation_hits"] += 1
                output, count = zero_output_tensors(output)
                state["ablation_zeroed_tensor_count"] += int(count)
            return output

        instrumented.register(replace(rule, executor=wrapped_executor))

    backend.lowering_rules = instrumented
    return state


def instrument_model_modules(model, node_ids, state):
    """Compatibility hooks for the frozen backend's ModuleDict executors."""

    if state.get("mode") != "module_hooks_plus_parent_counterfactual":
        return
    modules = getattr(model, "node_modules", {})
    target_ids = frozenset(node_ids)
    for node_id in target_ids:
        if node_id not in modules:
            continue

        def hook(_module, _inputs, output, _node_id=node_id):
            if state["capture"]:
                captured = []
                seen = set()
                for path, tensor in iter_output_tensors(output):
                    identity = id(tensor)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    if tensor.requires_grad:
                        tensor.retain_grad()
                    captured.append((path, tensor))
                state["captured"][_node_id] = captured
            if state["ablate_node_id"] == _node_id:
                state["ablation_hits"] += 1
                replacement, count = zero_output_tensors(output)
                state["ablation_zeroed_tensor_count"] += int(count)
                return replacement
            return output

        state["handles"].append(modules[node_id].register_forward_hook(hook))


def align_parent_state(candidate_model, parent_model):
    """Copy every shape-compatible shared parameter/buffer into the parent."""

    candidate_state = candidate_model.state_dict()
    parent_state = parent_model.state_dict()
    copied = []
    for name, value in tuple(parent_state.items()):
        source = candidate_state.get(name)
        if source is None or tuple(source.shape) != tuple(value.shape):
            continue
        parent_state[name] = source.detach().clone()
        copied.append(name)
    parent_model.load_state_dict(parent_state, strict=False)
    return tuple(sorted(copied))


def main():
    args = parse_args()
    started = time.time()
    project_root = Path(args.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import torch
    from equivariant_nas.dsl import Compiler, core_registry, reference_motif_registry
    from equivariant_nas.dsl.backends import E3NNGraphBackend
    from equivariant_nas.dsl.serialization import load_program, load_task_contract

    torch.manual_seed(int(args.seed))
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    program_path = Path(args.program).resolve()
    task = load_task_contract(str(Path(args.task_contract).resolve()))
    program = load_program(str(program_path))
    compiler = Compiler(core_registry(), reference_motif_registry())
    artifact = compiler.analyze(program, task)
    lowering = compiler.plan_lowering(program, task)
    if lowering.mode != "experimental_node_graph" or lowering.unsupported_nodes:
        raise RuntimeError("program lacks a fully supported experimental node-graph lowering")
    backend = E3NNGraphBackend(compiler.primitives)
    support = backend.support_report(artifact.expanded_program)
    if not support.supported:
        raise RuntimeError("backend support report rejected the program")

    parent_program_path = None
    parent_program = None
    parent_artifact = None
    parent_architecture_id = None
    parent_program_sha256 = None
    parent_comparison = {
        "status": "not_requested",
        "added_node_ids": (),
        "removed_node_ids": (),
        "modified_node_ids": (),
        "unchanged_node_ids": (),
    }
    if args.parent_program:
        parent_program_path = Path(args.parent_program).resolve()
        parent_program = load_program(str(parent_program_path))
        parent_artifact = compiler.analyze(parent_program, task)
        parent_architecture_id = parent_artifact.architecture_id
        parent_program_sha256 = sha256_file(parent_program_path)
        parent_comparison = {
            "status": "compared",
            **compare_expanded_nodes(
                parent_artifact.expanded_program,
                artifact.expanded_program,
            ),
        }
    added_node_ids = tuple(parent_comparison["added_node_ids"])
    instrumentation = instrument_lowering_executors(backend, added_node_ids)

    model = backend.build(artifact.expanded_program, artifact.inference).to(device=device, dtype=dtype)
    instrument_model_modules(model, added_node_ids, instrumentation)
    model.eval()
    input_types = {item.name: item.value_type for item in artifact.expanded_program.inputs}
    generator, features, positions, batch = build_probe(args.seed + 17, dtype, device)
    with torch.no_grad():
        reference = model_forward(model, input_types, features, positions, batch)
        repeated = model_forward(model, input_types, features, positions, batch)
    repeat_relative, repeat_absolute = errors(reference, repeated)

    parent_counterfactual = {
        "status": "not_requested",
        "shared_state_key_count": 0,
        "relative_effect": None,
        "maximum_absolute_effect": None,
        "prediction_allclose": None,
        "active": None,
        "scope": "fixed synthetic probe with all shape-compatible parent parameters copied from the candidate",
    }
    if parent_program is not None and parent_artifact is not None:
        parent_backend = E3NNGraphBackend(compiler.primitives)
        parent_model = parent_backend.build(
            parent_artifact.expanded_program,
            parent_artifact.inference,
        ).to(device=device, dtype=dtype)
        copied_keys = align_parent_state(model, parent_model)
        parent_model.eval()
        parent_input_types = {
            item.name: item.value_type for item in parent_artifact.expanded_program.inputs
        }
        with torch.no_grad():
            counterfactual_output = model_forward(
                parent_model,
                parent_input_types,
                features,
                positions,
                batch,
            )
        counterfactual_relative, counterfactual_absolute = errors(
            counterfactual_output,
            reference,
        )
        counterfactual_allclose = bool(
            torch.allclose(
                counterfactual_output.detach(),
                reference.detach(),
                rtol=1.0e-5 if dtype == torch.float32 else 1.0e-9,
                atol=1.0e-7 if dtype == torch.float32 else 1.0e-12,
            )
        )
        parent_counterfactual = {
            "status": "compared",
            "shared_state_key_count": len(copied_keys),
            "shared_state_keys": list(copied_keys),
            "relative_effect": counterfactual_relative,
            "maximum_absolute_effect": counterfactual_absolute,
            "prediction_allclose": counterfactual_allclose,
            "active": not counterfactual_allclose,
            "scope": "fixed synthetic probe with all shape-compatible parent parameters copied from the candidate",
        }

    rows = []
    with torch.no_grad():
        for index in range(int(args.rotations)):
            rotation = random_rotation(generator, dtype=dtype, device=device)
            transformed = model_forward(
                model,
                input_types,
                features,
                positions @ rotation.transpose(0, 1),
                batch,
            )
            relative, absolute = errors(reference, transformed)
            rows.append({"kind": "rotation", "index": index, "relative": relative, "absolute": absolute})

        for index in range(int(args.translations)):
            shift = torch.randn((1, 3), generator=generator, dtype=dtype).to(device)
            transformed = model_forward(model, input_types, features, positions + shift, batch)
            relative, absolute = errors(reference, transformed)
            rows.append({"kind": "translation", "index": index, "relative": relative, "absolute": absolute})

        for index in range(int(args.permutations)):
            chunks = []
            for graph_id in (0, 1):
                indices = torch.nonzero(batch == graph_id, as_tuple=False).view(-1).cpu()
                order = torch.randperm(indices.numel(), generator=generator)
                chunks.append(indices.index_select(0, order))
            permutation = torch.cat(chunks).to(device)
            transformed = model_forward(
                model,
                input_types,
                features.index_select(0, permutation),
                positions.index_select(0, permutation),
                batch.index_select(0, permutation),
            )
            relative, absolute = errors(reference, transformed)
            rows.append({"kind": "permutation", "index": index, "relative": relative, "absolute": absolute})

        for index in range(int(args.joint)):
            rotation = random_rotation(generator, dtype=dtype, device=device)
            shift = torch.randn((1, 3), generator=generator, dtype=dtype).to(device)
            chunks = []
            for graph_id in (0, 1):
                indices = torch.nonzero(batch == graph_id, as_tuple=False).view(-1).cpu()
                order = torch.randperm(indices.numel(), generator=generator)
                chunks.append(indices.index_select(0, order))
            permutation = torch.cat(chunks).to(device)
            joint_positions = (positions @ rotation.transpose(0, 1) + shift).index_select(0, permutation)
            transformed = model_forward(
                model,
                input_types,
                features.index_select(0, permutation),
                joint_positions,
                batch.index_select(0, permutation),
            )
            relative, absolute = errors(reference, transformed)
            rows.append({"kind": "joint", "index": index, "relative": relative, "absolute": absolute})

    model.zero_grad(set_to_none=True)
    instrumentation["captured"] = {}
    instrumentation["capture"] = True
    try:
        differentiable = model_forward(model, input_types, features, positions, batch)
    finally:
        instrumentation["capture"] = False
    differentiable.square().sum().backward()

    gradient_rows = []
    for name, parameter in model.named_parameters():
        if not any(
            name == "node_modules.{}".format(node_id)
            or name.startswith("node_modules.{}.".format(node_id))
            for node_id in added_node_ids
        ):
            continue
        norm = None if parameter.grad is None else float(parameter.grad.detach().norm().cpu())
        gradient_rows.append({"parameter": name, "gradient_norm": norm})
    nonzero_inserted_gradients = [
        item for item in gradient_rows
        if item["gradient_norm"] is not None
        and math.isfinite(item["gradient_norm"])
        and item["gradient_norm"] > 0.0
    ]
    activation_rows = []
    capture_rows = []
    for node_id in added_node_ids:
        tensors = instrumentation["captured"].get(node_id, [])
        capture_rows.append(
            {
                "node_id": node_id,
                "executor_observed": node_id in instrumentation["captured"],
                "tensor_leaf_count": len(tensors),
            }
        )
        for path, activation in tensors:
            gradient = getattr(activation, "grad", None)
            activation_rows.append(
                {
                    "node_id": node_id,
                    "tensor_path": path,
                    "activation_norm": float(activation.detach().norm().cpu()),
                    "activation_gradient_norm": (
                        None if gradient is None else float(gradient.detach().norm().cpu())
                    ),
                }
            )
    active_activations = [
        item for item in activation_rows
        if item["activation_gradient_norm"] is not None
        and math.isfinite(item["activation_gradient_norm"])
        and item["activation_gradient_norm"] > 0.0
    ]

    activity_rtol = 1.0e-5 if dtype == torch.float32 else 1.0e-9
    activity_atol = 1.0e-7 if dtype == torch.float32 else 1.0e-12
    ablation_effects = []
    for node_id in added_node_ids:
        instrumentation["ablate_node_id"] = node_id
        instrumentation["ablation_hits"] = 0
        instrumentation["ablation_zeroed_tensor_count"] = 0
        try:
            with torch.no_grad():
                ablated = model_forward(model, input_types, features, positions, batch)
            hit_count = int(instrumentation["ablation_hits"])
            zeroed_count = int(instrumentation["ablation_zeroed_tensor_count"])
            if hit_count == 0:
                row = {
                    "node_id": node_id,
                    "status": "unknown",
                    "reason": "lowering executor was not observed during the ablation forward",
                    "executor_hit_count": hit_count,
                    "zeroed_tensor_count": zeroed_count,
                }
            elif zeroed_count == 0:
                row = {
                    "node_id": node_id,
                    "status": "unknown",
                    "reason": "lowering output exposed no tensor leaf that could be zeroed",
                    "executor_hit_count": hit_count,
                    "zeroed_tensor_count": zeroed_count,
                }
            else:
                relative, absolute = errors(reference, ablated)
                unchanged = bool(
                    torch.allclose(
                        reference.detach(),
                        ablated.detach(),
                        rtol=activity_rtol,
                        atol=activity_atol,
                    )
                )
                row = {
                    "node_id": node_id,
                    "status": "inactive" if unchanged else "active",
                    "reason": (
                        "zero ablation did not change prediction beyond tolerance"
                        if unchanged
                        else "zero ablation changed prediction beyond tolerance"
                    ),
                    "executor_hit_count": hit_count,
                    "zeroed_tensor_count": zeroed_count,
                    "relative_effect": relative,
                    "maximum_absolute_effect": absolute,
                    "prediction_allclose": unchanged,
                }
        except Exception as exc:
            row = {
                "node_id": node_id,
                "status": "unknown",
                "reason": "zero ablation forward raised an exception",
                "executor_hit_count": int(instrumentation["ablation_hits"]),
                "zeroed_tensor_count": int(instrumentation["ablation_zeroed_tensor_count"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        finally:
            instrumentation["ablate_node_id"] = None
        ablation_effects.append(row)

    ablation_statuses = [item["status"] for item in ablation_effects]
    if "active" in ablation_statuses:
        computation_path_active = True
        computation_path_status = "active"
        computation_path_reliable = True
        computation_path_reason = "at least one added node has a prediction-changing zero ablation"
    elif parent_counterfactual.get("active") is True:
        computation_path_active = True
        computation_path_status = "active"
        computation_path_reliable = True
        computation_path_reason = (
            "candidate differs from the shared-weight parent counterfactual on the fixed probe"
        )
    elif added_node_ids and all(status == "inactive" for status in ablation_statuses):
        computation_path_active = False
        computation_path_status = "inactive"
        computation_path_reliable = True
        computation_path_reason = "all added-node zero ablations were observed and prediction-inert on this probe"
    else:
        computation_path_active = None
        computation_path_status = "unknown"
        computation_path_reliable = False
        computation_path_reason = (
            "no parent-derived added nodes were available"
            if not added_node_ids
            else "one or more added-node ablations lacked conclusive executable evidence"
        )

    relative_values = [item["relative"] for item in rows]
    absolute_values = [item["absolute"] for item in rows]
    threshold_relative = 1.0e-5 if dtype == torch.float32 else 1.0e-7
    near_zero_absolute_threshold = 5.0e-5 if dtype == torch.float32 else 1.0e-8
    reference_norm = float(reference.detach().norm().cpu())
    reference_maximum_absolute = float(reference.detach().abs().max().cpu())
    reference_is_near_zero = reference_norm <= 100.0 * torch.finfo(dtype).eps
    result = {
        "protocol": "v1-mutant-random-weight-equivariance-audit-v2",
        "program_path": str(program_path),
        "program_sha256": sha256_file(program_path),
        "architecture_id": artifact.architecture_id,
        "parent_program_path": None if parent_program_path is None else str(parent_program_path),
        "parent_program_sha256": parent_program_sha256,
        "parent_architecture_id": parent_architecture_id,
        "parent_comparison": parent_comparison,
        "parent_counterfactual": parent_counterfactual,
        "instrumented_added_node_ids": list(added_node_ids),
        "lowering": lowering.to_dict(),
        "backend_support": portable_dict(support),
        "device": str(device),
        "dtype": str(dtype),
        "seed": int(args.seed),
        "synthetic_graph_count": 2,
        "synthetic_node_count": int(features.shape[0]),
        "test_split_loaded": False,
        "training_started": False,
        "optimizer_constructed": False,
        "repeat_forward": {"relative": repeat_relative, "absolute": repeat_absolute},
        "reference_output": {
            "norm": reference_norm,
            "maximum_absolute": reference_maximum_absolute,
            "near_zero": bool(reference_is_near_zero),
        },
        "transform_checks": rows,
        "relative_error": describe(relative_values),
        "absolute_error": describe(absolute_values),
        "thresholds": {
            "relative": threshold_relative,
            "near_zero_absolute": near_zero_absolute_threshold,
            "rule": "relative gate for nonzero outputs; absolute gate is additionally required only when the reference output is numerically near zero",
        },
        "equivariance_passed": bool(
            max(relative_values, default=0.0) <= threshold_relative
            and (
                not reference_is_near_zero
                or max(absolute_values, default=0.0) <= near_zero_absolute_threshold
            )
        ),
        "inserted_parameter_gradients": gradient_rows,
        "nonzero_inserted_gradient_count": len(nonzero_inserted_gradients),
        "inserted_activation_gradients": activation_rows,
        "nonzero_inserted_activation_gradient_count": len(active_activations),
        "lowering_executor_captures": capture_rows,
        "instrumentation_mode": instrumentation.get("mode", "unknown"),
        "ablation_thresholds": {
            "relative": activity_rtol,
            "absolute": activity_atol,
            "rule": "torch.allclose on the baseline and per-node zeros_like-ablated predictions",
        },
        "ablation_effects": ablation_effects,
        "inserted_parameter_path_active": (
            None if not added_node_ids else bool(nonzero_inserted_gradients)
        ),
        "inserted_computation_path_active": computation_path_active,
        "inserted_computation_path_activity": {
            "status": computation_path_status,
            "value": computation_path_active,
            "reliable": computation_path_reliable,
            "reason": computation_path_reason,
            "scope": "the fixed synthetic probe; this is executable evidence, not a global proof",
        },
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "architecture_id": result["architecture_id"],
        "equivariance_passed": result["equivariance_passed"],
        "maximum_relative": result["relative_error"]["maximum"],
        "maximum_absolute": result["absolute_error"]["maximum"],
        "inserted_computation_path_active": result["inserted_computation_path_active"],
        "inserted_computation_path_activity_status": computation_path_status,
        "instrumented_added_node_count": len(added_node_ids),
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
