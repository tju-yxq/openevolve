#!/usr/bin/env python
"""Export forward/gradient evidence for explicit typed topology migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    core_registry,
    migrate_message_flow_to_explicit_topology,
    reference_motif_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.inference import TypeChecker
from equivariant_nas.dsl.motifs import expand_motifs
from equivariant_nas.dsl.serialization import dumps_program


EVIDENCE_VERSION = "evoequilang-explicit-topology-evidence@1"


def _source_program() -> ArchitectureProgram:
    group = GroupSpec.so3()
    node_type = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0", group.family))
    edge_sh_type = EquivariantType(group, Carrier.EDGE, Irreps.parse("1x0+1x1", group.family))
    hidden = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0+2x1", group.family))
    graph_scalar = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0", group.family))
    return ArchitectureProgram(
        "1.0.0",
        "explicit-topology-evidence",
        (InputPort("x", node_type), InputPort("edge_sh", edge_sh_type)),
        (
            Node(
                "message",
                "motif.v1_initial_message",
                {"x": ("input:x",), "edge_sh": ("input:edge_sh",)},
                {"hidden_irreps": str(hidden.irreps)},
            ),
            Node("scalar", "core.select_scalars", {"x": ("message",)}, {"multiplicity": 1}),
            Node("pool", "core.global_pool", {"x": ("scalar",)}),
        ),
        (OutputPort("prediction", "pool", graph_scalar),),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _max_error(left, right) -> float:
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def build_evidence(output: Path, *, pytest_summary: str = "") -> Dict[str, Any]:
    import torch

    torch.manual_seed(20260730)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    primitives = core_registry()
    motifs = reference_motif_registry()
    source = _source_program()
    legacy = expand_motifs(source, motifs)
    migrated = migrate_message_flow_to_explicit_topology(source, primitives, motifs)
    explicit = migrated.program
    legacy_inference = TypeChecker(primitives).check(legacy)
    explicit_inference = TypeChecker(primitives).check(explicit)
    backend = E3NNGraphBackend(primitives)
    support = backend.support_report(explicit)
    if not support.supported:
        raise RuntimeError("explicit-topology support report failed: {}".format(support.to_dict()))

    legacy_model = backend.build(legacy, legacy_inference).eval()
    explicit_model = backend.build(explicit, explicit_inference).eval()
    explicit_model.load_state_dict(legacy_model.state_dict())

    edge_src = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 3, 0, 2], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    x_legacy = torch.randn(4, 2, requires_grad=True)
    sh_legacy = torch.randn(edge_src.shape[0], 4, requires_grad=True)
    x_explicit = x_legacy.detach().clone().requires_grad_(True)
    sh_explicit = sh_legacy.detach().clone().requires_grad_(True)

    legacy_output = legacy_model(
        {"x": x_legacy, "edge_sh": sh_legacy},
        {"edge_src": edge_src, "edge_dst": edge_dst, "num_nodes": 4, "batch": batch, "num_graphs": 2},
    )["prediction"]
    explicit_output = explicit_model(
        {
            "x": x_explicit,
            "edge_sh": sh_explicit,
            "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
            "edge_to_node_index": {"indices": edge_dst, "target_size": 4},
            "batch_index": {"indices": batch, "target_size": 2},
        },
        {},
    )["prediction"]
    legacy_output.sum().backward()
    explicit_output.sum().backward()

    parameter_errors = {}
    explicit_parameters = dict(explicit_model.named_parameters())
    for name, parameter in legacy_model.named_parameters():
        other = explicit_parameters[name]
        parameter_errors[name] = _max_error(parameter.grad, other.grad)

    result = {
        "evidence_version": EVIDENCE_VERSION,
        "pytest_summary": pytest_summary,
        "source_architecture_id": migrated.manifest.source_architecture_id,
        "target_architecture_id": migrated.manifest.target_architecture_id,
        "migration_manifest_hash": migrated.manifest.content_hash(),
        "added_inputs": list(migrated.manifest.added_inputs),
        "replaced_nodes": [list(item) for item in migrated.manifest.replaced_nodes],
        "legacy_required_context": ["edge_src", "edge_dst", "num_nodes", "batch", "num_graphs"],
        "explicit_required_context": [],
        "forward_max_abs_error": _max_error(legacy_output, explicit_output),
        "input_gradient_max_abs_error": {
            "x": _max_error(x_legacy.grad, x_explicit.grad),
            "edge_sh": _max_error(sh_legacy.grad, sh_explicit.grad),
        },
        "parameter_gradient_max_abs_error": max(parameter_errors.values(), default=0.0),
        "parameter_gradient_errors": parameter_errors,
        "backend_support": support.to_dict(),
        "claims": {
            "hidden_index_context_removed_for_this_flow": True,
            "forward_aligned": True,
            "input_gradients_aligned": True,
            "parameter_gradients_aligned": True,
            "official_equiformer_v1_reproduced": False,
        },
    }
    if max(
        result["forward_max_abs_error"],
        *result["input_gradient_max_abs_error"].values(),
        result["parameter_gradient_max_abs_error"],
    ) > 1.0e-7:
        raise RuntimeError("explicit topology migration exceeded the 1e-7 alignment threshold")

    (output / "legacy_program.json").write_text(dumps_program(legacy), encoding="utf-8")
    (output / "explicit_program.json").write_text(dumps_program(explicit), encoding="utf-8")
    _write_json(output / "migration_manifest.json", migrated.manifest.to_dict())
    _write_json(output / "support_report.json", support.to_dict())
    _write_json(output / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("export-dsl-explicit-topology-evidence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pytest-summary", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_evidence(Path(args.output), pytest_summary=str(args.pytest_summary))
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "source_architecture_id": result["source_architecture_id"],
        "target_architecture_id": result["target_architecture_id"],
        "forward_max_abs_error": result["forward_max_abs_error"],
        "parameter_gradient_max_abs_error": result["parameter_gradient_max_abs_error"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
