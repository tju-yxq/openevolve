#!/usr/bin/env python
"""Audit current DSL architecture-family coverage and bounded mutation reachability."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from equivariant_nas.dsl import (  # noqa: E402
    ArchitectureProgram,
    Carrier,
    Compiler,
    DSLValidationError,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    PatchEdit,
    TypedPatch,
    apply_typed_patch,
    core_registry,
    equiformer_v1_capability_profile,
    import_equiformer_v1,
    reference_motif_registry,
    validate_region_transition,
    v1_region_registry,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend, baseline_spec  # noqa: E402
from equivariant_nas.dsl.backends.e3nn_backend import _SUPPORTED as E3NN_SUPPORTED  # noqa: E402


@dataclass(frozen=True)
class FamilySpec:
    family: str
    required_capabilities: Tuple[str, ...]
    witness: str = ""
    exact_importer: bool = False
    exact_backend: bool = False
    note: str = ""


def _capabilities() -> Mapping[str, Mapping[str, Any]]:
    registry = core_registry()
    primitives = {name.split("@", 1)[0] for name in registry.names()}

    def primitive(name: str) -> bool:
        return name in primitives

    def executable(name: str) -> bool:
        return "{}@1".format(name) in E3NN_SUPPORTED

    capabilities = {
        "relative_geometry": primitive("core.relative_position") and executable("core.relative_position"),
        "distance": primitive("core.distance") and executable("core.distance"),
        "radial_basis": primitive("core.radial_basis") and executable("core.radial_basis"),
        "spherical_harmonics": primitive("core.spherical_harmonics") and executable("core.spherical_harmonics"),
        "tensor_product": primitive("core.tensor_product") and executable("core.tensor_product"),
        "equivariant_linear": primitive("core.irrep_linear") and executable("core.irrep_linear"),
        "permutation_aggregation": primitive("core.segment_sum") and executable("core.segment_sum"),
        "scalar_vector_gate": primitive("core.gate") and executable("core.gate"),
        "invariant_attention": all(
            primitive(name) and executable(name)
            for name in ("core.invariant_compatibility", "core.segment_softmax", "core.invariant_weight")
        ),
        "generic_attention_runtime": True,
        "edge_latent_features": primitive("core.edge_lift") and executable("core.edge_lift"),
        "higher_order_irreps": primitive("core.tensor_product"),
        "so2_edge_frame_fusion": all(
            primitive(name)
            for name in (
                "core.to_edge_frame",
                "core.so2_convolution",
                "core.separable_s2_activation",
                "core.from_edge_frame",
            )
        ),
        "generic_scalar_mlp": all(
            primitive(name) and executable(name)
            for name in ("core.irrep_linear", "core.scalar_activation")
        ),
        # The following are explicit semantic gaps, not guesses based on names.
        "multihead_axis_semantics": False,
        "affine_coordinate_type": False,
        "angle_triplet_geometry": False,
        "dihedral_quadruplet_geometry": False,
        "symmetric_contraction": False,
        "periodic_lattice_geometry": False,
        "true_cutoff_envelope_backend": True,
        "cutoff_distance_binding": False,
        "equiformer_v1_full_importer": True,
        "equiformer_v2_full_importer": False,
    }
    evidence = {
        "relative_geometry": "core.relative_position has an e3nn graph runtime",
        "distance": "core.distance has an e3nn graph runtime",
        "radial_basis": "runtime is currently a Gaussian basis only",
        "spherical_harmonics": "core.spherical_harmonics lowers to e3nn.o3.spherical_harmonics",
        "tensor_product": "core.tensor_product lowers to FullyConnectedTensorProduct",
        "equivariant_linear": "core.irrep_linear lowers to e3nn.o3.Linear",
        "permutation_aggregation": "segment_sum/mean use destination-index reductions",
        "scalar_vector_gate": "core.gate is typed and executable",
        "invariant_attention": "single scalar attention path exists; explicit multi-head axes do not",
        "generic_attention_runtime": "core.segment_softmax now uses a differentiable pure-PyTorch segmented softmax",
        "edge_latent_features": "Carrier.EDGE and core.edge_lift are executable",
        "higher_order_irreps": "tensor products can construct higher degrees when legal coupling paths exist",
        "so2_edge_frame_fusion": "only a closed Equiformer V2 fusion pattern is numerically trusted",
        "generic_scalar_mlp": "scalar-only irrep_linear and scalar_activation can be composed",
        "multihead_axis_semantics": "axes are stored in EquivariantType but no head split/merge primitive exists",
        "affine_coordinate_type": "positions and translation-invariant vectors share the same linear irrep type",
        "angle_triplet_geometry": "no typed angle, triplet-index or line-graph primitive is registered",
        "dihedral_quadruplet_geometry": "no typed dihedral or quadruplet interaction primitive is registered",
        "symmetric_contraction": "no MACE-style symmetric contraction primitive or certified expansion exists",
        "periodic_lattice_geometry": "periodicity metadata exists but no lattice-offset/minimum-image primitive exists",
        "true_cutoff_envelope_backend": "core.cutoff_envelope now executes a smooth compact-support polynomial",
        "cutoff_distance_binding": "the primitive still has one x port, so a separate distance cannot gate arbitrary radial features",
        "equiformer_v1_full_importer": "import_equiformer_v1 plus exact reference/constructor/hybrid lowering exists",
        "equiformer_v2_full_importer": "only a closed SO(2) operator fusion exists; there is no full V2 importer",
    }
    return {
        key: {"available": bool(value), "evidence": evidence[key]}
        for key, value in capabilities.items()
    }


def _family_specs() -> Tuple[FamilySpec, ...]:
    return (
        FamilySpec("Tensor Field Network", ("relative_geometry", "radial_basis", "spherical_harmonics", "tensor_product", "permutation_aggregation"), "tfn_message"),
        FamilySpec("SE(3)-Transformer", ("relative_geometry", "spherical_harmonics", "tensor_product", "invariant_attention", "generic_attention_runtime", "multihead_axis_semantics"), note="single-head semantic path exists, but its generic runtime dependency and exact multi-head semantics are incomplete"),
        FamilySpec("EGNN", ("relative_geometry", "distance", "generic_scalar_mlp", "permutation_aggregation", "affine_coordinate_type"), "egnn_coordinate", note="numeric coordinate update works, but the type system does not distinguish affine positions"),
        FamilySpec("PaiNN", ("relative_geometry", "distance", "scalar_vector_gate", "permutation_aggregation"), note="core scalar/vector mechanism vocabulary is present; no faithful importer"),
        FamilySpec("NequIP", ("relative_geometry", "radial_basis", "spherical_harmonics", "tensor_product", "scalar_vector_gate", "true_cutoff_envelope_backend", "cutoff_distance_binding"), note="the numerical envelope exists, but its current one-port signature cannot gate arbitrary radial features by a separate distance"),
        FamilySpec("Allegro", ("relative_geometry", "radial_basis", "spherical_harmonics", "tensor_product", "edge_latent_features", "true_cutoff_envelope_backend", "cutoff_distance_binding"), note="local edge-latent vocabulary exists, but exact cutoff binding is incomplete"),
        FamilySpec("MACE", ("relative_geometry", "radial_basis", "spherical_harmonics", "tensor_product", "symmetric_contraction"), note="higher-order tensor products exist but MACE symmetric contraction is absent"),
        FamilySpec("DimeNet", ("distance", "radial_basis", "angle_triplet_geometry"), note="triplet/angle carrier semantics are absent"),
        FamilySpec("GemNet", ("distance", "radial_basis", "angle_triplet_geometry", "dihedral_quadruplet_geometry"), note="triplet and quadruplet geometry are absent"),
        FamilySpec("SEGNN", ("relative_geometry", "spherical_harmonics", "tensor_product", "scalar_vector_gate"), note="mechanism vocabulary is present; no exact network reconstruction"),
        FamilySpec("Equiformer V1", ("equiformer_v1_full_importer", "invariant_attention", "tensor_product"), "equiformer_v1_reference", True, True),
        FamilySpec("Equiformer V2/eSCN", ("so2_edge_frame_fusion", "equiformer_v2_full_importer"), note="trusted closed SO(2) path exists, not a full architecture importer"),
    )


def _complete_edges(node_count: int):
    source = []
    target = []
    for dst in range(node_count):
        for src in range(node_count):
            if src != dst:
                source.append(src)
                target.append(dst)
    return source, target


def _tfn_program() -> ArchitectureProgram:
    group = GroupSpec.o3()
    positions = EquivariantType(group, Carrier.NODE, Irreps.parse("1x1o", "O3"), measure="length")
    features = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0e+1x1o", "O3"))
    output = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0e+1x1o+1x2e", "O3"))
    return ArchitectureProgram(
        "1.0.0",
        "tfn_mechanism_witness",
        (InputPort("positions", positions), InputPort("features", features)),
        (
            Node("relative", "core.relative_position", {"source": ("input:positions",), "target": ("input:positions",)}),
            Node("distance", "core.distance", {"vector": ("relative",)}),
            Node("radial", "core.radial_basis", {"distance": ("distance",)}, {"num_basis": 4, "cutoff": 5.0}),
            Node("radial_weight", "core.irrep_linear", {"x": ("radial",)}, {"out_irreps": "1x0e"}),
            Node("harmonics", "core.spherical_harmonics", {"direction": ("relative",)}, {"lmax": 2}),
            Node("weighted_harmonics", "core.invariant_weight", {"weight": ("radial_weight",), "value": ("harmonics",)}),
            Node("lift", "core.edge_lift", {"x": ("input:features",)}, {"endpoint": "source"}),
            Node("product", "core.tensor_product", {"left": ("lift",), "right": ("weighted_harmonics",)}, {"out_irreps": str(output.irreps)}),
            Node("aggregate", "core.segment_sum", {"x": ("product",)}),
        ),
        (OutputPort("out", "aggregate", output),),
    )


def _egnn_coordinate_program() -> ArchitectureProgram:
    group = GroupSpec.o3()
    positions = EquivariantType(group, Carrier.NODE, Irreps.parse("1x1o", "O3"), measure="length")
    return ArchitectureProgram(
        "1.0.0",
        "egnn_coordinate_mechanism_witness",
        (InputPort("positions", positions),),
        (
            Node("relative", "core.relative_position", {"source": ("input:positions",), "target": ("input:positions",)}),
            Node("distance", "core.distance", {"vector": ("relative",)}),
            Node("basis", "core.radial_basis", {"distance": ("distance",)}, {"num_basis": 4, "cutoff": 5.0}),
            Node("weight", "core.irrep_linear", {"x": ("basis",)}, {"out_irreps": "1x0e"}),
            Node("activate", "core.scalar_activation", {"x": ("weight",)}, {"activation": "tanh"}),
            Node("weighted", "core.invariant_weight", {"weight": ("activate",), "value": ("relative",)}),
            Node("update", "core.segment_sum", {"x": ("weighted",)}),
            Node("coordinate", "core.residual_add", {"left": ("input:positions",), "right": ("update",)}),
        ),
        (OutputPort("positions_out", "coordinate", positions),),
    )


def _relative_error(reference, actual):
    return float(((actual - reference).norm() / reference.norm().clamp_min(1.0e-12)).detach().cpu())


def _attention_dependency_audit() -> Mapping[str, Any]:
    import torch

    group = GroupSpec.o3()
    edge_scalar = EquivariantType(group, Carrier.EDGE, Irreps.parse("1x0e", "O3"))
    program = ArchitectureProgram(
        "1.0.0",
        "attention_dependency_probe",
        (InputPort("logits", edge_scalar),),
        (Node("softmax", "core.segment_softmax", {"logits": ("input:logits",)}),),
        (OutputPort("out", "softmax", edge_scalar),),
    )
    registry = core_registry()
    compiler = Compiler(registry)
    artifact = compiler.analyze(program)
    backend = E3NNGraphBackend(registry)
    support = backend._support_report(artifact.expanded_program)
    model = backend.build(artifact.expanded_program, artifact.inference)
    runtime_passed = True
    runtime_error = ""
    try:
        model(
            {"logits": torch.randn(4, 1)},
            {"edge_dst": torch.tensor([0, 0, 1, 1], dtype=torch.long)},
        )
    except Exception as exc:
        runtime_passed = False
        runtime_error = "{}: {}".format(type(exc).__name__, exc)
    return {
        "support_report_claimed_supported": support.supported,
        "torch_geometric_present": importlib.util.find_spec("torch_geometric") is not None,
        "runtime_passed": runtime_passed,
        "runtime_error": runtime_error,
        "false_positive_support_report": support.supported and not runtime_passed,
        "implementation": "pure_torch_segment_softmax",
    }


def _run_witness(name: str, seed: int = 20260730) -> Mapping[str, Any]:
    import torch
    from e3nn import o3

    registry = core_registry()
    compiler = Compiler(registry, reference_motif_registry())
    if name == "equiformer_v1_reference":
        program = import_equiformer_v1(baseline_spec())
        artifact = compiler.analyze(program)
        lowering = compiler.plan_lowering(program)
        return {
            "compiled": True,
            "runtime_executed": False,
            "passed": lowering.mode == "exact_reference",
            "architecture_id": artifact.architecture_id,
            "lowering_mode": lowering.mode,
            "note": "full numerical build requires the configured official Equiformer V1 source tree and QM9 runtime",
        }

    program = _tfn_program() if name == "tfn_message" else _egnn_coordinate_program()
    artifact = compiler.analyze(program)
    torch.manual_seed(seed)
    model = E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference).double().eval()
    node_count = 6
    edge_src, edge_dst = _complete_edges(node_count)
    edge_src = torch.tensor(edge_src, dtype=torch.long)
    edge_dst = torch.tensor(edge_dst, dtype=torch.long)
    context = {"edge_src": edge_src, "edge_dst": edge_dst, "num_nodes": node_count}
    positions = torch.randn(node_count, 3, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    translation = torch.tensor([[0.7, -1.1, 0.3]], dtype=torch.float64)

    with torch.no_grad():
        if name == "tfn_message":
            feature_type = program.inputs[1].value_type
            features = torch.randn(node_count, feature_type.irreps.dimension, dtype=torch.float64)
            input_action = o3.Irreps(str(feature_type.irreps)).D_from_matrix(rotation)
            output_type = program.outputs[0].expected_type
            output_action = o3.Irreps(str(output_type.irreps)).D_from_matrix(rotation)
            reference = model({"positions": positions, "features": features}, context)["out"]
            transformed = model(
                {
                    "positions": positions @ rotation.transpose(0, 1) + translation,
                    "features": features @ input_action.transpose(0, 1),
                },
                context,
            )["out"]
            expected = reference @ output_action.transpose(0, 1)
        else:
            reference = model({"positions": positions}, context)["positions_out"]
            transformed = model(
                {"positions": positions @ rotation.transpose(0, 1) + translation}, context
            )["positions_out"]
            expected = reference @ rotation.transpose(0, 1) + translation
        permutation = torch.randperm(node_count)
        if name == "tfn_message":
            permuted = model(
                {
                    "positions": positions.index_select(0, permutation),
                    "features": features.index_select(0, permutation),
                },
                context,
            )["out"]
        else:
            permuted = model(
                {"positions": positions.index_select(0, permutation)}, context
            )["positions_out"]
        permutation_expected = reference.index_select(0, permutation)

        inversion = -torch.eye(3, dtype=torch.float64)
        if name == "tfn_message":
            input_inversion = o3.Irreps(str(feature_type.irreps)).D_from_matrix(inversion)
            output_inversion = o3.Irreps(str(output_type.irreps)).D_from_matrix(inversion)
            inverted = model(
                {
                    "positions": positions @ inversion.transpose(0, 1),
                    "features": features @ input_inversion.transpose(0, 1),
                },
                context,
            )["out"]
            inversion_expected = reference @ output_inversion.transpose(0, 1)
        else:
            inverted = model(
                {"positions": positions @ inversion.transpose(0, 1)}, context
            )["positions_out"]
            inversion_expected = reference @ inversion.transpose(0, 1)
    rotation_translation_error = _relative_error(expected, transformed)
    permutation_error = _relative_error(permutation_expected, permuted)
    inversion_error = _relative_error(inversion_expected, inverted)
    error = max(rotation_translation_error, permutation_error, inversion_error)
    return {
        "compiled": True,
        "runtime_executed": True,
        "passed": error < 1.0e-7,
        "architecture_id": artifact.architecture_id,
        "relative_error": error,
        "rotation_translation_relative_error": rotation_translation_error,
        "permutation_relative_error": permutation_error,
        "inversion_relative_error": inversion_error,
        "lowering_mode": compiler.plan_lowering(program).mode,
    }


def audit_expressivity(seed: int = 20260730) -> Mapping[str, Any]:
    capabilities = _capabilities()
    witnesses = {
        name: _run_witness(name, seed)
        for name in ("tfn_message", "egnn_coordinate", "equiformer_v1_reference")
    }
    dependency_audit = _attention_dependency_audit()
    rows = []
    for spec in _family_specs():
        missing = tuple(
            capability
            for capability in spec.required_capabilities
            if not capabilities[capability]["available"]
        )
        witness = witnesses.get(spec.witness)
        if spec.exact_importer and spec.exact_backend and not missing and witness and witness["passed"]:
            status = "exact_reference_covered"
        elif missing:
            status = "partial_missing_semantics"
        elif witness and witness["passed"]:
            status = "executable_mechanism_witness"
        else:
            status = "mechanism_vocabulary_only"
        rows.append(
            {
                "family": spec.family,
                "status": status,
                "required_capabilities": list(spec.required_capabilities),
                "missing_capabilities": list(missing),
                "witness": spec.witness,
                "witness_result": witness,
                "exact_importer": spec.exact_importer,
                "exact_backend": spec.exact_backend,
                "note": spec.note,
            }
        )
    counts = Counter(item["status"] for item in rows)
    return {
        "capabilities": capabilities,
        "witnesses": witnesses,
        "backend_dependency_audit": dependency_audit,
        "families": rows,
        "counts": dict(counts),
        "exact_coverage_rate": counts["exact_reference_covered"] / len(rows),
        "mechanism_or_better_rate": sum(
            item["status"] in ("exact_reference_covered", "executable_mechanism_witness", "mechanism_vocabulary_only")
            for item in rows
        ) / len(rows),
    }


def _factor_options():
    profile = equiformer_v1_capability_profile()
    factors = tuple(item for item in profile.enabled_factors if item.parameter_paths)
    return factors, tuple((factor.identity_option,) + factor.alternative_options for factor in factors)


def _state_program(reference: ArchitectureProgram, state: Tuple[int, int, int, int]) -> ArchitectureProgram:
    factors, options = _factor_options()
    parameters = dict(reference.parameters)
    for factor, factor_options, selected in zip(factors, options, state[:3]):
        option = factor_options[selected]
        for path in factor.parameter_paths:
            parameters[path] = option[path.rsplit(".", 1)[1]]
    program = replace(reference, parameters=parameters)
    if state[3] == 1:
        nodes = tuple(
            replace(
                node,
                op="motif.v1_multilevel_readout",
                inputs={"terminal": ("scalar_readout",), "aux": ("block3",)},
                attrs={},
            )
            if node.id == "graph_pool"
            else node
            for node in program.nodes
        )
        program = replace(program, nodes=nodes)
    return program


def _transition_patch(parent, parent_id: str, source_state, target_state):
    differing = [index for index, (left, right) in enumerate(zip(source_state, target_state)) if left != right]
    if len(differing) != 1:
        raise ValueError("bounded transition must change exactly one factor")
    coordinate = differing[0]
    factors, options = _factor_options()
    if coordinate < 3:
        factor = factors[coordinate]
        option = options[coordinate][target_state[coordinate]]
        edits = tuple(
            PatchEdit("change_parameters", path, {"value": option[path.rsplit(".", 1)[1]]})
            for path in factor.parameter_paths
        )
        return factor.factor_id, TypedPatch(
            "1.0",
            parent_id,
            parent.language_version,
            {"factor_id": factor.factor_id, "claim": "bounded reachability edge"},
            factor.parameter_paths,
            edits,
        )

    graph_pool = next(node for node in parent.nodes if node.id == "graph_pool")
    if target_state[3] == 1:
        replacement = replace(
            graph_pool,
            op="motif.v1_multilevel_readout",
            inputs={"terminal": ("scalar_readout",), "aux": ("block3",)},
            attrs={},
        )
    else:
        replacement = Node(
            "graph_pool",
            "core.global_pool",
            {"x": ("scalar_readout",)},
            declared_types={"out": parent.outputs[0].expected_type},
        )
    return "F6.3", TypedPatch(
        "1.0",
        parent_id,
        parent.language_version,
        {"factor_id": "F6.3", "claim": "bounded reachability edge"},
        ("graph_pool", "output:prediction"),
        (PatchEdit("replace_node", "graph_pool", {"node": replacement.to_dict()}),),
    )


def _diagnostic_code(exc: Exception) -> str:
    if isinstance(exc, DSLValidationError) and exc.diagnostics:
        return exc.diagnostics[0].code
    return type(exc).__name__


def _reachable(start, adjacency):
    distance = {start: 0}
    queue = deque((start,))
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, ()):
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    return distance


def _strong_components(states, adjacency):
    # The graph has only 72 states; mutual reachability is clearer and less
    # error-prone here than embedding a separate SCC dependency.
    reachability = {state: set(_reachable(state, adjacency)) for state in states}
    unassigned = set(states)
    components = []
    while unassigned:
        seed = min(unassigned)
        component = {other for other in unassigned if other in reachability[seed] and seed in reachability[other]}
        components.append(tuple(sorted(component)))
        unassigned -= component
    return tuple(components)


def audit_reachability() -> Mapping[str, Any]:
    compiler = Compiler(core_registry(), reference_motif_registry())
    reference = import_equiformer_v1(baseline_spec())
    factors, options = _factor_options()
    state_ranges = tuple(range(len(item)) for item in options) + (range(2),)
    states = tuple(itertools.product(*state_ranges))
    programs = {state: _state_program(reference, state) for state in states}
    artifacts = {state: compiler.analyze(program) for state, program in programs.items()}
    lowerings = {state: compiler.plan_lowering(program).mode for state, program in programs.items()}
    adjacency = {state: set() for state in states}
    rejected = Counter()
    attempted = 0
    for source in states:
        for coordinate, choices in enumerate(state_ranges):
            for choice in choices:
                if choice == source[coordinate]:
                    continue
                target = tuple(choice if index == coordinate else value for index, value in enumerate(source))
                attempted += 1
                factor_id, patch = _transition_patch(
                    programs[source], artifacts[source].architecture_id, source, target
                )
                try:
                    child = apply_typed_patch(
                        programs[source],
                        patch,
                        compiler.primitives,
                        expected_parent_id=artifacts[source].architecture_id,
                        validate_child_with_core_registry=False,
                    )
                    region = next(item for item in v1_region_registry(programs[source]) if item.factor_id == factor_id)
                    validate_region_transition(programs[source], child, region)
                    child_artifact = compiler.analyze(child)
                    lowering = compiler.plan_lowering(child)
                    if lowering.mode != region.backend_capability:
                        raise DSLValidationError([]) if False else RuntimeError(
                            "lowering_mismatch:{}->{}".format(region.backend_capability, lowering.mode)
                        )
                    if child_artifact.architecture_id != artifacts[target].architecture_id:
                        raise RuntimeError("target_architecture_mismatch")
                    adjacency[source].add(target)
                except Exception as exc:
                    code = _diagnostic_code(exc)
                    if str(exc).startswith("lowering_mismatch"):
                        code = str(exc)
                    rejected[code] += 1

    start = (0, 0, 0, 0)
    distances = _reachable(start, adjacency)
    components = _strong_components(states, adjacency)
    sink_states = tuple(sorted(state for state in states if not adjacency[state]))
    mode_counts = Counter(lowerings.values())
    return {
        "space": {
            "factor_ids": [factor.factor_id for factor in factors] + ["F6.3"],
            "option_counts": [len(item) for item in options] + [2],
            "state_count": len(states),
            "reference_state": list(start),
        },
        "lowering_mode_counts": dict(mode_counts),
        "attempted_directed_single_factor_edges": attempted,
        "admitted_edges": sum(len(values) for values in adjacency.values()),
        "rejected_edges_by_reason": dict(rejected),
        "reachable_from_reference": len(distances),
        "reachable_rate_from_reference": len(distances) / len(states),
        "maximum_shortest_path_from_reference": max(distances.values()),
        "strong_component_count": len(components),
        "strong_component_sizes": sorted((len(item) for item in components), reverse=True),
        "strongly_connected": len(components) == 1,
        "sink_state_count": len(sink_states),
        "sink_states": [list(item) for item in sink_states],
        "all_state_architecture_ids_unique": len({item.architecture_id for item in artifacts.values()}) == len(states),
        "interpretation": (
            "Every preregistered state is reachable from the reference if constructor edits happen before the optional hybrid readout, "
            "but the directed mutation graph is not strongly connected."
        ),
    }


def _write_report(path: Path, expressivity, reachability) -> None:
    lines = [
        "# 当前DSL表达能力与变异可达性审计",
        "",
        "> 本报告检验两个有界命题：当前代码对哪些代表性等变网络已经具有精确或机制级表达证据；当前正式V1预注册空间中的合法状态是否能由现有Typed Patch和区域门控到达。它不宣称覆盖无限的等变网络空间。",
        "",
        "## 一、结论摘要",
        "",
        "- 代表性网络族共审计{}个，具有完整导入器和精确后端证据的只有{}个。".format(
            len(expressivity["families"]), expressivity["counts"].get("exact_reference_covered", 0)
        ),
        "- TFN消息传递和EGNN式坐标更新已经构造可执行机制见证，并通过数值变换测试；这不等于完整复现原论文网络。",
        "- 正式V1有界空间包含{}个预注册状态，从参考父代出发可达{}个，但状态图不是强连通图。".format(
            reachability["space"]["state_count"], reachability["reachable_from_reference"]
        ),
        "- 当前搜索具有“从固定参考父代覆盖预注册组合”的能力，但不具备“从任意已进化父代继续到任意目标”的遍历性。",
        "",
        "## 二、代表性等变网络表达能力",
        "",
        "| 网络族 | 当前证据等级 | 缺失语义 | 说明 |",
        "|---|---|---|---|",
    ]
    for item in expressivity["families"]:
        lines.append(
            "| {} | `{}` | {} | {} |".format(
                item["family"],
                item["status"],
                "、".join(item["missing_capabilities"]) or "无已登记缺口",
                item["note"] or "—",
            )
        )
    lines.extend(
        [
            "",
            "证据等级的含义：",
            "",
            "- `exact_reference_covered`：存在完整导入器和受信任精确Lowering。",
            "- `executable_mechanism_witness`：核心计算机制能够构造、编译并执行数值等变测试，但尚未完整复现原网络。",
            "- `mechanism_vocabulary_only`：所需核心词汇存在，但没有完整重建和数值一致性证据。",
            "- `partial_missing_semantics`：存在明确缺失的类型语义、几何载体或后端实现。",
            "",
            "其中Equiformer V1本轮验证了导入器与`exact_reference`规划，但没有重新执行官方完整模型前向；它属于现有精确后端代码证据，不是本轮新增的数值等价复现实验。",
            "",
            "### 2.1 可执行机制见证",
            "",
            "| 见证程序 | 结果 | 相对误差 | Lowering |",
            "|---|---:|---:|---|",
        ]
    )
    for name, result in expressivity["witnesses"].items():
        lines.append(
            "| `{}` | {} | {} | `{}` |".format(
                name,
                "通过" if result["passed"] else "失败",
                "{:.3e}".format(result["relative_error"]) if "relative_error" in result else "未执行",
                result["lowering_mode"],
            )
        )
    lines.extend(
        [
            "",
            "### 2.2 关键表达缺口",
            "",
            "1. `EquivariantType.axes`虽然能保存轴名称，但没有head拆分、合并和逐head约束，因此不能证明SE(3)-Transformer的精确多头语义。",
            "2. 位置与普通一阶向量都使用线性不可约表示，缺少仿射坐标类型；EGNN式坐标更新可数值运行，但平移协变性不是由类型系统完整证明的。",
            "3. 没有角度/三元组、二面角/四元组载体，无法精确表示DimeNet和GemNet的信息流。",
            "4. 没有MACE式对称收缩的受信任原语或展开证明。",
            "5. 周期性目前只是GroupSpec元数据，没有晶格偏移和最小镜像几何原语。",
            "6. 第一版通用Lowering已经实现真实多项式cutoff，但原语仍只有一个`x`端口，尚不能表达“用独立距离对任意径向特征施加包络”的完整绑定语义。",
            "7. 第一版通用Lowering已经将`segment_softmax`改为纯PyTorch实现，不再隐式依赖`torch_geometric`。",
            "",
            "### 2.3 后端依赖真实性",
            "",
            "| 检查项 | 结果 |",
            "|---|---:|",
            "| 支持报告声称`segment_softmax`可用 | {} |".format("是" if expressivity["backend_dependency_audit"]["support_report_claimed_supported"] else "否"),
            "| 当前环境存在`torch_geometric` | {} |".format("是" if expressivity["backend_dependency_audit"]["torch_geometric_present"] else "否"),
            "| 实际前向成功 | {} |".format("是" if expressivity["backend_dependency_audit"]["runtime_passed"] else "否"),
            "| 是否构成支持能力错报 | {} |".format("是" if expressivity["backend_dependency_audit"]["false_positive_support_report"] else "否"),
            "| 当前实现 | `{}` |".format(expressivity["backend_dependency_audit"]["implementation"]),
            "",
            "## 三、正式V1有界变异空间可达性",
            "",
            "状态空间由F2.2径向编码、F4.4注意力头数、F5.3归一化和F6.3读出四个因子组成：",
            "",
            "$$",
            r"4\times3\times3\times2=72.",
            "$$",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            "| 状态数 | {} |".format(reachability["space"]["state_count"]),
            "| 从参考父代可达状态 | {} ({:.1f}%) |".format(
                reachability["reachable_from_reference"], 100 * reachability["reachable_rate_from_reference"]
            ),
            "| 参考父代到任意状态最大最短步数 | {} |".format(reachability["maximum_shortest_path_from_reference"]),
            "| 尝试的有向单因子边 | {} |".format(reachability["attempted_directed_single_factor_edges"]),
            "| 被门控接受的边 | {} |".format(reachability["admitted_edges"]),
            "| 强连通分量数 | {} |".format(reachability["strong_component_count"]),
            "| 最大强连通分量规模 | {} |".format(reachability["strong_component_sizes"][0]),
            "| 无出边状态数 | {} |".format(reachability["sink_state_count"]),
            "| 是否强连通 | {} |".format("是" if reachability["strongly_connected"] else "否"),
            "",
            "### 3.1 结果解释",
            "",
            "从固定参考父代出发，72个预注册组合全部可达，前提是先完成所有构造器修改，最后再应用多层读出。这证明当前正式V1切片具有从初始父代出发的覆盖性。",
            "",
            "但它不具有遍历性。多层读出使Lowering变为`exact_hybrid`，此后构造器区域仍要求`exact_constructor`，所以后续构造器变异被拒绝；读出区域白名单只允许多层读出，也不允许恢复`core.global_pool`。此外，最后一个构造器因子恢复基线会使Lowering回到`exact_reference`，同样不等于区域要求的`exact_constructor`。",
            "",
            "因此，当前更准确的结论是：",
            "",
            "> 当前变异系统能从固定Equiformer V1参考父代覆盖全部72个预注册组合，但不能从任意中间架构继续到达任意目标架构。",
            "",
            "## 四、后续工作规划",
            "",
            "### P0：先修复会造成错误结论的验证缺口",
            "",
            "1. 增加输出活性和参数梯度活性门控，拒绝不影响输出的死分支。",
            "2. 修复后端支持检查，把`torch_geometric`、`torch_scatter`及融合后端依赖纳入真实前向探针。",
            "3. 扩展严格重写，识别`select_scalars`与线性图聚合可交换等已知代数等价。",
            "4. 将区域能力从单字符串相等改为能力集合或偏序，例如`exact_hybrid`应包含受信任构造器修改能力。",
            "5. 为每个结构变异提供合法逆变异，至少保证正式有界空间强连通。",
            "",
            "### P1：完成代表性网络重建测试集",
            "",
            "1. 优先完成TFN、EGNN、PaiNN、NequIP和SE(3)-Transformer五个导入器或精确模板。",
            "2. 每个网络同时验证DSL图、类型推导、参数映射、前向输出和旋转/平移/置换响应与原实现一致。",
            "3. 将DimeNet/GemNet所需角度载体、MACE对称收缩、多头轴语义和仿射坐标类型加入语言设计。",
            "4. 修复真实cutoff、径向基族和周期晶格几何后端。",
            "",
            "### P2：证明有界变异完备性",
            "",
            "1. 明确定义有界程序空间：群、最大$l$、节点数、深度、载体和资源上限。",
            "2. 建立可逆基础变异集：算子替换、分支增删、重连、表示迁移、Block增删和Motif展开/折叠。",
            "3. 自动构造状态图，要求只有一个强连通分量；若空间过大，则对分层抽样子空间做连通性证明和随机游走覆盖测试。",
            "4. 保留小概率全局重启或枚举提议，使每个有界合法程序具有非零采样概率。",
            "",
            "### P3：再评价能否发现更优网络",
            "",
            "1. 把若干已知较优架构作为隐藏目标，测试从弱父代的命中率和最短发现步数。",
            "2. 批量运行LLM变异，报告合法率、规范唯一率、机制新颖率、行为新颖率和档案覆盖率。",
            "3. 加入固定预算短训练，最终指标使用合法、等变、新颖且性能有竞争力的候选比例。",
            "4. 分开报告“搜索空间中可达”和“有限预算下实际找到”，不能用后者替代前者。",
            "",
            "## 五、复现命令",
            "",
            "```powershell",
            "python scripts/run_dsl_coverage_reachability_audit.py --output reports/dsl_coverage_reachability_20260730",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(output: Path, seed: int = 20260730):
    output.mkdir(parents=True, exist_ok=True)
    expressivity = audit_expressivity(seed)
    reachability = audit_reachability()
    payload = {
        "protocol": "dsl-coverage-reachability-audit@1",
        "seed": seed,
        "expressivity": expressivity,
        "reachability": reachability,
    }
    (output / "coverage_reachability_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(output / "当前DSL表达能力与变异可达性审计.md", expressivity, reachability)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/dsl_coverage_reachability_20260730"))
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    result = run_audit(args.output, args.seed)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "expressivity_counts": result["expressivity"]["counts"],
                "reachability": {
                    key: result["reachability"][key]
                    for key in (
                        "reachable_from_reference",
                        "strong_component_count",
                        "sink_state_count",
                        "strongly_connected",
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    witnesses_ok = all(item["passed"] for item in result["expressivity"]["witnesses"].values())
    return 0 if witnesses_ok and result["reachability"]["reachable_from_reference"] == 72 else 1


if __name__ == "__main__":
    raise SystemExit(main())
