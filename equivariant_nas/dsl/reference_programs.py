"""Import trusted legacy architectures into the typed graph representation."""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from ..spec import ArchitectureSpec
from .ast import ArchitectureProgram, InputPort, Node, OutputPort
from .canonicalize import architecture_id
from .groups import GroupSpec
from .irreps import Irreps
from .motifs import expand_motifs
from .reference_motifs import reference_motif_registry
from .registry import core_registry
from .types import Carrier, EquivariantType
from .backends.equiformer_v1_constructor import baseline_constructor_parameters


def _as_so3(text: str) -> str:
    """Drop O(3) parity suffixes when importing the rotation-only V1 model."""

    import re

    return re.sub(r"(\d+)[eo]", r"\1", text)


def import_equiformer_v1(spec: ArchitectureSpec, *, task_contract: str = "qm9_alpha") -> ArchitectureProgram:
    """Create a typed representation-flow program for a legacy V1 spec.

    This graph makes the representation and aggregation semantics inspectable.
    Numerical lowering remains locked to the unchanged imported graph because
    the legacy builder is constructor-based rather than node-by-node.
    """

    spec.validate()
    group = GroupSpec.so3()
    input_type = EquivariantType(group, Carrier.NODE, Irreps.parse("5x0", "SO3"))
    sh_type = EquivariantType(
        group,
        Carrier.EDGE,
        Irreps.parse(_as_so3(spec.representation.spherical_harmonics_irreps()), "SO3"),
    )
    hidden = _as_so3(spec.representation.embedding_irreps())
    hidden_type = EquivariantType(group, Carrier.NODE, Irreps.parse(hidden, "SO3"))
    scalar_type = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0", "SO3"))
    graph_scalar = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0", "SO3"))
    nodes = []
    previous = "input:node_features"
    for index in range(spec.macro.num_layers):
        node_id = "block{}".format(index)
        op = "motif.v1_initial_message" if index == 0 else "motif.v1_residual_message"
        nodes.append(
            Node(
                node_id,
                op,
                {"x": (previous,), "edge_sh": ("input:edge_sh",)},
                {"hidden_irreps": hidden},
                declared_types={"out": hidden_type},
                annotations={"stage": index, "legacy_family": "EquiformerV1"},
            )
        )
        previous = node_id
    nodes.extend(
        (
            Node("scalar_readout", "core.select_scalars", {"x": (previous,)}, {"multiplicity": 1}, declared_types={"out": scalar_type}),
            Node("graph_pool", "core.global_pool", {"x": ("scalar_readout",)}, declared_types={"out": graph_scalar}),
        )
    )
    program = ArchitectureProgram(
        language_version="1.0.0",
        task_contract=task_contract,
        inputs=(InputPort("node_features", input_type), InputPort("edge_sh", sh_type)),
        nodes=tuple(nodes),
        outputs=(OutputPort("prediction", "graph_pool", graph_scalar),),
        parameters=baseline_constructor_parameters(spec),
        program_id="equiformer_v1_{}".format(spec.architecture_id()),
        annotations={
            "legacy_backend": "equiformer_v1",
            "legacy_architecture_spec": spec.to_dict(),
            "representation_scope": "type-and-connectivity; numerical backend remains constructor-locked",
        },
    )
    expanded = expand_motifs(program, reference_motif_registry())
    registry = core_registry()
    lock = architecture_id(expanded, registry)
    annotations = dict(program.annotations)
    annotations["legacy_lock_architecture_id"] = lock
    return replace(program, annotations=annotations)
