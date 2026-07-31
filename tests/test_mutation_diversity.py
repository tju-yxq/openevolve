from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "dsl_glm_mutation_novelty_20260730"
    / "scripts"
    / "analyze_mutation_diversity.py"
)
SPEC = importlib.util.spec_from_file_location("mutation_diversity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def node(node_id, op, inputs=None, attrs=None):
    return {
        "id": node_id,
        "op": op,
        "inputs": inputs or {},
        "attrs": attrs or {},
        "outputs": ["out"],
    }


def test_operator_version_suffix_is_semantically_normalized():
    assert MODULE.semantic_node(node("x", "core.irrep_linear")) == MODULE.semantic_node(
        node("x", "core.irrep_linear@1")
    )


def test_depth_agnostic_signature_ignores_only_block_index():
    parent = {
        "nodes": [
            node("block0", "core.identity"),
            node("block1", "core.identity"),
            node("scalar_readout", "core.identity", {"x": "block1"}),
            node("graph_pool", "core.global_pool", {"x": "scalar_readout"}),
        ],
        "outputs": [{"name": "prediction", "source": "graph_pool"}],
    }

    def child(depth):
        return {
            "nodes": parent["nodes"][:3]
            + [
                node("aux", "core.select_scalars@1", {"x": depth}),
                node(
                    "fused",
                    "core.residual_add@1",
                    {"left": "scalar_readout", "right": "aux"},
                ),
                node("graph_pool", "core.global_pool@1", {"x": "fused"}),
            ],
            "outputs": [{"name": "prediction", "source": "graph_pool"}],
        }

    left = child("block0")
    right = child("block1")
    assert MODULE.program_signature(parent, left, depth_agnostic=True) == MODULE.program_signature(
        parent, right, depth_agnostic=True
    )
    assert MODULE.program_signature(parent, left, depth_agnostic=False) != MODULE.program_signature(
        parent, right, depth_agnostic=False
    )

