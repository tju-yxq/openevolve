#!/usr/bin/env python
"""Quantify structural diversity of accepted audit-only DSL mutations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


BLOCK_RE = re.compile(r"^block\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("analyze-mutation-diversity")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def refs(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def root_ref(value: str) -> str:
    return str(value).split(":", 1)[0]


def normalize_op(value: str) -> str:
    return str(value).split("@", 1)[0]


def semantic_node(node: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "op": normalize_op(node.get("op", "")),
        "inputs": {
            str(port): sorted(refs(value))
            for port, value in sorted(dict(node.get("inputs", {})).items())
        },
        "attrs": dict(node.get("attrs", {})),
        "outputs": list(node.get("outputs", ())),
    }


def edges(program: Mapping[str, Any]) -> set:
    result = set()
    for node in program.get("nodes", ()):
        target = str(node.get("id", ""))
        for port, value in dict(node.get("inputs", {})).items():
            for source in refs(value):
                result.add((root_ref(source), target, str(port)))
    return result


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def boundary_label(node_id: str, *, depth_agnostic: bool) -> str:
    if BLOCK_RE.match(node_id):
        return "BOUNDARY:block" if depth_agnostic else f"BOUNDARY:{node_id}"
    if node_id == "scalar_readout":
        return "BOUNDARY:scalar_readout"
    return f"BOUNDARY:{node_id}"


def program_signature(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    *,
    depth_agnostic: bool,
) -> str:
    parent_nodes = {str(node["id"]): node for node in parent.get("nodes", ())}
    child_nodes = {str(node["id"]): node for node in child.get("nodes", ())}
    changed = {
        node_id
        for node_id in set(parent_nodes).intersection(child_nodes)
        if semantic_node(parent_nodes[node_id]) != semantic_node(child_nodes[node_id])
    }
    memo: Dict[str, str] = {}
    visiting = set()

    def visit(node_id: str) -> str:
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            return "CYCLE"
        if node_id not in child_nodes:
            return boundary_label(node_id, depth_agnostic=depth_agnostic)
        if node_id in parent_nodes and node_id not in changed:
            return boundary_label(node_id, depth_agnostic=depth_agnostic)
        visiting.add(node_id)
        node = child_nodes[node_id]
        input_payload = []
        for port, value in sorted(dict(node.get("inputs", {})).items()):
            input_payload.append(
                (
                    str(port),
                    sorted(visit(root_ref(reference)) for reference in refs(value)),
                )
            )
        payload = {
            "op": normalize_op(node.get("op", "")),
            "attrs": dict(node.get("attrs", {})),
            "inputs": input_payload,
        }
        signature = stable_hash(payload)
        visiting.remove(node_id)
        memo[node_id] = signature
        return signature

    outputs = [
        {
            "name": str(output.get("name", "")),
            "source": visit(str(output.get("source", ""))),
        }
        for output in child.get("outputs", ())
    ]
    return stable_hash(outputs)


def multiset_jaccard(left: Counter, right: Counter) -> float:
    keys = set(left).union(right)
    intersection = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return intersection / float(union) if union else 1.0


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "structural_diversity" / "summary.json"
    )
    summary = read_json(run_dir / "summary.json")
    parent = read_json(run_dir / "audit_parent.dsl.json")
    parent_nodes = {str(node["id"]): node for node in parent.get("nodes", ())}
    parent_edges = edges(parent)
    parent_size = max(1, len(parent_nodes) + len(parent_edges))
    multiseed_path = run_dir / "multi_seed_audit" / "summary.json"
    multiseed = read_json(multiseed_path) if multiseed_path.is_file() else {"candidates": []}
    multiseed_by_id = {
        str(item["architecture_id"]): item for item in multiseed.get("candidates", ())
    }

    rows = []
    for candidate in summary.get("candidates", ()):
        architecture_id = str(candidate["architecture_id"])
        child = read_json(Path(candidate["candidate_path"]))
        child_nodes = {str(node["id"]): node for node in child.get("nodes", ())}
        inserted_ids = sorted(set(child_nodes).difference(parent_nodes))
        removed_ids = sorted(set(parent_nodes).difference(child_nodes))
        changed_ids = sorted(
            node_id
            for node_id in set(parent_nodes).intersection(child_nodes)
            if semantic_node(parent_nodes[node_id]) != semantic_node(child_nodes[node_id])
        )
        child_edges = edges(child)
        edge_edits = len(parent_edges.symmetric_difference(child_edges))
        inserted_ops = Counter(
            normalize_op(child_nodes[node_id].get("op", "")) for node_id in inserted_ids
        )
        depth_taps = sorted(
            {
                root_ref(reference)
                for node_id in inserted_ids + changed_ids
                for value in dict(child_nodes[node_id].get("inputs", {})).values()
                for reference in refs(value)
                if BLOCK_RE.match(root_ref(reference))
            }
        )
        robust = multiseed_by_id.get(architecture_id, {})
        rows.append(
            {
                "architecture_id": architecture_id,
                "factor_id": str(candidate.get("factor_id", "")),
                "inserted_node_count": len(inserted_ids),
                "removed_node_count": len(removed_ids),
                "changed_existing_node_count": len(changed_ids),
                "edge_edit_count": edge_edits,
                "normalized_graph_edit_ratio": (
                    len(inserted_ids) + len(removed_ids) + len(changed_ids) + edge_edits
                )
                / float(parent_size),
                "inserted_operator_multiset": dict(sorted(inserted_ops.items())),
                "depth_taps": depth_taps,
                "depth_aware_macro_signature": program_signature(
                    parent, child, depth_agnostic=False
                ),
                "depth_agnostic_macro_signature": program_signature(
                    parent, child, depth_agnostic=True
                ),
                "multiseed_equivariance_passed": robust.get("all_equivariance_passed"),
                "multiseed_paths_active": robust.get("all_paths_active"),
                "multiseed_maximum_relative_error": robust.get("maximum_relative_error"),
                "multiseed_maximum_output_norm_ratio": robust.get(
                    "maximum_output_norm_ratio_to_parent"
                ),
                "multiseed_maximum_inserted_gradient": robust.get(
                    "maximum_inserted_gradient"
                ),
            }
        )

    groups = defaultdict(list)
    for row in rows:
        groups[row["depth_agnostic_macro_signature"]].append(row["architecture_id"])
    for row in rows:
        peers = groups[row["depth_agnostic_macro_signature"]]
        row["depth_agnostic_macro_group_size"] = len(peers)
        row["depth_agnostic_macro_peers"] = peers
        row["variation_class"] = (
            "macro_topology_unique"
            if len(peers) == 1
            else "depth_or_attribute_variant_of_shared_macro_topology"
        )

    pairwise = []
    for index, left in enumerate(rows):
        left_ops = Counter(left["inserted_operator_multiset"])
        for right in rows[index + 1 :]:
            right_ops = Counter(right["inserted_operator_multiset"])
            pairwise.append(
                {
                    "left": left["architecture_id"],
                    "right": right["architecture_id"],
                    "same_factor": left["factor_id"] == right["factor_id"],
                    "same_depth_agnostic_macro": (
                        left["depth_agnostic_macro_signature"]
                        == right["depth_agnostic_macro_signature"]
                    ),
                    "operator_multiset_jaccard": multiset_jaccard(left_ops, right_ops),
                }
            )

    payload = {
        "protocol": "dsl-mutation-structural-diversity-v1",
        "run_dir": str(run_dir),
        "candidate_count": len(rows),
        "unique_architecture_id_count": len({row["architecture_id"] for row in rows}),
        "unique_depth_aware_macro_count": len(
            {row["depth_aware_macro_signature"] for row in rows}
        ),
        "unique_depth_agnostic_macro_count": len(groups),
        "macro_topology_unique_candidate_count": sum(
            1 for row in rows if row["depth_agnostic_macro_group_size"] == 1
        ),
        "shared_macro_topology_candidate_count": sum(
            1 for row in rows if row["depth_agnostic_macro_group_size"] > 1
        ),
        "training_started": False,
        "test_split_loaded": False,
        "candidates": rows,
        "pairwise": pairwise,
    }
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
