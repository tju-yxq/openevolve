"""Proof-carrying, semantics-preserving rewrites for canonical DSL programs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Tuple

from .ast import ArchitectureProgram, Node


@dataclass(frozen=True)
class RewriteRuleDescriptor:
    rule_id: str
    version: int
    equivalence: str
    proof_basis: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "version": self.version,
            "equivalence": self.equivalence,
            "proof_basis": self.proof_basis,
        }


@dataclass(frozen=True)
class RewriteStep:
    rule_id: str
    rule_version: int
    equivalence: str
    proof_basis: str
    before_fingerprint: str
    after_fingerprint: str
    affected_nodes: Tuple[str, ...]
    details: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "equivalence": self.equivalence,
            "proof_basis": self.proof_basis,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "affected_nodes": list(self.affected_nodes),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class StrictRewriteResult:
    program: ArchitectureProgram
    trace: Tuple[RewriteStep, ...]
    registry_hash: str


_IDENTITY = RewriteRuleDescriptor(
    "core.eliminate_identity",
    1,
    "exact_function",
    "core.identity@1 is the parameter-free identity map; substituting its sole source preserves values and gradients",
)

_RESIDUAL_COMMUTATIVITY = RewriteRuleDescriptor(
    "core.canonicalize_residual_operands",
    1,
    "exact_binary_operation",
    "core.residual_add@1 applies one binary addition to equal typed values; exchanging the two operands preserves the operation",
)

STRICT_REWRITE_RULES = (_IDENTITY, _RESIDUAL_COMMUTATIVITY)


def strict_rewrite_registry_hash() -> str:
    encoded = json.dumps(
        [rule.to_dict() for rule in STRICT_REWRITE_RULES],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _program_fingerprint(program: ArchitectureProgram) -> str:
    payload = program.to_dict()
    payload.pop("program_id", None)
    payload.pop("annotations", None)
    for node in payload["nodes"]:
        node.pop("annotations", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rewrite_reference(reference: str, node_id: str, replacement: str) -> str:
    if reference == node_id or reference == "{}:out".format(node_id):
        return replacement
    return reference


def _eliminate_one_identity(program: ArchitectureProgram):
    for target in sorted(program.nodes, key=lambda item: item.id):
        qualified = target.op if "@" in target.op else "{}@1".format(target.op)
        sources = target.inputs.get("x", ())
        if not (
            qualified == "core.identity@1"
            and set(target.inputs) == {"x"}
            and len(sources) == 1
            and not target.attrs
            and target.outputs == ("out",)
        ):
            continue
        replacement = sources[0]
        nodes = []
        for node in program.nodes:
            if node.id == target.id:
                continue
            inputs = {
                port: tuple(_rewrite_reference(reference, target.id, replacement) for reference in references)
                for port, references in node.inputs.items()
            }
            nodes.append(replace(node, inputs=inputs))
        outputs = tuple(
            replace(output, source=_rewrite_reference(output.source, target.id, replacement))
            for output in program.outputs
        )
        return replace(program, nodes=tuple(nodes), outputs=outputs), (target.id,), {
            "removed_node": target.id,
            "replacement_reference": replacement,
        }
    return None


def _value_fingerprint(reference: str, nodes: Mapping[str, Node], memo: Dict[str, str]) -> str:
    root, separator, output_name = reference.partition(":")
    if root == "input" or root not in nodes:
        payload = {"external": reference}
    else:
        if root not in memo:
            node = nodes[root]
            qualified = node.op if "@" in node.op else "{}@1".format(node.op)
            encoded_inputs = {
                port: tuple(_value_fingerprint(item, nodes, memo) for item in references)
                for port, references in sorted(node.inputs.items())
            }
            if qualified == "core.residual_add@1" and set(encoded_inputs) == {"left", "right"}:
                pair = sorted((encoded_inputs["left"], encoded_inputs["right"]))
                encoded_inputs = {"left": pair[0], "right": pair[1]}
            node_payload = {
                "op": qualified,
                "inputs": encoded_inputs,
                "attrs": dict(node.attrs),
                "outputs": tuple(node.outputs),
                "declared_types": {key: value.to_dict() for key, value in sorted(node.declared_types.items())},
            }
            memo[root] = hashlib.sha256(
                json.dumps(node_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        payload = {"node": memo[root], "output": output_name or "out"}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonicalize_one_residual(program: ArchitectureProgram):
    nodes = {node.id: node for node in program.nodes}
    memo: Dict[str, str] = {}
    for target in sorted(program.nodes, key=lambda item: item.id):
        qualified = target.op if "@" in target.op else "{}@1".format(target.op)
        left = target.inputs.get("left", ())
        right = target.inputs.get("right", ())
        if qualified != "core.residual_add@1" or len(left) != 1 or len(right) != 1:
            continue
        left_key = (_value_fingerprint(left[0], nodes, memo), left[0])
        right_key = (_value_fingerprint(right[0], nodes, memo), right[0])
        if left_key <= right_key:
            continue
        inputs = dict(target.inputs)
        inputs["left"], inputs["right"] = right, left
        rewritten = tuple(replace(node, inputs=inputs) if node.id == target.id else node for node in program.nodes)
        return replace(program, nodes=rewritten), (target.id,), {
            "node": target.id,
            "before": {"left": left[0], "right": right[0]},
            "after": {"left": right[0], "right": left[0]},
        }
    return None


def apply_strict_rewrites(program: ArchitectureProgram, *, max_steps: int = 10000) -> StrictRewriteResult:
    """Apply only registered proof-carrying rules to a deterministic fixed point."""

    current = program
    trace = []
    implementations = (
        (_IDENTITY, _eliminate_one_identity),
        (_RESIDUAL_COMMUTATIVITY, _canonicalize_one_residual),
    )
    for _ in range(max_steps):
        changed = False
        for descriptor, implementation in implementations:
            before = _program_fingerprint(current)
            outcome = implementation(current)
            if outcome is None:
                continue
            rewritten, affected_nodes, details = outcome
            after = _program_fingerprint(rewritten)
            if before == after:
                continue
            trace.append(
                RewriteStep(
                    descriptor.rule_id,
                    descriptor.version,
                    descriptor.equivalence,
                    descriptor.proof_basis,
                    before,
                    after,
                    tuple(affected_nodes),
                    dict(details),
                )
            )
            current = rewritten
            changed = True
            break
        if not changed:
            return StrictRewriteResult(current, tuple(trace), strict_rewrite_registry_hash())
    raise RuntimeError("strict rewrite system did not reach a fixed point within max_steps")
