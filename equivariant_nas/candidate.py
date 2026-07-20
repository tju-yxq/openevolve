"""Safe candidate representation and factor-local patch application."""

from __future__ import annotations

import ast
import json
import pprint
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping

from .spec import ArchitectureSpec, EvolutionFactor, SpecValidationError


CANDIDATE_VARIABLE = "ARCHITECTURE_SPEC"


def extract_literal_spec(program_path: str) -> ArchitectureSpec:
    """Parse a candidate without importing or executing LLM-generated code."""

    source = Path(program_path).read_text(encoding="utf-8")
    return extract_literal_spec_source(source, filename=program_path)


def extract_literal_spec_source(source: str, filename: str = "<candidate>") -> ArchitectureSpec:
    tree = ast.parse(source, filename=filename)
    assignments = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == CANDIDATE_VARIABLE for target in targets):
                assignments.append(node.value)
        elif isinstance(node, ast.Pass):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # A module docstring is harmless and never executed by this parser.
            continue
        else:
            raise SpecValidationError(
                "candidate contains executable statement {}".format(type(node).__name__)
            )
    if len(assignments) != 1:
        raise SpecValidationError(
            "candidate must define exactly one {} literal".format(CANDIDATE_VARIABLE)
        )
    try:
        value = ast.literal_eval(assignments[0])
    except (ValueError, TypeError, SyntaxError) as exc:
        raise SpecValidationError("candidate spec is not a literal: {}".format(exc))
    if not isinstance(value, dict):
        raise SpecValidationError("candidate spec must be a dictionary")
    return ArchitectureSpec.from_dict(value)


def render_candidate(spec: ArchitectureSpec) -> str:
    spec.validate()
    payload = pprint.pformat(spec.to_dict(), sort_dicts=True, width=100)
    return "# Trusted evaluator parses this literal without executing code.\n{} = {}\n".format(
        CANDIDATE_VARIABLE, payload
    )


def apply_factor_patch(
    parent: ArchitectureSpec,
    selected_factor: EvolutionFactor,
    patch: Mapping[str, Any],
) -> ArchitectureSpec:
    """Apply a complete replacement for exactly one factor."""

    data = parent.to_dict()
    key = selected_factor.value.lower()
    if key == "representation":
        target = "representation"
    elif key == "operator":
        target = "operator"
    elif key == "action":
        target = "action"
    elif key == "macro":
        target = "macro"
    else:
        raise SpecValidationError("unsupported factor {}".format(selected_factor))
    data[target] = dict(patch)
    child = ArchitectureSpec.from_dict(data)
    parent.assert_factor_local_change(child, selected_factor)
    return child


def parse_llm_patch(text: str, parent: ArchitectureSpec, selected: EvolutionFactor):
    """Parse the strict JSON response used by the factor-local editor."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    value = json.loads(stripped)
    if set(value) != {"selected_factor", "reasoning", "patch"}:
        raise SpecValidationError("LLM response must contain selected_factor, reasoning, patch")
    if value["selected_factor"] != selected.value:
        raise SpecValidationError("LLM changed the router-selected factor")
    child = apply_factor_patch(parent, selected, value["patch"])
    return child, str(value["reasoning"])
