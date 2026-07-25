"""JSON Schema exported to LLM clients and external validators."""

from __future__ import annotations

from typing import Any, Dict


def architecture_program_schema() -> Dict[str, Any]:
    group = {
        "type": "object",
        "additionalProperties": False,
        "required": ["family", "dimension"],
        "properties": {
            "family": {"enum": ["SO2", "O2", "Cyclic2D", "Dihedral2D", "SO3", "O3"]},
            "dimension": {"enum": [2, 3]},
            "order": {"type": "integer", "minimum": 0},
            "translation": {"enum": ["none", "relative_coordinates", "explicit"]},
            "permutation": {"enum": ["none", "node_set"]},
            "periodicity": {"enum": ["none", "lattice"]},
        },
    }
    value_type = {
        "type": "object",
        "additionalProperties": False,
        "required": ["group", "carrier", "irreps"],
        "properties": {
            "group": group,
            "carrier": {"enum": ["node", "edge", "graph", "grid", "pair"]},
            "irreps": {"type": "string", "minLength": 1},
            "frame": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"kind": {"enum": ["global", "edge", "local", "invariant"]}, "reference": {"type": "string"}},
            },
            "axes": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "dtype": {"enum": ["float16", "bfloat16", "float32", "float64"]},
            "measure": {"type": "string"},
            "equivariance_level": {"type": "integer", "minimum": 0, "maximum": 3},
        },
    }
    reference = {
        "oneOf": [
            {"type": "string", "minLength": 1},
            {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        ]
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://evoequilang.local/schema/architecture-program-1.json",
        "title": "EvoEquiLang ArchitectureProgram",
        "type": "object",
        "additionalProperties": False,
        "required": ["language_version", "task_contract", "inputs", "nodes", "outputs"],
        "properties": {
            "language_version": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+"},
            "program_id": {"type": "string"},
            "task_contract": {"type": "string", "minLength": 1},
            "parameters": {"type": "object"},
            "inputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "type"],
                    "properties": {"name": {"type": "string", "minLength": 1}, "type": value_type},
                },
            },
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "op", "inputs"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1, "pattern": "^[^:]+$"},
                        "op": {"type": "string", "minLength": 1},
                        "inputs": {"type": "object", "additionalProperties": reference},
                        "attrs": {"type": "object"},
                        "outputs": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string"}},
                        "declared_types": {"type": "object", "additionalProperties": value_type},
                        "annotations": {"type": "object"},
                    },
                },
            },
            "outputs": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "source", "type"],
                    "properties": {"name": {"type": "string"}, "source": {"type": "string"}, "type": value_type},
                },
            },
            "constraints": {"type": "array", "items": {"type": "object"}},
            "annotations": {"type": "object"},
        },
    }
