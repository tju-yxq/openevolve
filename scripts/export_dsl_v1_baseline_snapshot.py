#!/usr/bin/env python
"""Export the immutable pre-v2 DSL language and implementation baseline.

The snapshot is deliberately independent from any model training run.  It
captures the language registry, motif registry, lowering rule registry,
compiler semantics, source identities, and the exact files that define the
current implementation.  Later migrations can therefore prove what changed
instead of relying on a dirty-worktree commit alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SNAPSHOT_VERSION = "evoequilang-v1-baseline-snapshot@1"
V1_PRIMITIVE_NAMES = frozenset({
    "core.identity@1", "core.irrep_linear@1", "core.irrep_concat@1", "core.residual_add@1",
    "core.tensor_product@1", "core.scalar_activation@1", "core.invariant_weight@1",
    "core.edge_lift@1", "core.segment_sum@1", "core.global_pool@1", "core.select_scalars@1",
    "core.to_edge_frame@1", "core.from_edge_frame@1", "core.irrep_slice@1",
    "core.change_multiplicity@1", "core.relative_position@1", "core.distance@1",
    "core.radial_basis@1", "core.cutoff_envelope@1", "core.spherical_harmonics@1",
    "core.norm_activation@1", "core.gate@1", "core.equivariant_norm@1",
    "core.stochastic_depth@1", "core.invariant_dropout@1", "core.segment_mean@1",
    "core.segment_softmax@1", "core.invariant_compatibility@1", "core.so2_convolution@1",
    "core.s2_activation@1", "core.separable_s2_activation@1", "core.s2_swiglu@1",
    "core.equivariant_merge_norm@1",
})


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_identity(root: Path) -> Dict[str, Any]:
    root = root.resolve()
    if not (root / ".git").exists():
        return {
            "root": str(root),
            "available": False,
            "branch": "",
            "commit": "",
            "tracked_dirty": None,
            "untracked_count": None,
        }
    status_lines = tuple(
        line for line in _run_git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
        if line
    )
    tracked_dirty = any(not line.startswith("??") for line in status_lines)
    return {
        "root": str(root),
        "available": True,
        "branch": _run_git(root, "branch", "--show-current"),
        "commit": _run_git(root, "rev-parse", "HEAD"),
        "tracked_dirty": tracked_dirty,
        "untracked_count": sum(1 for line in status_lines if line.startswith("??")),
        "status_paths": [line[3:] for line in status_lines],
    }


def _semantic_files(project: Path) -> Sequence[Path]:
    roots = (
        project / "equivariant_nas" / "dsl",
        project / "tests",
    )
    files = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if root.name == "tests" and not (
                path.name.startswith("test_dsl_")
                or path.name == "test_equiformer_v3_compositional_support.py"
                or path.name == "conftest.py"
            ):
                continue
            files.append(path.resolve())
    return tuple(sorted(files, key=lambda item: item.relative_to(project).as_posix()))


def _primitive_payload(definition: Any) -> Dict[str, Any]:
    return {
        "name": definition.name,
        "version": definition.version,
        "qualified_name": definition.qualified_name,
        "input_ports": list(definition.input_ports),
        "output_ports": list(definition.output_ports),
        "type_rule": getattr(definition.type_rule, "__name__", "anonymous"),
        "group_families": list(definition.group_families),
        "certificate_level": int(definition.certificate_level),
        "backend_keys": list(definition.backend_keys),
        "description": definition.description,
        "required_attrs": list(definition.required_attrs),
        "optional_attrs": dict(definition.optional_attrs),
        "attribute_aliases": dict(definition.attribute_aliases),
        "motif_parameter_attrs": list(definition.motif_parameter_attrs),
        "semantic_constraints": list(definition.semantic_constraints),
        "edit_guidance": list(definition.edit_guidance),
        "parameter_rule": getattr(definition.parameter_rule, "__name__", "") if definition.parameter_rule else "",
        "content_hash": definition.content_hash(),
    }


def build_snapshot(
    project_root: Path,
    *,
    pytest_summary: str = "",
    pytest_command: str = "python -m pytest -q",
    snapshot_version: str = SNAPSHOT_VERSION,
    evidence_context: Mapping[str, Any] | None = None,
    primitive_scope: str = "legacy_v1",
) -> Dict[str, Any]:
    project = project_root.resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))

    from equivariant_nas.dsl.backends.e3nn_backend import (
        E3NNGraphBackend,
        build_e3nn_lowering_registry,
    )
    from equivariant_nas.dsl.canonicalize import (
        BACKEND_SEMANTICS_VERSION,
        COMPILER_SEMANTICS_VERSION,
    )
    from equivariant_nas.dsl.reference_motifs import reference_motif_registry
    from equivariant_nas.dsl.registry import core_registry
    from equivariant_nas.dsl.rewrites import strict_rewrite_registry_hash
    from equivariant_nas.dsl.types import VALUE_TYPE_SCHEMA_VERSION
    from equivariant_nas.dsl.parameters import PARAMETER_CONTRACT_SCHEMA_VERSION

    primitives = core_registry()
    if primitive_scope not in ("legacy_v1", "all"):
        raise ValueError("primitive_scope must be legacy_v1 or all")
    primitive_names = (
        tuple(name for name in primitives.names() if name in V1_PRIMITIVE_NAMES)
        if primitive_scope == "legacy_v1"
        else primitives.names()
    )
    primitive_items = [_primitive_payload(primitives.resolve(name)) for name in primitive_names]
    motifs = reference_motif_registry()
    motif_payload = motifs.to_dict()
    lowering_payload = build_e3nn_lowering_registry().audit()
    if primitive_scope == "legacy_v1":
        rules = [item for item in lowering_payload["rules"] if item["primitive"] in V1_PRIMITIVE_NAMES]
        exactness_counts: Dict[str, int] = {}
        for item in rules:
            exactness_counts[item["exactness"]] = exactness_counts.get(item["exactness"], 0) + 1
        lowering_payload = dict(
            lowering_payload,
            rule_count=len(rules),
            exactness_counts=exactness_counts,
            rules=rules,
        )
    file_hashes = {
        path.relative_to(project).as_posix(): _sha256_file(path)
        for path in _semantic_files(project)
    }
    source_roots = {
        "equiformer_v1": project.parent / "equiformer",
        "equiformer_v2": project.parent / "equiformer_v3",
        "equiformer_v3": project.parent / "equiformer_v3_official",
    }
    semantic_payload = {
        "snapshot_version": str(snapshot_version),
        "primitive_scope": primitive_scope,
        "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
        "backend_neutral_semantics_version": BACKEND_SEMANTICS_VERSION,
        "value_type_schema_version": VALUE_TYPE_SCHEMA_VERSION,
        "parameter_contract_schema_version": PARAMETER_CONTRACT_SCHEMA_VERSION,
        "e3nn_backend_semantics_version": E3NNGraphBackend.semantic_version,
        "strict_rewrite_registry_hash": strict_rewrite_registry_hash(),
        "primitive_count": len(primitive_items),
        "primitives": primitive_items,
        "motif_count": len(motif_payload),
        "motifs": motif_payload,
        "lowering": lowering_payload,
        "semantic_file_sha256": file_hashes,
    }
    semantic_snapshot_id = _sha256_bytes(_json_bytes(semantic_payload))
    return {
        **semantic_payload,
        "semantic_snapshot_id": semantic_snapshot_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "project_git": git_identity(project),
        "official_source_git": {
            name: git_identity(path) for name, path in source_roots.items()
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "tests": {
            "command": pytest_command,
            "summary": pytest_summary,
        },
        "evidence_context": dict(evidence_context or {}),
    }


def write_snapshot(output: Path, payload: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("export-dsl-v1-baseline-snapshot")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    parser.add_argument("--pytest-summary", default="")
    parser.add_argument("--pytest-command", default="python -m pytest -q")
    parser.add_argument("--snapshot-version", default=SNAPSHOT_VERSION)
    parser.add_argument("--milestone", default="")
    parser.add_argument("--plan-items", default="")
    parser.add_argument("--primitive-scope", choices=("legacy_v1", "all"), default="legacy_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_snapshot(
        Path(args.project_root),
        pytest_summary=str(args.pytest_summary),
        pytest_command=str(args.pytest_command),
        snapshot_version=str(args.snapshot_version),
        evidence_context={
            "milestone": str(args.milestone),
            "plan_items": [
                item.strip() for item in str(args.plan_items).split(",") if item.strip()
            ],
        },
        primitive_scope=str(args.primitive_scope),
    )
    output = Path(args.output).resolve()
    write_snapshot(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "semantic_snapshot_id": payload["semantic_snapshot_id"],
                "primitive_count": payload["primitive_count"],
                "motif_count": payload["motif_count"],
                "lowering_rule_count": payload["lowering"]["rule_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
