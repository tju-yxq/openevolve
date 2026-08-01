#!/usr/bin/env python
"""Export the exhaustive canonical DSL search-surface audit bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from equivariant_nas.dsl import (  # noqa: E402
    GroupSpec,
    LanguageVersion,
    core_registry,
    default_canonical_search_surface,
    reference_motif_registry,
    select_active_vocabulary,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def export(repo: Path, output: Path, pytest_summary: str) -> dict:
    primitives = core_registry()
    motifs = reference_motif_registry()
    surface = default_canonical_search_surface(primitives, motifs)
    language = LanguageVersion.from_registries(
        "2.2.0",
        "",
        primitives,
        motifs,
        "2026-07-31T00:00:00+08:00",
        {"purpose": "canonical-search-surface-audit"},
    )
    o3 = select_active_vocabulary(
        language,
        GroupSpec.o3(),
        primitives,
        motifs,
        search_surface=surface,
    )
    so3 = select_active_vocabulary(
        language,
        GroupSpec("SO3", 3),
        primitives,
        motifs,
        search_surface=surface,
    )

    status = _git(repo, "status", "--short")
    identity = {
        "repository": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "worktree_status_line_count": len(status.splitlines()),
        "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "search_surface_version": surface.version,
        "search_surface_hash": surface.content_hash(),
        "language_registry_hash": language.registry_hash(),
    }
    summary = {
        "status": "canonical_search_surface_v1_complete",
        "pytest_summary": pytest_summary,
        "identity": identity,
        "counts": surface.to_dict()["counts"],
        "claims": {
            "all_registry_primitives_classified_exactly_once": True,
            "all_reference_motifs_classified_exactly_once": True,
            "llm_generation_surface_is_smaller_than_execution_registry": True,
            "trusted_completion_adapters_are_separate_from_llm_generation": True,
            "compatibility_and_fusion_entries_cannot_be_newly_inserted_by_search_engine": True,
            "concrete_runtime_versions_do_not_define_novelty_identity": True,
            "unified_canonical_core_operator_implementations_complete": False,
        },
        "next_required_work": [
            "replace versioned realization groups with unified canonical core signatures",
            "decompose fused S2/Grid operators into M4 primitives",
            "use canonical family identity in structural novelty reports",
        ],
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "canonical_search_surface.json", surface.to_dict())
    _write_json(output / "o3_active_vocabulary.json", o3.to_dict())
    _write_json(output / "so3_active_vocabulary.json", so3.to_dict())
    _write_json(output / "code_identity.json", identity)
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser("export-dsl-search-surface-evidence")
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytest-summary", default="")
    args = parser.parse_args()
    summary = export(args.repo.resolve(), args.output.resolve(), args.pytest_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
