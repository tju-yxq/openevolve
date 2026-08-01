"""Immutable runtime identity for formal DSL experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity(root: Path) -> Dict[str, Any]:
    if not (root / ".git").exists():
        return {"root": str(root), "commit": "", "dirty": None}
    def run(*args):
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
    return {
        "root": str(root),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def build_runtime_manifest(
    *,
    project_root: str,
    equiformer_root: str,
    program_path: str,
    architecture_id: str,
    lowering_plan: Mapping[str, Any],
    task_contract_hash: str,
    data_path: str,
    train_subset_sha256: str,
    critical_files: Iterable[str],
    equiformer_v3_root: str = "",
) -> Dict[str, Any]:
    project = Path(project_root).resolve()
    files = {}
    for item in critical_files:
        path = (project / item).resolve()
        files[item] = _sha256(path) if path.is_file() else "missing"
    manifest = {
        "manifest_version": "formal-v1-runtime-manifest@1",
        "program_sha256": _sha256(Path(program_path).resolve()),
        "program_id": architecture_id,
        "lowering_plan": dict(lowering_plan),
        "task_contract_hash": task_contract_hash,
        "project_git": _git_identity(project),
        "equiformer_git": _git_identity(Path(equiformer_root).resolve()),
        "equiformer_v3_git": (
            _git_identity(Path(equiformer_v3_root).resolve())
            if equiformer_v3_root
            else {"root": "", "commit": "", "dirty": None}
        ),
        "critical_file_sha256": files,
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "dependencies": {
            "torch": _module_version("torch"),
            "e3nn": _module_version("e3nn"),
            "timm": _module_version("timm"),
        },
        "data": {
            "path": str(Path(data_path).resolve()),
            "train_subset_sha256": train_subset_sha256,
        },
    }
    executable_payload = {
        "program_id": architecture_id,
        "lowering_plan": dict(lowering_plan),
        "project_commit": manifest["project_git"]["commit"],
        "equiformer_commit": manifest["equiformer_git"]["commit"],
        "equiformer_v3_commit": manifest["equiformer_v3_git"]["commit"],
        "critical_file_sha256": files,
    }
    manifest["executable_id"] = hashlib.sha256(
        json.dumps(executable_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return manifest


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
