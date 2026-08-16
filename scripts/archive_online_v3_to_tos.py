#!/usr/bin/env python
"""Create a content manifest and archive a completed online run to TOS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from equivariant_nas.dsl.online_state import atomic_write_json


def assert_archive_ready(run_root: str | Path) -> dict:
    root = Path(run_root)
    state = json.loads((root / "state" / "state.json").read_text(encoding="utf-8"))
    if state.get("stage") != "ready_for_tos_archive" or state.get("test_evaluated") is not True:
        raise RuntimeError("archive is allowed only after the single final Test and ready_for_tos_archive gate")
    return state


def build_manifest(run_root: str | Path) -> dict:
    root = Path(run_root).resolve()
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and "archive_manifest" not in item.name):
        entries.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"schema_version": "equinas-tos-manifest@1", "file_count": len(entries), "total_bytes": sum(item["size"] for item in entries), "files": entries}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True); parser.add_argument("--tos-prefix", required=True); parser.add_argument("--tos-command", default=os.environ.get("EQUINAS_TOS_UPLOAD_COMMAND", "")); parser.add_argument("--manifest-only", action="store_true"); args = parser.parse_args()
    root = Path(args.run_root).resolve(); state = assert_archive_ready(root); manifest = build_manifest(root)
    manifest_path = root / "archive_manifest.json"; atomic_write_json(manifest_path, manifest)
    if args.manifest_only:
        print(json.dumps({"manifest": str(manifest_path), "file_count": manifest["file_count"]})); return
    if not args.tos_command:
        raise RuntimeError("set --tos-command or EQUINAS_TOS_UPLOAD_COMMAND; credentials must remain outside the repository")
    completed = subprocess.run(args.tos_command.format(source=str(root), destination=args.tos_prefix), shell=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"TOS upload command exited {completed.returncode}")
    state["stage"] = "tos_archived"; state["tos_prefix"] = args.tos_prefix; atomic_write_json(root / "state" / "state.json", state)


if __name__ == "__main__": main()
