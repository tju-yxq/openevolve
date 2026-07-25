#!/usr/bin/env python
"""Migrate only the formal-V1 generation-attempt ceiling after audited exhaustion."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, payload):
    path = Path(path)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return encoded


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def migrate(run_root, search_dir, *, old_attempts, new_attempts, reason):
    root = Path(run_root).resolve()
    search = Path(search_dir).resolve()
    config_path = root / "formal_v1_config.json"
    protocol_path = root / "protocol.json"
    ready_path = root / "READY_TO_LAUNCH.json"
    summary_path = search / "summary.json"
    evolution_path = search / "evolution.jsonl"
    missing = [str(path) for path in (config_path, protocol_path, ready_path, summary_path, evolution_path) if not path.is_file()]
    if missing:
        raise RuntimeError("generation-attempt migration lacks required artifacts: {}".format(missing))
    if int(new_attempts) <= int(old_attempts):
        raise ValueError("generation-attempt ceiling must strictly increase")
    if not str(reason).strip():
        raise ValueError("generation-attempt migration requires a reason")

    summary = read_json(summary_path)
    if summary.get("status") != "generation_attempts_exhausted":
        raise RuntimeError("search is not in generation_attempts_exhausted state")
    if int(summary.get("last_completed_iteration", -1)) < int(old_attempts):
        raise RuntimeError("original generation attempts are not fully recorded")
    if int(summary.get("valid_candidate_count", 0)) >= int(summary.get("valid_candidate_target", 8)):
        raise RuntimeError("candidate cohort is already complete")

    rows = [json.loads(line) for line in evolution_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any((row.get("metrics") or {}).get("test_evaluated") is True for row in rows):
        raise RuntimeError("generation-attempt migration refuses Test-contaminated search history")

    config = read_json(config_path)
    protocol = read_json(protocol_path)
    observed = int(config.get("max_generation_attempts", old_attempts))
    if observed != int(old_attempts):
        raise RuntimeError("frozen config generation-attempt ceiling mismatch")
    old_config_bytes = config_path.read_bytes()
    old_protocol_bytes = protocol_path.read_bytes()
    config["max_generation_attempts"] = int(new_attempts)
    new_config_bytes = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    protocol["max_generation_attempts"] = int(new_attempts)
    protocol["config_sha256"] = sha256(new_config_bytes)
    new_protocol_bytes = json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")

    failures = {}
    for row in rows:
        code = str(row.get("error", "")).split(":", 1)[0]
        if code:
            failures[code] = failures.get(code, 0) + 1
    migration = {
        "kind": "formal_generation_attempt_extension",
        "old_maximum_generation_attempts": int(old_attempts),
        "new_maximum_generation_attempts": int(new_attempts),
        "last_completed_iteration": int(summary["last_completed_iteration"]),
        "valid_candidate_count": int(summary.get("valid_candidate_count", 0)),
        "valid_candidate_target": int(summary.get("valid_candidate_target", 8)),
        "failure_code_counts": failures,
        "reason": str(reason).strip(),
        "training_protocol_changed": False,
        "selection_protocol_changed": False,
        "test_evaluated": False,
        "old_formal_config_sha256": sha256(old_config_bytes),
        "new_formal_config_sha256": sha256(new_config_bytes),
        "old_protocol_sha256": sha256(old_protocol_bytes),
        "new_protocol_sha256": sha256(new_protocol_bytes),
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }
    archive = root / "protocol_migrations"
    archive.mkdir(parents=True, exist_ok=True)
    for label, digest, payload in (
        ("formal_v1_config", migration["old_formal_config_sha256"], old_config_bytes),
        ("protocol", migration["old_protocol_sha256"], old_protocol_bytes),
    ):
        previous = archive / "{}_{}.previous.json".format(label, digest[:12])
        if not previous.exists():
            previous.write_bytes(payload)
    atomic_json(config_path, config)
    atomic_json(protocol_path, protocol)
    ready = read_json(ready_path)
    ready["protocol"] = protocol
    ready.setdefault("protocol_migrations", []).append(migration)
    atomic_json(ready_path, ready)
    with (archive / "migrations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(migration, ensure_ascii=False, sort_keys=True) + "\n")
    return migration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--old-attempts", type=int, required=True)
    parser.add_argument("--new-attempts", type=int, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    result = migrate(
        args.run_root,
        args.search_dir,
        old_attempts=args.old_attempts,
        new_attempts=args.new_attempts,
        reason=args.reason,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
