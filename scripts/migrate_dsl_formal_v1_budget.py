#!/usr/bin/env python
"""Apply an auditable, budget-only migration to an existing formal-V1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, payload):
    path = Path(path)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return encoded


def used_gpu_hours(ledger_path):
    path = Path(ledger_path)
    if not path.exists():
        return 0.0
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sum(float(item.get("gpu_seconds", 0.0)) for item in records) / 3600.0


def migrate(run_root, *, old_hours, new_hours, reason):
    root = Path(run_root).resolve()
    frozen_config_path = root / "formal_v1_config.json"
    protocol_path = root / "protocol.json"
    ready_path = root / "READY_TO_LAUNCH.json"
    required = (frozen_config_path, protocol_path, ready_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("formal-V1 budget migration lacks frozen artifacts: {}".format(missing))
    if float(new_hours) <= float(old_hours):
        raise ValueError("budget migration must strictly increase the GPU-hour ceiling")
    if not str(reason).strip():
        raise ValueError("budget migration requires a non-empty reason")

    frozen_config = read_json(frozen_config_path)
    protocol = read_json(protocol_path)
    ready = read_json(ready_path)
    observed = {
        "formal_v1_config": float(frozen_config.get("gpu_budget_hours", -1.0)),
        "protocol": float(protocol.get("gpu_budget_hours", -1.0)),
    }
    if any(abs(value - float(old_hours)) > 1.0e-12 for value in observed.values()):
        raise RuntimeError("budget migration source ceiling mismatch: {}".format(observed))
    if protocol.get("test_during_search") is not False:
        raise RuntimeError("budget migration refuses a run without explicit Test isolation")

    ledger_path = Path(protocol.get("budget_ledger", root / "budget_ledger.jsonl"))
    used = used_gpu_hours(ledger_path)
    if used > float(old_hours) + 1.0e-12:
        raise RuntimeError("recorded GPU use already exceeds the declared source ceiling")

    old_config_bytes = frozen_config_path.read_bytes()
    old_protocol_bytes = protocol_path.read_bytes()
    frozen_config["gpu_budget_hours"] = float(new_hours)
    new_config_bytes = json.dumps(frozen_config, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    protocol["gpu_budget_hours"] = float(new_hours)
    protocol["config_sha256"] = sha256_bytes(new_config_bytes)
    new_protocol_bytes = json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")

    migration = {
        "kind": "formal_v1_gpu_budget_ceiling_increase",
        "old_gpu_budget_hours": float(old_hours),
        "new_gpu_budget_hours": float(new_hours),
        "used_gpu_hours_at_migration": used,
        "reason": str(reason).strip(),
        "training_protocol_changed": False,
        "selection_protocol_changed": False,
        "test_evaluated": False,
        "old_formal_config_sha256": sha256_bytes(old_config_bytes),
        "new_formal_config_sha256": sha256_bytes(new_config_bytes),
        "old_protocol_sha256": sha256_bytes(old_protocol_bytes),
        "new_protocol_sha256": sha256_bytes(new_protocol_bytes),
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }

    archive = root / "protocol_migrations"
    archive.mkdir(parents=True, exist_ok=True)
    old_config_copy = archive / "formal_v1_config_{}.previous.json".format(migration["old_formal_config_sha256"][:12])
    old_protocol_copy = archive / "protocol_{}.previous.json".format(migration["old_protocol_sha256"][:12])
    if not old_config_copy.exists():
        old_config_copy.write_bytes(old_config_bytes)
    if not old_protocol_copy.exists():
        old_protocol_copy.write_bytes(old_protocol_bytes)

    atomic_json(frozen_config_path, frozen_config)
    atomic_json(protocol_path, protocol)
    ready["protocol"] = protocol
    ready.setdefault("protocol_migrations", []).append(migration)
    atomic_json(ready_path, ready)
    with (archive / "migrations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(migration, ensure_ascii=False, sort_keys=True) + "\n")
    return migration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--old-hours", type=float, required=True)
    parser.add_argument("--new-hours", type=float, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    migration = migrate(
        args.run_root,
        old_hours=args.old_hours,
        new_hours=args.new_hours,
        reason=args.reason,
    )
    print(json.dumps(migration, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
