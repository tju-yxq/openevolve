"""Crash-safe state machine for the online 20k/80k/250k experiment."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


STAGES = ("evolve_20k", "freeze_20k_selection", "train_80k", "freeze_80k_selection", "train_250k", "freeze_winner", "evaluate_test", "ready_for_tos_archive", "tos_archived", "complete")


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class OnlineEvolutionState:
    def __init__(self, path: str | Path, *, protocol_hash: str, valid_target: int = 60, maximum_attempts: int = 180):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data["protocol_hash"] != protocol_hash:
                raise RuntimeError("resume protocol hash mismatch")
        else:
            self.data = {
                "schema_version": "online-v3-state@1", "protocol_hash": protocol_hash,
                "stage": "evolve_20k", "valid_child_target": valid_target,
                "maximum_generation_attempts": maximum_attempts, "generation_attempts": 0,
                "completed_20k": [], "completed_80k": [], "completed_250k": [],
                "test_evaluated": False, "created_at": _now(), "updated_at": _now(),
            }
            self.save()

    def save(self) -> None:
        self.data["updated_at"] = _now()
        atomic_write_json(self.path, self.data)

    def add_generation_attempt(self, record: Mapping[str, Any]) -> bool:
        if self.data["stage"] != "evolve_20k":
            raise RuntimeError("generation is closed")
        self.data["generation_attempts"] += 1
        if self.data["generation_attempts"] > self.data["maximum_generation_attempts"]:
            raise RuntimeError("maximum generation attempts exhausted")
        accepted = bool(record.get("valid")) and int(record.get("endpoint_step", 0)) == 20000 and record.get("test_evaluated") is False
        if accepted and str(record.get("protocol_hash", "")) != self.data["protocol_hash"]:
            raise ValueError("candidate protocol hash mismatch")
        ids = {str(item["architecture_id"]) for item in self.data["completed_20k"]}
        if accepted and str(record.get("architecture_id", "")) not in ids:
            self.data["completed_20k"].append(dict(record))
        self.data.pop("current_candidate", None)
        self.save()
        return accepted

    def set_current_candidate(self, candidate: Mapping[str, Any]) -> None:
        if self.data["stage"] != "evolve_20k":
            raise RuntimeError("cannot set a pending candidate after generation closes")
        self.data["current_candidate"] = dict(candidate)
        self.save()

    def completed_ids(self, fidelity: int) -> set[str]:
        return {str(item["architecture_id"]) for item in self.data[f"completed_{fidelity // 1000}k"]}

    def record_training(self, fidelity: int, record: Mapping[str, Any]) -> bool:
        expected_stage = {80000: "train_80k", 250000: "train_250k"}[fidelity]
        if self.data["stage"] != expected_stage:
            raise RuntimeError(f"cannot record {fidelity} during {self.data['stage']}")
        if int(record.get("endpoint_step", 0)) != fidelity or record.get("test_evaluated") is not False:
            raise ValueError("invalid or test-leaking training result")
        if str(record.get("protocol_hash", "")) != self.data["protocol_hash"]:
            raise ValueError("candidate protocol hash mismatch")
        key = f"completed_{fidelity // 1000}k"
        architecture_id = str(record["architecture_id"])
        if architecture_id in self.completed_ids(fidelity):
            return False
        self.data[key].append(dict(record)); self.save(); return True

    def transition(self, expected: str, next_stage: str, **updates: Any) -> None:
        if self.data["stage"] != expected or next_stage not in STAGES:
            raise RuntimeError(f"invalid transition {self.data['stage']} -> {next_stage}")
        if STAGES.index(next_stage) != STAGES.index(expected) + 1:
            raise RuntimeError("state transitions cannot skip stages")
        self.data.update(updates); self.data["stage"] = next_stage; self.save()

    def record_test(self, result: Mapping[str, Any]) -> None:
        if self.data["stage"] != "evaluate_test" or self.data.get("test_evaluated"):
            raise RuntimeError("Test may run exactly once after winner freeze")
        if str(result.get("architecture_id", "")) != str(self.data.get("winner", {}).get("architecture_id", "")):
            raise ValueError("Test result is not for the frozen winner")
        self.data["test_result"] = dict(result); self.data["test_evaluated"] = True; self.save()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
