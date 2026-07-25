"""Append-only SQLite evidence store for programs, prompts, proofs, and evaluations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .ast import ArchitectureProgram
from .canonicalize import architecture_id, canonical_json
from .completion import CompletionDistance, HoleSink, TypedHole
from .diagnostics import DSLValidationError, Diagnostic
from .inference import InferenceResult
from .language import LanguageVersion, VocabularyDecision
from .patch import TypedPatch
from .registry import PrimitiveRegistry
from .task import TaskContract


_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS task_contracts (
    task_hash TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    contract_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS language_versions (
    version TEXT PRIMARY KEY,
    parent_version TEXT NOT NULL,
    registry_hash TEXT NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate_programs (
    architecture_id TEXT PRIMARY KEY,
    language_version TEXT NOT NULL,
    task_hash TEXT NOT NULL REFERENCES task_contracts(task_hash),
    canonical_program_json TEXT NOT NULL,
    source_program_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS patches (
    patch_id TEXT PRIMARY KEY,
    parent_architecture_id TEXT NOT NULL REFERENCES candidate_programs(architecture_id),
    child_architecture_id TEXT,
    patch_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS compiler_runs (
    run_id TEXT PRIMARY KEY,
    architecture_id TEXT NOT NULL REFERENCES candidate_programs(architecture_id),
    compiler_version TEXT NOT NULL,
    status TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL,
    obligations_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id TEXT PRIMARY KEY,
    architecture_id TEXT NOT NULL REFERENCES candidate_programs(architecture_id),
    split TEXT NOT NULL,
    fidelity_steps INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    resources_json TEXT NOT NULL,
    final_audit INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prompt_runs (
    prompt_run_id TEXT PRIMARY KEY,
    architecture_id TEXT NOT NULL REFERENCES candidate_programs(architecture_id),
    role TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_json TEXT NOT NULL,
    response_text TEXT NOT NULL,
    vocabulary_json TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL,
    visible_splits_json TEXT NOT NULL,
    token_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS completion_runs (
    completion_id TEXT PRIMARY KEY,
    architecture_id TEXT NOT NULL REFERENCES candidate_programs(architecture_id),
    language_registry_hash TEXT NOT NULL,
    hole_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    materialized_patch_json TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_architecture ON evaluations(architecture_id, split, fidelity_steps);
CREATE INDEX IF NOT EXISTS idx_prompt_architecture ON prompt_runs(architecture_id, role);
CREATE INDEX IF NOT EXISTS idx_completion_architecture ON completion_runs(architecture_id, status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()[:20]
    return "{}:{}".format(prefix, digest)


class EvidenceStore:
    def __init__(self, path: str):
        self.path = str(Path(path))
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def register_task(self, task: TaskContract) -> str:
        task_hash = task.content_hash()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO task_contracts VALUES (?, ?, ?, ?)",
                (task_hash, task.task_id, _json(task.to_dict()), _now()),
            )
        return task_hash

    def register_language(self, language: LanguageVersion) -> None:
        record = {
            "version": language.version,
            "parent_version": language.parent_version,
            "primitive_names": list(language.primitive_names),
            "motif_names": list(language.motif_names),
            "frozen_at": language.frozen_at,
            "metadata": dict(language.metadata),
            "primitive_hashes": dict(language.primitive_hashes),
            "motif_hashes": dict(language.motif_hashes),
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO language_versions VALUES (?, ?, ?, ?, ?)",
                (language.version, language.parent_version, language.registry_hash(), _json(record), _now()),
            )

    def add_candidate(
        self,
        program: ArchitectureProgram,
        task: TaskContract,
        registry: PrimitiveRegistry,
    ) -> str:
        task_hash = self.register_task(task)
        candidate_id = architecture_id(program, registry, task_contract_hash=task_hash)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO candidate_programs VALUES (?, ?, ?, ?, ?, ?)",
                (candidate_id, program.language_version, task_hash, canonical_json(program, registry), _json(program.to_dict()), _now()),
            )
        return candidate_id

    def add_compiled_candidate(self, artifact, task: TaskContract) -> str:
        """Store source and expanded forms under the compiler's semantic id."""

        task_hash = self.register_task(task)
        candidate_id = artifact.architecture_id
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO candidate_programs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    candidate_id,
                    artifact.source_program.language_version,
                    task_hash,
                    canonical_json(artifact.expanded_program, None),
                    _json(artifact.source_program.to_dict()),
                    _now(),
                ),
            )
        return candidate_id

    def add_patch(self, patch: TypedPatch, *, child_architecture_id: str = "") -> str:
        payload = patch.to_dict()
        patch_id = _content_id("patch", payload)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO patches VALUES (?, ?, ?, ?, ?)",
                (patch_id, patch.parent_architecture_id, child_architecture_id or None, _json(payload), _now()),
            )
        return patch_id

    def add_compiler_run(
        self,
        architecture_id_value: str,
        compiler_version: str,
        status: str,
        *,
        diagnostics: Sequence[Diagnostic] = (),
        inference: Optional[InferenceResult] = None,
    ) -> str:
        payload = {
            "architecture_id": architecture_id_value,
            "compiler_version": compiler_version,
            "status": status,
            "diagnostics": [item.to_dict() for item in diagnostics],
            "obligations": [item.to_dict() for item in inference.obligations] if inference else [],
            "nonce": _now(),
        }
        run_id = _content_id("compile", payload)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO compiler_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, architecture_id_value, compiler_version, status, _json(payload["diagnostics"]), _json(payload["obligations"]), payload["nonce"]),
            )
        return run_id

    def add_completion_run(
        self,
        architecture_id_value: str,
        language_registry_hash: str,
        hole: TypedHole,
        result: CompletionDistance,
        *,
        sink: Optional[HoleSink] = None,
        materialized_patch: Optional[TypedPatch] = None,
        status: str,
    ) -> str:
        allowed_statuses = {"reachable", "unreachable", "materialized", "applied", "rejected"}
        if status not in allowed_statuses:
            raise DSLValidationError([
                Diagnostic("E_STORE_004", "unknown completion status", actual=status, details={"allowed": sorted(allowed_statuses)})
            ])
        if status in {"materialized", "applied"} and materialized_patch is None:
            raise DSLValidationError([
                Diagnostic("E_STORE_005", "materialized completion status requires a typed patch", actual=status)
            ])
        request = {"hole": hole.to_dict(), "sink": sink.to_dict() if sink is not None else None}
        identity = {
            "architecture_id": architecture_id_value,
            "language_registry_hash": language_registry_hash,
            "request": request,
        }
        completion_id = _content_id("completion", identity)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO completion_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(completion_id) DO UPDATE SET
                    result_json=excluded.result_json,
                    materialized_patch_json=excluded.materialized_patch_json,
                    status=excluded.status
                """,
                (
                    completion_id,
                    architecture_id_value,
                    language_registry_hash,
                    _json(request),
                    _json(result.to_dict()),
                    _json(materialized_patch.to_dict()) if materialized_patch is not None else None,
                    status,
                    _now(),
                ),
            )
        return completion_id

    def get_completion_run(self, completion_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT architecture_id, language_registry_hash, hole_json, result_json,
                       materialized_patch_json, status, created_at
                FROM completion_runs WHERE completion_id=?
                """,
                (completion_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "completion_id": completion_id,
            "architecture_id": row[0],
            "language_registry_hash": row[1],
            "request": json.loads(row[2]),
            "result": json.loads(row[3]),
            "materialized_patch": json.loads(row[4]) if row[4] is not None else None,
            "status": row[5],
            "created_at": row[6],
        }

    def add_evaluation(
        self,
        architecture_id_value: str,
        *,
        split: str,
        fidelity_steps: int,
        seed: int,
        metrics: Mapping[str, Any],
        resources: Mapping[str, Any],
        final_audit: bool = False,
    ) -> str:
        if split not in ("train", "validation", "test"):
            raise DSLValidationError([Diagnostic("E_STORE_001", "unknown evaluation split", actual=split)])
        if split == "test" and not final_audit:
            raise DSLValidationError([Diagnostic("E_STORE_002", "test evaluations require an explicit final-audit flag")])
        payload = {
            "architecture_id": architecture_id_value,
            "split": split,
            "fidelity_steps": int(fidelity_steps),
            "seed": int(seed),
            "metrics": dict(metrics),
            "resources": dict(resources),
            "final_audit": bool(final_audit),
            "nonce": _now(),
        }
        evaluation_id = _content_id("eval", payload)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (evaluation_id, architecture_id_value, split, int(fidelity_steps), int(seed), _json(metrics), _json(resources), int(final_audit), payload["nonce"]),
            )
        return evaluation_id

    def add_prompt_run(
        self,
        architecture_id_value: str,
        *,
        role: str,
        model: str,
        prompt: Mapping[str, str],
        response_text: str,
        vocabulary: VocabularyDecision,
        evidence_ids: Sequence[str],
        visible_splits: Sequence[str],
        token_usage: Mapping[str, int],
    ) -> str:
        if "test" in visible_splits:
            raise DSLValidationError([Diagnostic("E_STORE_003", "test evidence cannot be attached to an LLM prompt")])
        payload = {
            "architecture_id": architecture_id_value,
            "role": role,
            "model": model,
            "prompt": dict(prompt),
            "response": response_text,
            "vocabulary": vocabulary.to_dict(),
            "evidence_ids": list(evidence_ids),
            "visible_splits": list(visible_splits),
            "token_usage": dict(token_usage),
            "nonce": _now(),
        }
        prompt_run_id = _content_id("prompt", payload)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO prompt_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    prompt_run_id,
                    architecture_id_value,
                    role,
                    model,
                    _json(prompt),
                    response_text,
                    _json(vocabulary.to_dict()),
                    _json(list(evidence_ids)),
                    _json(list(visible_splits)),
                    _json(token_usage),
                    payload["nonce"],
                ),
            )
        return prompt_run_id

    def candidate_generation_evidence(self, architecture_id_value: str) -> Tuple[Dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT evaluation_id, split, fidelity_steps, seed, metrics_json, resources_json FROM evaluations WHERE architecture_id=? AND split IN ('train','validation') ORDER BY created_at",
                (architecture_id_value,),
            ).fetchall()
        return tuple(
            {
                "evaluation_id": row[0],
                "split": row[1],
                "fidelity_steps": row[2],
                "seed": row[3],
                "metrics": json.loads(row[4]),
                "resources": json.loads(row[5]),
            }
            for row in rows
        )

    def candidate_exists(self, architecture_id_value: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM candidate_programs WHERE architecture_id=?",
                (architecture_id_value,),
            ).fetchone()
        return row is not None
