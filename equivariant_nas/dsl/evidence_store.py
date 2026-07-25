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
from .motifs import MotifRegistry
from .patch import TypedPatch
from .registry import PrimitiveRegistry
from .rewrites import RewriteStep
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
CREATE TABLE IF NOT EXISTS rewrite_runs (
    rewrite_run_id TEXT PRIMARY KEY,
    architecture_id TEXT NOT NULL REFERENCES candidate_programs(architecture_id),
    compiler_version TEXT NOT NULL,
    rewrite_registry_hash TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS motif_occurrences (
    occurrence_id TEXT PRIMARY KEY,
    architecture_id TEXT NOT NULL REFERENCES candidate_programs(architecture_id),
    lineage_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    language_registry_hash TEXT NOT NULL,
    rewrite_registry_hash TEXT NOT NULL,
    subgraph_hash TEXT NOT NULL,
    node_ids_json TEXT NOT NULL,
    boundary_json TEXT NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS motif_proposals (
    proposal_id TEXT PRIMARY KEY,
    motif_hash TEXT NOT NULL,
    parent_language_version TEXT NOT NULL,
    language_registry_hash TEXT NOT NULL,
    rewrite_registry_hash TEXT NOT NULL,
    canonical_form_hash TEXT NOT NULL,
    source_architecture_ids_json TEXT NOT NULL,
    proposal_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS language_replay_runs (
    replay_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL REFERENCES motif_proposals(proposal_id),
    architecture_id TEXT NOT NULL REFERENCES candidate_programs(architecture_id),
    occurrence_id TEXT NOT NULL REFERENCES motif_occurrences(occurrence_id),
    before_semantic_id TEXT NOT NULL,
    after_semantic_id TEXT NOT NULL,
    passed INTEGER NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS motif_admission_runs (
    admission_run_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL REFERENCES motif_proposals(proposal_id),
    boundary_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    reasons_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_architecture ON evaluations(architecture_id, split, fidelity_steps);
CREATE INDEX IF NOT EXISTS idx_prompt_architecture ON prompt_runs(architecture_id, role);
CREATE INDEX IF NOT EXISTS idx_completion_architecture ON completion_runs(architecture_id, status);
CREATE INDEX IF NOT EXISTS idx_rewrite_architecture ON rewrite_runs(architecture_id, compiler_version);
CREATE INDEX IF NOT EXISTS idx_motif_occurrence_architecture ON motif_occurrences(architecture_id, subgraph_hash);
CREATE INDEX IF NOT EXISTS idx_motif_proposal_status ON motif_proposals(parent_language_version, status);
CREATE INDEX IF NOT EXISTS idx_language_replay_proposal ON language_replay_runs(proposal_id, passed);
CREATE INDEX IF NOT EXISTS idx_motif_admission_proposal ON motif_admission_runs(proposal_id, accepted);
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

    def register_language(self, language: LanguageVersion, motifs: Optional[MotifRegistry] = None) -> None:
        if language.motif_names and motifs is None:
            raise DSLValidationError([
                Diagnostic("E_STORE_009", "language snapshots with motifs require the complete motif registry")
            ])
        if motifs is not None:
            missing = set(language.motif_names) - set(motifs.names())
            if missing:
                raise DSLValidationError([
                    Diagnostic("E_STORE_010", "motif registry cannot materialize the language snapshot", details={"missing": sorted(missing)})
                ])
            drifted = {
                name: {
                    "expected": language.motif_hashes[name],
                    "actual": motifs.resolve(name).content_hash(),
                }
                for name in language.motif_names
                if name in language.motif_hashes and language.motif_hashes[name] != motifs.resolve(name).content_hash()
            }
            if drifted:
                raise DSLValidationError([
                    Diagnostic("E_STORE_011", "motif definitions differ from the frozen language hashes", details={"motifs": drifted})
                ])
        record = {
            "version": language.version,
            "parent_version": language.parent_version,
            "primitive_names": list(language.primitive_names),
            "motif_names": list(language.motif_names),
            "frozen_at": language.frozen_at,
            "metadata": dict(language.metadata),
            "primitive_hashes": dict(language.primitive_hashes),
            "motif_hashes": dict(language.motif_hashes),
            "motif_definitions": {
                name: motifs.resolve(name).to_dict()
                for name in language.motif_names
            } if motifs is not None else {},
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO language_versions VALUES (?, ?, ?, ?, ?)",
                (language.version, language.parent_version, language.registry_hash(), _json(record), _now()),
            )

    def get_language_snapshot(self, version: str) -> Optional[Tuple[LanguageVersion, MotifRegistry]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT registry_hash, record_json FROM language_versions WHERE version=?",
                (version,),
            ).fetchone()
        if row is None:
            return None
        record = json.loads(row[1])
        language = LanguageVersion(
            str(record["version"]),
            str(record["parent_version"]),
            tuple(str(item) for item in record.get("primitive_names", ())),
            tuple(str(item) for item in record.get("motif_names", ())),
            str(record["frozen_at"]),
            dict(record.get("metadata", {})),
            dict(record.get("primitive_hashes", {})),
            dict(record.get("motif_hashes", {})),
        )
        if language.registry_hash() != row[0]:
            raise DSLValidationError([
                Diagnostic("E_STORE_007", "stored language record does not match its registry hash", actual=version)
            ])
        motifs = MotifRegistry.from_dict(record.get("motif_definitions", {}))
        missing = set(language.motif_names) - set(motifs.names())
        if missing:
            raise DSLValidationError([
                Diagnostic(
                    "E_STORE_008",
                    "stored language snapshot lacks complete motif definitions",
                    details={"missing": sorted(missing)},
                )
            ])
        return language, motifs

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
        rewrite_trace: Sequence[RewriteStep] = (),
        rewrite_registry_hash: str = "",
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
        if rewrite_registry_hash:
            self.add_rewrite_run(
                architecture_id_value,
                compiler_version,
                rewrite_registry_hash,
                rewrite_trace,
            )
        return run_id

    def add_rewrite_run(
        self,
        architecture_id_value: str,
        compiler_version: str,
        rewrite_registry_hash: str,
        trace: Sequence[RewriteStep],
    ) -> str:
        trace_payload = [item.to_dict() for item in trace]
        identity = {
            "architecture_id": architecture_id_value,
            "compiler_version": compiler_version,
            "rewrite_registry_hash": rewrite_registry_hash,
            "trace": trace_payload,
        }
        rewrite_run_id = _content_id("rewrite", identity)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO rewrite_runs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    rewrite_run_id,
                    architecture_id_value,
                    compiler_version,
                    rewrite_registry_hash,
                    _json(trace_payload),
                    _now(),
                ),
            )
        return rewrite_run_id

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

    def add_motif_occurrence(self, occurrence) -> str:
        payload = occurrence.to_dict()
        boundary = {
            "inputs": payload["boundary_inputs"],
            "outputs": payload["boundary_outputs"],
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO motif_occurrences
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    occurrence.occurrence_id,
                    occurrence.architecture_id,
                    occurrence.lineage_id,
                    occurrence.task_id,
                    occurrence.language_registry_hash,
                    occurrence.rewrite_registry_hash,
                    occurrence.topology_hash,
                    _json(list(occurrence.node_ids)),
                    _json(boundary),
                    _json(payload),
                    _now(),
                ),
            )
        return occurrence.occurrence_id

    def add_motif_proposal(
        self,
        proposal,
        *,
        parent_language_version: str,
        language_registry_hash: str,
        rewrite_registry_hash: str,
        status: str = "proposed",
    ) -> str:
        allowed = {"proposed", "rejected", "admissible", "published"}
        if status not in allowed:
            raise DSLValidationError([
                Diagnostic("E_STORE_006", "unknown motif proposal status", actual=status, details={"allowed": sorted(allowed)})
            ])
        payload = proposal.to_dict()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO motif_proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(proposal_id) DO UPDATE SET status=excluded.status
                """,
                (
                    proposal.proposal_id,
                    proposal.motif.content_hash(),
                    parent_language_version,
                    language_registry_hash,
                    rewrite_registry_hash,
                    proposal.canonical_form_hash,
                    _json(list(proposal.source_architecture_ids)),
                    _json(payload),
                    status,
                    _now(),
                ),
            )
        return proposal.proposal_id

    def add_language_replay(self, result) -> str:
        payload = result.to_dict()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO language_replay_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.replay_id,
                    result.proposal_id,
                    result.architecture_id,
                    result.occurrence_id,
                    result.before_semantic_id,
                    result.after_semantic_id,
                    int(result.passed),
                    _json(payload),
                    _now(),
                ),
            )
        return result.replay_id

    def add_motif_admission(self, proposal_id: str, boundary_id: str, decision, evidence) -> str:
        payload = {
            "proposal_id": proposal_id,
            "boundary_id": boundary_id,
            "decision": decision.to_dict(),
            "evidence": evidence.to_dict(),
        }
        admission_run_id = _content_id("motif-admission", payload)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO motif_admission_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    admission_run_id,
                    proposal_id,
                    boundary_id,
                    decision.policy_version,
                    int(decision.accepted),
                    _json(list(decision.reasons)),
                    _json(evidence.to_dict()),
                    _now(),
                ),
            )
        return admission_run_id

    def get_motif_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT proposal_json, status, parent_language_version, language_registry_hash,
                       rewrite_registry_hash, created_at
                FROM motif_proposals WHERE proposal_id=?
                """,
                (proposal_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "proposal_id": proposal_id,
            "proposal": json.loads(row[0]),
            "status": row[1],
            "parent_language_version": row[2],
            "language_registry_hash": row[3],
            "rewrite_registry_hash": row[4],
            "created_at": row[5],
        }

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
