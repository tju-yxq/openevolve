"""Resumable orchestration core for one-child-at-a-time online evolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .fidelity_aware_parent_sampler import FidelityAwareParentSampler
from .online_promotion import fidelity_trust_report, select_20k_to_80k, select_80k_to_250k
from .online_state import OnlineEvolutionState, atomic_write_json

Generator = Callable[[int, list[Mapping[str, Any]], Mapping[str, Any] | None], Mapping[str, Any]]
Trainer = Callable[[Mapping[str, Any], int, str], Mapping[str, Any]]
TestEvaluator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class OnlineV3Controller:
    def __init__(self, run_root: str | Path, protocol, *, generator: Generator, trainer: Trainer, test_evaluator: TestEvaluator):
        self.root = Path(run_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.protocol = protocol
        search = protocol.raw["search"]
        self.state = OnlineEvolutionState(
            self.root / "state" / "state.json", protocol_hash=protocol.content_hash,
            valid_target=int(search["valid_child_target"]), maximum_attempts=int(search["maximum_generation_attempts"]),
        )
        self.generator, self.trainer, self.test_evaluator = generator, trainer, test_evaluator

    def run(self, *, stop_after_stage: str = "ready_for_tos_archive") -> Mapping[str, Any]:
        while True:
            stage = self.state.data["stage"]
            if stage == stop_after_stage or stage in {"tos_archived", "complete"}:
                return self.state.data
            getattr(self, "_" + stage)()

    def _evolve_20k(self) -> None:
        target = int(self.state.data["valid_child_target"])
        while len(self.state.data["completed_20k"]) < target:
            attempt = int(self.state.data["generation_attempts"]) + 1
            population = list(self.state.data["completed_20k"])
            pending = self.state.data.get("current_candidate")
            if pending:
                candidate = dict(pending)
            else:
                sampler = FidelityAwareParentSampler(self.protocol.raw["parent_sampling"], seed=int(self.protocol.raw["seed"]) + attempt)
                selection = sampler.sample(population, current_island=(attempt - 1) % int(self.protocol.raw["search"]["num_islands"])) if population else None
                parent = selection.candidate if selection else None
                try:
                    candidate = dict(self.generator(attempt, population, parent))
                except Exception as error:
                    self.state.add_generation_attempt({"architecture_id": "", "valid": False, "endpoint_step": 0, "test_evaluated": False, "error_type": type(error).__name__, "error": str(error)[:4000]})
                    continue
                candidate.update({
                    "island": (attempt - 1) % int(self.protocol.raw["search"]["num_islands"]),
                    "parent_sampling_strategy": selection.strategy if selection else "initial_seed",
                    "parent_sampling_eligible_ids": list(selection.eligible_architecture_ids) if selection else [],
                    "parent_sampling_weights": list(selection.normalized_weights) if selection else [],
                })
                self.state.set_current_candidate(candidate)
            architecture_id = str(candidate.get("architecture_id", ""))
            if not architecture_id:
                self.state.add_generation_attempt({**candidate, "valid": False, "endpoint_step": 0, "test_evaluated": False})
                continue
            if architecture_id in self.state.completed_ids(20000):
                self.state.add_generation_attempt({**candidate, "valid": False, "endpoint_step": 0, "test_evaluated": False, "failure_stage": "duplicate"})
                continue
            result = dict(self.trainer(candidate, 20000, ""))
            merged = self._fidelity_record({**candidate, **result, "architecture_id": architecture_id}, 20000)
            self.state.add_generation_attempt(merged)
        self.state.transition("evolve_20k", "freeze_20k_selection")

    def _freeze_20k_selection(self) -> None:
        snapshot = select_20k_to_80k(self.state.data["completed_20k"], protocol_hash=self.protocol.content_hash, seed=int(self.protocol.raw["seed"]))
        atomic_write_json(self.root / "selections" / "selection_20k_to_80k.json", snapshot)
        self.state.transition("freeze_20k_selection", "train_80k", promoted_to_80k=snapshot["selected"], selection_20k_sha256=snapshot["snapshot_sha256"])

    def _train_80k(self) -> None:
        done = self.state.completed_ids(80000)
        for candidate in self.state.data["promoted_to_80k"]:
            if str(candidate["architecture_id"]) in done:
                continue
            checkpoint = _validated_checkpoint(candidate, expected_step=20000)
            result = self.trainer(candidate, 80000, checkpoint)
            self.state.record_training(80000, self._fidelity_record({**candidate, **result, "architecture_id": candidate["architecture_id"]}, 80000))
        if len(self.state.data["completed_80k"]) != 15:
            raise RuntimeError("80k stage did not complete exactly 15 candidates")
        self.state.transition("train_80k", "freeze_80k_selection")

    def _freeze_80k_selection(self) -> None:
        snapshot = select_80k_to_250k(self.state.data["completed_80k"], protocol_hash=self.protocol.content_hash)
        trust = fidelity_trust_report(self.state.data["completed_20k"], self.state.data["completed_80k"])
        atomic_write_json(self.root / "selections" / "selection_80k_to_250k.json", snapshot)
        atomic_write_json(self.root / "reports" / "trust_20k_to_80k.json", trust)
        self.state.transition("freeze_80k_selection", "train_250k", promoted_to_250k=snapshot["selected"], selection_80k_sha256=snapshot["snapshot_sha256"], trust_20k_to_80k=trust)

    def _train_250k(self) -> None:
        done = self.state.completed_ids(250000)
        for candidate in self.state.data["promoted_to_250k"]:
            if str(candidate["architecture_id"]) in done:
                continue
            checkpoint = _validated_checkpoint(candidate, expected_step=80000)
            result = self.trainer(candidate, 250000, checkpoint)
            self.state.record_training(250000, self._fidelity_record({**candidate, **result, "architecture_id": candidate["architecture_id"]}, 250000))
        if len(self.state.data["completed_250k"]) != 10:
            raise RuntimeError("250k stage did not complete exactly 10 candidates")
        self.state.transition("train_250k", "freeze_winner")

    def _freeze_winner(self) -> None:
        ranked = sorted(self.state.data["completed_250k"], key=lambda item: (float(item["validation_alpha_mae"]), str(item["architecture_id"])))
        winner = dict(ranked[0])
        atomic_write_json(self.root / "selections" / "winner_250k_validation.json", winner)
        self.state.transition("freeze_winner", "evaluate_test", winner=winner)

    def _evaluate_test(self) -> None:
        result = dict(self.test_evaluator(self.state.data["winner"]))
        self.state.record_test(result)
        self.state.transition("evaluate_test", "ready_for_tos_archive")

    def _fidelity_record(self, record: Mapping[str, Any], endpoint: int) -> dict[str, Any]:
        value = dict(record)
        checkpoint = str(value.get("checkpoint") or value.get("checkpoint_last") or "")
        checkpoint_global_step = int(value.get("checkpoint_global_step") or value.get("endpoint_step") or endpoint)
        value["checkpoint"] = checkpoint
        value["checkpoint_last"] = checkpoint
        value["checkpoint_global_step"] = checkpoint_global_step
        metrics = dict(value.get("metrics_by_fidelity", {}))
        metrics[str(endpoint)] = {
            "validation_alpha_mae": float(value["validation_alpha_mae"]),
            "checkpoint": checkpoint,
            "endpoint_step": endpoint,
            "training_data": str(value.get("training_data", "")),
        }
        dataset = self.protocol.raw["dataset"]
        value.update({
            "protocol_hash": self.protocol.content_hash,
            "highest_completed_fidelity": endpoint,
            "metrics_by_fidelity": metrics,
            "dataset_id": dataset["dataset_id"],
            "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
            "quarter_subset_sha256": dataset["quarter_subset_sha256"],
            "equivariance_contract_sha256": dataset["equivariance_contract_sha256"],
        })
        # Full compiler/lowering evidence remains in each pipeline run's result.json.
        # Keeping it again in the controller state makes generation IPC and every
        # atomic state update unnecessarily huge.
        for bulky_key in ("compiler_obligations", "lowering_plan", "backend_support", "traceback"):
            value.pop(bulky_key, None)
        return value


def _validated_checkpoint(candidate: Mapping[str, Any], *, expected_step: int) -> str:
    checkpoint = str(candidate.get("checkpoint", ""))
    if not checkpoint or int(candidate.get("checkpoint_global_step", candidate.get("endpoint_step", 0))) != expected_step:
        raise RuntimeError(f"candidate checkpoint is not an exact step-{expected_step} continuation boundary")
    return checkpoint
