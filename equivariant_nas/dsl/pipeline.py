"""Native evaluation pipeline for typed EvoEquiLang candidates."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..budget import BudgetLedger
from .constants import BASELINE_PARAMETERS
from .compiler import Compiler
from .registry import core_registry
from .reference_motifs import reference_motif_registry
from .serialization import load_program, load_task_contract
from .backends.qm9_model import build_qm9_dsl_model


def _count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _representation_statistics(value_types: Iterable[Any]) -> Dict[str, Any]:
    """Summarize only values carrying an explicit irrep decomposition.

    V3 programs also contain categorical species, topology, index, lattice and
    grid values.  Those values are part of the typed graph but do not expose an
    ``irreps`` field and must not be treated as representation tensors.
    """

    irrep_sets = [
        value_type.irreps
        for value_type in value_types
        if getattr(value_type, "irreps", None) is not None
    ]
    degrees = [irrep.degree for irreps in irrep_sets for _, irrep in irreps]
    higher_order = sum(
        multiplicity * irrep.dimension
        for irreps in irrep_sets
        for multiplicity, irrep in irreps
        if irrep.degree > 0
    )
    total_width = sum(irreps.dimension for irreps in irrep_sets)
    return {
        "lmax": max(degrees, default=0),
        "higher_order_fraction": higher_order / max(total_width, 1),
    }


def _relative_error(actual, expected) -> float:
    import math

    stable_scale = max(float(expected.norm().detach().cpu()), math.sqrt(max(expected.numel(), 1)) * 1.0e-6)
    return float((actual - expected).norm().detach().cpu()) / stable_scale


def _maximum_absolute_error(actual, expected) -> float:
    return float((actual - expected).abs().max().detach().cpu())


def _symmetry_gate_failed(report, relative_threshold: float, absolute_threshold: float) -> bool:
    """Reject material violations while tolerating float32 noise near zero outputs."""

    return (
        float(report["maximum"]) > float(relative_threshold)
        and float(report["maximum_absolute"]) > float(absolute_threshold)
    )


def _observable_symmetry_report(model, batch) -> Dict[str, Any]:
    import torch
    from e3nn import o3

    model.eval()
    with torch.no_grad():
        reference = model(
            f_in=batch.x,
            pos=batch.pos,
            batch=batch.batch,
            node_atom=batch.z,
            edge_d_index=batch.edge_d_index,
            edge_d_attr=batch.edge_d_attr,
        )
        errors = {}
        absolute_errors = {}
        for index in range(4):
            rotation = o3.rand_matrix(dtype=batch.pos.dtype, device=batch.pos.device)
            rotated = model(
                f_in=batch.x,
                pos=batch.pos @ rotation.transpose(0, 1),
                batch=batch.batch,
                node_atom=batch.z,
                edge_d_index=batch.edge_d_index,
                edge_d_attr=batch.edge_d_attr,
            )
            key = "rotation_{:02d}".format(index)
            errors[key] = _relative_error(rotated, reference)
            absolute_errors[key] = _maximum_absolute_error(rotated, reference)
        for index, translation in enumerate(((1.25, -0.75, 0.5), (-0.4, 0.9, 1.1))):
            translated = model(
                f_in=batch.x,
                pos=batch.pos + batch.pos.new_tensor([translation]),
                batch=batch.batch,
                node_atom=batch.z,
                edge_d_index=batch.edge_d_index,
                edge_d_attr=batch.edge_d_attr,
            )
            key = "translation_{:02d}".format(index)
            errors[key] = _relative_error(translated, reference)
            absolute_errors[key] = _maximum_absolute_error(translated, reference)
        graph_nodes = [torch.nonzero(batch.batch == graph_id, as_tuple=False).flatten() for graph_id in torch.unique(batch.batch, sorted=True)]
        permutations = (
            torch.cat([nodes.flip(0) for nodes in graph_nodes]),
            torch.cat([nodes.roll(1) for nodes in graph_nodes]),
        )
        for index, permutation in enumerate(permutations):
            inverse_permutation = torch.empty_like(permutation)
            inverse_permutation[permutation] = torch.arange(
                permutation.numel(), device=permutation.device
            )
            permuted_edge_index = inverse_permutation.index_select(
                0, batch.edge_d_index.reshape(-1)
            ).reshape_as(batch.edge_d_index)
            permuted = model(
                f_in=batch.x.index_select(0, permutation),
                pos=batch.pos.index_select(0, permutation),
                batch=batch.batch.index_select(0, permutation),
                node_atom=batch.z.index_select(0, permutation),
                edge_d_index=permuted_edge_index,
                edge_d_attr=batch.edge_d_attr,
            )
            key = "permutation_{:02d}".format(index)
            errors[key] = _relative_error(permuted, reference)
            absolute_errors[key] = _maximum_absolute_error(permuted, reference)
    return {
        "protocol_version": "formal-v1-symmetry@2",
        "molecule_count": int(reference.shape[0]),
        "errors": errors,
        "absolute_errors": absolute_errors,
        "maximum": max(errors.values()),
        "maximum_absolute": max(absolute_errors.values()),
        "reference_rms": float(reference.float().square().mean().sqrt().cpu()),
    }


def _gradient_health(model, batch, target_mean: float, target_std: float) -> Dict[str, Any]:
    import torch

    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(
        f_in=batch.x,
        pos=batch.pos,
        batch=batch.batch,
        node_atom=batch.z,
        edge_d_index=batch.edge_d_index,
        edge_d_attr=batch.edge_d_attr,
    ).squeeze()
    standardized_target = (batch.y[:, 1] - target_mean) / target_std
    loss = torch.nn.functional.l1_loss(prediction, standardized_target)
    loss.backward()
    gradients = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    finite = bool(gradients) and all(bool(torch.isfinite(item).all()) for item in gradients)
    squared_norm = sum(float(torch.sum(item.float() ** 2).cpu()) for item in gradients)
    maximum = max((float(item.abs().max().cpu()) for item in gradients), default=0.0)
    model.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach().cpu()),
        "global_l2_norm": squared_norm ** 0.5,
        "maximum_absolute_gradient": maximum,
        "gradients_present": bool(gradients),
        "all_finite": finite,
    }


def _numerical_health(model, batch, target_mean: float, target_std: float) -> Dict[str, Any]:
    import torch

    model.eval()
    with torch.no_grad():
        prediction = model(
            f_in=batch.x,
            pos=batch.pos,
            batch=batch.batch,
            node_atom=batch.z,
            edge_d_index=batch.edge_d_index,
            edge_d_attr=batch.edge_d_attr,
        ).squeeze()
    standardized_target = (batch.y[:, 1] - target_mean) / target_std
    report = {
        "all_finite": bool(torch.isfinite(prediction).all()),
        "prediction_mean": float(prediction.float().mean().cpu()),
        "prediction_std": float(prediction.float().std(unbiased=False).cpu()),
        "prediction_rms": float(torch.sqrt(torch.mean(prediction.float() ** 2)).cpu()),
        "prediction_max_abs": float(torch.max(torch.abs(prediction)).cpu()),
        "standardized_batch_mae": float(torch.mean(torch.abs(prediction - standardized_target)).cpu()),
        "limits": {
            "prediction_rms": 50.0,
            "prediction_max_abs": 100.0,
            "standardized_batch_mae": 100.0,
        },
    }
    return report


def _assert_numerical_health(report: Dict[str, Any], gradient: Dict[str, Any]) -> None:
    if not report["all_finite"]:
        raise ValueError("non-finite predictions in pre-training numerical gate")
    for name, limit in report["limits"].items():
        if float(report[name]) > float(limit):
            raise ValueError(
                "pre-training numerical health {} {:.6g} exceeds {:.6g}".format(
                    name, report[name], limit
                )
            )
    if not gradient["all_finite"]:
        raise ValueError("non-finite or missing gradients in pre-training gate")
    if float(gradient["global_l2_norm"]) > 1.0e5:
        raise ValueError("pre-training gradient norm exceeds calibrated safety limit")


def evaluate_dsl_candidate_pipeline(
    program_path: str,
    project_root: str,
    equiformer_root: str,
    data_path: str,
    max_steps: int = 0,
    seed: int = 0,
    parameter_ratio_limit: float = 1.2,
    symmetry_threshold: float = 1.0e-2,
    symmetry_warning_threshold: float = 5.0e-3,
    symmetry_absolute_threshold: float = 1.5e-2,
    symmetry_absolute_warning_threshold: float = 8.0e-3,
    run_symmetry: bool = True,
    gpu_budget_hours: Optional[float] = None,
    resume_checkpoint: str = "",
    batch_size: int = 64,
    train_subset_file: str = "",
    eval_interval_epochs: int = 0,
    data_epoch_origin_step: int = 0,
    allow_data_transition: bool = False,
    resume_model_only: bool = False,
    lr_schedule_origin_step: int = 0,
    equiformer_v2_root: str = "",
    equiformer_v3_root: str = "",
    task_contract_path: str = "",
    allow_experimental_generic_lowering: bool = False,
) -> Dict[str, Any]:
    project = Path(project_root)
    if equiformer_root not in sys.path:
        sys.path.insert(0, equiformer_root)
    program_file = Path(program_path).resolve()
    program = load_program(str(program_file))
    task = load_task_contract(task_contract_path) if task_contract_path else None
    compiler = Compiler(core_registry(), reference_motif_registry())
    artifact = compiler.analyze(program, task)
    is_v3_program = "equiformer_v3_spec" in program.parameters
    if is_v3_program:
        if not equiformer_v3_root:
            raise ValueError("V3 DSL pipeline requires equiformer_v3_root")
        from .backends import E3NNGraphBackend

        v3_backend = E3NNGraphBackend(core_registry(), equiformer_v3_root=equiformer_v3_root)
        support = v3_backend.support_report(program)
        if support.unsupported_nodes or support.composition_errors or support.missing_dependencies:
            raise ValueError("V3 program lacks complete generic Lowering support: {}".format(support.to_dict()))
        lowering = compiler.plan_lowering(program, task, graph_backend=v3_backend)
        formal_ranking_admitted = True
    else:
        lowering = compiler.plan_lowering(program, task)
        formal_ranking_admitted = lowering.mode in {
            "exact_reference",
            "exact_constructor",
            "exact_hybrid",
        }
    architecture_id = artifact.architecture_id
    subset_fingerprint = ""
    if train_subset_file:
        subset_path = Path(train_subset_file).resolve()
        if not subset_path.is_file():
            raise FileNotFoundError(subset_path)
        subset_fingerprint = hashlib.sha256(subset_path.read_bytes()).hexdigest()
        train_subset_file = str(subset_path)
    from .runtime_manifest import build_runtime_manifest, manifest_hash

    runtime_manifest = build_runtime_manifest(
        project_root=str(project),
        equiformer_root=equiformer_root,
        program_path=str(program_file),
        architecture_id=architecture_id,
        lowering_plan=lowering.to_dict(),
        task_contract_hash=task.content_hash() if task is not None else "unresolved-task-contract",
        data_path=data_path,
        train_subset_sha256=subset_fingerprint,
        critical_files=(
            "equivariant_nas/dsl/compiler.py",
            "equivariant_nas/dsl/pipeline.py",
            "equivariant_nas/dsl/backends/qm9_model.py",
            "equivariant_nas/dsl/backends/equiformer_v1_constructor.py",
            "equivariant_nas/training/fixed_step_trainer.py",
            "equivariant_nas/training/v3_qm9_runtime.py",
        ),
        equiformer_v3_root=equiformer_v3_root,
    )
    executable_id = runtime_manifest["executable_id"]
    runtime_manifest_sha256 = manifest_hash(runtime_manifest)
    protocol = {
        "pipeline_version": "dsl-formal-v1",
        "architecture_id": architecture_id,
        "executable_id": executable_id,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "run_symmetry": bool(run_symmetry),
        "symmetry_threshold": float(symmetry_threshold),
        "symmetry_absolute_threshold": float(symmetry_absolute_threshold),
        "parameter_ratio_limit": float(parameter_ratio_limit),
        "batch_size": int(batch_size),
        "train_subset_sha256": subset_fingerprint,
        "eval_interval_epochs": int(eval_interval_epochs),
        "data_epoch_origin_step": int(data_epoch_origin_step),
        "lr_schedule_origin_step": int(lr_schedule_origin_step),
        "equiformer_v2_commit": (
            "d5ad4be729b56f74012ebb7f097f77c5b00a1004" if equiformer_v2_root else ""
        ),
        "equiformer_v3_root": str(Path(equiformer_v3_root).resolve()) if equiformer_v3_root else "",
        "task_contract_hash": task.content_hash() if task is not None else "unresolved-task-contract",
        "lowering_plan_hash": lowering.content_hash(),
        "backend_semantics_version": lowering.backend_semantics_version,
        "allow_data_transition": bool(allow_data_transition),
        "resume_model_only": bool(resume_model_only),
        "allow_experimental_generic_lowering": bool(allow_experimental_generic_lowering),
    }
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    run_dir = project / "runs" / "dsl_candidates" / architecture_id / (
        "seed{}_steps{}_{}".format(seed, max_steps, protocol_id)
    )
    result_path = run_dir / "result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        cached_manifest_path = run_dir / "runtime_manifest.json"
        cached_manifest = json.loads(cached_manifest_path.read_text(encoding="utf-8")) if cached_manifest_path.exists() else None
        checkpoint = run_dir / "training" / "checkpoint_last.pth"
        if cached_manifest == runtime_manifest and (cached.get("valid") or not checkpoint.exists()):
            cached["cache_hit"] = True
            return cached
    run_dir.mkdir(parents=True, exist_ok=True)
    canonical_program = run_dir / "architecture.dsl.json"
    canonical_program.write_text(
        json.dumps(program.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "runtime_manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "protocol.json").write_text(
        json.dumps(dict(protocol, protocol_id=protocol_id), indent=2, sort_keys=True), encoding="utf-8"
    )
    result: Dict[str, Any] = {
        "architecture_id": architecture_id,
        "program_id": architecture_id,
        "executable_id": executable_id,
        "protocol_id": protocol_id,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "language_version": program.language_version,
        "candidate_format": "evoequilang",
        "valid": False,
        "combined_score": -1.0e9,
        "fidelity_steps": int(max_steps),
        "seed": int(seed),
        "cache_hit": False,
        "run_dir": str(run_dir),
        "test_evaluated": False,
        "lowering_plan": lowering.to_dict(),
        "lowering_plan_hash": lowering.content_hash(),
        "experimental_generic_lowering": bool(
            lowering.mode == "experimental_node_graph"
            and allow_experimental_generic_lowering
        ),
        "selection_eligible": formal_ranking_admitted,
        "formal_ranking_admitted": formal_ranking_admitted,
    }
    resolved_budget = float(gpu_budget_hours) if gpu_budget_hours is not None else float(
        os.environ.get("NAS_GPU_BUDGET_HOURS", "5.0")
    )
    ledger = BudgetLedger(
        os.environ.get("NAS_BUDGET_LEDGER", str(project / "runs" / "budget_ledger.jsonl")),
        resolved_budget,
    )
    started = time.perf_counter()
    gpu_started = None
    try:
        stage_name = "dsl_steps{}".format(max_steps)
        fallback_seconds_per_step = float(os.environ.get("NAS_FALLBACK_SECONDS_PER_STEP", "0.54"))
        ledger.require_available(
            ledger.estimate_stage_gpu_hours(
                stage_name,
                fallback_gpu_hours=max_steps * fallback_seconds_per_step / 3600.0
                + (0.02 if run_symmetry else 0.0),
            )
        )
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if lowering.mode == "representation_only":
            raise ValueError("representation-flow-only V1 programs are not trainable candidates")
        def build_candidate_model():
            if is_v3_program:
                from equivariant_nas.training.v3_qm9_runtime import build_lowered_v3_qm9_model

                return build_lowered_v3_qm9_model(
                    program,
                    equiformer_v3_root=equiformer_v3_root,
                )
            return build_qm9_dsl_model(
                program,
                compiler,
                radius=float(program.parameters.get("radius", 5.0)),
                equiformer_root=equiformer_root,
                equiformer_v2_root=equiformer_v2_root or None,
                task=task,
                allow_experimental_generic_lowering=allow_experimental_generic_lowering,
            )

        model = build_candidate_model()
        parameter_count = _count_parameters(model)
        parameter_ratio = parameter_count / BASELINE_PARAMETERS
        representation_statistics = _representation_statistics(
            artifact.inference.value_types.values()
        )
        result.update({
            "parameter_count": parameter_count,
            "parameter_ratio": parameter_ratio,
            "lmax": representation_statistics["lmax"],
            "num_layers": len(artifact.expanded_program.nodes),
            "higher_order_fraction": representation_statistics["higher_order_fraction"],
            "compiler_obligations": [item.to_dict() for item in artifact.inference.obligations],
            "backend_family": getattr(model, "backend_family", lowering.backend_family),
            "backend_semantics": getattr(model, "backend_semantics_version", lowering.backend_semantics_version),
            "lowering_mode": getattr(model, "lowering_mode", lowering.mode),
            "generic_lowering_admission": getattr(model, "generic_lowering_admission", {}),
        })
        if parameter_ratio > parameter_ratio_limit:
            raise ValueError(
                "parameter ratio {:.4f} exceeds {:.4f}".format(parameter_ratio, parameter_ratio_limit)
            )

        batch = None
        if run_symmetry or max_steps > 0:
            from datasets.pyg.qm9 import QM9
            from torch_geometric.loader import DataLoader
            from torch.utils.data import Subset

            dataset = QM9(data_path, "train", feature_type="one_hot")
            if train_subset_file:
                with np.load(train_subset_file) as payload:
                    dataset = Subset(dataset, np.asarray(payload["train_local_indices"], dtype=np.int64).tolist())
            batch = next(iter(DataLoader(dataset, batch_size=16))).to("cuda")
            model = model.to("cuda")
            gpu_started = time.perf_counter()
            base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
            if isinstance(dataset, Subset):
                target_indices = torch.as_tensor(dataset.indices, dtype=torch.long)
                target_values = base_dataset.data.y[target_indices, 1].float()
            else:
                target_values = base_dataset.data.y[:, 1].float()
            target_mean = float(target_values.mean())
            target_std = float(target_values.std())
            numerical = _numerical_health(model, batch, target_mean, target_std)
            gradient = _gradient_health(model, batch, target_mean, target_std)
            result["pretrain_numerical_health"] = numerical
            result["gradient_health"] = gradient
            _assert_numerical_health(numerical, gradient)
        if run_symmetry:
            symmetry = _observable_symmetry_report(model, batch)
            result["pretrain_symmetry_report"] = symmetry
            result["pretrain_max_symmetry_error"] = symmetry["maximum"]
            result["pretrain_max_absolute_symmetry_error"] = symmetry["maximum_absolute"]
            result["pretrain_symmetry_warning"] = _symmetry_gate_failed(
                symmetry,
                symmetry_warning_threshold,
                symmetry_absolute_warning_threshold,
            )
            if _symmetry_gate_failed(symmetry, symmetry_threshold, symmetry_absolute_threshold):
                raise ValueError(
                    "pre-training symmetry errors relative={:.6g}, absolute={:.6g} "
                    "exceed thresholds relative={:.6g}, absolute={:.6g}".format(
                        symmetry["maximum"],
                        symmetry["maximum_absolute"],
                        symmetry_threshold,
                        symmetry_absolute_threshold,
                    )
                )
            (run_dir / "pretrain_symmetry_report.json").write_text(
                json.dumps(symmetry, indent=2, sort_keys=True), encoding="utf-8"
            )
        if batch is not None:
            del model
            torch.cuda.empty_cache()

        if max_steps > 0:
            train_dir = run_dir / "training"
            automatic_resume = train_dir / "checkpoint_last.pth"
            effective_resume = str(resume_checkpoint or (automatic_resume if automatic_resume.exists() else ""))
            configured_python = os.environ.get("EQUIFORMER_PYTHON", "")
            default_python = "/home/20262202788/conda-envs/equiformer/bin/python"
            training_python = configured_python or (default_python if Path(default_python).is_file() else sys.executable)
            command = [
                training_python,
                "-u",
                "-m",
                "equivariant_nas.training.fixed_step_trainer",
                "--output-dir", str(train_dir),
                "--dsl-program", str(canonical_program),
                "--equiformer-root", equiformer_root,
                "--input-irreps", "5x0e",
                "--target", "1",
                "--data-path", data_path,
                "--feature-type", "one_hot",
                "--batch-size", str(batch_size),
                "--max-steps", str(max_steps),
                "--reference-steps-per-epoch", "859",
                "--eval-interval-steps", str(max_steps),
                "--eval-interval-epochs", str(eval_interval_epochs),
                "--data-epoch-origin-step", str(data_epoch_origin_step),
                "--lr-schedule-origin-step", str(lr_schedule_origin_step),
                "--checkpoint-interval-steps", "859",
                "--epochs", "300",
                "--radius", str(program.parameters.get("radius", 5.0)),
                "--num-basis", "128",
                "--drop-path", "0.0",
                "--weight-decay", "5e-3",
                "--lr", "5e-4",
                "--min-lr", "1e-6",
                "--workers", "4",
                "--print-freq", "100",
                "--seed", str(seed),
                "--no-model-ema",
                "--no-amp",
            ]
            if is_v3_program:
                reference_index = command.index("--reference-steps-per-epoch") + 1
                checkpoint_index = command.index("--checkpoint-interval-steps") + 1
                command[reference_index] = "0"
                command[checkpoint_index] = "0"
            if task_contract_path:
                command.extend(["--dsl-task-contract", str(Path(task_contract_path).resolve())])
            if equiformer_v2_root:
                command.extend(["--equiformer-v2-root", equiformer_v2_root])
            if equiformer_v3_root:
                command.extend(["--equiformer-v3-root", equiformer_v3_root])
            if allow_experimental_generic_lowering:
                command.append("--allow-experimental-generic-lowering")
            if train_subset_file:
                command.extend(["--train-subset-file", train_subset_file])
            if effective_resume:
                command.extend(["--resume-step", effective_resume])
            if allow_data_transition:
                command.append("--allow-data-transition")
            if resume_model_only:
                command.append("--resume-model-only")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = project_root
            environment["EQUIFORMER_ROOT"] = equiformer_root
            if equiformer_v2_root:
                environment["EQUIFORMER_V2_ROOT"] = equiformer_v2_root
            gpu_started = gpu_started or time.perf_counter()
            requested_timeout = int(
                os.environ.get(
                    "NAS_TRAINING_TIMEOUT_SECONDS",
                    max(1800, int(max_steps * 4.0 + 1800)),
                )
            )
            timeout = min(
                requested_timeout,
                max(1, int(ledger.remaining_gpu_hours() * 3600.0 - (time.perf_counter() - gpu_started))),
            )
            with (run_dir / "trainer_console.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError("trainer exited with code {}".format(completed.returncode))
            summary = json.loads((train_dir / "training_summary.json").read_text(encoding="utf-8"))
            metric_lines = [
                json.loads(line)
                for line in (train_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            endpoint = metric_lines[-1]
            validation_mae = float(endpoint["val_mae"])
            result.update({
                "validation_alpha_mae": validation_mae,
                "best_validation_alpha_mae": float(summary["best_val_mae"]),
                "best_step": int(summary["best_step"]),
                "endpoint_step": int(endpoint["global_step"]),
                "training_time_sec": float(summary["training_time_sec"]),
                "training_wall_time_sec": float(summary["wall_time_sec_current_job"]),
                "combined_score": -validation_mae,
                "resumed_from": effective_resume,
                "batch_size": int(summary.get("batch_size", batch_size)),
                "training_dataset_id": str(summary.get("training_dataset_id", "")),
                "training_dataset_size": int(summary.get("training_dataset_size", 0)),
                "start_global_step": int(summary.get("start_global_step", 0)),
                "steps_executed_current_job": int(summary.get("steps_executed_current_job", 0)),
                "data_epoch_origin_step": int(summary.get("data_epoch_origin_step", 0)),
                "test_evaluated": bool(summary.get("test_evaluated", False)),
            })
            if result["test_evaluated"]:
                raise ValueError("search pipeline must not evaluate the test split")
            if run_symmetry:
                audited = build_candidate_model().to("cuda")
                checkpoint = torch.load(train_dir / "checkpoint_last.pth", map_location="cpu")
                audited.load_state_dict(checkpoint["model"])
                post = _observable_symmetry_report(audited, batch)
                result["posttrain_symmetry_report"] = post
                result["max_symmetry_error"] = post["maximum"]
                result["max_absolute_symmetry_error"] = post["maximum_absolute"]
                result["symmetry_warning"] = _symmetry_gate_failed(
                    post,
                    symmetry_warning_threshold,
                    symmetry_absolute_warning_threshold,
                )
                (run_dir / "posttrain_symmetry_report.json").write_text(
                    json.dumps(post, indent=2, sort_keys=True), encoding="utf-8"
                )
                if _symmetry_gate_failed(post, symmetry_threshold, symmetry_absolute_threshold):
                    raise ValueError(
                        "post-training symmetry errors relative={:.6g}, absolute={:.6g} "
                        "exceed thresholds relative={:.6g}, absolute={:.6g}".format(
                            post["maximum"],
                            post["maximum_absolute"],
                            symmetry_threshold,
                            symmetry_absolute_threshold,
                        )
                    )
        else:
            result["combined_score"] = 0.0
        result["valid"] = True
        result["failure_stage"] = ""
    except Exception as exc:
        result.update({
            "valid": False,
            "combined_score": -1.0e9,
            "failure_stage": "dsl_pipeline",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        })
    finally:
        elapsed = time.perf_counter() - started
        gpu_seconds = time.perf_counter() - gpu_started if gpu_started is not None else 0.0
        result["evaluation_wall_time_sec"] = elapsed
        result["charged_gpu_seconds"] = gpu_seconds
        try:
            ledger.append({
                "architecture_id": architecture_id,
                "stage": "dsl_steps{}".format(max_steps),
                "seed": seed,
                "gpu_seconds": gpu_seconds,
                "valid": bool(result.get("valid")),
            })
        except Exception as budget_exc:
            result["budget_error"] = str(budget_exc)
        result["used_gpu_hours"] = ledger.used_gpu_hours()
        checkpoint = run_dir / "training" / "checkpoint_last.pth"
        result["checkpoint_last"] = str(checkpoint) if checkpoint.exists() else ""
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
