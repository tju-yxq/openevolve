"""Stage-one feasibility and short-training evaluation pipeline."""

from __future__ import annotations

import json
import hashlib
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from torch_geometric.loader import DataLoader

from .budget import BudgetExceeded, BudgetLedger
from .builder import build_equiformer, count_trainable_parameters
from .candidate import extract_literal_spec
from .diagnostics import symmetry_report
from .evaluation import BASELINE_PARAMETERS


def evaluate_candidate_pipeline(
    program_path: str,
    project_root: str,
    equiformer_root: str,
    data_path: str,
    max_steps: int = 0,
    seed: int = 0,
    parameter_ratio_limit: float = 1.2,
    symmetry_threshold: float = 2.5e-1,
    symmetry_warning_threshold: float = 1.0e-2,
    run_symmetry: bool = True,
    gpu_budget_hours: Optional[float] = None,
    resume_checkpoint: str = "",
) -> Dict[str, Any]:
    """Evaluate one architecture with hard gates before optional training."""

    project = Path(project_root)
    spec = extract_literal_spec(program_path)
    architecture_id = spec.architecture_id()
    evaluation_protocol = json.dumps(
        {
            "pipeline_version": 5,
            "seed": seed,
            "max_steps": max_steps,
            "run_symmetry": run_symmetry,
            "symmetry_threshold": symmetry_threshold,
            "symmetry_warning_threshold": symmetry_warning_threshold,
            "parameter_ratio_limit": parameter_ratio_limit,
            "resume_checkpoint": str(resume_checkpoint or ""),
        },
        sort_keys=True,
    )
    protocol_id = hashlib.sha256(evaluation_protocol.encode("utf-8")).hexdigest()[:10]
    run_dir = project / "runs" / "candidates" / architecture_id / (
        "seed{}_steps{}_{}".format(seed, max_steps, protocol_id)
    )
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["cache_hit"] = True
        return result

    run_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "architecture_id": architecture_id,
        "valid": False,
        "combined_score": -1.0e9,
        "fidelity_steps": int(max_steps),
        "seed": int(seed),
        "cache_hit": False,
    }
    resolved_budget_hours = (
        float(gpu_budget_hours)
        if gpu_budget_hours is not None
        else float(os.environ.get("NAS_GPU_BUDGET_HOURS", "5.0"))
    )
    ledger_path = os.environ.get(
        "NAS_BUDGET_LEDGER", str(project / "runs" / "budget_ledger.jsonl")
    )
    ledger = BudgetLedger(ledger_path, resolved_budget_hours)
    started = time.perf_counter()
    gpu_started = None
    try:
        # Repeated fidelities use observed median ledger cost with a 20% safety
        # margin. New fidelities fall back to a conservative static estimate.
        stage_name = "steps{}".format(max_steps)
        fallback_gpu_hours = max_steps * 0.00015 + (0.02 if run_symmetry else 0.0)
        estimated_gpu_hours = ledger.estimate_stage_gpu_hours(
            stage_name, fallback_gpu_hours=fallback_gpu_hours
        )
        ledger.require_available(estimated_gpu_hours)
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = build_equiformer(spec, equiformer_root)
        parameter_count = count_trainable_parameters(model)
        parameter_ratio = parameter_count / BASELINE_PARAMETERS
        result.update(
            {
                "parameter_count": parameter_count,
                "parameter_ratio": parameter_ratio,
                "lmax": spec.representation.lmax,
                "num_layers": spec.macro.num_layers,
                "higher_order_fraction": spec.representation.capacity_profile()[
                    "higher_order_fraction"
                ],
            }
        )
        if parameter_ratio > parameter_ratio_limit:
            raise ValueError(
                "parameter ratio {:.4f} exceeds {:.4f}".format(
                    parameter_ratio, parameter_ratio_limit
                )
            )

        if run_symmetry:
            from .builder import add_equiformer_to_path

            add_equiformer_to_path(equiformer_root)
            from datasets.pyg.qm9 import QM9

            gpu_started = time.perf_counter()
            # Static symmetry/gradient gates use training molecules only. The
            # validation split remains reserved for endpoint model selection.
            dataset = QM9(data_path, "train", feature_type="one_hot")
            batch = next(iter(DataLoader(dataset, batch_size=2))).to("cuda")
            model = model.to("cuda")
            report = symmetry_report(model, batch, rotations=2, translations=2)
            report_dict = report.to_dict()
            observable_symmetry_values = [
                report.rotation_invariance.maximum,
                report.translation_invariance.maximum,
                report.permutation_invariance.maximum,
            ]
            layerwise_symmetry_values = [
                item.maximum for item in report.layerwise_rotation_equivariance.values()
            ]
            maximum_symmetry_error = max(observable_symmetry_values)
            maximum_layerwise_error = max(layerwise_symmetry_values, default=0.0)
            result["pretrain_max_symmetry_error"] = maximum_symmetry_error
            result["pretrain_max_layerwise_symmetry_error"] = maximum_layerwise_error
            result["pretrain_symmetry_warning"] = (
                maximum_symmetry_error > symmetry_warning_threshold
                or maximum_layerwise_error > symmetry_warning_threshold
            )
            result["pretrain_symmetry_report"] = report_dict
            (run_dir / "pretrain_symmetry_report.json").write_text(
                json.dumps(report_dict, indent=2, sort_keys=True), encoding="utf-8"
            )
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
            diagnostic_loss = torch.nn.functional.l1_loss(
                prediction, batch.y[:, 1]
            )
            diagnostic_loss.backward()
            gradients = [
                parameter.grad.detach()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            gradients_finite = bool(gradients) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            squared_norm = sum(
                float(torch.sum(gradient.float() ** 2).cpu()) for gradient in gradients
            )
            maximum_gradient = max(
                (float(gradient.abs().max().cpu()) for gradient in gradients),
                default=0.0,
            )
            result["gradient_health"] = {
                "loss": float(diagnostic_loss.detach().cpu()),
                "global_l2_norm": squared_norm ** 0.5,
                "maximum_absolute_gradient": maximum_gradient,
                "gradients_present": bool(gradients),
                "all_finite": gradients_finite,
            }
            if not gradients_finite:
                raise ValueError("non-finite or missing gradients in pre-training gate")
            model.zero_grad(set_to_none=True)
            del model
            torch.cuda.empty_cache()

        if max_steps > 0:
            spec_path = run_dir / "architecture_spec.json"
            spec_path.write_text(spec.canonical_json(), encoding="utf-8")
            train_dir = run_dir / "training"
            command = [
                "/home/20262202788/conda-envs/equiformer/bin/python",
                "-u",
                "-m",
                "equivariant_nas.training.fixed_step_trainer",
                "--output-dir",
                str(train_dir),
                "--architecture-spec",
                str(spec_path),
                "--equiformer-root",
                equiformer_root,
                "--input-irreps",
                "5x0e",
                "--target",
                "1",
                "--data-path",
                data_path,
                "--feature-type",
                "one_hot",
                "--batch-size",
                "64",
                "--max-steps",
                str(max_steps),
                "--reference-steps-per-epoch",
                "859",
                "--eval-interval-steps",
                str(max_steps),
                "--epochs",
                "300",
                "--radius",
                "5.0",
                "--num-basis",
                "128",
                "--drop-path",
                "0.0",
                "--weight-decay",
                "5e-3",
                "--lr",
                "5e-4",
                "--min-lr",
                "1e-6",
                "--workers",
                "4",
                "--print-freq",
                "100",
                "--seed",
                str(seed),
                "--no-model-ema",
                "--no-amp",
            ]
            if resume_checkpoint:
                command.extend(["--resume-step", str(resume_checkpoint)])
            environment = os.environ.copy()
            environment["PYTHONPATH"] = project_root
            environment["EQUIFORMER_ROOT"] = equiformer_root
            gpu_started = gpu_started or time.perf_counter()
            requested_timeout = max(1800, int(max_steps * 2.0))
            remaining_budget_seconds = (
                ledger.remaining_gpu_hours() * 3600.0
                - (time.perf_counter() - gpu_started)
            )
            if remaining_budget_seconds <= 1.0:
                raise BudgetExceeded("no GPU budget remains for training")
            training_timeout = min(
                requested_timeout, max(1, int(remaining_budget_seconds))
            )
            with (run_dir / "trainer_console.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=training_timeout,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError("trainer exited with code {}".format(completed.returncode))
            summary = json.loads(
                (train_dir / "training_summary.json").read_text(encoding="utf-8")
            )
            metric_lines = [
                line
                for line in (train_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            endpoint_metrics = json.loads(metric_lines[-1])
            validation_mae = float(endpoint_metrics["val_mae"])
            result.update(
                {
                    "validation_alpha_mae": validation_mae,
                    "best_validation_alpha_mae": float(summary["best_val_mae"]),
                    "best_step": int(summary["best_step"]),
                    "endpoint_step": int(endpoint_metrics["global_step"]),
                    "training_time_sec": float(summary["training_time_sec"]),
                    "training_wall_time_sec": float(summary["wall_time_sec_current_job"]),
                    "step_time_ms": 1000.0
                    * float(summary["wall_time_sec_current_job"])
                    / max(
                        1,
                        int(
                            summary.get(
                                "steps_executed_current_job",
                                endpoint_metrics["global_step"],
                            )
                        ),
                    ),
                    "combined_score": -validation_mae,
                    "resumed_from": str(resume_checkpoint or ""),
                }
            )
            if run_symmetry:
                audited_model = build_equiformer(spec, equiformer_root).to("cuda")
                checkpoint = torch.load(
                    train_dir / "checkpoint_last.pth", map_location="cpu"
                )
                audited_model.load_state_dict(checkpoint["model"])
                post_report = symmetry_report(
                    audited_model, batch, rotations=5, translations=3
                )
                post_dict = post_report.to_dict()
                post_observable = [
                    post_report.rotation_invariance.maximum,
                    post_report.translation_invariance.maximum,
                    post_report.permutation_invariance.maximum,
                ]
                post_layerwise = [
                    item.maximum
                    for item in post_report.layerwise_rotation_equivariance.values()
                ]
                post_maximum = max(post_observable)
                post_layerwise_maximum = max(post_layerwise, default=0.0)
                result["max_symmetry_error"] = post_maximum
                result["max_layerwise_symmetry_error"] = post_layerwise_maximum
                result["symmetry_warning"] = (
                    post_maximum > symmetry_warning_threshold
                    or post_layerwise_maximum > symmetry_warning_threshold
                )
                result["symmetry_report"] = post_dict
                result["posttrain_symmetry_report"] = post_dict
                (run_dir / "posttrain_symmetry_report.json").write_text(
                    json.dumps(post_dict, indent=2, sort_keys=True), encoding="utf-8"
                )
                (run_dir / "symmetry_report.json").write_text(
                    json.dumps(post_dict, indent=2, sort_keys=True), encoding="utf-8"
                )
                del audited_model
                torch.cuda.empty_cache()
                if post_maximum > symmetry_threshold:
                    raise ValueError(
                        "post-training observable symmetry error {:.6g} exceeds "
                        "catastrophic threshold {:.6g}".format(
                            post_maximum, symmetry_threshold
                        )
                    )
        else:
            result["combined_score"] = 0.0

        result["valid"] = True
        result["failure_stage"] = ""
    except Exception as exc:
        result.update(
            {
                "valid": False,
                "combined_score": -1.0e9,
                "failure_stage": "pipeline",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }
        )
    finally:
        elapsed = time.perf_counter() - started
        result["evaluation_wall_time_sec"] = elapsed
        gpu_seconds = time.perf_counter() - gpu_started if gpu_started is not None else 0.0
        result["charged_gpu_seconds"] = gpu_seconds
        try:
            ledger.append(
                {
                    "architecture_id": architecture_id,
                    "stage": "steps{}".format(max_steps),
                    "seed": seed,
                    "gpu_seconds": gpu_seconds,
                    "valid": bool(result.get("valid")),
                }
            )
        except Exception as budget_exc:
            result["budget_error"] = str(budget_exc)
        result["used_gpu_hours"] = ledger.used_gpu_hours()
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
