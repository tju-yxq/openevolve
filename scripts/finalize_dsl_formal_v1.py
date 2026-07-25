#!/usr/bin/env python
"""Freeze the validation winner, evaluate Test once, and build formal-V1 reports."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def selection_payload(state, state_path):
    if state.get("stage") not in ("completed_validation_selection", "final_completed"):
        raise RuntimeError("validation selection is not frozen")
    winner = state.get("winner_by_validation") or {}
    metrics = winner.get("metrics_250000") or {}
    if not winner.get("program") or not winner.get("checkpoint_250000"):
        raise RuntimeError("validation winner lacks program or full-fidelity checkpoint")
    if metrics.get("test_evaluated") is not False:
        raise RuntimeError("winner was exposed to Test before selection freeze")
    return {
        "frozen_at": now(),
        "selection_split": "validation",
        "test_evaluated_before_freeze": False,
        "architecture_id": winner.get("architecture_id"),
        "program_id": winner.get("program_id"),
        "executable_id": winner.get("executable_id"),
        "program": str(Path(winner["program"]).resolve()),
        "program_sha256": sha256(winner["program"]),
        "checkpoint": str(Path(winner["checkpoint_250000"]).resolve()),
        "checkpoint_sha256": sha256(winner["checkpoint_250000"]),
        "validation_mae": float(metrics["validation_alpha_mae"]),
        "multifidelity_state_sha256": sha256(state_path),
    }


def ensure_freeze(root, state, state_path):
    path = root / "selection_freeze.json"
    proposed = selection_payload(state, state_path)
    if path.exists():
        existing = read_json(path)
        immutable = (
            "architecture_id", "program_id", "program_sha256", "checkpoint_sha256", "validation_mae"
        )
        if any(existing.get(key) != proposed.get(key) for key in immutable):
            raise RuntimeError("frozen validation selection changed before final Test")
        return existing
    write_json(path, proposed)
    return proposed


def run_final_test(args, root, freeze):
    output = root / "final_test" / "training"
    summary_path = output / "training_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary.get("test_evaluated") is True and summary.get("evaluation_only") is True:
            return summary
    command = [
        args.python, "-u", "-m", "equivariant_nas.training.fixed_step_trainer",
        "--output-dir", str(output),
        "--dsl-program", freeze["program"],
        "--dsl-task-contract", args.task_contract,
        "--equiformer-root", args.equiformer_root,
        "--input-irreps", "5x0e",
        "--target", "1",
        "--data-path", args.data_path,
        "--feature-type", "one_hot",
        "--batch-size", "32",
        "--max-steps", "250000",
        "--reference-steps-per-epoch", "859",
        "--eval-interval-steps", "250000",
        "--checkpoint-interval-steps", "859",
        "--epochs", "300",
        "--radius", "5.0",
        "--num-basis", "128",
        "--drop-path", "0.0",
        "--weight-decay", "5e-3",
        "--lr", "5e-4",
        "--min-lr", "1e-6",
        "--workers", "4",
        "--print-freq", "100",
        "--seed", str(args.seed),
        "--resume-step", freeze["checkpoint"],
        "--evaluate-test",
        "--evaluation-only",
        "--no-model-ema",
        "--no-amp",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = args.project_root
    environment["EQUIFORMER_ROOT"] = args.equiformer_root
    output.mkdir(parents=True, exist_ok=True)
    with (root / "final_test" / "console.log").open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError("final Test evaluation exited with code {}".format(completed.returncode))
    summary = read_json(summary_path)
    if summary.get("test_evaluated") is not True or summary.get("evaluation_only") is not True:
        raise RuntimeError("final evaluator did not prove evaluation-only Test access")
    if int(summary.get("steps_executed_current_job", -1)) != 0:
        raise RuntimeError("final Test evaluation performed optimizer steps")
    return summary


def collect_run_materials(root, search_dir, state):
    evolution = read_jsonl(search_dir / "evolution.jsonl")
    candidates = []
    failures = []
    for record in evolution:
        if int(record.get("iteration", 0)) <= 0:
            continue
        metrics = record.get("metrics") or {}
        candidate = {
            "iteration": record.get("iteration"),
            "architecture_id": record.get("architecture_id", metrics.get("architecture_id")),
            "factor_id": (record.get("region_audit") or {}).get("factor_id", ""),
            "valid": bool(metrics.get("valid")),
            "test_evaluated": metrics.get("test_evaluated"),
            "validation_alpha_mae": metrics.get("validation_alpha_mae"),
            "checkpoint_last": metrics.get("checkpoint_last", ""),
            "program": next(
                (str(item) for item in (search_dir / "candidates").glob("iteration_{:04d}_*.dsl.json".format(int(record["iteration"])))),
                "",
            ),
        }
        candidates.append(candidate)
        if record.get("error") or not metrics.get("valid"):
            failures.append(record)
    write_json(root / "candidate_index.json", candidates)
    write_jsonl(root / "failure_evidence.jsonl", failures)

    promotions = []
    for stage, key, metrics_key in (
        (8000, "candidates_8000", "metrics_8000"),
        (80000, "candidates_80000", "metrics_80000"),
        (250000, "candidates_250000", "metrics_250000"),
    ):
        for rank, item in enumerate(state.get(key, []), 1):
            metrics = item.get(metrics_key) or {}
            promotions.append({
                "stage_steps": stage,
                "rank": rank,
                "architecture_id": item.get("architecture_id"),
                "factor_id": item.get("factor_id", ""),
                "validation_alpha_mae": metrics.get("validation_alpha_mae"),
                "selection_split": "validation",
                "test_evaluated": metrics.get("test_evaluated"),
            })
    write_jsonl(root / "promotion_history.jsonl", promotions)
    return candidates, promotions


def make_plots(root, state):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    stage_maps = {
        8000: state.get("candidates_8000", []),
        80000: state.get("candidates_80000", []),
        250000: state.get("candidates_250000", []),
    }
    metric_keys = {8000: "metrics_8000", 80000: "metrics_80000", 250000: "metrics_250000"}
    by_architecture = {}
    for stage, items in stage_maps.items():
        for item in items:
            metrics = item.get(metric_keys[stage]) or {}
            by_architecture.setdefault(item.get("architecture_id"), {"factor": item.get("factor_id", "")})[stage] = metrics.get("validation_alpha_mae")
    plt.figure(figsize=(8, 5))
    for architecture_id, values in sorted(by_architecture.items()):
        xs = [stage for stage in (8000, 80000, 250000) if values.get(stage) is not None]
        ys = [values[stage] for stage in xs]
        plt.plot(xs, ys, marker="o", label="{} {}".format(values.get("factor", ""), architecture_id[:8]))
    plt.xscale("log")
    plt.xlabel("Optimizer step")
    plt.ylabel("Validation MAE")
    plt.title("Formal V1 validation MAE by fidelity")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(assets / "mae_vs_step.png", dpi=180)
    plt.close()

    factor_values = {}
    for item in state.get("candidates_8000", []):
        value = (item.get("metrics_8000") or {}).get("validation_alpha_mae")
        if value is not None:
            factor_values.setdefault(item.get("factor_id", "unknown"), []).append(float(value))
    labels = sorted(factor_values)
    values = [min(factor_values[label]) for label in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylabel("Best 8k validation MAE")
    plt.title("Formal V1 factor comparison")
    plt.tight_layout()
    plt.savefig(assets / "factor_comparison.png", dpi=180)
    plt.close()


def write_reports(root, state, freeze, test_summary, candidates):
    winner = state["winner_by_validation"]
    validation_mae = float(freeze["validation_mae"])
    test_mae = float(test_summary["endpoint_test_mae"])
    report = """# 等变DSL正式V1实验报告

## 最终结果

- Validation冻结赢家：`{architecture}`
- 叶子因子：`{factor}`
- 250000-step Validation MAE：`{validation:.8f}`
- 冻结后一次性Test MAE：`{test:.8f}`
- Test评估执行Optimizer step：`0`
- 搜索期间Test参与：`false`
- 总生成候选记录：`{candidate_count}`

## 协议

正式周期使用QM9极化率、Seed 201、batch size 32。8k和80k使用固定训练集1/4，250k切换完整训练集；所有晋级只依据Validation。选择规则和赢家写入`selection_freeze.json`后，才执行一次Evaluation-only Test。

## 图表

![不同保真度Validation MAE](assets/mae_vs_step.png)

![因子比较](assets/factor_comparison.png)
""".format(
        architecture=freeze["architecture_id"],
        factor=winner.get("factor_id", ""),
        validation=validation_mae,
        test=test_mae,
        candidate_count=len(candidates),
    )
    (root / "final_report.md").write_text(report, encoding="utf-8")
    body = "".join(
        "<h2>{}</h2><p>{}</p>".format(html.escape(section.splitlines()[0]), html.escape("\n".join(section.splitlines()[1:])))
        for section in report.split("## ")[1:]
    )
    page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>等变DSL正式V1实验报告</title></head><body><h1>等变DSL正式V1实验报告</h1>{}<p><img src="assets/mae_vs_step.png" alt="MAE曲线"></p><p><img src="assets/factor_comparison.png" alt="因子比较"></p></body></html>""".format(body)
    (root / "final_report.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--multifidelity-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--equiformer-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--task-contract", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--seed", type=int, default=201)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    search_dir = Path(args.search_dir).resolve()
    state_path = Path(args.multifidelity_root).resolve() / "state.json"
    state = read_json(state_path)
    freeze = ensure_freeze(root, state, state_path)
    test_summary = run_final_test(args, root, freeze)
    candidates, _promotions = collect_run_materials(root, search_dir, state)
    make_plots(root, state)
    write_reports(root, state, freeze, test_summary, candidates)
    state["stage"] = "final_completed"
    state["final_test"] = {
        "evaluated_at": now(),
        "endpoint_validation_mae": test_summary["endpoint_validation_mae"],
        "endpoint_test_mae": test_summary["endpoint_test_mae"],
        "evaluation_only": True,
        "steps_executed_current_job": 0,
    }
    write_json(state_path, state)
    write_json(root / "final_result.json", {
        "status": "completed",
        "selection_freeze": freeze,
        "final_test": state["final_test"],
        "test_evaluated": True,
        "reports": [str(root / "final_report.md"), str(root / "final_report.html")],
    })
    print(json.dumps({"status": "completed", "test_mae": test_summary["endpoint_test_mae"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
