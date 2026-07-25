#!/usr/bin/env python
"""Freeze the validation winner, evaluate Test once, and build formal-V1 reports."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
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


def copy_atomic(source, target):
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def link_or_copy_atomic(source, target):
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(str(source), str(temporary))
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(target)


def copy_training_evidence(run_dir, target):
    """Archive one fidelity run without relying on its original absolute path later."""
    if not run_dir:
        return False
    run_dir = Path(str(run_dir))
    target = Path(target)
    if not run_dir.is_dir():
        return False
    for name in (
        "architecture.dsl.json",
        "protocol.json",
        "runtime_manifest.json",
        "pretrain_symmetry_report.json",
        "posttrain_symmetry_report.json",
        "result.json",
    ):
        source = run_dir / name
        if source.is_file():
            copy_atomic(source, target / name)
    training = run_dir / "training"
    for name in ("progress.json", "metrics.jsonl", "training_summary.json", "checkpoint_last.pth"):
        source = training / name
        if source.is_file():
            link_or_copy_atomic(source, target / "training" / name)
    return True


def validation_trajectory(run_dir, endpoint_step=None, endpoint_mae=None):
    """Return deduplicated validation observations from a trainer run."""
    points = {}
    path = Path(str(run_dir)) / "training" / "metrics.jsonl" if run_dir else None
    if path is not None and path.is_file():
        for row in read_jsonl(path):
            step = row.get("global_step", row.get("step"))
            value = row.get("val_mae", row.get("validation_alpha_mae"))
            if step is not None and value is not None:
                points[int(step)] = float(value)
    if endpoint_step is not None and endpoint_mae is not None:
        points[int(endpoint_step)] = float(endpoint_mae)
    return sorted(points.items())


def selection_payload(state, state_path):
    if state.get("stage") not in ("completed_validation_selection", "final_completed"):
        raise RuntimeError("validation selection is not frozen")
    winner = state.get("winner_by_validation") or {}
    metrics = winner.get("metrics_250000") or {}
    parent = state.get("parent_baseline") or {}
    parent_metrics = parent.get("metrics_250000") or {}
    if not winner.get("program") or not winner.get("checkpoint_250000"):
        raise RuntimeError("validation winner lacks program or full-fidelity checkpoint")
    if metrics.get("test_evaluated") is not False:
        raise RuntimeError("winner was exposed to Test before selection freeze")
    if (
        parent_metrics.get("valid") is not True
        or parent_metrics.get("test_evaluated") is not False
        or int(parent_metrics.get("endpoint_step", 0)) != 250000
        or not Path(str(parent.get("checkpoint_250000", ""))).is_file()
    ):
        raise RuntimeError("full-fidelity parent baseline is incomplete or exposed to Test")
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
        "parent_validation_mae": float(parent_metrics["validation_alpha_mae"]),
        "validation_mae_delta_vs_parent": float(metrics["validation_alpha_mae"])
        - float(parent_metrics["validation_alpha_mae"]),
        "candidate_beats_parent": float(metrics["validation_alpha_mae"])
        < float(parent_metrics["validation_alpha_mae"]),
        "parent_checkpoint_sha256": sha256(parent["checkpoint_250000"]),
        "multifidelity_state_sha256": sha256(state_path),
    }


def ensure_freeze(root, state, state_path):
    path = root / "selection_freeze.json"
    proposed = selection_payload(state, state_path)
    if path.exists():
        existing = read_json(path)
        immutable = (
            "architecture_id",
            "program_id",
            "program_sha256",
            "checkpoint_sha256",
            "validation_mae",
            "parent_validation_mae",
            "parent_checkpoint_sha256",
            "candidate_beats_parent",
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
    write_jsonl(root / "evolution.jsonl", evolution)

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
    write_json(root / "parent_baseline.json", state.get("parent_baseline") or {})
    return candidates, promotions


def materialize_candidate_evidence(root, search_dir, state=None):
    parent_program = search_dir.parent / "initial_program.dsl.json"
    records = [item for item in read_jsonl(search_dir / "evolution.jsonl") if int(item.get("iteration", 0)) > 0]
    for record in records:
        metrics = record.get("metrics") or {}
        architecture_id = str(record.get("architecture_id", metrics.get("architecture_id", "unknown")))
        iteration = int(record["iteration"])
        material = root / "candidate_materials" / "iteration_{:04d}_{}".format(iteration, architecture_id)
        if parent_program.is_file():
            copy_atomic(parent_program, material / "parent.dsl.json")
        candidates = sorted((search_dir / "candidates").glob("iteration_{:04d}_*.dsl.json".format(iteration)))
        if candidates:
            copy_atomic(candidates[0], material / "candidate.dsl.json")
        write_json(material / "typed_patch.json", record.get("patch") or {})
        write_json(material / "factor_router.json", record.get("router_response") or {})
        write_json(material / "factor_critic.json", record.get("critic_response") or {})
        write_json(material / "synthesizer.json", {
            "planner_response": record.get("planner_response") or {},
            "typed_patch": record.get("patch") or {},
        })
        write_json(material / "generation_record.json", record)
        write_json(material / "compiler_report.json", {
            "region_audit": record.get("region_audit") or {},
            "compiler_obligations": metrics.get("compiler_obligations") or [],
            "valid": metrics.get("valid"),
            "failure_stage": metrics.get("failure_stage", ""),
        })
        write_json(material / "lowering_plan.json", metrics.get("lowering_plan") or (record.get("region_audit") or {}).get("lowering_plan") or {})
        run_dir = Path(str(metrics.get("run_dir", "")))
        if run_dir.is_dir():
            runtime_manifest = run_dir / "runtime_manifest.json"
            if runtime_manifest.is_file():
                copy_atomic(runtime_manifest, material / "runtime_manifest.json")
            symmetry = run_dir / "posttrain_symmetry_report.json"
            if not symmetry.is_file():
                symmetry = run_dir / "pretrain_symmetry_report.json"
            if symmetry.is_file():
                copy_atomic(symmetry, material / "symmetry_audit.json")
            training = run_dir / "training"
            for name in ("progress.json", "metrics.jsonl", "checkpoint_last.pth"):
                source = training / name
                if source.is_file():
                    link_or_copy_atomic(source, material / "training" / name)
            result = run_dir / "result.json"
            if result.is_file():
                copy_atomic(result, material / "result.json")
        if not (material / "result.json").exists():
            write_json(material / "result.json", metrics)

    if state is None:
        return
    architecture_materials = root / "candidate_materials" / "by_architecture"
    for stage, key, metrics_key in (
        (8000, "candidates_8000", "metrics_8000"),
        (80000, "candidates_80000", "metrics_80000"),
        (250000, "candidates_250000", "metrics_250000"),
    ):
        for item in state.get(key, []):
            architecture_id = str(item.get("architecture_id", "unknown"))
            metrics = item.get(metrics_key) or {}
            target = architecture_materials / architecture_id / "steps_{}".format(stage)
            copy_training_evidence(metrics.get("run_dir", ""), target)
            write_json(target / "state_record.json", item)
    parent = state.get("parent_baseline") or {}
    parent_target = root / "candidate_materials" / "parent_baseline"
    for stage in (8000, 80000, 250000):
        metrics = parent.get("metrics_{}".format(stage)) or {}
        if metrics:
            target = parent_target / "steps_{}".format(stage)
            copy_training_evidence(metrics.get("run_dir", ""), target)
            write_json(target / "state_record.json", {"metrics_{}".format(stage): metrics})


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
    trajectories = {}
    for stage, items in stage_maps.items():
        for item in items:
            metrics = item.get(metric_keys[stage]) or {}
            architecture_id = item.get("architecture_id")
            by_architecture.setdefault(architecture_id, {"factor": item.get("factor_id", "")})[stage] = metrics.get("validation_alpha_mae")
            trajectory = validation_trajectory(
                metrics.get("run_dir", ""),
                metrics.get("endpoint_step", stage),
                metrics.get("validation_alpha_mae"),
            )
            trajectories.setdefault(architecture_id, {}).update(dict(trajectory))
    parent = state.get("parent_baseline") or {}
    parent_values = {"factor": "Parent"}
    for stage in (8000, 80000, 250000):
        value = (parent.get("metrics_{}".format(stage)) or {}).get("validation_alpha_mae")
        if value is not None:
            parent_values[stage] = value
        metrics = parent.get("metrics_{}".format(stage)) or {}
        trajectory = validation_trajectory(
            metrics.get("run_dir", ""),
            metrics.get("endpoint_step", stage),
            metrics.get("validation_alpha_mae"),
        )
        trajectories.setdefault("parent_baseline", {}).update(dict(trajectory))
    if len(parent_values) > 1:
        by_architecture["parent_baseline"] = parent_values
    plt.figure(figsize=(8, 5))
    for architecture_id, values in sorted(by_architecture.items()):
        points = sorted((trajectories.get(architecture_id) or {}).items())
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
        else:
            xs = [stage for stage in (8000, 80000, 250000) if values.get(stage) is not None]
            ys = [values[stage] for stage in xs]
        plt.plot(xs, ys, marker="o", label="{} {}".format(values.get("factor", ""), architecture_id[:8]))
    plt.xscale("log")
    plt.xlabel("Optimizer step")
    plt.ylabel("Validation MAE")
    plt.title("Formal V1 validation MAE trajectory")
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
    parent_validation_mae = float(freeze["parent_validation_mae"])
    validation_delta = float(freeze["validation_mae_delta_vs_parent"])
    test_mae = float(test_summary["endpoint_test_mae"])
    promotion_rows = []
    for stage, key, metrics_key in (
        (8000, "candidates_8000", "metrics_8000"),
        (80000, "candidates_80000", "metrics_80000"),
        (250000, "candidates_250000", "metrics_250000"),
    ):
        for rank, item in enumerate(state.get(key, []), 1):
            metrics = item.get(metrics_key) or {}
            promotion_rows.append(
                "|{}|{}|`{}`|`{}`|{:.8f}|{}|".format(
                    stage,
                    rank,
                    item.get("architecture_id", ""),
                    item.get("factor_id", ""),
                    float(metrics.get("validation_alpha_mae", float("nan"))),
                    str(metrics.get("test_evaluated", "")).lower(),
                )
            )
    promotion_table = "\n".join(promotion_rows)
    report = """# 等变DSL正式V1实验报告

## 最终结果

- Validation冻结赢家：`{architecture}`
- 叶子因子：`{factor}`
- 250000-step Validation MAE：`{validation:.8f}`
- Equiformer V1父代250000-step Validation MAE：`{parent_validation:.8f}`
- 相对父代Validation MAE差值：`{validation_delta:+.8f}`
- 候选是否优于父代：`{candidate_beats_parent}`
- 冻结后一次性Test MAE：`{test:.8f}`
- Test评估执行Optimizer step：`0`
- 搜索期间Test参与：`false`
- 总生成候选记录：`{candidate_count}`

## 协议

正式周期使用QM9极化率、Seed 201、batch size 32。8k和80k使用固定训练集1/4，250k切换完整训练集；所有晋级只依据Validation。选择规则和赢家写入`selection_freeze.json`后，才执行一次Evaluation-only Test。

## 图表

![完整Validation MAE轨迹](assets/mae_vs_step.png)

![因子比较](assets/factor_comparison.png)

## 晋级记录

|阶段终点|阶段内排名|候选ID|因子|Validation MAE|Test参与|
|---:|---:|---|---|---:|---|
{promotion_table}

## 可回溯材料

- `selection_freeze.json`：冻结的Validation赢家、程序和Checkpoint哈希；
- `promotion_history.jsonl`：8k、80k和250k的Validation晋级历史；
- `candidate_materials/by_architecture`：每个晋级候选在各保真度的Runtime Manifest、Validation轨迹、Checkpoint和结果；
- `candidate_materials/parent_baseline`：父代同协议训练证据；
- `final_test/training/training_summary.json`：唯一一次Evaluation-only Test证据。
""".format(
        architecture=freeze["architecture_id"],
        factor=winner.get("factor_id", ""),
        validation=validation_mae,
        parent_validation=parent_validation_mae,
        validation_delta=validation_delta,
        candidate_beats_parent=str(bool(freeze["candidate_beats_parent"])).lower(),
        test=test_mae,
        candidate_count=len(candidates),
        promotion_table=promotion_table,
    )
    (root / "final_report.md").write_text(report, encoding="utf-8")
    try:
        import markdown
        body = markdown.markdown(report, extensions=["tables", "fenced_code"])
    except ImportError:
        body = "<pre>{}</pre>".format(html.escape(report))
    page = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>等变DSL正式V1实验报告</title>
<style>
body{{max-width:1080px;margin:36px auto;padding:0 24px;color:#000;background:#fff;font-family:Arial,"Microsoft YaHei",sans-serif;line-height:1.7}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #bbb;padding:7px 9px;text-align:left}}
img{{max-width:100%;height:auto}}code{{color:#000;background:#f3f3f3;padding:1px 4px}}
</style>
</head>
<body>{}</body>
</html>
""".format(body)
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
    materialize_candidate_evidence(root, search_dir, state)
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
    write_json(root / "state.json", state)
    write_json(root / "heartbeat.json", {
        "status": "final_completed",
        "updated_at": now(),
        "test_evaluated": True,
        "architecture_id": freeze["architecture_id"],
    })
    (root / "STATUS.md").write_text(
        "# 等变DSL正式V1\n\n- 阶段：`final_completed`\n- Validation赢家：`{}`\n- 最终Test MAE：`{:.8f}`\n- Test访问次数：`1`\n".format(
            freeze["architecture_id"], float(test_summary["endpoint_test_mae"])
        ),
        encoding="utf-8",
    )
    archive = Path(args.multifidelity_root).resolve() / "full_fidelity_archive.jsonl"
    if archive.is_file():
        copy_atomic(archive, root / "full_fidelity_archive.jsonl")
    for required_log in (root / "controller.log", root / "supervisor.log"):
        required_log.touch(exist_ok=True)
    write_json(root / "final_result.json", {
        "status": "completed",
        "selection_freeze": freeze,
        "final_test": state["final_test"],
        "parent_baseline": state.get("parent_baseline"),
        "candidate_beats_parent": freeze["candidate_beats_parent"],
        "validation_mae_delta_vs_parent": freeze["validation_mae_delta_vs_parent"],
        "test_evaluated": True,
        "reports": [str(root / "final_report.md"), str(root / "final_report.html")],
    })
    print(json.dumps({"status": "completed", "test_mae": test_summary["endpoint_test_mae"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
