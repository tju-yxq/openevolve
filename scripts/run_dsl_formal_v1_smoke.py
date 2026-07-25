#!/usr/bin/env python
"""Run a resumable real-runtime smoke matrix for the formal DSL V1 slice."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from equivariant_nas.dsl import (
    Compiler,
    PatchEdit,
    TypedPatch,
    apply_typed_patch,
    core_registry,
    equiformer_v1_capability_profile,
    import_equiformer_v1,
    reference_motif_registry,
    validate_region_transition,
    v1_region_registry,
)
from equivariant_nas.dsl.serialization import dumps_program
from equivariant_nas.pipeline import evaluate_candidate_pipeline
from equivariant_nas.spec import baseline_spec


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _constructor_child(parent, compiler, factor):
    option = dict(factor.alternative_options[0])
    patch = TypedPatch(
        "1.0",
        compiler.analyze(parent).architecture_id,
        parent.language_version,
        {
            "factor_id": factor.factor_id,
            "claim": "formal V1 deterministic smoke alternative",
        },
        factor.parameter_paths,
        tuple(
            PatchEdit(
                "change_parameters",
                path,
                {"value": option[path.rsplit(".", 1)[1]]},
            )
            for path in factor.parameter_paths
        ),
    )
    child = apply_typed_patch(
        parent,
        patch,
        compiler.primitives,
        expected_parent_id=compiler.analyze(parent).architecture_id,
        validate_child_with_core_registry=False,
    )
    return child, patch


def _readout_child(parent, compiler, factor):
    pool = parent.nodes[-1]
    replacement = replace(
        pool,
        op="motif.v1_multilevel_readout",
        inputs={"terminal": ("scalar_readout",), "aux": ("block3",)},
        attrs={},
    )
    patch = TypedPatch(
        "1.0",
        compiler.analyze(parent).architecture_id,
        parent.language_version,
        {
            "factor_id": factor.factor_id,
            "claim": "formal V1 deterministic multilevel readout smoke alternative",
        },
        ("graph_pool", "output:prediction"),
        (PatchEdit("replace_node", "graph_pool", {"node": replacement.to_dict()}),),
        ({"kind": "node_op_is", "node_id": "graph_pool", "op": "core.global_pool"},),
        (
            {
                "kind": "node_op_is",
                "node_id": "graph_pool",
                "op": "motif.v1_multilevel_readout",
            },
        ),
        {"accuracy": "hypothesis_only"},
    )
    child = apply_typed_patch(
        parent,
        patch,
        compiler.primitives,
        expected_parent_id=compiler.analyze(parent).architecture_id,
        validate_child_with_core_registry=False,
    )
    return child, patch


def materialize_candidates(output: Path):
    compiler = Compiler(core_registry(), reference_motif_registry())
    parent = import_equiformer_v1(baseline_spec())
    profile = equiformer_v1_capability_profile()
    regions = {item.factor_id: item for item in v1_region_registry(parent)}
    candidates = [("parent", "Equiformer V1父代", parent, None, None)]
    for factor in profile.enabled_factors:
        if factor.parameter_paths:
            child, patch = _constructor_child(parent, compiler, factor)
        else:
            child, patch = _readout_child(parent, compiler, factor)
        audit = validate_region_transition(parent, child, regions[factor.factor_id])
        candidates.append(
            (
                "factor_{}".format(factor.factor_id.replace(".", "_")),
                "{}替代候选".format(factor.factor_id),
                child,
                patch,
                audit,
            )
        )

    materialized = []
    for key, label, program, patch, audit in candidates:
        artifact = compiler.analyze(program)
        lowering = compiler.plan_lowering(program)
        program_path = output / "programs" / (key + ".dsl.json")
        program_path.parent.mkdir(parents=True, exist_ok=True)
        program_path.write_text(dumps_program(program), encoding="utf-8")
        evidence = {
            "key": key,
            "label": label,
            "factor_id": "" if patch is None else str(patch.hypothesis.get("factor_id", "")),
            "program_id": artifact.architecture_id,
            "program_path": str(program_path),
            "patch": None if patch is None else patch.to_dict(),
            "region_audit": audit,
            "lowering_plan": lowering.to_dict(),
        }
        _write_json(output / "candidate_evidence" / (key + ".json"), evidence)
        materialized.append(evidence)
    _write_json(output / "candidate_index.json", materialized)
    return materialized


def _report(output: Path, summary):
    rows = []
    for item in summary["candidates"]:
        result = item["result"]
        rows.append(
            "|{}|{}|{}|{}|{}|{}|{}|".format(
                item["label"],
                item["factor_id"] or "父代",
                result.get("lowering_mode", ""),
                "通过" if result.get("valid") else "失败",
                "{:.6f}".format(float(result["validation_alpha_mae"]))
                if "validation_alpha_mae" in result
                else "未训练",
                "{:.3e}/{:.3e}".format(
                    float(result.get("pretrain_max_symmetry_error", float("nan"))),
                    float(result.get("pretrain_max_absolute_symmetry_error", float("nan"))),
                ),
                str(result.get("test_evaluated", False)).lower(),
            )
        )
    content = """# 正式V1短链路Smoke验证报告

## 结论

- 运行状态：`{status}`
- 代码提交：`{commit}`
- 候选数量：`{count}`
- 每个候选训练步数：`{steps}`
- Batch size：`{batch_size}`
- 数据：QM9训练集固定1/4子集
- Test参与：`false`

## 候选验证结果

|候选|因子|Lowering模式|门禁|训练终点Validation MAE|等变最大误差（相对/绝对）|Test参与|
|---|---|---|---|---:|---:|---|
{rows}

## 验证范围

本报告验证父代和四个正式V1开放因子的确定性替代候选是否能够完成真实官方后端构建、Runtime Manifest绑定、16个QM9分子的旋转/平移/置换等变审计、数值健康检查、梯度门禁以及可选短训练。它不是正式8→4→2搜索结果，也不用于证明某个因子已经优于父代。
""".format(
        status=summary["status"],
        commit=summary["project_commit"],
        count=len(summary["candidates"]),
        steps=summary["max_steps"],
        batch_size=summary["batch_size"],
        rows="\n".join(rows),
    )
    (output / "正式V1短链路Smoke验证报告.md").write_text(content, encoding="utf-8")


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    state = _load_json(state_path, {}) or {}
    candidates = materialize_candidates(output)
    results = dict(state.get("results", {}))
    common = {
        "project_root": args.project_root,
        "equiformer_root": args.equiformer_root,
        "data_path": args.data_path,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "train_subset_file": args.train_subset_file,
        "eval_interval_epochs": args.eval_interval_epochs,
        "dsl_task_contract": args.task_contract,
        "run_symmetry": True,
    }
    for candidate in candidates:
        key = candidate["key"]
        _write_json(
            state_path,
            {
                "status": "running",
                "current_candidate": key,
                "updated_at": _now(),
                "results": results,
            },
        )
        result = evaluate_candidate_pipeline(program_path=candidate["program_path"], **common)
        results[key] = result
        _write_json(output / "results" / (key + ".json"), result)
        if result.get("test_evaluated"):
            raise RuntimeError("Smoke验证访问了锁定的Test集")
        if not result.get("valid"):
            raise RuntimeError("{}验证失败：{}".format(key, result.get("error", "未知错误")))
        if args.max_steps > 0:
            if int(result.get("endpoint_step", -1)) != args.max_steps:
                raise RuntimeError("{}未到达注册训练终点".format(key))
            if not Path(result.get("checkpoint_last", "")).is_file():
                raise RuntimeError("{}缺少checkpoint_last.pth".format(key))

    commit = subprocess.check_output(
        ["git", "-C", args.project_root, "rev-parse", "HEAD"], text=True
    ).strip()
    combined = [dict(candidate, result=results[candidate["key"]]) for candidate in candidates]
    summary = {
        "status": "completed",
        "completed_at": _now(),
        "project_commit": commit,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "test_evaluated": False,
        "candidates": combined,
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        state_path,
        {
            "status": "completed",
            "current_candidate": "",
            "updated_at": _now(),
            "results": results,
            "summary": str(output / "summary.json"),
        },
    )
    _report(output, summary)
    print(json.dumps({"status": "completed", "summary": str(output / "summary.json")}, ensure_ascii=False))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--equiformer-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--train-subset-file", required=True)
    parser.add_argument("--task-contract", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--eval-interval-epochs", type=int, default=10)
    return parser


if __name__ == "__main__":
    run(get_parser().parse_args())
