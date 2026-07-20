#!/usr/bin/env python
"""Generate the auditable Chinese stage-one report and figures."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from equivariant_nas.fidelity_trust import assess_fidelity_trust
from equivariant_nas.trajectory import response_fingerprint


ROOT = Path("/home/20262202788/equivariant-nas")
RUNS = ROOT / "runs"
OUT = ROOT / "reports" / "stage1"
ASSETS = OUT / "assets"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_optional(path, default):
    path = Path(path)
    return load(path) if path.exists() else default


def jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def endpoint(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return json.loads([line for line in lines if line.strip()][-1])


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def git_dirty(path):
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        )
        return bool(output.strip())
    except Exception:
        return None


def draw_workflow():
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.2, 3.2, 2.0, 1.0, "OpenEvolve\nislands + MAP-Elites", "#DCEAF7"),
        (2.6, 3.2, 1.8, 1.0, "ECFR\nfactor router", "#E8DDF3"),
        (4.8, 3.2, 1.8, 1.0, "SPARK-style RC\nevidence + risk", "#FBE5D6"),
        (7.0, 3.2, 1.8, 1.0, "SPARK-style SAR\nfactor-local patch", "#FBE5D6"),
        (9.2, 3.2, 1.6, 1.0, "SPAG\ntyped compiler", "#DDF1E4"),
        (11.2, 3.2, 1.6, 1.0, "Equiformer\ntrusted builder", "#DDF1E4"),
        (8.3, 1.0, 2.0, 1.0, "symmetry / params /\nforward-backward gates", "#FFF1CC"),
        (5.3, 1.0, 2.2, 1.0, "QM9 alpha validation\nphase-aligned fidelity", "#FFF1CC"),
        (2.4, 1.0, 2.1, 1.0, "SCFTG\nrank trust decision", "#F4CCCC"),
        (0.1, 1.0, 1.6, 1.0, "credit + archive\nupdate", "#DCEAF7"),
    ]
    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.04", fc=color, ec="#444", lw=1.2
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)
    arrows = [
        ((2.2, 3.7), (2.6, 3.7)), ((4.4, 3.7), (4.8, 3.7)),
        ((6.6, 3.7), (7.0, 3.7)), ((8.8, 3.7), (9.2, 3.7)),
        ((10.8, 3.7), (11.2, 3.7)), ((12.0, 3.2), (9.4, 2.0)),
        ((8.3, 1.5), (7.5, 1.5)), ((5.3, 1.5), (4.5, 1.5)),
        ((2.4, 1.5), (1.7, 1.5)), ((0.9, 2.0), (1.1, 3.2)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.set_title("OpenEvolve × SPARK-style reasoning × symmetry-preserving Equiformer NAS", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSETS / "framework_workflow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    draw_workflow()
    test_output_path = OUT / "test_output.txt"
    test_summary = "尚未写入最终测试记录"
    if test_output_path.exists():
        lines = [line for line in test_output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            test_summary = next(
                (line for line in reversed(lines) if "passed" in line or "failed" in line),
                lines[-1],
            )
    training_summaries = [
        load(path) for path in RUNS.glob("candidates/*/*/training/training_summary.json")
    ]
    test_evaluated_count = sum(
        bool(item.get("test_evaluated", False)) for item in training_summaries
    )

    gate = load_optional(RUNS / "gate1_training.json", []) + load_optional(
        RUNS / "gate1_training_additional.json", []
    )
    gate = sorted(gate, key=lambda item: item["validation_alpha_mae"])
    baseline_300 = next(item for item in gate if item["candidate_name"] == "baseline")

    factor_records = jsonl(RUNS / "smoke_factorized_seed44" / "evolution.jsonl")
    factor_children = [item for item in factor_records if item.get("iteration", 0) > 0]
    factor_valid = [item for item in factor_children if item.get("metrics", {}).get("valid")]
    factor_best = min(factor_valid, key=lambda item: item["metrics"]["validation_alpha_mae"])
    random_summary = load(RUNS / "random_seed44" / "summary.json")
    random_valid = [item for item in random_summary["records"] if item["metrics"].get("valid")]
    random_best = min(random_valid, key=lambda item: item["metrics"]["validation_alpha_mae"])
    schema_records = jsonl(RUNS / "smoke_schema_seed43" / "evolution.jsonl")[1:]
    schema_before_repair = jsonl(RUNS / "smoke_qd_seed46" / "evolution.jsonl")[1:]
    schema_after_repair = jsonl(RUNS / "smoke_qd_repair_seed48" / "evolution.jsonl")[1:]
    qd_run = (
        RUNS / "smoke_qd_repair_seed51"
        if (RUNS / "smoke_qd_repair_seed51" / "summary.json").exists()
        else RUNS / "smoke_qd_repair_seed48"
    )
    qd_metadata = load_optional(qd_run / "database" / "metadata.json", {})
    qd_summary = load_optional(qd_run / "summary.json", {})
    qd_records = jsonl(qd_run / "evolution.jsonl")
    qd_metrics = [
        item.get("metrics", {})
        for item in qd_records
        if item.get("metrics", {}).get("valid")
    ]
    qd_children = [item for item in qd_records if item.get("iteration", 0) > 0]
    qd_repair_count = sum(len(item.get("repair_history", [])) for item in qd_children)
    qd_cells = sum(
        len(feature_map) for feature_map in qd_metadata.get("island_feature_maps", [])
    )
    def observed_range(key, default=0.0):
        values = [float(item[key]) for item in qd_metrics if key in item]
        return (min(values), max(values)) if values else (default, default)

    qd_lmax_range = observed_range("lmax")
    qd_higher_range = observed_range("higher_order_fraction")
    qd_parameter_range = observed_range("parameter_ratio")

    fidelity = {
        "baseline": {300: baseline_300["validation_alpha_mae"]},
        "factorized": {300: factor_best["metrics"]["validation_alpha_mae"]},
        "random": {300: random_best["metrics"]["validation_alpha_mae"]},
    }
    paths_1000 = {
        "baseline": RUNS / "candidates/8639c8a64dad5d25/seed0_steps1000_cc934a9505/training/metrics.jsonl",
        "factorized": RUNS / "candidates/36a56d97fa7703cc/seed0_steps1000_2ed4a18ecf/training/metrics.jsonl",
        "random": RUNS / "candidates/63ee926702488f11/seed0_steps1000_e5a30ac6a8/training/metrics.jsonl",
    }
    for name, path in paths_1000.items():
        if path.exists():
            fidelity[name][1000] = endpoint(path)["val_mae"]
    for item in load_optional(RUNS / "promotion_5000.json", []):
        key = {
            "baseline": "baseline",
            "factorized_best_300": "factorized",
            "random_best_300": "random",
        }[item["name"]]
        fidelity[key][5000] = item["validation_alpha_mae"]

    fingerprints = {
        name: response_fingerprint(points).to_dict()
        for name, points in fidelity.items()
        if all(step in points for step in (300, 1000, 5000))
    }
    trust_report = load_optional(RUNS / "fidelity_trust_300_to_5000.json", None)
    if trust_report is None and all(5000 in fidelity[name] for name in fidelity):
        trust_report = assess_fidelity_trust(
            {name: values[300] for name, values in fidelity.items()},
            {name: values[5000] for name, values in fidelity.items()},
            top_k=1,
            minimum_cohort=8,
        ).to_dict()

    stable_records = jsonl(RUNS / "stable_factorized_seed45" / "evolution.jsonl")
    stable_children = [item for item in stable_records if item.get("iteration", 0) > 0]
    stable_valid = [item for item in stable_children if item.get("metrics", {}).get("valid")]
    stable_best = min(
        stable_valid, key=lambda item: item["metrics"]["validation_alpha_mae"], default=None
    )
    stable_random = load_optional(RUNS / "stable_random_seed45" / "summary.json", {})
    stable_random_valid = [
        item for item in stable_random.get("records", []) if item.get("metrics", {}).get("valid")
    ]
    stable_random_best = min(
        stable_random_valid,
        key=lambda item: item["metrics"]["validation_alpha_mae"],
        default=None,
    )
    interaction_records = load_optional(
        RUNS / "stage1_interaction_ablation" / "results.json", []
    )
    resolved_interaction = next(
        (item for item in interaction_records if item.get("status") == "resolved"),
        None,
    )
    trained_joint_audit = load_optional(
        RUNS / "stage1_interaction_ablation" / "trained_joint_symmetry.json", {}
    )
    trained_joint_report = trained_joint_audit.get("report", {})

    ledger = jsonl(RUNS / "budget_ledger.jsonl")
    budget_hours = sum(float(item.get("gpu_seconds", 0.0)) for item in ledger) / 3600.0
    cumulative = []
    running = 0.0
    for item in sorted(ledger, key=lambda row: row.get("timestamp", 0.0)):
        running += float(item.get("gpu_seconds", 0.0)) / 3600.0
        cumulative.append(running)

    labels = [item["candidate_name"] for item in gate]
    values = [item["validation_alpha_mae"] for item in gate]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), values, color="#4C78A8")
    ax.axhline(baseline_300["validation_alpha_mae"], color="#E45756", linestyle="--", label="baseline")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Validation alpha MAE (a0^3) at 300 steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "gate1_300step_mae.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, points in fidelity.items():
        xs = sorted(points)
        ax.plot(xs, [points[x] for x in xs], marker="o", linewidth=2, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Optimizer steps")
    ax.set_ylabel("Endpoint validation alpha MAE (a0^3)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "fidelity_trajectory.png", dpi=180)
    plt.close(fig)

    if stable_records:
        names = ["baseline"] + ["evo-{}".format(item["iteration"]) for item in stable_valid]
        maes = [stable_records[0]["metrics"]["validation_alpha_mae"]] + [
            item["metrics"]["validation_alpha_mae"] for item in stable_valid
        ]
        if stable_random_valid:
            names += ["random-{}".format(item["iteration"]) for item in stable_random_valid]
            maes += [item["metrics"]["validation_alpha_mae"] for item in stable_random_valid]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        colors = ["#E45756"] + ["#4C78A8"] * len(stable_valid) + ["#72B7B2"] * len(stable_random_valid)
        ax.bar(range(len(names)), maes, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_ylabel("Validation alpha MAE (a0^3) at 5,000 steps")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(ASSETS / "stable_5000_comparison.png", dpi=180)
        plt.close(fig)

    if resolved_interaction:
        contrast = resolved_interaction["interaction_contrast"]
        names = ["baseline", "drop-path only", "operator only", "joint"]
        values = [
            contrast["baseline_mae"],
            contrast["factor_a_only_mae"],
            contrast["factor_b_only_mae"],
            contrast["joint_mae"],
        ]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(range(4), values, color=["#4C78A8", "#E45756", "#72B7B2", "#54A24B"])
        ax.set_xticks(range(4))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("Validation alpha MAE (a0^3) at 5,000 steps")
        ax.set_title("ACTION × OPERATOR counterfactual contrast")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(ASSETS / "interaction_contrast.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(1, len(cumulative) + 1), cumulative, color="#F58518", linewidth=2)
    ax.axhline(5.0, color="#E45756", linestyle="--", label="stage-one hard cap")
    ax.set_xlabel("Ledger event")
    ax.set_ylabel("Cumulative A100-hours")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "budget_ledger.png", dpi=180)
    plt.close(fig)

    gate_rows = [
        (
            item["candidate_name"],
            "{:.4f}".format(item["validation_alpha_mae"]),
            "{:,}".format(item["parameter_count"]),
            "{:.1f}".format(item["training_time_sec"]),
        )
        for item in gate
    ]
    fidelity_rows = [
        (name,) + tuple("{:.4f}".format(fidelity[name][step]) if step in fidelity[name] else "—" for step in (300, 1000, 5000))
        for name in ("baseline", "factorized", "random")
    ]
    fingerprint_rows = [
        (
            name,
            "{:.3f}".format(values["lr_shock_ratio"]),
            "{:.3f}".format(values["warmup_recovery_ratio"]),
            "{:.3f}".format(values["early_to_warmup_rank_risk"]),
        )
        for name, values in fingerprints.items()
    ]
    stable_rows = []
    if stable_records:
        stable_rows.append(("baseline", "BASELINE", "{:.6f}".format(stable_records[0]["metrics"]["validation_alpha_mae"]), "3,531,715"))
    stable_rows.extend(
        (
            "evo-{}".format(item["iteration"]),
            item.get("selected_factor", ""),
            "{:.6f}".format(item["metrics"]["validation_alpha_mae"]),
            "{:,}".format(item["metrics"]["parameter_count"]),
        )
        for item in stable_valid
    )
    stable_rows.extend(
        (
            "random-{}".format(item["iteration"]),
            item.get("factor", ""),
            "{:.6f}".format(item["metrics"]["validation_alpha_mae"]),
            "{:,}".format(item["metrics"]["parameter_count"]),
        )
        for item in stable_random_valid
    )

    trust_text = "尚无完整跨保真度数据。"
    if trust_report:
        trust_text = (
            "300→5000 steps 的 Spearman={:.3f}、Kendall τ={:.3f}、top-1 recall={:.3f}、"
            "normalized selection regret={:.3f}；信任判定为 **{}**（{}）。"
        ).format(
            trust_report["spearman"],
            trust_report["kendall_tau"],
            trust_report["top_k_recall"],
            trust_report["normalized_selection_regret"],
            "PASS" if trust_report["trustworthy"] else "FAIL",
            trust_report["reason"],
        )

    paired_stop = (RUNS / "stable_factorized_seed45" / "paired_budget_stop.json").exists()
    stable_statement = "修订后的 5,000-step 搜索尚在运行。"
    if paired_stop:
        stable_statement = "按配对预算设计，自进化侧在完成 {} 个候选后有意停止。".format(
            len(stable_children)
        )
    elif len(stable_children) == 4:
        stable_statement = "修订后的 4 候选 5,000-step 搜索已经完成。"
    random_statement = "匹配的 5,000-step 随机对照尚未完成。"
    if stable_random.get("iterations"):
        random_statement = "匹配随机对照已完成 {} 个提案、{} 个训练合法候选。".format(
            stable_random["iterations"], stable_random.get("valid", 0)
        )
    stable_pilot_decision = "等待匹配随机对照完成。"
    if stable_best and stable_random_best:
        evo_mae = stable_best["metrics"]["validation_alpha_mae"]
        random_mae = stable_random_best["metrics"]["validation_alpha_mae"]
        baseline_mae = stable_records[0]["metrics"]["validation_alpha_mae"]
        if evo_mae < random_mae and evo_mae < baseline_mae:
            stable_pilot_decision = (
                "best-so-far pilot 为 GO：自进化最佳 {:.6f}，随机最佳 {:.6f}，"
                "baseline {:.6f}；但每侧仅 2 个训练合法候选，不能作统计优越性声明。"
            ).format(evo_mae, random_mae, baseline_mae)
        else:
            stable_pilot_decision = (
                "pilot 未显示自进化 best-so-far 优势：自进化最佳 {:.6f}，"
                "随机最佳 {:.6f}，baseline {:.6f}；扩大搜索前应先调整策略。"
            ).format(evo_mae, random_mae, baseline_mae)
    interaction_statement = "交互反事实尚未完成。"
    gradient_gate_statement = "新梯度门尚无完成的训练候选记录。"
    if resolved_interaction:
        contrast = resolved_interaction["interaction_contrast"]
        interaction_statement = (
            "operator-only MAE 为 **{:.6f} a₀³**；OPERATOR 在无 drop-path 时的增益为 "
            "**{:.6f}**，在 drop-path=0.05 条件下的增益为 **{:.6f}**，"
            "difference-in-differences epistasis 为 **{:.6f}**。ACTION-only 增益为 "
            "**{:.6f}**，且 joint 比 operator-only 高 **{:.6f}** MAE；因此负 epistasis "
            "表示 OPERATOR 对有害 ACTION 的救援强于加性预期，并不表示保留 ACTION 后的 joint 最优。"
        ).format(
            contrast["factor_b_only_mae"],
            contrast["factor_b_gain_without_a"],
            contrast["factor_b_gain_with_a"],
            contrast["epistasis_mae"],
            contrast["factor_a_gain"],
            contrast["joint_mae"] - contrast["factor_b_only_mae"],
        )
        gradient_health = resolved_interaction.get("counterfactual_metrics", {}).get(
            "gradient_health", {}
        )
        if gradient_health:
            post_symmetry = resolved_interaction.get("counterfactual_metrics", {}).get(
                "posttrain_symmetry_report", {}
            )
            symmetry_suffix = ""
            if post_symmetry:
                symmetry_suffix = (
                    " 训练后 rotation max={:.6f}、translation max={:.6f}、"
                    "permutation={:.6f}；低于 0.25 硬阈值，但 rotation 略高于 0.01 warning。"
                ).format(
                    post_symmetry["rotation_invariance"]["maximum"],
                    post_symmetry["translation_invariance"]["maximum"],
                    post_symmetry["permutation_invariance"]["maximum"],
                )
            gradient_gate_statement = (
                "反事实候选的训练前梯度门：loss={:.6f}，global L2 norm={:.6f}，"
                "max |grad|={:.6f}，all_finite={}。{}"
            ).format(
                gradient_health["loss"],
                gradient_health["global_l2_norm"],
                gradient_health["maximum_absolute_gradient"],
                gradient_health["all_finite"],
                symmetry_suffix,
            )
    trained_joint_statement = "联合候选尚无独立 checkpoint 对称性复核。"
    if trained_joint_report:
        trained_joint_statement = (
            "联合候选 checkpoint 的 5 次旋转复核：rotation max={:.6f}，"
            "translation max={:.6f}，permutation={:.6f}。"
        ).format(
            trained_joint_report["rotation_invariance"]["maximum"],
            trained_joint_report["translation_invariance"]["maximum"],
            trained_joint_report["permutation_invariance"]["maximum"],
        )

    factor_gain = 100.0 * (
        baseline_300["validation_alpha_mae"] - factor_best["metrics"]["validation_alpha_mae"]
    ) / baseline_300["validation_alpha_mae"]
    schema_valid_rate = 100.0 * sum(
        bool(item.get("metrics", {}).get("valid")) for item in schema_records
    ) / max(1, len(schema_records))
    schema_before_rate = 100.0 * sum(
        bool(item.get("metrics", {}).get("valid")) for item in schema_before_repair
    ) / max(1, len(schema_before_repair))
    schema_after_rate = 100.0 * sum(
        bool(item.get("metrics", {}).get("valid")) for item in schema_after_repair
    ) / max(1, len(schema_after_repair))

    report = f"""# OpenEvolve × SPARK × Equiformer：等变神经网络架构自进化框架第一阶段报告

> 数据集：QM9；任务：极化率 α（`target=1`，MAE 单位为 a₀³）  
> 硬件：单卡 NVIDIA A100-SXM4-80GB  
> 搜索协议：batch size 64、验证集选择、测试集锁定  
> 最终协议：257,700 optimizer steps × 3 seeds，仅在候选冻结并单独批准后运行  
> 第一阶段预算：累计硬上限 5 A100-hours

## 1. 结论摘要

第一阶段已经形成真实可运行的融合框架，而不是把三个仓库串成脚本：OpenEvolve 提供 islands、lineage、MAP-Elites 和候选采样；SPARK 启发的 RC/SAR 将“反思”和“结构修改”分离；Equiformer 被封装为可信构建器，LLM 只能修改类型化的等变架构描述，不能修改数据、损失、优化器、预算或评测器。

本阶段最重要的研究发现不是“短训练已经找到更好模型”，而是：**QM9 α 上的极短训练排名会在学习率相位变化后系统性反转。** 因而低 fidelity 必须先通过排名信任检验，不能天然拥有候选晋级权。框架据此新增 LRPF 与 SCFTG，使错误代理能够被自动识别和关闭。

当前证据支持“工程与方法原型成立”，但不支持“已经达到 CCF-A oral”或“最终性能超过论文”。这两种结论需要更大候选池、匹配基线、多 seed 和完整训练。

## 2. 融合工作流

![Framework workflow](assets/framework_workflow.png)

核心闭环：

1. **OpenEvolve substrate**：保存谱系、islands 和质量多样性 archive。
2. **ECFR**：只选择 `REPRESENTATION / OPERATOR / ACTION / MACRO` 中一个因子。
3. **RC**：LLM 只分析已有证据、建议方向和风险，不产生代码。
4. **SAR**：LLM 只返回所选因子的完整 replacement patch。
5. **SPAG**：AST 字面量解析、schema/irrep 约束、去重和参数量上限。
6. **可信 Equiformer builder**：候选描述映射到官方算子，不执行 LLM 代码。
7. **SAPF**：等变性、平移/置换不变性、forward/backward 和资源门控先于训练。
8. **LRPF + SCFTG**：学习率相位对齐，并用跨 fidelity 排名统计决定代理能否用于晋级。
9. **父子信用与 archive 更新**：只用同 fidelity 验证 MAE 和真实效率数据更新路由器。

## 3. 方法创新点

### 3.1 SPAG：对称性保持架构语法

基因型不是任意 Python，而是具有 irrep 语义的 `ArchitectureSpec`。未知字段、可执行语句、不合法角动量通道以及超过 1.2× baseline 参数量的候选在训练前被拒绝。baseline spec 精确构建出官方 **3,531,715** 参数模型。

### 3.2 ECFR：证据校准的因子路由

一次父子变化只归属于一个因子，使性能变化具有局部因果解释。路由器综合父子验证 MAE、真实 step time、参数减少、合法率、不确定性和层级对称性漂移。第一阶段审计修复了“参数比例被误当作延迟”的问题，并开始显式记录 `step_time_ms`。

### 3.3 等变容量质量多样性地图

OpenEvolve 的 MAP-Elites 坐标被改为 `lmax`、高阶通道比例、参数比例和网络深度，而不是源代码长度。这使 archive 保留不同角动量容量与计算规模的精英，而不只是单一 MAE 最优点。

零训练 smoke 中共有 **{qd_summary.get('unique_architectures', '—')}** 个唯一架构，并占据 **{qd_cells}** 个等变容量 cell。为避免 OpenEvolve 动态 min–max 导致旧 cell 随生成顺序失真，分箱尺度预注册为 `lmax 1–3`、高阶比例 `0–1`、参数比 `0.5–1.2`、深度 `3–8`；本次实际观测覆盖 `lmax={qd_lmax_range[0]:.0f}–{qd_lmax_range[1]:.0f}`、高阶通道比例 `{qd_higher_range[0]:.4f}–{qd_higher_range[1]:.4f}` 和参数比例 `{qd_parameter_range[0]:.4f}–{qd_parameter_range[1]:.4f}`。4 个新候选全部合法，其中 **{qd_repair_count}** 次 schema 错误由同因子 compiler repair 实时修复。该结果验证了 OpenEvolve archive 和修复闭环确实运行，而非仅在文档中声明。

### 3.4 LRPF：学习率相位感知 fidelity

300、1,000 和 5,000 steps 分别处于不同优化器相位。框架记录跃迁前 MAE、跃迁冲击比、warmup 恢复比和 early-rank risk，禁止把不同相位的历史最优值混为同一 fidelity 端点。

### 3.5 SCFTG：自校准 fidelity 信任门

低成本代理只有在最小共享候选池上同时通过 Spearman、Kendall τ、top-k recall 和 selection regret 阈值，才获得晋级权；否则只保留为轨迹数据。该机制直接针对“快速但误导”的等变 NAS 搜索。

### 3.6 科学语义与统一修复门

schema 编译失败、重复架构和 evaluator 拒绝均进入同一因子内的修复循环；修复不能改变路由器选定的因子。同时，确定性语义守卫阻止两类物理错误进入 lineage：把 QM9 α 错称为二阶张量目标，以及由标量输出错误推出高阶隐藏 irreps 无用。`irreps_head` 被明确限定为内部注意力表示，而非输出头。

### 3.7 IACC：交互感知反事实信用

因子局部修改并不意味着因素效应可加。若一个候选显著救援了相对祖先已经退化的父代，IACC 自动提出 sibling 反事实：把新因子 patch 直接应用到祖先，移除前一个因子变化。2×2 difference-in-differences 将新因子的主效应与 ACTION×OPERATOR 等交互效应分离；在反事实完成前，该次 MAE 信用保持 provisional，不直接强化路由器。

## 4. 安全性、可复现性与预算

- 已审计 **{len(training_summaries)}** 个训练 summary，`test_evaluated=true` 数量为 **{test_evaluated_count}**；搜索期间测试集保持锁定。
- LLM 密钥仅保存在服务器 mode-600 的 `~/.config/openevolve/apis.env`；代码与报告不含密钥。网络使用 SSH 反向代理 `127.0.0.1:12356`，未使用火山引擎付费网际快车。
- batch size 固定为 64，总 step 轴与原 batch-128 的 859 steps/epoch 学习率协议一致。
- 固定 seed 和按 data-cycle 派生的 shuffle；断点恢复参数最大差异约 `7.2e-6`，符合 CUDA scatter 原子归约的非位级确定性。
- 训练前执行参数量、构建、对称性和预算检查。
- 最终标量输出的旋转/平移/置换误差是硬门；layerwise hook profile 仅作 warning。第一阶段发现未校准的内部 irrep 变换公式也会把官方 Gaussian baseline 标为约 0.40，因此不能把该 profile 直接当结构失效证据。
- Bessel 家族还出现“随机初始化 sibling 的输出旋转误差 0.738，而训练后联合候选为 0.00926”的差异。因此训练前 profile 只预警；真正决定 archive 资格的是同 fidelity 训练完成后的 checkpoint 审计。
- 最终自动化测试记录：`{test_summary}`。
- 当前累计计费 **{budget_hours:.3f} A100-hours**；硬上限 5 小时。
- 预算估计使用同 fidelity 历史中位耗时并加 20% 安全裕量。

![Budget ledger](assets/budget_ledger.png)

代码版本：Equiformer `{git_revision(Path('/home/20262202788/equiformer'))}`（dirty={git_dirty(Path('/home/20262202788/equiformer'))}）；OpenEvolve `{git_revision(Path('/home/20262202788/openevolve'))}`（dirty={git_dirty(Path('/home/20262202788/openevolve'))}）；SPARK `{git_revision(Path('/home/20262202788/SPARK'))}`（dirty={git_dirty(Path('/home/20262202788/SPARK'))}）。Equiformer/OpenEvolve 的 dirty 状态包含本实验适配器、配置、缓存或数据；最终 manifest 对交付代码和证据逐文件计算 SHA-256。

## 5. Gate 1：搜索空间覆盖实验

{table(["候选", "300-step Val MAE (a₀³)", "参数量", "训练秒"], gate_rows)}

![Gate-1 results](assets/gate1_300step_mae.png)

这些结果证明四类结构决策能够改变优化行为和计算规模，但 300-step 指标不能作为最终架构优劣证据。

## 6. 初始自进化与随机搜索

- 改进后的 schema-only LLM 候选合法率：**{schema_valid_rate:.1f}%**。
- 新质量多样性地图首次 smoke 在缺少 compiler repair 时合法率为 **{schema_before_rate:.1f}%**；统一 compiler/duplicate repair 后，8 个候选合法率为 **{schema_after_rate:.1f}%**。
- 300-step 自进化最佳：**{factor_best['metrics']['validation_alpha_mae']:.4f}**。
- 300-step random best：**{random_best['metrics']['validation_alpha_mae']:.4f}**。
- 300-step baseline：**{baseline_300['validation_alpha_mae']:.4f}**。
- 自进化短期相对 baseline 改善：**{factor_gain:.1f}%**。

历史最佳父子链为 `alpha_drop 0.2→0.1`，再将 `radial_hidden [64,64]→[32,32]`；参数减少约 7.5%。但该候选在 5,000 steps 退化，因此它只能支持“早期优化动力学 insight”，不能支持最终架构结论。

## 7. 跨 fidelity 排名反转

{table(["方法", "300 steps (a₀³)", "1,000 steps", "5,000 steps"], fidelity_rows)}

![Fidelity trajectory](assets/fidelity_trajectory.png)

{table(["方法", "LR shock", "warmup recovery", "early-rank risk"], fingerprint_rows)}

{trust_text}

这意味着当前 300-step 代理被 SCFTG 明确判为 **NO-GO**。特别是 300-step winner 到 5,000 steps 反而最差，而随机候选成为最好；隐藏这一点会产生错误的 NAS 结论。

## 8. Warmup-end 匹配实验

{stable_statement} {random_statement}

{table(["候选", "修改因子", "5,000-step Val MAE (a₀³)", "参数量"], stable_rows) if stable_rows else '尚无完成记录。'}

{'![Stable 5000 comparison](assets/stable_5000_comparison.png)' if stable_records else ''}

该实验只用于判断搜索策略是否值得扩大，不用于宣称最终测试性能。本配对实验仅完成 {len(stable_children)} 次自进化评估，尚未覆盖全部因子，也不足以验证 ECFR 的自适应路由优势；后续必须在独立批准的更大候选池中运行超过四次迭代。

**匹配 pilot 判定：** {stable_pilot_decision}

{trained_joint_statement}

## 9. IACC 交互反事实

第二个自进化候选形成“退化后救援”链：baseline → `drop_path=0.05` → `drop_path=0.05 + Bessel/64 bases`。IACC 不把救援幅度直接全部记给 OPERATOR，而是补跑移除 ACTION 改动的 sibling。

{interaction_statement}

{gradient_gate_statement}

{'![Interaction contrast](assets/interaction_contrast.png)' if resolved_interaction else ''}

## 10. 当前可以成立与不能成立的结论

可以成立：

- OpenEvolve、SPARK 式双阶段推理和可信等变架构编译器形成了真实闭环。
- 类型化 genotype 显著提高候选合法率，并阻止 LLM 修改实验协议。
- 极短 fidelity 会被学习率相位系统性误导；LRPF/SCFTG 是由实验反例驱动的必要机制。
- 搜索期间没有使用测试集，预算有可审计硬约束。

不能成立：

- 尚不能证明 ECFR 比匹配随机搜索具有统计显著优势。
- 尚不能证明对称性漂移可以预测最终 MAE。
- 尚不能证明任何候选在 257,700 steps、3 seeds 下超过 Equiformer 论文结果。
- 尚不能声称达到 CCF-A oral；当前是具备明确创新假设和负结果证据的第一阶段研究原型。

## 11. 第一阶段 Go/No-Go

- **框架工程：GO。** 安全 genotype、真实 Equiformer 构建、OpenEvolve archive、RC/SAR、预算与测试集隔离均已实现。
- **300/1,000-step accuracy proxy：NO-GO。** 排名证据不可信。
- **5,000-step 稳定搜索策略：{stable_pilot_decision}**
- **257,700-step 完整训练：NO-GO（当前阶段）。** 只有稳定搜索产生可信候选且完成预注册消融后才单独批准。

## 12. 复现入口与证据目录

- 方法说明：`METHOD.md`
- 搜索入口：`scripts/run_factorized_evolution.py`
- 随机基线：`scripts/run_random_search.py`
- 可信评测：`equivariant_nas/pipeline.py`
- fixed-step trainer：`equivariant_nas/training/fixed_step_trainer.py`
- fidelity trust：`equivariant_nas/fidelity_trust.py`
- 预算账本：`runs/budget_ledger.jsonl`
- 固定分箱 QD/repair smoke：`runs/smoke_qd_repair_seed51/`
- 300→5,000 fidelity 信任报告：`runs/fidelity_trust_300_to_5000.json`
- 第一阶段稳定搜索：`runs/stable_factorized_seed45/`
- 第一阶段匹配随机搜索：`runs/stable_random_seed45/`
- 交互反事实：`runs/stage1_interaction_ablation/`

## 13. 直接相关文献与软件

1. Yi-Lun Liao, Tess Smidt. *Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs*. ICLR 2023. [OpenReview](https://openreview.net/forum?id=KwmPfARgOTD), [arXiv:2206.11990](https://arxiv.org/abs/2206.11990).
2. Asankhaya Sharma. *OpenEvolve: an open-source evolutionary coding agent*. GitHub software, 2025. [Repository](https://github.com/algorithmicsuperintelligence/openevolve).
3. Zhen Liu et al. *Structured Progressive Knowledge Activation for LLM-Driven Neural Architecture Search*. arXiv preprint 2605.04057, 2026. [arXiv](https://arxiv.org/abs/2605.04057). 本报告将其作为 SPARK 方法来源，不把预印本状态表述为正式会议录用。
"""
    path = OUT / "stage1_report.md"
    path.write_text(report, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
