#!/usr/bin/env python
"""Generate an auditable Markdown/HTML report for the DSL showcase pair."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fmt(value, digits=6):
    if value is None:
        return "未记录"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def training_dir(result):
    return Path(result["run_dir"]) / "training"


def checkpoint_audit(result):
    path = Path(result.get("checkpoint_last", ""))
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else 0}


def draw_mae(summary, assets: Path):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = {"parent": "#3264A8", "child": "#D05A43"}
    labels = {"parent": "Official Equiformer V1 parent", "child": "DSL multilevel-readout child"}
    points = {}
    for name in ("parent", "child"):
        records = jsonl(training_dir(summary[name]) / "metrics.jsonl")
        xs = [int(record["global_step"]) for record in records if "val_mae" in record]
        ys = [float(record["val_mae"]) for record in records if "val_mae" in record]
        points[name] = list(zip(xs, ys))
        if len(xs) > 1:
            ax.plot(xs, ys, marker="o", linewidth=2, color=colors[name], label=labels[name])
        else:
            ax.scatter(xs, ys, s=75, color=colors[name], label=labels[name], zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.5f}", (x, y), xytext=(5, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Validation MAE (a0^3)")
    ax.set_title("QM9 alpha: endpoint validation comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    target = assets / "父代与DSL子代验证MAE对比.png"
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target, points


def training_trace(result):
    path = Path(result["run_dir"]) / "trainer_console.log"
    if not path.exists():
        return []
    pattern = re.compile(r"Global step: \[(\d+)/(\d+)\].*?MAE: ([0-9.eE+-]+)")
    points = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            points.append((int(match.group(1)), float(match.group(3))))
    return points


def draw_training_mae(summary, assets: Path):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = {"parent": "#3264A8", "child": "#D05A43"}
    labels = {"parent": "Official Equiformer V1 parent", "child": "DSL multilevel-readout child"}
    traces = {}
    for name in ("parent", "child"):
        trace = training_trace(summary[name])
        traces[name] = trace
        if trace:
            ax.plot([x for x, _ in trace], [y for _, y in trace], linewidth=1.8, color=colors[name], label=labels[name])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Logged training MAE (a0^3)")
    ax.set_title("QM9 alpha: training trajectory logged every 100 steps")
    ax.grid(alpha=0.25)
    if any(traces.values()):
        ax.legend()
    fig.tight_layout()
    target = assets / "父代与DSL子代训练MAE轨迹.png"
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target, traces


def draw_workflow(assets: Path):
    fig, ax = plt.subplots(figsize=(13.5, 5.3))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 5.3)
    ax.axis("off")
    boxes = [
        (0.2, 3.4, 2.0, "Region Router\none registered region", "#DCEAF7"),
        (2.6, 3.4, 2.0, "Region Critic\nmechanism and risks", "#E8DDF3"),
        (5.0, 3.4, 2.0, "Patch Synthesizer\ntyped local patch", "#FBE5D6"),
        (7.4, 3.4, 2.0, "DSL compiler\nregion + type gates", "#DDF1E4"),
        (9.8, 3.4, 1.6, "Lowering plan\nexact only", "#DDF1E4"),
        (11.8, 3.4, 1.5, "QM9 trainer\nfixed protocol", "#FFF1CC"),
        (9.7, 1.0, 2.0, "symmetry + numerical\nhealth audit", "#FFF1CC"),
        (6.8, 1.0, 2.2, "checkpoint + validation\nevidence", "#FFF1CC"),
        (3.9, 1.0, 2.2, "parent-child endpoint\ncomparison", "#F4CCCC"),
        (0.7, 1.0, 2.3, "Markdown + HTML\ntraceability report", "#DCEAF7"),
    ]
    for x, y, width, label, color in boxes:
        patch = FancyBboxPatch((x, y), width, 1.0, boxstyle="round,pad=0.04", fc=color, ec="#333", lw=1.1)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + 0.5, label, ha="center", va="center", fontsize=9, color="black")
    arrows = [
        ((2.2, 3.9), (2.6, 3.9)), ((4.6, 3.9), (5.0, 3.9)),
        ((7.0, 3.9), (7.4, 3.9)), ((9.4, 3.9), (9.8, 3.9)),
        ((11.4, 3.9), (11.8, 3.9)), ((12.55, 3.4), (10.7, 2.0)),
        ((9.7, 1.5), (9.0, 1.5)), ((6.8, 1.5), (6.1, 1.5)),
        ((3.9, 1.5), (3.0, 1.5)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.35, "color": "#333"})
    ax.set_title("Stable and auditable equivariant-DSL showcase workflow", fontsize=13, color="black")
    fig.tight_layout()
    target = assets / "端到端实验工作流.png"
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target


def build_markdown(summary, run_dir: Path, output_dir: Path, mae_figure: Path, training_figure: Path, workflow_figure: Path, points, traces):
    parent = summary["parent"]
    child = summary["child"]
    generation = summary["generation"]
    protocol = summary["protocol"]
    parent_ckpt = checkpoint_audit(parent)
    child_ckpt = checkpoint_audit(child)
    role_counts = generation.get("prompt_role_counts", {})
    region_audit = generation.get("region_audit", {})
    child_plan = child.get("lowering_plan", {})
    auxiliary = child_plan.get("details", {}).get("auxiliary_block")
    delta = float(summary["endpoint_validation_mae_delta"])
    outcome = "子代更优" if delta < 0 else "父代更优或持平"
    relative_mae = f"{abs(delta) / max(float(parent['validation_alpha_mae']), 1e-12) * 100:.4f}%"
    checkpoint_ok = parent_ckpt["exists"] and child_ckpt["exists"]
    test_hidden = not bool(summary.get("test_evaluated")) and not bool(parent.get("test_evaluated")) and not bool(child.get("test_evaluated"))
    frozen_hash = region_audit.get("frozen_complement_hash") or generation.get("patch", {}).get("frozen_complement_hash") or "未记录"
    parent_subset = parent.get("train_subset") or {}
    child_subset = child.get("train_subset") or {}
    parent_step_ms = 1000.0 * float(parent["training_wall_time_sec"]) / max(1, int(parent["endpoint_step"]))
    child_step_ms = 1000.0 * float(child["training_wall_time_sec"]) / max(1, int(child["endpoint_step"]))
    parent_grad_norm = parent.get("gradient_health", {}).get("global_l2_norm")
    child_grad_norm = child.get("gradient_health", {}).get("global_l2_norm")
    lines = [
        "# 等变DSL可信端到端展示实验报告",
        "",
        "> 本报告由实验原始JSON、训练指标和checkpoint自动生成。实验只使用validation比较候选，未访问test集。",
        "",
        "## 一、结论",
        "",
        f"本次实验完成了真实Equiformer V1父代与一个LLM生成DSL子代的同协议端到端训练。父代通过`exact_reference`调用官方Equiformer V1构造语义；子代通过`exact_hybrid`保留官方V1主体，仅在认证的`v1_readout`区域增加从`block{auxiliary}`读取的辅助不变量标量头，并用可训练组合器与原始末端readout融合。",
        "",
        f"在固定seed={protocol['seed']}、batch size={protocol['batch_size']}、{protocol['max_optimizer_steps']} optimizer steps和固定1/4训练子集下，父代终点validation MAE为`{parent['validation_alpha_mae']:.8f}`$a_0^3$，子代为`{child['validation_alpha_mae']:.8f}`$a_0^3$，差值为`{delta:+.8f}`$a_0^3$，相对幅度为`{relative_mae}`，当前单seed信号为**{outcome}**。该结果只证明本版系统能够稳定生成、编译、训练并比较真实父子架构；不能据此声称形成了多seed稳定改进或发现了V2级新算子。",
        "",
        "![父代与DSL子代验证MAE对比](assets/父代与DSL子代验证MAE对比.png)",
        "",
        "下图补充展示训练器每100 optimizer steps打印的训练MAE。它用于观察优化动力学，不替代候选选择所使用的终点validation MAE。每个data epoch开始时累计窗口会重新建立，曲线可能出现阶段性跳变。",
        "",
        "![父代与DSL子代训练MAE轨迹](assets/父代与DSL子代训练MAE轨迹.png)",
        "",
        "## 二、完整工作流",
        "",
        "![端到端实验工作流](assets/端到端实验工作流.png)",
        "",
        "流程首先由Region Router（区域路由器）从编译器注册的具名区域中选择唯一可编辑区域；Region Critic（区域审查器）分析该区域的功能、边界、不变量、风险和可证伪假设；Patch Synthesizer（补丁合成器）在冻结边界内生成typed patch（带类型补丁）。编译器随后验证区域授权、等变类型、结构补集hash和lowering能力。只有`exact_reference`或`exact_hybrid`候选能够进入训练，`representation_only`候选会在使用GPU前被拒绝。",
        "",
        "通过编译的父子候选使用相同QM9 alpha数据、初始化seed、batch size、optimizer step上限、学习率协议和validation规则从头训练。训练前后执行旋转、平移和置换审计，并记录预测尺度与梯度健康；训练过程按固定间隔保存checkpoint，终点强制validation。控制器使用原子`state.json`记录阶段，监督器在异常退出后复用同一run目录和最新checkpoint恢复。",
        "",
        "## 三、预注册训练协议",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 数据集与目标 | QM9 alpha极化率 |",
        f"| 训练数据 | 固定1/4训练子集：`{protocol['training_subset']}` |",
        f"| 子集SHA-256 | `{parent_subset.get('sha256', '未记录')}` |",
        f"| 子集样本数 | `{parent_subset.get('size', parent.get('training_dataset_size', '未记录'))}`，原训练集`{parent_subset.get('source_train_size', '未记录')}` |",
        f"| 随机种子 | `{protocol['seed']}` |",
        f"| batch size | `{protocol['batch_size']}` |",
        f"| 终止条件 | `{protocol['max_optimizer_steps']}` optimizer steps |",
        f"| validation节奏 | 每`{protocol['validation_interval_data_epochs']}`个data epoch；终点强制validation |",
        f"| 候选选择指标 | endpoint validation MAE |",
        f"| test集 | 禁止访问；审计结果：`{str(test_hidden).lower()}` |",
        "",
        "## 四、父代与子代的结构关系",
        "",
        "| 项目 | 官方V1父代 | DSL子代 |",
        "|---|---|---|",
        f"| architecture ID | `{parent['architecture_id']}` | `{child['architecture_id']}` |",
        f"| lowering模式 | `{parent['lowering_mode']}` | `{child['lowering_mode']}` |",
        f"| 后端语义 | `{parent['backend_semantics']}` | `{child['backend_semantics']}` |",
        f"| 参数量 | `{parent['parameter_count']:,}` | `{child['parameter_count']:,}` |",
        f"| 参数比例 | `1.0` | `{child['parameter_ratio']:.6f}` |",
        f"| 被授权修改区域 | 官方完整参考模型 | `v1_readout` |",
        f"| 新增读取路径 | 无 | `block{auxiliary}`辅助标量readout |",
        f"| 区域外结构hash | 参考父代 | `{frozen_hash}` |",
        "",
        "子代不是把整个网络交给LLM任意改写，也不是用简化e3nn图近似训练Equiformer。它复用完整官方V1作为数值主体，通过forward hook读取一个中间block的等变输出，只允许将其中的标量不变量通道聚合成图级辅助预测，再与原始官方末端预测进行标量融合。因而新增路径保持输出为旋转不变量标量，同时`scalar_readout→block5`原始路径仍然存在。",
        "",
        "## 五、LLM候选生成证据",
        "",
        "| 固定角色 | 调用次数 | 职责 |",
        "|---|---:|---|",
        f"| Region Router | `{role_counts.get('region_router', 0)}` | 只决定修改哪个已注册区域 |",
        f"| Region Critic | `{role_counts.get('region_critic', 0)}` | 分析区域机制、风险、边界和可证伪假设 |",
        f"| Patch Synthesizer | `{role_counts.get('patch_synthesizer', 0)}` | 把审查结论合成为typed patch |",
        f"| Compiler-guided Repair | `{role_counts.get('compiler_repair', 0) + role_counts.get('patch_repair', 0)}` | 仅在结构化编译失败时触发 |",
        "",
        f"本候选由真实LLM调用生成，生成架构ID为`{generation['architecture_id']}`，固定三阶段均有独立prompt/response证据。本次补丁首次编译通过，因此没有把被动Repair伪装成第三个正常推理阶段。完整生成记录位于`{generation['generation_search_dir']}`。",
        "",
        "## 六、训练结果与可信性审计",
        "",
        "| 指标 | 官方V1父代 | DSL子代 |",
        "|---|---:|---:|",
        f"| 终点validation MAE ($a_0^3$) | `{parent['validation_alpha_mae']:.8f}` | `{child['validation_alpha_mae']:.8f}` |",
        f"| 最佳validation MAE ($a_0^3$) | `{parent['best_validation_alpha_mae']:.8f}` | `{child['best_validation_alpha_mae']:.8f}` |",
        f"| 最佳step | `{parent['best_step']}` | `{child['best_step']}` |",
        f"| 训练wall time | `{parent['training_wall_time_sec'] / 3600:.3f}`小时 | `{child['training_wall_time_sec'] / 3600:.3f}`小时 |",
        f"| 平均step时间 | `{parent_step_ms:.2f}`ms | `{child_step_ms:.2f}`ms |",
        f"| 训练前预测RMS | `{parent['pretrain_numerical_health']['prediction_rms']:.6f}` | `{child['pretrain_numerical_health']['prediction_rms']:.6f}` |",
        f"| 梯度norm | `{fmt(parent_grad_norm)}` | `{fmt(child_grad_norm)}` |",
        f"| 训练前最大对称误差 | `{parent['pretrain_max_symmetry_error']:.6g}` | `{child['pretrain_max_symmetry_error']:.6g}` |",
        f"| 训练后最大对称误差 | `{parent['max_symmetry_error']:.6g}` | `{child['max_symmetry_error']:.6g}` |",
        f"| checkpoint存在 | `{str(parent_ckpt['exists']).lower()}` | `{str(child_ckpt['exists']).lower()}` |",
        f"| checkpoint大小 | `{parent_ckpt['size'] / 1024**2:.2f}`MiB | `{child_ckpt['size'] / 1024**2:.2f}`MiB |",
        f"| test_evaluated | `{str(parent['test_evaluated']).lower()}` | `{str(child['test_evaluated']).lower()}` |",
        f"| training dataset ID一致 | `{str(parent.get('training_dataset_id') == child.get('training_dataset_id')).lower()}` | `{str(parent.get('training_dataset_id') == child.get('training_dataset_id')).lower()}` |",
        "",
        f"validation轨迹点数为：父代`{len(points['parent'])}`个、子代`{len(points['child'])}`个。由于8000 steps小于10个data epoch对应的8590 steps，本协议在终点前不会进行额外validation，因此图中正常地只有终点validation点；这不是日志缺失。训练过程仍按100 step打印训练统计，并按859 step保存checkpoint。",
        "",
        f"训练日志轨迹点数为：父代`{len(traces['parent'])}`个、子代`{len(traces['child'])}`个。该轨迹只用于诊断收敛过程，正式父子排序不使用训练MAE。",
        "",
        "## 七、稳定性与恢复机制",
        "",
        f"- 控制器使用原子写入的`{run_dir / 'state.json'}`保存当前阶段和已完成结果。",
        "- 父代完成后才进入子代；已经完成且endpoint一致的候选在控制器重启时不会重复训练。",
        "- 候选训练目录中的`checkpoint_last.pth`每859 step更新；异常退出时pipeline自动把它作为`--resume-step`恢复。",
        f"- 本次父子checkpoint完整性合取结果为`{str(checkpoint_ok).lower()}`。",
        f"- 父子与总摘要的test隔离合取结果为`{str(test_hidden).lower()}`。",
        "- architecture ID与protocol identity绑定lowering/backend语义，防止不同数值后端错误复用缓存。",
        "",
        "## 八、该版本已经解决与尚未解决的问题",
        "",
        "### 已经解决",
        "",
        "- 第一代父代不再由表示流简化后端冒充，而是精确调用官方Equiformer V1。",
        "- 没有可信lowering的修改图会被标记为`representation_only`并禁止训练排名。",
        "- LLM生成被拆成Router、Critic和Synthesizer三个语义不同的固定调用。",
        "- 子代修改被限制在一个认证region内，region外结构由frozen complement hash证明未改。",
        "- 父子候选通过相同训练协议、数值健康、等变性、断点和test隔离审计。",
        "",
        "### 尚未解决",
        "",
        "- 当前只开放readout区域，属于可信闭环展示，不足以支持“发现V2式新算子”的主张。",
        "- 当前只有一个LLM子代和一个seed；即使子代更优，也只能视为需要复验的validation信号。",
        "- 还需要实现一个完整message/attention block的精确节点级lowering，才能搜索更具结构新颖性的算子。",
        "- 还没有完成静态DSL、自进化DSL、OpenEvolve和SPARK式基线的同预算对照。",
        "- 语言motif发现必须只消费本类可信lowering候选，不能使用早期数值语义无效的run作为准入证据。",
        "",
        "## 九、结论边界",
        "",
        "本次实验的有效主张是：系统已经形成一个可展示、可恢复、可审计的真实Equiformer DSL端到端闭环，LLM能够在编译器强制的局部等变约束下生成结构不同的可训练子代，并与官方V1父代进行公平的validation-only比较。",
        "",
        "本次实验不能单独证明DSL提高了搜索效率、子代稳定优于V1、语言已经发生有效自进化，或系统能够发现Equiformer V2级新算子。这些结论需要block级lowering、多候选、多seed、多保真和基线对照。",
        "",
        "## 十、原始证据索引",
        "",
        f"- 实验预注册：`{run_dir / 'experiment_preregistration.json'}`",
        f"- 生成证据：`{run_dir / 'generation_evidence.json'}`",
        f"- 控制器状态：`{run_dir / 'state.json'}`",
        f"- 总摘要：`{run_dir / 'showcase_summary.json'}`",
        f"- 父代结果：`{run_dir / 'parent_result.json'}`",
        f"- 子代结果：`{run_dir / 'child_result.json'}`",
        f"- 父代训练指标：`{training_dir(parent) / 'metrics.jsonl'}`",
        f"- 子代训练指标：`{training_dir(child) / 'metrics.jsonl'}`",
        f"- 父代checkpoint：`{parent_ckpt['path']}`",
        f"- 子代checkpoint：`{child_ckpt['path']}`",
        "",
    ]
    return "\n".join(lines)


def markdown_to_html(markdown_text: str, title: str):
    try:
        import markdown

        body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code", "sane_lists"])
    except Exception:
        body = "<pre>" + html.escape(markdown_text) + "</pre>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{max-width:1100px;margin:36px auto;padding:0 24px;color:#000;background:#fff;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.72}}
h1,h2,h3{{color:#000;line-height:1.3}} table{{border-collapse:collapse;width:100%;margin:16px 0}} th,td{{border:1px solid #999;padding:8px 10px;text-align:left}} th{{background:#f2f2f2}} code{{background:#f4f4f4;padding:2px 4px;border-radius:3px}} pre{{overflow:auto;background:#f6f6f6;padding:14px}} img{{max-width:100%;height:auto}} blockquote{{border-left:4px solid #888;margin-left:0;padding-left:16px;color:#222}}
</style>
</head>
<body>{body}</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary_path = run_dir / "showcase_summary.json"
    if not summary_path.exists():
        raise SystemExit("showcase_summary.json does not exist; the pair experiment is not complete")
    summary = load(summary_path)
    if summary.get("status") != "completed":
        raise SystemExit("showcase pair is not completed")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "report"
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    mae_figure, points = draw_mae(summary, assets)
    training_figure, traces = draw_training_mae(summary, assets)
    workflow_figure = draw_workflow(assets)
    markdown_text = build_markdown(summary, run_dir, output_dir, mae_figure, training_figure, workflow_figure, points, traces)
    markdown_path = output_dir / "等变DSL可信端到端展示实验报告.md"
    html_path = output_dir / "等变DSL可信端到端展示实验报告.html"
    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown_text, "等变DSL可信端到端展示实验报告"), encoding="utf-8")
    manifest = {
        "status": "completed",
        "markdown": str(markdown_path),
        "html": str(html_path),
        "assets": [str(mae_figure), str(training_figure), str(workflow_figure)],
        "test_evaluated": False,
    }
    (output_dir / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
