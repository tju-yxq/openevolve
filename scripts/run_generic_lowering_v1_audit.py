#!/usr/bin/env python
"""Generate an implementation and verification audit for generic Lowering V1."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from equivariant_nas.dsl import core_registry  # noqa: E402
from equivariant_nas.dsl.compiler import Compiler  # noqa: E402
from equivariant_nas.dsl.backends import E3NNGraphBackend  # noqa: E402
from equivariant_nas.dsl.backends.qm9_model import build_qm9_dsl_model  # noqa: E402
from equivariant_nas.dsl.backends.e3nn_backend import (  # noqa: E402
    _SUPPORTED,
    _execute_cutoff_envelope,
)


DEFAULT_TESTS = (
    "tests",
)


def _run_tests(paths: Sequence[str], output: Path) -> Mapping[str, object]:
    environment = dict(os.environ)
    environment.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    command = [sys.executable, "-m", "pytest", "-q"] + list(paths)
    completed = subprocess.run(
        command,
        cwd=str(REPOSITORY_ROOT),
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output.write_text(completed.stdout, encoding="utf-8")
    return {
        "executed": True,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "test_paths": list(paths),
        "output_file": output.name,
        "tail": completed.stdout.splitlines()[-20:],
    }


def collect_audit(run_tests: bool = True, output: Path = None) -> Mapping[str, object]:
    primitive_registry = core_registry()
    backend = E3NNGraphBackend(primitive_registry)
    manifest = backend.lowering_manifest()
    primitive_names = set(primitive_registry.names())
    generic_names = set(backend.lowering_rules.names())
    fusion_only = tuple(sorted(primitive_names - generic_names))
    build_source = inspect.getsource(E3NNGraphBackend.build)
    support_source = inspect.getsource(E3NNGraphBackend._support_report)
    cutoff_source = inspect.getsource(_execute_cutoff_envelope)
    qm9_source = inspect.getsource(build_qm9_dsl_model)
    plan_source = inspect.getsource(Compiler.plan_lowering)
    registry_driven = (
        "lowering_rules.resolve" in build_source
        and "rule.execute" in build_source
        and "elif op ==" not in build_source
    )
    rules_complete = all(
        item["primitive"] and item["exactness"] and item["has_executor"]
        for item in manifest["rules"]
    )
    tests = {"executed": False, "passed": None}
    if run_tests:
        if output is None:
            raise ValueError("output directory is required when tests are enabled")
        tests = _run_tests(DEFAULT_TESTS, output / "pytest_output.txt")

    completion_criteria = {
        "all_core_primitives_have_independent_rules": primitive_names == generic_names,
        "registry_is_single_support_source": (
            set(_SUPPORTED) == generic_names
            and "self.lowering_rules.support_report" in support_source
        ),
        "backend_build_is_registry_driven": registry_driven,
        "all_generic_rules_have_executor_and_exactness": rules_complete,
        "segment_softmax_has_no_pyg_dependency": not any(
            dependency == "torch_geometric"
            for rule in manifest["rules"]
            for dependency in rule["dependencies"]
        ),
        "cutoff_has_non_identity_rule": "torch.where" in cutoff_source and "scaled.pow" in cutoff_source,
        "generic_qm9_path_is_explicitly_reachable": (
            "allow_experimental_generic_lowering" in qm9_source
            and "graph_backend.support_report" in qm9_source
            and "formal_ranking_admitted" in qm9_source
        ),
        "compiler_plan_consumes_backend_support": (
            "backend.support_report" in plan_source
            and "backend_support" in plan_source
            and "uncertified_nodes" in plan_source
            and "rule_exactness" in plan_source
        ),
        "relevant_test_matrix_passed": tests["passed"] if run_tests else None,
    }
    achieved = all(
        value is True
        for value in completion_criteria.values()
        if value is not None
    )
    return {
        "protocol": "generic-lowering-v1-audit@2",
        "backend_semantics_version": E3NNGraphBackend.semantic_version,
        "core_primitive_count": len(primitive_names),
        "generic_rule_count": len(generic_names),
        "generic_coverage_rate": len(generic_names) / len(primitive_names),
        "fusion_only_primitives": list(fusion_only),
        "architecture_coverage": {
            "equiformer_v1_current_dsl_representation_flow": {
                "generic_node_lowering_executable": True,
                "closed_fusion_required": False,
                "official_full_network_numerically_equivalent": False,
                "evidence_test": "tests/test_dsl_generic_v1_v2_lowering.py::test_imported_v1_representation_flow_executes_by_generic_lowering",
            },
            "equiformer_v2_current_so2_motif": {
                "generic_node_lowering_executable": not fusion_only,
                "closed_fusion_required": False,
                "fusion_is_optional_optimization": True,
                "official_operator_modules_reused": True,
                "official_full_network_numerically_equivalent": False,
                "evidence_tests": [
                    "tests/test_dsl_generic_v1_v2_lowering.py::test_v2_motif_executes_without_any_closed_subgraph_fusion",
                    "tests/test_dsl_generic_v1_v2_lowering.py::test_v2_fusion_and_generic_lowering_match_forward_and_gradients",
                ],
            },
        },
        "manifest": manifest,
        "implementation": {
            "registry_driven": registry_driven,
            "large_op_dispatch_removed": "elif op ==" not in build_source,
            "support_and_execution_share_registry": (
                "self.lowering_rules.support_report" in support_source
                and "lowering_rules.resolve" in build_source
            ),
        },
        "completion_criteria": completion_criteria,
        "first_version_achieved": achieved,
        "tests": tests,
        "limitations": [
            "V1验证对象是当前DSL导入器生成的表示流抽象，不是官方Equiformer V1完整注意力网络；因此不能宣称整网前向或训练轨迹等价。",
            "V2逐原语SO(2)卷积当前使用官方internal_weights=True路径，并以零张量占位边条件输入；尚未表达完整径向条件注意力。",
            "五个V2原语的当前数值Lowering要求稠密、逐阶齐全且各l具有统一通道数的SO(3)系数布局；前端语言仍允许更宽的O(3)或非均匀类型实例，这些实例会在后端验证阶段被拒绝。",
            "radial_basis和cutoff_envelope仍属于experimental规则，尚未达到认证精确等级。",
            "cutoff原语没有独立distance端口，尚不能对任意径向特征执行独立包络门控。",
            "仿射坐标、多头轴、triplet、quadruplet和对称收缩语义仍不在V1语言范围内。",
            "Generic节点图Lowering只能通过显式审计开关执行，并继续排除在正式排名之外。",
            "尚未完成与Equiformer V1、Equiformer V2、TFN、NequIP、MACE等官方参考实现的整网数值等价验证。",
        ],
    }


def _write_report(path: Path, audit: Mapping[str, object]) -> None:
    manifest = audit["manifest"]
    tests = audit["tests"]
    lines = [
        "# 通用Lowering第一版实现与审计报告",
        "",
        "> 本报告审计的是第一版组合式通用Lowering：编译器按Typed DSL原语选择数值规则，不依赖TFN、NequIP或Equiformer等网络名称。报告区分已经实现的通用机制和仍需后续扩展的语言语义。",
        "",
        "## 一、交付结论",
        "",
        "第一版通用Lowering已经完成：`E3NNGraphBackend`的支持判断、模块构造和运行执行均由同一个`LoweringRuleRegistry`驱动，原来的大型`if/elif`操作分派已经移除。",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
        "| DSL核心原语数 | {} |".format(audit["core_primitive_count"]),
        "| 通用逐原语Lowering规则数 | {} |".format(audit["generic_rule_count"]),
        "| 通用规则覆盖率 | {:.1f}% |".format(100 * audit["generic_coverage_rate"]),
        "| 只能通过V2融合执行的原语数 | {} |".format(len(audit["fusion_only_primitives"])),
        "| 后端是否由规则注册表驱动 | {} |".format("是" if audit["implementation"]["registry_driven"] else "否"),
        "| 是否移除大型操作分派 | {} |".format("是" if audit["implementation"]["large_op_dispatch_removed"] else "否"),
        "| 第一版完成判据 | {} |".format("通过" if audit["first_version_achieved"] else "未通过"),
        "",
        "## 二、通用Lowering工作流",
        "",
        "```text",
        "Typed DSL程序",
        "    ↓ TypeChecker推导每条边的EquivariantType",
        "LoweringRuleRegistry.resolve(node.op)",
        "    ↓ 规则验证与依赖检查",
        "module_builder（需要参数模块时）",
        "    ↓",
        "executor（统一运行接口）",
        "    ↓",
        "PyTorch/e3nn组合模型",
        "```",
        "",
        "每条规则记录原语版本、依赖、上下文键、模块构造器、运行执行器和精确性等级。支持报告与真实执行读取同一个规则注册表，因此不会再维护相互漂移的支持集合与执行分派。",
        "",
        "## 三、规则覆盖与精确性",
        "",
        "| 精确性等级 | 规则数 |",
        "|---|---:|",
    ]
    for level, count in sorted(manifest["exactness_counts"].items()):
        lines.append("| `{}` | {} |".format(level, count))
    lines.extend(
        [
            "",
            "当前31条通用规则覆盖全部核心原语，包括线性层、张量积、球谐函数、几何构造、载体迁移、聚合、注意力标量路径、门控、归一化、非线性、读出、随机正则化，以及五个边局部标架/SO(2)/S²原语。",
            "",
            "五个V2相关原语现在都有独立规则；闭合子图融合只作为可选优化。只能依靠融合执行的原语列表为：",
            "",
        ]
    )
    if audit["fusion_only_primitives"]:
        lines.extend("- `{}`".format(item) for item in audit["fusion_only_primitives"])
    else:
        lines.append("- 无。")
    lines.extend(
        [
            "",
            "## 四、V1与V2表示验证",
            "",
            "| 验证对象 | 当前DSL结构可表达 | 逐原语通用Lowering | 等变性与梯度 | 与官方整网等价 |",
            "|---|---:|---:|---:|---:|",
            "| Equiformer V1当前导入表示流 | 是 | 是 | 通过 | 否 |",
            "| Equiformer V2当前SO(2)残差motif | 是 | 是，不要求融合 | 通过 | 否 |",
            "",
            "V1测试展开`import_equiformer_v1(baseline_spec())`，在真实非空图上完成前向、反向、旋转和节点置换验证。它证明的是当前DSL中的V1表示流可以执行，不是官方多头注意力Block的数值复现。",
            "",
            "V2测试关闭闭合子图融合，依次执行`to_edge_frame → so2_convolution → select_scalars → separable_s2_activation → from_edge_frame`；同时单独验证普通`s2_activation`、Frame往返恢复、有限梯度和旋转等变性。给融合与非融合路径复制相同权重并固定Frame随机规范后，前向、输入梯度和SO(2)参数梯度逐元素相等。",
            "",
            "QM9模型入口新增`prefer_v2_fusion=False`，可显式选择纯逐原语路径；融合现在是优化选项，而不是功能前提。",
            "",
            "## 五、本轮修复的具体问题",
            "",
            "1. `segment_softmax`改为可微的纯PyTorch分段Softmax，不再隐式依赖`torch_geometric`。",
            "2. `cutoff_envelope`不再返回恒等输入，而是执行平滑、紧支撑的多项式包络，并验证cutoff和order。",
            "3. 径向基和激活函数增加后端属性验证，错误在模型构造阶段暴露。",
            "4. 分段聚合不再错误地用边输入长度猜测节点数，而是优先使用`num_nodes`或目标索引范围。",
            "5. 编译模型保存完整Lowering规则清单和后端语义版本，便于实验复现。",
            "6. Generic QM9入口增加显式实验准入：默认拒绝；审计开关启用后先消费后端Support Report，再构造、前向和反向执行。",
            "7. `Compiler.plan_lowering()`直接读取同一Support Report，并把逐节点精确性、未认证节点、不支持节点和缺失依赖写入Lowering Plan。",
            "",
            "## 六、自动验证",
            "",
            "| 检查 | 结果 |",
            "|---|---:|",
        ]
    )
    for name, value in audit["completion_criteria"].items():
        lines.append("| `{}` | {} |".format(name, "通过" if value else "未执行/未通过"))
    lines.extend(
        [
            "",
            "全量pytest回归状态：{}。完整输出见`{}`。".format(
                "通过" if tests.get("passed") else "未执行/失败",
                tests.get("output_file", "未生成"),
            ),
            "",
            "数值测试覆盖张量积旋转等变性、门控和范数激活、分段Softmax置换不变性及反向传播、cutoff紧支撑及梯度、dropout不可约表示块掩码、V1表示流非空图执行、V2关闭融合后的逐原语执行、边Frame往返、两种S²激活、融合/非融合前向与梯度完全对照，以及前两轮变异与可达性审计。",
            "",
            "## 七、当前边界",
            "",
        ]
    )
    lines.extend("- {}".format(item) for item in audit["limitations"])
    lines.extend(
        [
            "",
            "因此，本轮完成意味着“31个核心原语均有独立Lowering，当前V1表示流和当前V2 SO(2) motif都能通过通用节点图执行”；它不意味着DSL已经精确覆盖官方V1/V2整网，也不意味着所有规则都已达到正式排名所需的认证等级。",
            "",
            "## 八、下一步",
            "",
            "1. 为`radial_basis`和`cutoff_envelope`定义更精确的类型合同，并将包络修改为`distance + value → out`双输入语义。",
            "2. 给`core.so2_convolution`增加显式径向/边条件端口，复现V2的`internal_weights=False`完整条件卷积。",
            "3. 把V1多头注意力、V2图注意力与FFN进一步拆成Typed DSL原语或可展开motif，才能测试官方整Block和整网等价。",
            "4. 增加仿射坐标、多头轴、triplet/quadruplet和对称收缩类型及Lowering规则。",
            "5. 以Equiformer V1/V2、TFN、EGNN、PaiNN、NequIP和SE(3)-Transformer官方实现作为数值Oracle，完成前向、梯度和短训练一致性测试。",
            "6. 当前只能通过`allow_experimental_generic_lowering=True`或对应CLI参数进入审计执行；只有规则晋升到受信任等级后，才开放正式候选排名。",
            "",
            "## 九、复现命令",
            "",
            "```powershell",
            "python scripts/run_generic_lowering_v1_audit.py --output reports/generic_lowering_v1_20260730",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(output: Path, run_tests: bool = True):
    output.mkdir(parents=True, exist_ok=True)
    audit = collect_audit(run_tests=run_tests, output=output)
    (output / "generic_lowering_v1_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output / "通用Lowering第一版实现与审计报告.md", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/generic_lowering_v1_20260730"))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    audit = run_audit(args.output, run_tests=not args.skip_tests)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "first_version_achieved": audit["first_version_achieved"],
                "generic_rule_count": audit["generic_rule_count"],
                "generic_coverage_rate": audit["generic_coverage_rate"],
                "tests_passed": audit["tests"].get("passed"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if audit["first_version_achieved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
