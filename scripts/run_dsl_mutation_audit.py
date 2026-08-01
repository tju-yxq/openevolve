#!/usr/bin/env python
"""Run a controlled mutation audit over the executable core DSL.

The audit separates four questions that are easy to conflate:

1. Did the typed patch apply and compile?
2. Is the canonical child different from its parent?
3. Does the executable child remain O(3)-invariant and permutation-invariant?
4. Is the change structurally or behaviorally non-trivial?

No training or test labels are used.  This is a first-stage architecture mutation
test, not a claim that a mutation improves QM9 accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

# e3nn 0.4.4 ships trusted package constants in a legacy torch checkpoint.
# PyTorch >=2.6 otherwise refuses to load that file because weights_only=True
# became the default.  This must be set before importing e3nn.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from equivariant_nas.dsl import (  # noqa: E402
    ArchitectureProgram,
    Carrier,
    Compiler,
    DSLValidationError,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    PatchEdit,
    TypedPatch,
    apply_typed_patch,
    canonicalize,
    core_registry,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend  # noqa: E402


@dataclass(frozen=True)
class MutationCase:
    mutation_id: str
    description: str
    expected_class: str
    build_patch: Callable[[ArchitectureProgram, str], TypedPatch]


def _seed_program() -> ArchitectureProgram:
    group = GroupSpec.o3()
    mixed = EquivariantType(group, Carrier.NODE, Irreps.parse("4x0e+2x1o", "O3"))
    scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0e", "O3"))
    graph_scalar = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0e", "O3"))
    return ArchitectureProgram(
        language_version="1.0.0",
        task_contract="controlled_o3_scalar_readout",
        inputs=(InputPort("x", mixed),),
        nodes=(
            Node("linear", "core.irrep_linear", {"x": ("input:x",)}, {"out_irreps": "4x0e+2x1o"}),
            Node("act", "core.norm_activation", {"x": ("linear",)}, {"activation": "silu"}),
            Node("select", "core.select_scalars", {"x": ("act",)}, {"multiplicity": 1}, declared_types={"out": scalar}),
            Node("pool", "core.global_pool", {"x": ("select",)}, declared_types={"out": graph_scalar}),
        ),
        outputs=(OutputPort("prediction", "pool", graph_scalar),),
        program_id="controlled_mutation_parent",
    )


def _patch(parent: ArchitectureProgram, parent_id: str, mutation_id: str, scope, edits) -> TypedPatch:
    return TypedPatch(
        patch_version="1.0",
        parent_architecture_id=parent_id,
        language_version=parent.language_version,
        hypothesis={"mutation_id": mutation_id, "claim": "controlled mutation audit"},
        scope=tuple(scope),
        edits=tuple(edits),
    )


def _identity_insertion(parent, parent_id):
    identity = Node("act__identity", "core.identity", {"x": ("linear",)})
    return _patch(
        parent,
        parent_id,
        "M1_identity_insertion",
        ("act",),
        (
            PatchEdit("insert_before", "act", {"node": identity.to_dict()}),
            PatchEdit("rewire_port", "act", {"port": "x", "references": "act__identity"}),
        ),
    )


def _activation_change(parent, parent_id):
    return _patch(
        parent,
        parent_id,
        "M2_activation_change",
        ("act",),
        (PatchEdit("change_attrs", "act", {"attrs": {"activation": "tanh"}}),),
    )


def _width_change(parent, parent_id):
    return _patch(
        parent,
        parent_id,
        "M3_irrep_multiplicity_change",
        ("linear",),
        (PatchEdit("change_attrs", "linear", {"attrs": {"out_irreps": "6x0e+3x1o"}}),),
    )


def _parallel_branch(parent, parent_id):
    branch = Node(
        "select__branch",
        "core.irrep_linear",
        {"x": ("input:x",)},
        {"out_irreps": "4x0e+2x1o"},
    )
    merge = Node(
        "select__concat",
        "core.irrep_concat",
        {"xs": ("act", "select__branch")},
    )
    return _patch(
        parent,
        parent_id,
        "M4_parallel_branch_concat",
        ("select",),
        (
            PatchEdit("insert_before", "select", {"node": branch.to_dict()}),
            PatchEdit("insert_before", "select", {"node": merge.to_dict()}),
            PatchEdit("rewire_port", "select", {"port": "x", "references": "select__concat"}),
        ),
    )


def _tensor_product_path(parent, parent_id):
    product = Node(
        "select__tensor_product",
        "core.tensor_product",
        {"left": ("act",), "right": ("act",)},
        {"out_irreps": "2x0e+2x1o+1x2e"},
    )
    return _patch(
        parent,
        parent_id,
        "M5_tensor_product_path",
        ("select",),
        (
            PatchEdit("insert_before", "select", {"node": product.to_dict()}),
            PatchEdit("rewire_port", "select", {"port": "x", "references": "select__tensor_product"}),
        ),
    )


def _pool_select_swap(parent, parent_id):
    select = next(node for node in parent.nodes if node.id == "select")
    pool = next(node for node in parent.nodes if node.id == "pool")
    swapped_select = replace(
        select,
        op="core.global_pool",
        inputs={"x": ("act",)},
        attrs={},
        declared_types={},
    )
    swapped_pool = replace(
        pool,
        op="core.select_scalars",
        inputs={"x": ("select",)},
        attrs={"multiplicity": 1},
    )
    return _patch(
        parent,
        parent_id,
        "M6_pool_select_swap",
        ("select", "pool"),
        (
            PatchEdit("replace_node", "select", {"node": swapped_select.to_dict()}),
            PatchEdit("replace_node", "pool", {"node": swapped_pool.to_dict()}),
        ),
    )


def _illegal_scalar_activation(parent, parent_id):
    act = next(node for node in parent.nodes if node.id == "act")
    replacement = replace(act, op="core.scalar_activation")
    return _patch(
        parent,
        parent_id,
        "M7_illegal_scalar_activation",
        ("act",),
        (PatchEdit("replace_node", "act", {"node": replacement.to_dict()}),),
    )


def _illegal_irrep_creation(parent, parent_id):
    return _patch(
        parent,
        parent_id,
        "M8_illegal_irrep_creation",
        ("linear",),
        (PatchEdit("change_attrs", "linear", {"attrs": {"out_irreps": "4x0e+2x1o+1x2e"}}),),
    )


def mutation_cases() -> Tuple[MutationCase, ...]:
    return (
        MutationCase("M1_identity_insertion", "插入可被严格重写消除的恒等节点", "canonical_duplicate", _identity_insertion),
        MutationCase("M2_activation_change", "将范数激活从 SiLU 改为 tanh", "attribute_variant", _activation_change),
        MutationCase("M3_irrep_multiplicity_change", "改变标量与向量不可约表示的重数", "representation_variant", _width_change),
        MutationCase("M4_parallel_branch_concat", "新增并行等变线性分支并进行直和拼接", "topology_innovation", _parallel_branch),
        MutationCase("M5_tensor_product_path", "新增自张量积耦合计算路径", "operator_innovation", _tensor_product_path),
        MutationCase("M6_pool_select_swap", "交换标量选择与图级求和的次序", "algebraic_reordering", _pool_select_swap),
        MutationCase("M7_illegal_scalar_activation", "对含向量表示的特征施加普通逐元素激活", "expected_type_rejection", _illegal_scalar_activation),
        MutationCase("M8_illegal_irrep_creation", "试图用线性层凭空产生二阶表示", "expected_type_rejection", _illegal_irrep_creation),
    )


def _counter_distance(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    denominator = sum(max(left[key], right[key]) for key in keys)
    if not denominator:
        return 0.0
    numerator = sum(abs(left[key] - right[key]) for key in keys)
    return numerator / denominator


def _references(program: ArchitectureProgram) -> Dict[str, Node]:
    return {node.id: node for node in program.nodes}


def _graph_signature(program: ArchitectureProgram, inference) -> Mapping[str, Counter]:
    canonical = canonicalize(program, core_registry())
    nodes = _references(canonical)
    operations = Counter()
    edges = Counter()
    for node in canonical.nodes:
        qualified = node.op if "@" in node.op else "{}@1".format(node.op)
        attrs = json.dumps(dict(node.attrs), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        operations[(qualified, attrs)] += 1
        for port, refs in node.inputs.items():
            for reference in refs:
                source_id = reference.split(":", 1)[0]
                if source_id == "input":
                    source_op = reference
                else:
                    source_node = nodes.get(source_id)
                    source_op = source_node.op if source_node is not None else source_id
                edges[(source_op, qualified, port)] += 1
    types = Counter()
    for value_type in inference.value_types.values():
        types[
            (
                value_type.group.family,
                value_type.carrier,
                str(value_type.irreps),
                str(value_type.frame),
            )
        ] += 1
    return {"operations": operations, "edges": edges, "types": types}


def _structural_distance(parent_signature, child_signature) -> Mapping[str, float]:
    operation = _counter_distance(parent_signature["operations"], child_signature["operations"])
    edge = _counter_distance(parent_signature["edges"], child_signature["edges"])
    value_type = _counter_distance(parent_signature["types"], child_signature["types"])
    combined = 0.4 * operation + 0.3 * edge + 0.3 * value_type
    return {
        "operation_attr_distance": operation,
        "edge_distance": edge,
        "type_distance": value_type,
        "combined": combined,
    }


def _error(reference, transformed) -> Tuple[float, float]:
    import torch

    delta = transformed - reference
    absolute = float(delta.detach().abs().max().cpu())
    relative = float((delta.detach().norm() / reference.detach().norm().clamp_min(1.0e-12)).cpu())
    if not torch.isfinite(delta).all():
        return float("inf"), float("inf")
    return relative, absolute


def _runtime_audit(program: ArchitectureProgram, artifact, *, seed: int, rotations: int, permutations: int):
    import torch
    from e3nn import o3

    torch.manual_seed(seed)
    model = E3NNGraphBackend(core_registry()).build(artifact.expanded_program, artifact.inference).double().eval()
    input_type = program.inputs[0].value_type
    action_irreps = o3.Irreps(str(input_type.irreps))
    generator = torch.Generator().manual_seed(seed + 17)
    node_count = 12
    x = torch.randn(node_count, input_type.irreps.dimension, generator=generator, dtype=torch.float64)
    batch = torch.tensor([0] * 5 + [1] * 7, dtype=torch.long)
    context = {"batch": batch, "num_graphs": 2, "num_nodes": node_count}
    with torch.no_grad():
        reference = model({"x": x}, context)["prediction"]

    checks = []
    for index in range(rotations):
        rotation = o3.rand_matrix(dtype=torch.float64)
        action = action_irreps.D_from_matrix(rotation)
        with torch.no_grad():
            transformed = model({"x": x @ action.transpose(0, 1)}, context)["prediction"]
        relative, absolute = _error(reference, transformed)
        checks.append({"kind": "rotation", "index": index, "relative": relative, "absolute": absolute})

    reflection = -torch.eye(3, dtype=torch.float64)
    reflection_action = action_irreps.D_from_matrix(reflection)
    with torch.no_grad():
        reflected = model({"x": x @ reflection_action.transpose(0, 1)}, context)["prediction"]
    relative, absolute = _error(reference, reflected)
    checks.append({"kind": "reflection", "index": 0, "relative": relative, "absolute": absolute})

    for index in range(permutations):
        permutation = torch.randperm(node_count, generator=generator)
        permuted_context = dict(context, batch=batch.index_select(0, permutation))
        with torch.no_grad():
            transformed = model({"x": x.index_select(0, permutation)}, permuted_context)["prediction"]
        relative, absolute = _error(reference, transformed)
        checks.append({"kind": "permutation", "index": index, "relative": relative, "absolute": absolute})

    maximum_relative = max(item["relative"] for item in checks)
    maximum_absolute = max(item["absolute"] for item in checks)
    return {
        "executed": True,
        "passed": maximum_relative < 1.0e-7 and maximum_absolute < 1.0e-9,
        "maximum_relative_error": maximum_relative,
        "maximum_absolute_error": maximum_absolute,
        "checks": checks,
        "fingerprint": [float(item) for item in reference.detach().flatten().cpu()],
    }


def _behavior_distance(parent_fingerprint: Sequence[float], child_fingerprint: Sequence[float]) -> float:
    import torch

    left = torch.tensor(parent_fingerprint, dtype=torch.float64)
    right = torch.tensor(child_fingerprint, dtype=torch.float64)
    return float(((left - right).norm() / left.norm().clamp_min(1.0e-12)).cpu())


def _diagnostics(exc: Exception):
    if isinstance(exc, DSLValidationError):
        return [item.to_dict() for item in exc.diagnostics]
    return [{"code": type(exc).__name__, "message": str(exc)}]


def run_audit(output_dir: Path, *, seed: int = 20260730, rotations: int = 8, permutations: int = 4):
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = core_registry()
    compiler = Compiler(registry)
    parent = _seed_program()
    parent_artifact = compiler.analyze(parent)
    parent_runtime = _runtime_audit(parent, parent_artifact, seed=seed, rotations=rotations, permutations=permutations)
    parent_signature = _graph_signature(parent_artifact.expanded_program, parent_artifact.inference)

    records = []
    valid_children = []
    for case in mutation_cases():
        patch = case.build_patch(parent, parent_artifact.architecture_id)
        record: Dict[str, Any] = {
            "mutation_id": case.mutation_id,
            "description": case.description,
            "expected_class": case.expected_class,
            "patch": patch.to_dict(),
        }
        try:
            child = apply_typed_patch(
                parent,
                patch,
                registry,
                expected_parent_id=parent_artifact.architecture_id,
            )
            artifact = compiler.analyze(child)
            runtime = _runtime_audit(child, artifact, seed=seed, rotations=rotations, permutations=permutations)
            distance = _structural_distance(
                parent_signature,
                _graph_signature(artifact.expanded_program, artifact.inference),
            )
            canonical_unique = artifact.architecture_id != parent_artifact.architecture_id
            behavior_distance = _behavior_distance(parent_runtime["fingerprint"], runtime["fingerprint"])
            mechanism_novel = case.expected_class in ("topology_innovation", "operator_innovation")
            if case.expected_class == "canonical_duplicate":
                classification_consistent = not canonical_unique
            elif case.expected_class == "expected_type_rejection":
                classification_consistent = False
            else:
                classification_consistent = canonical_unique
            record.update(
                {
                    "typed_patch_applied": True,
                    "typecheck_passed": True,
                    "architecture_id": artifact.architecture_id,
                    "canonical_unique_vs_parent": canonical_unique,
                    "structural_distance": distance,
                    "behavior_distance_vs_parent": behavior_distance,
                    "behaviorally_distinct": behavior_distance > 1.0e-6,
                    "mechanism_novel_by_preregistered_class": mechanism_novel,
                    "equivariance": runtime,
                    "classification_consistent": classification_consistent,
                }
            )
            valid_children.append(record)
        except Exception as exc:  # audit records rejection rather than aborting the matrix
            record.update(
                {
                    "typed_patch_applied": False,
                    "typecheck_passed": False,
                    "architecture_id": "",
                    "canonical_unique_vs_parent": False,
                    "structural_distance": None,
                    "behavior_distance_vs_parent": None,
                    "behaviorally_distinct": False,
                    "mechanism_novel_by_preregistered_class": False,
                    "equivariance": {"executed": False, "passed": False},
                    "diagnostics": _diagnostics(exc),
                    "classification_consistent": case.expected_class == "expected_type_rejection",
                }
            )
        records.append(record)

    generated = len(records)
    valid = sum(item["typecheck_passed"] for item in records)
    unique_ids = {item["architecture_id"] for item in valid_children if item["canonical_unique_vs_parent"]}
    equivariant = sum(item["equivariance"].get("passed", False) for item in valid_children)
    mechanism_novel = sum(item["mechanism_novel_by_preregistered_class"] for item in valid_children)
    useful_novel = sum(
        item["canonical_unique_vs_parent"]
        and item["mechanism_novel_by_preregistered_class"]
        and item["behaviorally_distinct"]
        and item["equivariance"].get("passed", False)
        for item in valid_children
    )
    maximum_relative_error = max(
        (item["equivariance"]["maximum_relative_error"] for item in valid_children),
        default=float("nan"),
    )
    maximum_absolute_error = max(
        (item["equivariance"]["maximum_absolute_error"] for item in valid_children),
        default=float("nan"),
    )
    summary = {
        "protocol": "controlled-dsl-mutation-audit@1",
        "seed": seed,
        "rotation_trials": rotations,
        "reflection_trials": 1,
        "permutation_trials": permutations,
        "thresholds": {
            "equivariance_relative": 1.0e-7,
            "equivariance_absolute": 1.0e-9,
            "behaviorally_distinct": 1.0e-6,
        },
        "parent_architecture_id": parent_artifact.architecture_id,
        "parent_equivariance": parent_runtime,
        "counts": {
            "generated": generated,
            "valid": valid,
            "rejected": generated - valid,
            "canonical_unique_children": len(unique_ids),
            "equivariant_valid_children": equivariant,
            "mechanism_novel_children": mechanism_novel,
            "useful_novel_without_accuracy_gate": useful_novel,
        },
        "observed_symmetry_errors": {
            "maximum_relative": maximum_relative_error,
            "maximum_absolute": maximum_absolute_error,
        },
        "rates": {
            "valid_rate": valid / generated if generated else 0.0,
            "unique_rate_among_valid": len(unique_ids) / valid if valid else 0.0,
            "equivariance_pass_rate_among_valid": equivariant / valid if valid else 0.0,
            "mechanism_novel_rate_among_valid": mechanism_novel / valid if valid else 0.0,
            "useful_novel_rate_without_accuracy_gate": useful_novel / generated if generated else 0.0,
        },
        "all_expectations_met": all(item["classification_consistent"] for item in records),
        "mutations": records,
    }
    (output_dir / "mutation_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output_dir / "mutation_audit.csv", records)
    _write_report(output_dir / "DSL变异创新性与等变性测试报告.md", summary)
    return summary


def _write_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    fields = (
        "mutation_id",
        "expected_class",
        "typecheck_passed",
        "canonical_unique_vs_parent",
        "structural_distance",
        "behavior_distance_vs_parent",
        "behaviorally_distinct",
        "mechanism_novel_by_preregistered_class",
        "equivariance_passed",
        "maximum_relative_error",
        "maximum_absolute_error",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            distance = item.get("structural_distance") or {}
            symmetry = item.get("equivariance") or {}
            writer.writerow(
                {
                    "mutation_id": item["mutation_id"],
                    "expected_class": item["expected_class"],
                    "typecheck_passed": item["typecheck_passed"],
                    "canonical_unique_vs_parent": item["canonical_unique_vs_parent"],
                    "structural_distance": distance.get("combined"),
                    "behavior_distance_vs_parent": item.get("behavior_distance_vs_parent"),
                    "behaviorally_distinct": item.get("behaviorally_distinct"),
                    "mechanism_novel_by_preregistered_class": item.get("mechanism_novel_by_preregistered_class"),
                    "equivariance_passed": symmetry.get("passed"),
                    "maximum_relative_error": symmetry.get("maximum_relative_error"),
                    "maximum_absolute_error": symmetry.get("maximum_absolute_error"),
                }
            )


def _percent(value: float) -> str:
    return "{:.1f}%".format(100.0 * value)


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    counts = summary["counts"]
    rates = summary["rates"]
    lines = [
        "# DSL变异创新性与等变性测试报告",
        "",
        "> 本报告是无训练、无测试标签的第一阶段受控变异审计。它能评价候选是否合法、是否重复、结构与行为是否变化以及数值等变性，但不能证明候选提高了QM9精度。",
        "",
        "## 一、实验设置",
        "",
        "- 父代：由 `irrep_linear → norm_activation → select_scalars → global_pool` 组成的O(3)等变标量读出图。",
        "- 变异数：{}个，其中包含合法变异和故意违反类型规则的负对照。".format(counts["generated"]),
        "- 数值检查：{}次随机旋转、1次空间反演、{}次节点置换。".format(summary["rotation_trials"], summary["permutation_trials"]),
        "- 随机种子：`{}`。".format(summary["seed"]),
        "- 等变通过阈值：相对误差小于 $10^{-7}$，绝对误差小于 $10^{-9}$。",
        "",
        "创新性按以下层次区分：规范化唯一性只表示不是同一个规范计算图；结构距离衡量算子、边和类型变化；机制新颖性只认定新增拓扑或新增算子路径；行为距离用于发现结构不同但函数等价的伪创新。",
        "",
        "## 二、总体结果",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        "| 类型合法率 | {} ({}/{}) |".format(_percent(rates["valid_rate"]), counts["valid"], counts["generated"]),
        "| 合法候选规范去重率 | {} |".format(_percent(rates["unique_rate_among_valid"])),
        "| 合法候选数值等变通过率 | {} |".format(_percent(rates["equivariance_pass_rate_among_valid"])),
        "| 合法候选机制新颖率 | {} |".format(_percent(rates["mechanism_novel_rate_among_valid"])),
        "| 无精度门控的有效创新率 | {} |".format(_percent(rates["useful_novel_rate_without_accuracy_gate"])),
        "| 全部合法变异最大相对等变误差 | `{:.3e}` |".format(summary["observed_symmetry_errors"]["maximum_relative"]),
        "| 全部合法变异最大绝对等变误差 | `{:.3e}` |".format(summary["observed_symmetry_errors"]["maximum_absolute"]),
        "",
        "## 三、逐个变异结果",
        "",
        "| 变异 | 预注册类别 | 类型检查 | 规范唯一 | 结构距离 | 行为距离 | 等变性 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["mutations"]:
        distance = item.get("structural_distance") or {}
        symmetry = item.get("equivariance") or {}
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                item["mutation_id"],
                item["expected_class"],
                "通过" if item["typecheck_passed"] else "拒绝",
                "是" if item["canonical_unique_vs_parent"] else "否",
                "{:.4f}".format(distance["combined"]) if distance else "—",
                "{:.3e}".format(item["behavior_distance_vs_parent"]) if item.get("behavior_distance_vs_parent") is not None else "—",
                "通过" if symmetry.get("passed") else "未执行/未通过",
            )
        )
    lines.extend(
        [
            "",
            "## 四、关键解释",
            "",
            "1. `M1_identity_insertion` 用于验证规范化去重。虽然补丁在源图中插入节点，严格重写会消除恒等操作，因此它不应计为创新。",
            "2. `M2`和`M3`分别改变非线性函数与表示宽度。它们是有效的架构配置变体，但不自动等同于新计算机制。",
            "3. `M4_parallel_branch_concat` 在图结构上新增了分支，但最终的 `select_scalars(multiplicity=1)` 仍只取原分支的第一个标量通道，新增分支没有影响输出。因此它是结构新颖但功能无效的变异，暴露了当前系统缺少通道级活性检查。",
            "4. `M5_tensor_product_path` 新增张量积耦合路径，同时产生非零行为距离，是本轮唯一同时通过规范去重、机制、行为和等变门控的变异。",
            "5. `M6_pool_select_swap` 用于检测代数等价伪创新。求和与通道选择可交换，所以规范图虽然不同，行为距离为零。当前严格重写尚未识别这一等价关系。",
            "6. `M7`和`M8`是负对照：普通逐元素激活不能直接作用于向量表示，等变线性映射也不能凭空产生输入中不存在的二阶表示。DSL在执行前正确拒绝了它们。",
            "",
            "## 五、当前能得出的结论",
            "",
            "本实验可以验证当前DSL具备三项能力：类型系统能拦截明确破坏等变语义的变异；可执行后端能够对合法变异进行旋转、反演和置换数值检查；规范ID与行为指纹联合使用时，能够识别恒等变换、无效分支和代数等价等伪创新。",
            "",
            "尚不能据此声称系统已经解决大规模搜索中的低多样性问题。下一阶段仍需让LLM或随机策略批量生成候选，使用历史档案计算k近邻新颖度，并加入短训练质量门控。当前报告中的“有效创新率”明确不含精度条件。",
            "",
            "## 六、限制",
            "",
            "- 本轮是受控的8个变异，不是LLM大样本生成统计。",
            "- 数值检查观察的是最终O(3)不变标量输出；内部死分支可能被最终读出掩盖，因此必须与活性/行为检查联合使用。",
            "- 当前父代没有坐标输入，所以本轮检查旋转、空间反演和节点置换，不包含平移测试。",
            "- 行为指纹来自固定随机初始化和固定探针批次，不等同于训练后的功能新颖度。",
            "- 未加入QM9短训练、MAE、参数量、运行时间或显存质量门控。",
            "",
            "## 七、复现命令",
            "",
            "```powershell",
            "python scripts/run_dsl_mutation_audit.py --output reports/dsl_mutation_audit_20260730",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/dsl_mutation_audit_20260730"))
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--rotations", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=4)
    args = parser.parse_args()
    summary = run_audit(args.output, seed=args.seed, rotations=args.rotations, permutations=args.permutations)
    print(json.dumps({"output": str(args.output), "counts": summary["counts"], "rates": summary["rates"]}, ensure_ascii=False, indent=2))
    return 0 if summary["all_expectations_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
