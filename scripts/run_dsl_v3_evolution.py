#!/usr/bin/env python
"""Run a resumable 10-round first-pass evolution over a lowered V3 Typed DSL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_V3_ROOT = PROJECT_ROOT.parent / "equiformer_v3_official"

from equivariant_nas.dsl import (
    BACKEND_SEMANTICS_VERSION,
    COMPILER_SEMANTICS_VERSION,
    V3_FIRST_ROUND_MUTATION_VERSION,
    V3_STRUCTURAL_MUTATION_VERSION,
    Compiler,
    DSLValidationError,
    TypedPatch,
    TypeChecker,
    apply_v3_first_round_patch,
    apply_v3_structural_patch,
    architecture_id,
    build_v3_first_round_patch,
    build_v3_structural_patch,
    canonicalize,
    choose_deterministic_v3_action,
    choose_deterministic_v3_structural_action,
    core_registry,
    parse_v3_critic_response,
    parse_v3_router_response,
    v3_evolution_state,
    v3_first_round_mutation_catalog,
    v3_mutation_regions,
    v3_program_from_spec,
    v3_program_spec,
    v3_region_by_id,
    v3_structural_mutation_catalog,
    v3_structural_regions,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend, EquiformerV3Spec, V3_REFERENCE_COMMIT
from equivariant_nas.dsl.serialization import dumps_program, load_program


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _load_mapping(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("configuration must decode to one mapping")
    return value


def _load_seed(args):
    if args.seed_program:
        return load_program(args.seed_program), {"source": "seed_program", "path": str(Path(args.seed_program).resolve())}
    payload = _load_mapping(Path(args.model_config))
    spec, manifest = EquiformerV3Spec.from_official_config(payload)
    return v3_program_from_spec(spec), {
        "source": "official_model_config",
        "path": str(Path(args.model_config).resolve()),
        "import_manifest": manifest.to_dict(),
    }


def _candidate_state_for_action(program, action, *, mutation_mode="probability", registry=None):
    if mutation_mode == "structural":
        registry = registry or core_registry()
        patch = build_v3_structural_patch(program, action.action_id, registry)
        child, _ = apply_v3_structural_patch(program, patch, registry)
        return architecture_id(child, registry)
    values = json.loads(v3_evolution_state(program))
    values[action.field] = action.value
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _available_actions(program, seen_states, *, mutation_mode="probability", registry=None):
    seen = set(seen_states)
    if mutation_mode == "structural":
        registry = registry or core_registry()
        catalog = v3_structural_mutation_catalog(program)
    else:
        catalog = v3_first_round_mutation_catalog(program)
    return tuple(
        action
        for action in catalog
        if _candidate_state_for_action(
            program,
            action,
            mutation_mode=mutation_mode,
            registry=registry,
        ) not in seen
    )


def _chat_endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _extract_json_object(text: str):
    stripped = str(text).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response must be one JSON object")
    return value


def _call_glm(args, prompt):
    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise RuntimeError("missing LLM credential environment variable {}".format(args.api_key_env))
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    request = urllib.request.Request(
        _chat_endpoint(args.api_base),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    return {
        "response": _extract_json_object(content),
        "latency_seconds": time.time() - started,
        "transport": "openai_compatible",
        "model": args.model,
        "raw_response": content,
    }


def _deterministic_envelope(response, role):
    return {
        "response": dict(response),
        "latency_seconds": 0.0,
        "transport": "deterministic_test_protocol",
        "model": "",
        "role": role,
        "raw_response": json.dumps(response, ensure_ascii=False, sort_keys=True),
    }


def _router_prompt(program, actions, history, round_index, *, mutation_mode="probability"):
    spec = v3_program_spec(program)
    regions = v3_structural_regions(actions) if mutation_mode == "structural" else v3_mutation_regions(actions)
    if mutation_mode == "structural":
        current_state = {
            item.field: sorted({action.current_value for action in actions if action.field == item.field})
            for item in regions
        }
        goal = "选择一个最值得审查的 Equiformer V3 局部结构因子与其唯一编辑区域"
    else:
        current_state = {item.field: float(getattr(spec, item.field)) for item in regions}
        goal = "选择一个最值得审查的 Equiformer V3 局部叶子因子与其唯一编辑区域"
    payload = {
        "round": round_index,
        "mutation_mode": mutation_mode,
        "goal": goal,
        "current_parent_state": current_state,
        "regions": [item.to_dict(actions=actions) for item in regions],
        "history": [
            {
                "round": item.get("round"),
                "factor_id": item.get("factor_id"),
                "region_id": item.get("region_id"),
                "action_id": item.get("action_id"),
            }
            for item in history[-6:]
            if int(item.get("round", 0)) > 0
        ],
        "rules": [
            "只选择一个已列出的 factor_id 与它唯一拥有的 region_id",
            "本阶段不得选择具体 action_id、数值、补丁或代码",
            "所有收益均是待训练验证的预期，不得声称已经提高精度",
        ],
        "response_schema": {
            "factor_id": "exact factor_id from regions",
            "region_id": "the uniquely owned exact region_id",
            "rationale": "why this factor has high information value",
            "evidence_refs": [],
            "expected_value": "one falsifiable expected benefit",
            "risk": "main risk",
        },
    }
    return {
        "system": (
            "你是 Typed 等变架构搜索的 Factor Router。只决定叶子因子和局部区域，"
            "不得提出补丁、动作值或代码。只输出严格匹配 response_schema 的 JSON 对象。"
        ),
        "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
    }


def _critic_prompt(program, actions, region, router, history, round_index):
    selected = tuple(item for item in actions if item.field == region.field)
    node_ids = {target.node_id for action in selected for target in action.targets}
    nodes = {node.id: node for node in program.nodes}
    payload = {
        "round": round_index,
        "router_decision": dict(router),
        "selected_region": region.to_dict(actions=selected),
        "local_parent_ast": [nodes[node_id].to_dict() for node_id in sorted(node_ids) if node_id in nodes],
        "locally_absent_authorized_nodes": sorted(node_id for node_id in node_ids if node_id not in nodes),
        "recent_history": [
            {
                "round": item.get("round"),
                "action_id": item.get("action_id"),
                "hypothesis": item.get("hypothesis", {}),
            }
            for item in history[-6:]
            if int(item.get("round", 0)) > 0
        ],
        "rules": [
            "只分析 router 已选择的 factor_id、region_id 与局部边界",
            "提出可证伪机制和有序编辑计划，但不得生成 Typed Patch 或源代码",
            "不得扩大到另一个 mutation family",
            "必须明确保持等变类型和冻结的 V3 张量维度；若插入或删除参数化原语，必须显式报告参数树变化",
        ],
        "response_schema": {
            "factor_id": region.factor_id,
            "region_id": region.region_id,
            "claim": "one falsifiable local hypothesis",
            "mechanism": "why this local computation graph may affect expressivity, optimization, or generalization",
            "edit_plan": ["ordered abstract edits inside the selected family"],
            "preserved_invariants": ["contracts that must remain true"],
            "evidence_refs": [],
            "uncertainty": "main uncertainty",
            "risk": "main correctness or optimization risk",
            "acceptance_metrics": ["training or validation measurements that can falsify the claim"],
        },
    }
    return {
        "system": (
            "你是独立的 V3 Region Critic。只审查 Router 冻结的局部区域，给出机制与编辑计划，"
            "不得生成补丁或改变 factor/region。只输出严格 JSON 对象。"
        ),
        "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
    }


def _critic_repair_prompt(original_prompt, rejected, diagnostics, region, immutable_mechanism):
    schema = json.loads(original_prompt["user"])["response_schema"]
    required_keys = list(schema)
    rejected_keys = set(rejected) if isinstance(rejected, dict) else set()
    missing_keys = [name for name in required_keys if name not in rejected_keys]
    unknown_keys = sorted(rejected_keys - set(required_keys))
    payload = {
        "instruction": (
            "仅修复 JSON 协议错误。不得改变 factor_id、region_id、原始 mechanism 或科学干预方向；"
            "不得加入补丁、代码、Markdown 或额外字段。最终对象必须逐一包含 required_keys 中的全部键。"
        ),
        "required_keys": required_keys,
        "missing_keys_that_must_be_added": missing_keys,
        "unknown_keys_that_must_be_removed": unknown_keys,
        "immutable_factor_id": region.factor_id,
        "immutable_region_id": region.region_id,
        "immutable_mechanism": immutable_mechanism,
        "response_schema": schema,
        "required_field_example": {
            "acceptance_metrics": ["validation metric", "Energy rotation error", "Force rotation error"]
        },
        "rejected_response": rejected,
        "diagnostics": diagnostics,
    }
    return {
        "system": (
            "你是严格的 Region Critic JSON 协议修复器。只输出一个 JSON 对象。"
            "遗漏 required_keys 中任何一个键都视为修复失败。"
        ),
        "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
    }


def _patch_hypothesis(critic):
    return {
        name: critic[name]
        for name in (
            "claim",
            "mechanism",
            "edit_plan",
            "preserved_invariants",
            "evidence_refs",
            "uncertainty",
            "risk",
            "acceptance_metrics",
        )
    }


def _allowed_typed_patches(program, actions, registry, critic, *, mutation_mode="probability"):
    hypothesis = _patch_hypothesis(critic)
    builder = build_v3_structural_patch if mutation_mode == "structural" else build_v3_first_round_patch
    return {
        action.action_id: builder(
            program,
            action.action_id,
            registry,
            hypothesis=hypothesis,
        )
        for action in actions
    }


def _synthesizer_prompt(program, region, router, critic, allowed_patches, round_index):
    payload = {
        "round": round_index,
        "router_decision": dict(router),
        "critic_plan": dict(critic),
        "immutable_factor_id": region.factor_id,
        "immutable_region_id": region.region_id,
        "immutable_mechanism": critic["mechanism"],
        "allowed_typed_patches": [
            {"action_id": action_id, "typed_patch": patch.to_dict()}
            for action_id, patch in allowed_patches.items()
        ],
        "rules": [
            "从 allowed_typed_patches 中选择一个完整 Typed Patch 并逐字段原样输出 typed_patch 对象",
            "不得输出外层 action_id 包装、Markdown、解释或额外字段",
            "不得改变 factor_id、region_id、mechanism、scope、edits、前后置条件或父代身份",
            "效果仍是待训练假设，不得宣称已经获得性能提升",
        ],
    }
    return {
        "system": (
            "你是 V3 Patch Synthesizer。依据冻结的 Router 和 Critic 决策，从编译器认证的候选中"
            "输出一个完整 Typed Patch JSON 对象；不得扩大范围或改写补丁。"
        ),
        "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
    }


def _compiler_repair_prompt(original_prompt, rejected, diagnostics, region, critic, allowed_patches):
    payload = {
        "instruction": (
            "根据编译器诊断重写完整 Typed Patch。只能逐字段复制 allowed_typed_patches 中的一个；"
            "factor_id、region_id、mechanism、scope 和科学干预方向全部冻结。"
        ),
        "immutable_factor_id": region.factor_id,
        "immutable_region_id": region.region_id,
        "immutable_mechanism": critic["mechanism"],
        "diagnostics": diagnostics,
        "rejected_response": rejected,
        "original_synthesizer_request": json.loads(original_prompt["user"]),
        "allowed_typed_patches": [patch.to_dict() for patch in allowed_patches.values()],
    }
    return {
        "system": "你是编译器引导的 Typed Patch 修复器。只输出一个完整 JSON 补丁对象。",
        "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
    }


def _error_diagnostics(error):
    if isinstance(error, DSLValidationError):
        return [item.to_dict() for item in error.diagnostics]
    return [{"code": "E_V3_WORKFLOW", "message": str(error), "error_type": type(error).__name__}]


def _parse_synthesized_patch(value, allowed_patches, region, critic):
    patch = TypedPatch.from_dict(value)
    action_id = str(patch.hypothesis.get("action_id", ""))
    expected = allowed_patches.get(action_id)
    if expected is None:
        raise ValueError("synthesizer selected an action outside the routed region: {!r}".format(action_id))
    if str(patch.hypothesis.get("factor_id", "")) != region.factor_id:
        raise ValueError("synthesizer changed the routed factor")
    if str(patch.hypothesis.get("region_id", "")) != region.region_id:
        raise ValueError("synthesizer changed the routed region")
    if str(patch.hypothesis.get("mechanism", "")) != str(critic["mechanism"]):
        raise ValueError("synthesizer changed the critic mechanism")
    if patch.to_dict() != expected.to_dict():
        raise ValueError("synthesizer changed the compiler-certified patch envelope, scope, or edits")
    return patch, action_id


def _write_stage_exchange(round_dir, stem, prompt, envelope):
    _write_json(round_dir / "{}_prompt.json".format(stem), prompt)
    _write_json(round_dir / "{}_response.json".format(stem), envelope)


def _run_three_stage_round(args, program, actions, history, seen_states, round_index, round_dir, registry):
    mutation_mode = getattr(args, "mutation_mode", "probability")
    regions = v3_structural_regions(actions) if mutation_mode == "structural" else v3_mutation_regions(actions)
    router_prompt = _router_prompt(
        program,
        actions,
        history,
        round_index,
        mutation_mode=mutation_mode,
    )
    if args.selection_mode == "glm":
        router_envelope = _call_glm(args, router_prompt)
    else:
        region = regions[(round_index - 1) % len(regions)]
        router_envelope = _deterministic_envelope({
            "factor_id": region.factor_id,
            "region_id": region.region_id,
            "rationale": "确定性覆盖当前 V3 局部变异因子",
            "evidence_refs": [],
            "expected_value": "验证三阶段协议和 Typed Patch 闭环",
            "risk": "尚未经过训练排序",
        }, "factor_router")
    _write_stage_exchange(round_dir, "router", router_prompt, router_envelope)
    router = parse_v3_router_response(router_envelope["response"], regions)
    region = next(item for item in regions if item.region_id == router["region_id"])
    routed_actions = tuple(item for item in actions if item.field == region.field)

    critic_prompt = _critic_prompt(program, routed_actions, region, router, history, round_index)
    if args.selection_mode == "glm":
        critic_envelope = _call_glm(args, critic_prompt)
    else:
        structural = mutation_mode == "structural"
        critic_envelope = _deterministic_envelope({
            "factor_id": region.factor_id,
            "region_id": region.region_id,
            "claim": (
                "重组 {} 局部数据流可能改变表示能力与优化路径".format(region.field)
                if structural
                else "调整 {} 可能改变正则化强度与优化稳定性".format(region.field)
            ),
            "mechanism": (
                "只重写编译器授权的局部原语图，同时保持等变类型和 V3 张量维度合同"
                if structural
                else "只改变训练期随机路径概率，同时保持所有等变类型与参数形状不变"
            ),
            "edit_plan": (
                ["选择一个未访问的结构重写", "应用可逆 Typed Patch 并执行通用 Lowering 与等变审计"]
                if structural
                else ["选择一个未访问的合法概率", "同步更新全部重复运行时节点与 V3 规格字段"]
            ),
            "preserved_invariants": (
                ["等变类型不变", "V3 张量维度不变", "局部作用域不扩张"]
                if structural
                else ["等变类型不变", "参数形状不变", "重复节点同步修改"]
            ),
            "evidence_refs": [],
            "uncertainty": "短期结构检查不能预测训练后的精度",
            "risk": (
                "局部重组虽满足类型合同，但可能降低信息流质量或优化稳定性"
                if structural
                else "随机正则过强可能导致欠拟合"
            ),
            "acceptance_metrics": ["验证集指标", "Energy 旋转误差", "Force 旋转误差"],
        }, "region_critic")
    _write_stage_exchange(round_dir, "critic", critic_prompt, critic_envelope)
    critic_repair_count = 0
    immutable_mechanism = ""
    rejected_critic = critic_envelope["response"]
    try:
        critic = parse_v3_critic_response(rejected_critic, region)
    except DSLValidationError as error:
        if args.selection_mode != "glm":
            raise
        if isinstance(rejected_critic, dict):
            immutable_mechanism = str(rejected_critic.get("mechanism", "")).strip()
        last_error = error
        critic = None
        for repair_index in range(1, args.critic_repair_attempts + 1):
            repair_prompt = _critic_repair_prompt(
                critic_prompt,
                rejected_critic,
                _error_diagnostics(last_error),
                region,
                immutable_mechanism,
            )
            repair_envelope = _call_glm(args, repair_prompt)
            _write_stage_exchange(round_dir, "critic_repair_{:03d}".format(repair_index), repair_prompt, repair_envelope)
            rejected_critic = repair_envelope["response"]
            try:
                critic = parse_v3_critic_response(
                    rejected_critic,
                    region,
                    immutable_mechanism=immutable_mechanism,
                )
                critic_repair_count = repair_index
                break
            except DSLValidationError as repair_error:
                last_error = repair_error
        if critic is None:
            raise last_error

    allowed_patches = _allowed_typed_patches(
        program,
        routed_actions,
        registry,
        critic,
        mutation_mode=mutation_mode,
    )
    synth_prompt = _synthesizer_prompt(program, region, router, critic, allowed_patches, round_index)
    if args.selection_mode == "glm":
        synth_envelope = _call_glm(args, synth_prompt)
    else:
        if mutation_mode == "structural":
            preferred = choose_deterministic_v3_structural_action(
                program,
                round_index=round_index,
                seen_states=seen_states,
                registry=registry,
            )
        else:
            preferred = choose_deterministic_v3_action(
                program,
                round_index=round_index,
                seen_states=seen_states,
            )
        if preferred.action_id not in allowed_patches:
            preferred = routed_actions[0]
        synth_envelope = _deterministic_envelope(
            allowed_patches[preferred.action_id].to_dict(),
            "patch_synthesizer",
        )
    _write_stage_exchange(round_dir, "synthesizer", synth_prompt, synth_envelope)

    rejected_patch = synth_envelope["response"]
    compiler_repair_count = 0
    last_error = None
    for attempt in range(args.compiler_repair_attempts + 1):
        try:
            patch, action_id = _parse_synthesized_patch(rejected_patch, allowed_patches, region, critic)
            if mutation_mode == "structural":
                child, child_spec = apply_v3_structural_patch(program, patch, registry)
                state = architecture_id(child, registry)
            else:
                child, child_spec = apply_v3_first_round_patch(program, patch, registry)
                state = v3_evolution_state(child)
            if state in set(seen_states):
                raise ValueError("synthesized patch recreated an ancestor state")
            action = next(item for item in routed_actions if item.action_id == action_id)
            compiler_repair_count = attempt
            return {
                "router": router,
                "critic": critic,
                "patch": patch,
                "action": action,
                "child": child,
                "child_spec": child_spec,
                "state": state,
                "router_response": router_envelope["response"],
                "critic_response": critic,
                "synthesizer_response": rejected_patch,
                "critic_repair_count": critic_repair_count,
                "compiler_repair_count": compiler_repair_count,
            }
        except (DSLValidationError, ValueError, RuntimeError) as error:
            last_error = error
            if args.selection_mode != "glm" or attempt >= args.compiler_repair_attempts:
                raise
            repair_index = attempt + 1
            repair_prompt = _compiler_repair_prompt(
                synth_prompt,
                rejected_patch,
                _error_diagnostics(error),
                region,
                critic,
                allowed_patches,
            )
            repair_envelope = _call_glm(args, repair_prompt)
            _write_stage_exchange(round_dir, "compiler_repair_{:03d}".format(repair_index), repair_prompt, repair_envelope)
            rejected_patch = repair_envelope["response"]
    raise last_error


def _index(indices, target_size):
    return {"indices": indices, "target_size": target_size}


def _runtime_validation(model, program, *, seed: int):
    import torch

    atomic_numbers = torch.tensor([1, 6, 8, 14], dtype=torch.long)
    base_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1], [0.3, 1.2, 0.4], [-0.4, 0.6, 1.3]],
        dtype=torch.float32,
    )
    source = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    target = torch.tensor([1, 2, 3, 0, 2, 1], dtype=torch.long)
    batch = torch.zeros(atomic_numbers.numel(), dtype=torch.long)
    lattice = torch.tensor(
        [[5.0, 0.0, 0.0], [0.1, 5.2, 0.0], [0.0, 0.2, 5.4]],
        dtype=torch.float32,
    )
    lattice_shift = torch.zeros((source.numel(), 3), dtype=torch.long)
    input_names = {item.name for item in program.inputs}

    def inputs(positions, cell=lattice):
        values = {
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "source_index": _index(source, source.numel()),
            "target_index": _index(target, target.numel()),
            "target_segment": _index(target, atomic_numbers.numel()),
            "batch": _index(batch, 1),
        }
        if "lattice" in input_names:
            values["lattice"] = cell
        if "lattice_shift" in input_names:
            values["lattice_shift"] = lattice_shift
        return values

    model.eval()
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        reference = model(inputs(base_positions), {})
        rotated = model(inputs(base_positions @ rotation.T, lattice @ rotation.T), {})
    energy_error = float((reference["energy"] - rotated["energy"]).abs().max().item())
    force_error = float((reference["forces"] @ rotation.T - rotated["forces"]).abs().max().item())
    if energy_error > 2.0e-5 or force_error > 3.0e-4:
        raise RuntimeError(
            "rotation audit failed: energy_error={} force_error={}".format(energy_error, force_error)
        )

    positions = base_positions.clone().requires_grad_(True)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    outputs = model(inputs(positions), {})
    loss = outputs["energy"].square().sum() + outputs["forces"].square().sum()
    loss.backward()
    missing_gradients = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
    nonfinite_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if positions.grad is None or not torch.isfinite(positions.grad).all():
        raise RuntimeError("position gradient is absent or non-finite")
    if missing_gradients or nonfinite_gradients:
        raise RuntimeError(
            "parameter gradient audit failed: missing={} nonfinite={}".format(
                missing_gradients, nonfinite_gradients
            )
        )
    return {
        "energy_shape": list(outputs["energy"].shape),
        "force_shape": list(outputs["forces"].shape),
        "loss": float(loss.detach().item()),
        "rotation_energy_max_abs_error": energy_error,
        "rotation_force_max_abs_error": force_error,
        "position_gradient_finite": True,
        "trainable_parameter_gradients_complete": True,
    }


def _validate_candidate(program, registry, backend, *, validation_level: str, seed: int):
    inference = TypeChecker(registry).check(program)
    support = backend.support_report(program)
    if support.unsupported_nodes or support.composition_errors or support.missing_dependencies:
        raise RuntimeError("candidate lacks complete generic lowering support: {}".format(support.to_dict()))
    plan = Compiler(registry).plan_lowering(program, graph_backend=backend)
    result = {
        "status": "passed",
        "validation_level": validation_level,
        "architecture_id": architecture_id(program, registry),
        "official_spec_id": v3_program_spec(program).architecture_id(),
        "node_count": len(program.nodes),
        "typed_node_count": len(inference.node_order),
        "outputs": [item.name for item in program.outputs],
        "backend_support": support.to_dict(),
        "lowering_plan": plan.to_dict(),
        "constructor_bypass": False,
        "checkpoint_inheritance": "parameter shapes unchanged; checkpoint mapping still requires source/target config audit",
    }
    if validation_level in {"build", "full"}:
        model = backend.build(program, inference)
        forbidden = {
            "EquiformerV3_OC",
            "TransBlockV3",
            "EquivariantGraphAttention",
            "ScalarFeedForwardNetwork",
        }
        bypass = sorted({type(module).__name__ for module in model.modules()} & forbidden)
        if bypass:
            raise RuntimeError("official constructor bypass detected: {}".format(bypass))
        result["trainable_parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        result["official_constructor_types_present"] = bypass
        if validation_level == "full":
            result["runtime"] = _runtime_validation(model, program, seed=seed)
    return result


def _read_history(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prepare_round_dir(output: Path, round_index: int, *, resume: bool) -> Path:
    """Create a round directory while preserving any failed incomplete attempt."""

    round_dir = output / "round_{:03d}".format(round_index)
    if round_dir.exists():
        if not resume:
            raise RuntimeError("round evidence directory already exists: {}".format(round_dir))
        failed_root = output / "failed_attempts"
        failed_root.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while True:
            archived = failed_root / "round_{:03d}_attempt_{:03d}".format(round_index, attempt)
            if not archived.exists():
                round_dir.replace(archived)
                break
            attempt += 1
    round_dir.mkdir(parents=True, exist_ok=False)
    return round_dir


def run(args):
    output = Path(args.output).resolve()
    evolution_path = output / "evolution.jsonl"
    seed, seed_source = _load_seed(args)
    registry = core_registry()
    seed_id = architecture_id(seed, registry)
    TypeChecker(registry).check(seed)
    if len(seed.outputs) != 2 or {item.name for item in seed.outputs} != {"energy", "forces"}:
        raise ValueError("first-round V3 evolution requires the direct Energy+Force program")

    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError("output directory is nonempty; pass --resume to continue the same lineage")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    mutation_mode = getattr(args, "mutation_mode", "structural")
    mutation_version = (
        V3_STRUCTURAL_MUTATION_VERSION
        if mutation_mode == "structural"
        else V3_FIRST_ROUND_MUTATION_VERSION
    )
    manifest = {
        "mutation_mode": mutation_mode,
        "mutation_version": mutation_version,
        "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
        "backend_semantics_version": BACKEND_SEMANTICS_VERSION,
        "official_v3_commit": V3_REFERENCE_COMMIT,
        "seed_architecture_id": seed_id,
        "seed_source": seed_source,
        "selection_mode": args.selection_mode,
        "workflow": [
            "factor_router",
            "region_critic",
            "patch_synthesizer",
            "conditional_compiler_repair",
        ],
        "model": args.model if args.selection_mode == "glm" else "",
        "api_key_env": args.api_key_env if args.selection_mode == "glm" else "",
        "credential_serialized": False,
        "validation_level": args.validation_level,
        "equiformer_v3_root": str(Path(args.equiformer_v3_root).resolve()) if args.equiformer_v3_root else "",
        "round_target": args.rounds,
        "created_at": _now(),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen_keys = (
            "mutation_mode",
            "mutation_version",
            "compiler_semantics_version",
            "backend_semantics_version",
            "official_v3_commit",
            "seed_architecture_id",
            "selection_mode",
            "workflow",
            "model",
            "validation_level",
            "equiformer_v3_root",
        )
        changed = {key: (existing.get(key), manifest.get(key)) for key in frozen_keys if existing.get(key) != manifest.get(key)}
        if changed:
            raise RuntimeError("resume manifest identity changed: {}".format(changed))
        existing["round_target"] = max(int(existing.get("round_target", 0)), int(args.rounds))
        manifest = existing
    _write_json(manifest_path, manifest)

    seed_path = output / "seed.dsl.json"
    if not seed_path.exists():
        seed_path.write_text(dumps_program(seed), encoding="utf-8")
    history = _read_history(evolution_path)
    if not history:
        _append_jsonl(evolution_path, {
            "round": 0,
            "kind": "seed",
            "architecture_id": seed_id,
            "official_spec_id": v3_program_spec(seed).architecture_id(),
            "candidate_path": str(seed_path),
            "state": seed_id if mutation_mode == "structural" else v3_evolution_state(seed),
            "created_at": _now(),
        })
        history = _read_history(evolution_path)

    completed = max(int(item.get("round", 0)) for item in history)
    if completed:
        last = max((item for item in history if int(item.get("round", 0)) == completed), key=lambda item: item["round"])
        parent = load_program(last["candidate_path"])
    else:
        parent = seed
    seen_states = [str(item["state"]) for item in history if item.get("state")]
    backend = E3NNGraphBackend(
        registry,
        equiformer_v3_root=args.equiformer_v3_root,
    )

    for round_index in range(completed + 1, args.rounds + 1):
        actions = _available_actions(
            parent,
            seen_states,
            mutation_mode=mutation_mode,
            registry=registry,
        )
        if not actions:
            raise RuntimeError("no unseen first-round V3 mutations remain at round {}".format(round_index))
        round_dir = _prepare_round_dir(output, round_index, resume=args.resume)
        workflow = _run_three_stage_round(
            args,
            parent,
            actions,
            history,
            seen_states,
            round_index,
            round_dir,
            registry,
        )
        action = workflow["action"]
        patch = workflow["patch"]
        child = workflow["child"]
        child_spec = workflow["child_spec"]
        child_id = architecture_id(child, registry)
        state = workflow["state"]
        validation = _validate_candidate(
            child,
            registry,
            backend,
            validation_level=args.validation_level,
            seed=args.seed + round_index,
        )
        candidate_path = round_dir / "candidate.dsl.json"
        canonical_path = round_dir / "candidate.canonical.dsl.json"
        candidate_path.write_text(dumps_program(child), encoding="utf-8")
        canonical_path.write_text(dumps_program(canonicalize(child, registry)), encoding="utf-8")
        _write_json(round_dir / "patch.json", patch.to_dict())
        _write_json(round_dir / "lowering_manifest.json", validation)
        record = {
            "round": round_index,
            "kind": "candidate",
            "parent_architecture_id": patch.parent_architecture_id,
            "architecture_id": child_id,
            "official_spec_id": child_spec.architecture_id(),
            "action_id": action.action_id,
            "mutation_family": action.field,
            "mutation_mode": mutation_mode,
            "mutation_level": getattr(action, "level", "attribute"),
            "factor_id": workflow["router"]["factor_id"],
            "region_id": workflow["router"]["region_id"],
            "hypothesis": dict(patch.hypothesis),
            "router_response": workflow["router_response"],
            "critic_response": workflow["critic_response"],
            "synthesizer_response": workflow["synthesizer_response"],
            "critic_repair_count": workflow["critic_repair_count"],
            "compiler_repair_count": workflow["compiler_repair_count"],
            "workflow": [
                "factor_router",
                "region_critic",
                "patch_synthesizer",
                "conditional_compiler_repair",
            ],
            "selected_by": args.selection_mode,
            "candidate_path": str(candidate_path),
            "canonical_path": str(canonical_path),
            "patch_path": str(round_dir / "patch.json"),
            "lowering_manifest_path": str(round_dir / "lowering_manifest.json"),
            "state": state,
            "validation_level": args.validation_level,
            "created_at": _now(),
        }
        _append_jsonl(evolution_path, record)
        history.append(record)
        seen_states.append(state)
        parent = child
        _write_json(output / "heartbeat.json", {
            "status": "running" if round_index < args.rounds else "complete",
            "completed_rounds": round_index,
            "round_target": args.rounds,
            "latest_architecture_id": child_id,
            "updated_at": _now(),
        })

    candidates = [item for item in history if int(item.get("round", 0)) > 0]
    summary = {
        "status": "complete" if len(candidates) >= args.rounds else "partial",
        "requested_rounds": args.rounds,
        "completed_rounds": len(candidates),
        "unique_candidate_count": len({item["architecture_id"] for item in candidates}),
        "seed_architecture_id": seed_id,
        "final_architecture_id": architecture_id(parent, registry),
        "selection_mode": args.selection_mode,
        "selection_provenance": sorted({str(item.get("selected_by")) for item in candidates}),
        "workflow": [
            "factor_router",
            "region_critic",
            "patch_synthesizer",
            "conditional_compiler_repair",
        ],
        "total_critic_repairs": sum(int(item.get("critic_repair_count", 0)) for item in candidates),
        "total_compiler_repairs": sum(int(item.get("compiler_repair_count", 0)) for item in candidates),
        "validation_level": args.validation_level,
        "mutation_version": V3_FIRST_ROUND_MUTATION_VERSION,
        "candidate_records": candidates,
        "training_started": False,
        "ranking_claimed": False,
        "first_round_scope": "shape-preserving stochastic-path mutations only",
        "second_round_deferred": [
            "width/head/resolution/layer-count mutations",
            "arbitrary typed subgraph insertion or deletion",
            "partial checkpoint inheritance across shape changes",
            "stress head",
            "training-based selection and promotion",
        ],
        "completed_at": _now(),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def get_parser():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-config")
    source.add_argument("--seed-program")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument(
        "--mutation-mode",
        choices=("structural", "probability"),
        default="structural",
        help="Use structural Typed DSL rewrites by default; probability mode is retained only as an ablation.",
    )
    parser.add_argument("--selection-mode", choices=("glm", "deterministic"), default="glm")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--api-base", default=os.environ.get("GLM_API_BASE", "https://glm.llm.autos/v1"))
    parser.add_argument("--api-key-env", default="GLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--llm-attempts", type=int, default=3)
    parser.add_argument("--critic-repair-attempts", type=int, default=2)
    parser.add_argument("--compiler-repair-attempts", type=int, default=2)
    parser.add_argument("--allow-deterministic-fallback", action="store_true")
    parser.add_argument("--validation-level", choices=("static", "build", "full"), default="static")
    parser.add_argument(
        "--equiformer-v3-root",
        default=(
            os.environ.get("EQUIFORMER_V3_ROOT", "")
            or (str(DEFAULT_V3_ROOT) if DEFAULT_V3_ROOT.exists() else "")
        ),
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = get_parser().parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if not args.equiformer_v3_root:
        raise ValueError("V3 lowering validation requires --equiformer-v3-root")
    run(args)


if __name__ == "__main__":
    main()
