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
    Compiler,
    TypeChecker,
    apply_v3_first_round_patch,
    architecture_id,
    build_v3_first_round_patch,
    canonicalize,
    choose_deterministic_v3_action,
    core_registry,
    equiformer_v3_direct_model_program,
    v3_evolution_state,
    v3_first_round_mutation_catalog,
    v3_program_spec,
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
    return equiformer_v3_direct_model_program(spec), {
        "source": "official_model_config",
        "path": str(Path(args.model_config).resolve()),
        "import_manifest": manifest.to_dict(),
    }


def _candidate_state_for_action(program, action):
    values = json.loads(v3_evolution_state(program))
    values[action.field] = action.value
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _available_actions(program, seen_states):
    seen = set(seen_states)
    return tuple(
        action
        for action in v3_first_round_mutation_catalog(program)
        if _candidate_state_for_action(program, action) not in seen
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


def _select_with_glm(args, program, actions, history):
    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise RuntimeError("missing LLM credential environment variable {}".format(args.api_key_env))
    current_spec = v3_program_spec(program)
    request_payload = {
        "round": sum(1 for item in history if int(item.get("round", 0)) > 0) + 1,
        "goal": "从当前 Equiformer V3 Typed DSL 父代中选择一个值得训练的局部变异",
        "constraints": [
            "只能选择 available_actions 中的一个 action_id",
            "不得声称已经提高精度，效果只能作为待训练假设",
            "优先避免与历史轮次重复的变异逻辑",
            "这些动作保持参数形状与等变类型合同，但会改变训练期随机计算路径",
        ],
        "current_stochastic_spec": {
            field: getattr(current_spec, field)
            for field in (
                "alpha_drop",
                "attn_weights_drop",
                "value_drop",
                "drop_path_rate",
                "proj_drop",
                "ffn_drop",
            )
        },
        "history": [
            {
                "round": item.get("round"),
                "action_id": item.get("action_id"),
                "hypothesis": item.get("hypothesis", {}),
            }
            for item in history[-6:]
        ],
        "available_actions": [item.to_dict() for item in actions],
        "response_schema": {
            "action_id": "exact id copied from available_actions",
            "hypothesis": {
                "claim": "one concise Chinese hypothesis",
                "rationale": "why this local change may help",
                "risk": "main failure risk",
            },
        },
    }
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "你是等变神经网络架构变异规划器。只输出一个 JSON 对象，不要输出 Markdown。",
            },
            {
                "role": "user",
                "content": json.dumps(request_payload, ensure_ascii=False, sort_keys=True),
            },
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
    selected = _extract_json_object(content)
    selected["latency_seconds"] = time.time() - started
    selected["transport"] = "openai_compatible"
    selected["model"] = args.model
    selected["raw_response"] = content
    return selected


def _select_action(args, program, actions, history, seen_states, round_index):
    allowed = {item.action_id: item for item in actions}
    failures = []
    if args.selection_mode == "glm":
        for attempt in range(1, args.llm_attempts + 1):
            try:
                response = _select_with_glm(args, program, actions, history)
                action_id = str(response.get("action_id", ""))
                if action_id not in allowed:
                    raise ValueError("LLM selected unavailable action {!r}".format(action_id))
                hypothesis = response.get("hypothesis")
                if not isinstance(hypothesis, dict):
                    raise ValueError("LLM hypothesis must be one object")
                return allowed[action_id], hypothesis, {**response, "attempt": attempt, "selected_by": "glm"}
            except (OSError, KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError) as error:
                failures.append({"attempt": attempt, "error_type": type(error).__name__, "error": str(error)})
        if not args.allow_deterministic_fallback:
            raise RuntimeError("LLM selection failed: {}".format(failures))
    action = choose_deterministic_v3_action(
        program,
        round_index=round_index,
        seen_states=seen_states,
    )
    if action.action_id not in allowed:
        action = actions[0]
    hypothesis = {
        "claim": "确定性 smoke 选择 {}".format(action.description),
        "rationale": "覆盖第一轮形状保持变异目录并验证完整 DSL 闭环",
        "risk": "尚未经过真实训练排序",
    }
    return action, hypothesis, {
        "selected_by": "deterministic" if args.selection_mode == "deterministic" else "deterministic_fallback",
        "failures": failures,
    }


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
    manifest = {
        "mutation_version": V3_FIRST_ROUND_MUTATION_VERSION,
        "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
        "backend_semantics_version": BACKEND_SEMANTICS_VERSION,
        "official_v3_commit": V3_REFERENCE_COMMIT,
        "seed_architecture_id": seed_id,
        "seed_source": seed_source,
        "selection_mode": args.selection_mode,
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
            "mutation_version",
            "compiler_semantics_version",
            "backend_semantics_version",
            "official_v3_commit",
            "seed_architecture_id",
            "selection_mode",
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
            "state": v3_evolution_state(seed),
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
        actions = _available_actions(parent, seen_states)
        if not actions:
            raise RuntimeError("no unseen first-round V3 mutations remain at round {}".format(round_index))
        action, hypothesis, selection = _select_action(
            args,
            parent,
            actions,
            history,
            seen_states,
            round_index,
        )
        patch = build_v3_first_round_patch(parent, action.action_id, registry, hypothesis=hypothesis)
        child, child_spec = apply_v3_first_round_patch(parent, patch, registry)
        child_id = architecture_id(child, registry)
        state = v3_evolution_state(child)
        if state in seen_states:
            raise RuntimeError("selected mutation recreated an ancestor state")
        validation = _validate_candidate(
            child,
            registry,
            backend,
            validation_level=args.validation_level,
            seed=args.seed + round_index,
        )
        round_dir = output / "round_{:03d}".format(round_index)
        round_dir.mkdir(parents=True, exist_ok=False)
        candidate_path = round_dir / "candidate.dsl.json"
        canonical_path = round_dir / "candidate.canonical.dsl.json"
        candidate_path.write_text(dumps_program(child), encoding="utf-8")
        canonical_path.write_text(dumps_program(canonicalize(child, registry)), encoding="utf-8")
        _write_json(round_dir / "patch.json", patch.to_dict())
        _write_json(round_dir / "selection.json", selection)
        _write_json(round_dir / "lowering_manifest.json", validation)
        record = {
            "round": round_index,
            "kind": "candidate",
            "parent_architecture_id": patch.parent_architecture_id,
            "architecture_id": child_id,
            "official_spec_id": child_spec.architecture_id(),
            "action_id": action.action_id,
            "mutation_family": action.field,
            "hypothesis": hypothesis,
            "selected_by": selection.get("selected_by"),
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
    parser.add_argument("--selection-mode", choices=("glm", "deterministic"), default="glm")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--api-base", default=os.environ.get("GLM_API_BASE", "https://glm.llm.autos/v1"))
    parser.add_argument("--api-key-env", default="GLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--llm-attempts", type=int, default=3)
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
