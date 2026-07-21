#!/usr/bin/env python
"""OpenEvolve database + evidence-calibrated SPARK-style factor evolution."""

import argparse
import asyncio
import json
import os
import random
import sys
import uuid
from pathlib import Path

from equivariant_nas.candidate import (
    extract_literal_spec_source,
    parse_llm_patch,
    render_candidate,
)
from equivariant_nas.credit import factor_credit
from equivariant_nas.interaction import (
    rescue_requires_counterfactual,
    resolved_counterfactual_credits,
)
from equivariant_nas.router import (
    EvidenceCalibratedRouter,
    prompt_for_factor,
    prompt_for_reflection,
)
from equivariant_nas.semantics import validate_qm9_alpha_reasoning
from equivariant_nas.search_memory import (
    compact_layerwise_symmetry,
    compact_metrics,
    summarize_lineage,
)
from equivariant_nas.spec import EvolutionFactor


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def initialize_equivariant_feature_ranges(database):
    """Fix MAP-Elites scaling so cell meaning is not arrival-order dependent."""

    database.feature_stats = {
        "lmax": {"min": 1.0, "max": 3.0, "values": []},
        "higher_order_fraction": {"min": 0.0, "max": 1.0, "values": []},
        "parameter_ratio": {"min": 0.5, "max": 1.2, "values": []},
        "num_layers": {"min": 3.0, "max": 8.0, "values": []},
    }


def parse_json_response(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return json.loads(stripped)


async def generate_compilable_patch(
    ensemble,
    prompt,
    parent_spec,
    factor,
    seen,
    repair_attempts,
):
    """Generate a factor-local patch and repair compiler/duplicate failures."""

    response = await ensemble.generate_with_context(
        system_message=prompt["system"],
        messages=[{"role": "user", "content": prompt["user"]}],
    )
    history = []
    for attempt in range(repair_attempts + 1):
        try:
            child_spec, reasoning = parse_llm_patch(response, parent_spec, factor)
            validate_qm9_alpha_reasoning(reasoning)
            architecture_id = child_spec.architecture_id()
            if architecture_id in seen:
                raise ValueError("duplicate architecture {}".format(architecture_id))
            return child_spec, reasoning, response, history
        except Exception as exc:
            history.append(
                {
                    "stage": "compiler",
                    "attempt": attempt + 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "response": response,
                }
            )
            if attempt >= repair_attempts:
                raise
            correction = (
                prompt["user"]
                + "\n\nThe trusted compiler rejected the previous response.\n"
                + "Compiler error: "
                + str(exc)[:2000]
                + "\nPrevious response:\n"
                + response[:4000]
                + "\nReturn one corrected strict-JSON response. Change the same factor only."
            )
            response = await ensemble.generate_with_context(
                system_message=prompt["system"],
                messages=[{"role": "user", "content": correction}],
            )
    raise RuntimeError("unreachable patch generation state")


async def run(args):
    sys.path.insert(0, args.openevolve_root)
    sys.path.insert(0, str(Path(args.evaluator_file).parent))
    from openevolve.config import load_config
    from openevolve.database import Program, ProgramDatabase
    from openevolve.llm.ensemble import LLMEnsemble
    import evaluator as evaluator_adapter

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidates").mkdir(exist_ok=True)
    config = load_config(args.config)
    config.database.db_path = str(output / "database")
    config.database.in_memory = True
    # Equivariance-aware quality-diversity map: preserve elites across angular
    # order, higher-order capacity, scale and depth rather than source-code size.
    config.database.feature_dimensions = [
        "lmax",
        "higher_order_fraction",
        "parameter_ratio",
        "num_layers",
    ]
    config.database.feature_bins = {
        "lmax": 3,
        "higher_order_fraction": 4,
        "parameter_ratio": 4,
        "num_layers": 4,
    }
    config.database.num_islands = min(config.database.num_islands, 3)
    config.database.population_size = min(config.database.population_size, 100)
    config.database.archive_size = min(config.database.archive_size, 30)
    config.database.random_seed = args.seed
    for model in config.llm.models:
        model.random_seed = args.seed

    database = ProgramDatabase(config.database)
    initialize_equivariant_feature_ranges(database)
    ensemble = LLMEnsemble(config.llm.models)
    router = EvidenceCalibratedRouter(seed=args.seed)
    if args.router_mode == "evidence" and args.router_prior:
        router.load_prior(args.router_prior)
    factor_rng = random.Random(args.seed + 100003)
    applied_counterfactual_credits = set()
    os.environ["NAS_MAX_STEPS"] = str(args.max_steps)
    os.environ["NAS_SKIP_SYMMETRY"] = "1" if args.skip_symmetry else "0"
    initial_code = Path(args.initial_program).read_text(encoding="utf-8")
    initial_path = output / "candidates" / "iteration_0000.py"
    initial_path.write_text(initial_code, encoding="utf-8")
    if args.initial_metrics:
        initial_metrics = json.loads(Path(args.initial_metrics).read_text(encoding="utf-8"))
    else:
        initial_metrics = evaluator_adapter.evaluate(str(initial_path))
    initial = Program(
        id=str(uuid.uuid4()),
        code=initial_code,
        language="python",
        metrics=initial_metrics,
        iteration_found=0,
        metadata={"architecture_id": initial_metrics.get("architecture_id")},
    )
    database.add(initial, iteration=0)
    seen = {initial_metrics.get("architecture_id")}
    append_jsonl(
        output / "evolution.jsonl",
        {"iteration": 0, "kind": "initial", "metrics": initial_metrics},
    )

    proposal_limit = args.max_proposals if args.valid_target > 0 else args.iterations
    valid_children = 0
    completed_iterations = 0
    for iteration in range(1, proposal_limit + 1):
        completed_iterations = iteration
        parent, inspirations = database.sample(num_inspirations=2)
        parent_spec = extract_literal_spec_source(parent.code)
        memory = summarize_lineage(parent, database.get).to_dict()
        parent_prompt_metrics = compact_metrics(parent.metrics)
        if (
            args.router_mode == "evidence"
            and not args.disable_iacc
            and args.resolved_counterfactuals
        ):
            newly_resolved = resolved_counterfactual_credits(
                args.resolved_counterfactuals, applied_counterfactual_credits
            )
            for resolved in newly_resolved:
                resolved_factor = EvolutionFactor(resolved["selected_factor"])
                router.update(
                    resolved_factor,
                    valid=True,
                    mae_gain=resolved["mae_gain"],
                )
                applied_counterfactual_credits.add(resolved["key"])
                append_jsonl(
                    output / "counterfactual_credit_applied.jsonl",
                    dict(resolved, iteration_applied=iteration),
                )
        context = {
            "symmetry_drift": float(parent.metrics.get("max_symmetry_error", 0.0))
            / max(args.symmetry_reference, 1.0e-12),
            "relative_step_time": float(parent.metrics.get("step_time_ms", 0.0))
            / max(args.baseline_step_time_ms, 1.0e-12),
            "parameter_ratio": float(parent.metrics.get("parameter_ratio", 1.0)),
        }
        factor = (
            router.select(context)
            if args.router_mode == "evidence"
            else factor_rng.choice(list(EvolutionFactor))
        )
        artifacts = {
            "last_failure": parent.metrics.get("error", ""),
            "layerwise_symmetry_maxima": compact_layerwise_symmetry(parent.metrics),
        }
        reflection_prompt = prompt_for_reflection(
            factor=factor,
            parent_json=parent_spec.canonical_json(),
            metrics=parent_prompt_metrics,
            artifacts=artifacts,
            search_memory=memory,
        )
        prompt = prompt_for_factor(
            factor=factor,
            parent_json=parent_spec.canonical_json(),
            metrics=parent_prompt_metrics,
            inspirations=[
                {
                    "spec": extract_literal_spec_source(item.code).to_dict(),
                    "measured_metrics": compact_metrics(item.metrics),
                }
                for item in inspirations
            ],
            artifacts=artifacts,
            reflection={},
            search_memory=memory,
        )
        record = {
            "iteration": iteration,
            "parent_id": parent.id,
            "selected_factor": factor.value,
            "reflection_prompt": reflection_prompt,
            "search_memory": memory,
        }
        try:
            if args.skip_reflection:
                reflection_response = ""
                reflection = {
                    "direction": "No RC stage in this registered ablation.",
                    "evidence": ["SAR receives measured metrics and hard constraints only."],
                    "risk": "removing reflection may reduce semantic edit quality",
                }
            else:
                reflection_response = await ensemble.generate_with_context(
                    system_message=reflection_prompt["system"],
                    messages=[{"role": "user", "content": reflection_prompt["user"]}],
                )
                try:
                    reflection = parse_json_response(reflection_response)
                    validate_qm9_alpha_reasoning(json.dumps(reflection, sort_keys=True))
                except Exception as reflection_exc:
                    record["reflection_error"] = "{}: {}".format(
                        type(reflection_exc).__name__, str(reflection_exc)[:1000]
                    )
                    reflection = {
                        "direction": "Use only measured metrics and the selected-factor schema.",
                        "evidence": [
                            "reflection was rejected as non-JSON or scientifically inconsistent"
                        ],
                        "risk": "untrusted reflection; SAR must rely on measured metrics and hard constraints",
                    }
            prompt = prompt_for_factor(
                factor=factor,
                parent_json=parent_spec.canonical_json(),
                metrics=parent_prompt_metrics,
                inspirations=[
                    {
                        "spec": extract_literal_spec_source(item.code).to_dict(),
                        "measured_metrics": compact_metrics(item.metrics),
                    }
                    for item in inspirations
                ],
                artifacts=artifacts,
                reflection=reflection,
                search_memory=memory,
            )
            child_spec, reasoning, response, repair_history = await generate_compilable_patch(
                ensemble=ensemble,
                prompt=prompt,
                parent_spec=parent_spec,
                factor=factor,
                seen=seen,
                repair_attempts=args.repair_attempts,
            )
            architecture_id = child_spec.architecture_id()
            record["llm_response"] = response
            record["reflection_response"] = reflection_response
            record["reflection"] = reflection
            record["prompt"] = prompt
            record["reasoning"] = reasoning
            record["architecture_id"] = architecture_id
            child_code = render_candidate(child_spec)
            candidate_path = output / "candidates" / "iteration_{:04d}.py".format(iteration)
            candidate_path.write_text(child_code, encoding="utf-8")
            metrics = evaluator_adapter.evaluate(str(candidate_path))
            for repair_index in range(args.repair_attempts):
                if metrics.get("valid"):
                    break
                repair_artifacts = dict(artifacts)
                repair_artifacts["rejected_architecture"] = child_spec.canonical_json()
                repair_artifacts["compiler_or_evaluator_error"] = metrics.get("error", "")
                repair_reflection = dict(reflection)
                repair_reflection["repair_instruction"] = (
                    "Repair the rejected child while changing the same factor. "
                    "Respect the exact allowed values and parameter cap."
                )
                repair_prompt = prompt_for_factor(
                    factor=factor,
                    parent_json=parent_spec.canonical_json(),
                    metrics=parent_prompt_metrics,
                    inspirations=[
                        {
                            "spec": extract_literal_spec_source(item.code).to_dict(),
                            "measured_metrics": compact_metrics(item.metrics),
                        }
                        for item in inspirations
                    ],
                    artifacts=repair_artifacts,
                    reflection=repair_reflection,
                    search_memory=memory,
                )
                (
                    repaired_spec,
                    repaired_reasoning,
                    repair_response,
                    compiler_repairs,
                ) = await generate_compilable_patch(
                    ensemble=ensemble,
                    prompt=repair_prompt,
                    parent_spec=parent_spec,
                    factor=factor,
                    seen=seen,
                    repair_attempts=args.repair_attempts,
                )
                repaired_id = repaired_spec.architecture_id()
                repair_entry = {
                    "stage": "evaluator",
                    "attempt": repair_index + 1,
                    "previous_error": metrics.get("error", ""),
                    "response": repair_response,
                    "architecture_id": repaired_id,
                    "compiler_repairs": compiler_repairs,
                }
                child_spec = repaired_spec
                architecture_id = repaired_id
                reasoning = repaired_reasoning
                child_code = render_candidate(child_spec)
                candidate_path.write_text(child_code, encoding="utf-8")
                metrics = evaluator_adapter.evaluate(str(candidate_path))
                repair_entry["metrics"] = metrics
                repair_history.append(repair_entry)
            record["repair_history"] = repair_history
            record["architecture_id"] = architecture_id
            record["reasoning"] = reasoning
            credit = factor_credit(parent.metrics, metrics) if metrics.get("valid") else {}
            counterfactual_required = False
            ancestor = database.get(parent.parent_id) if parent.parent_id else None
            if (
                not args.disable_iacc
                and ancestor is not None
                and metrics.get("valid")
            ):
                required_keys = ("validation_alpha_mae",)
                if all(
                    key in candidate.metrics
                    for candidate in (ancestor, parent)
                    for key in required_keys
                ) and "validation_alpha_mae" in metrics:
                    counterfactual_required = rescue_requires_counterfactual(
                        ancestor.metrics["validation_alpha_mae"],
                        parent.metrics["validation_alpha_mae"],
                        metrics["validation_alpha_mae"],
                    )
            if counterfactual_required:
                request = {
                    "iteration": iteration,
                    "selected_factor": factor.value,
                    "ancestor_program_id": ancestor.id,
                    "ancestor_architecture_id": ancestor.metrics.get("architecture_id"),
                    "parent_program_id": parent.id,
                    "parent_architecture_id": parent.metrics.get("architecture_id"),
                    "child_architecture_id": architecture_id,
                    "ancestor_mae": ancestor.metrics["validation_alpha_mae"],
                    "parent_mae": parent.metrics["validation_alpha_mae"],
                    "child_mae": metrics["validation_alpha_mae"],
                    "ancestor_spec": extract_literal_spec_source(ancestor.code).to_dict(),
                    "factor_replacement": child_spec.to_dict()[factor.value.lower()],
                    "status": "required_before_mae_credit",
                }
                append_jsonl(output / "counterfactual_requests.jsonl", request)
                record["counterfactual_request"] = request
                record["credit_status"] = "provisional_interaction"
            else:
                record["credit_status"] = "resolved_parent_child"
            router.update(
                factor,
                valid=bool(metrics.get("valid")),
                mae_gain=(
                    0.0 if counterfactual_required else credit.get("mae_gain", 0.0)
                ),
                efficiency_gain=credit.get("parameter_reduction", 0.0)
                / 1_000_000.0,
            )
            if metrics.get("valid"):
                child = Program(
                    id=str(uuid.uuid4()),
                    code=child_code,
                    language="python",
                    parent_id=parent.id,
                    generation=parent.generation + 1,
                    metrics=metrics,
                    iteration_found=iteration,
                    metadata={
                        "selected_factor": factor.value,
                        "reasoning": reasoning,
                        "credit": credit,
                        "architecture_id": architecture_id,
                        "repair_attempts": len(repair_history),
                    },
                )
                database.add(child, iteration=iteration)
                seen.add(architecture_id)
                valid_children += 1
            record["metrics"] = metrics
            record["credit"] = credit
        except Exception as exc:
            router.update(factor, valid=False)
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)[:2000]
        append_jsonl(output / "evolution.jsonl", record)
        router.save(str(output / "router_state.json"))
        database.save(str(output / "database"), iteration=iteration)
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "factor": factor.value,
                    "valid": bool(record.get("metrics", {}).get("valid")),
                    "score": record.get("metrics", {}).get("combined_score"),
                    "error": record.get("error", ""),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.valid_target > 0 and valid_children >= args.valid_target:
            break

    best = database.get(database.best_program_id) if database.best_program_id else None
    summary = {
        "iterations": completed_iterations,
        "requested_valid_target": args.valid_target,
        "valid_children": valid_children,
        "unique_architectures": len(seen),
        "best_program_id": database.best_program_id,
        "best_metrics": best.metrics if best else None,
        "router": router.to_dict(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-program", required=True)
    parser.add_argument("--evaluator-file", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--openevolve-root", default="/home/20262202788/openevolve")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--valid-target", type=int, default=0)
    parser.add_argument("--max-proposals", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-symmetry", action="store_true")
    parser.add_argument("--symmetry-reference", type=float, default=0.0015)
    parser.add_argument("--baseline-step-time-ms", type=float, default=210.0)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--initial-metrics", default="")
    parser.add_argument(
        "--router-mode", choices=("evidence", "uniform"), default="evidence"
    )
    parser.add_argument("--skip-reflection", action="store_true")
    parser.add_argument("--disable-iacc", action="store_true")
    parser.add_argument("--router-prior", default="")
    parser.add_argument("--resolved-counterfactuals", default="")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
