#!/usr/bin/env python
"""Run resumable OpenEvolve population search over typed DSL programs."""

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from equivariant_nas.dsl import (
    BACKEND_SEMANTICS_VERSION,
    COMPILER_SEMANTICS_VERSION,
    Compiler,
    DSLGenerationEngine,
    EvidenceItem,
    EvidenceStore,
    LanguageVersion,
    core_registry,
    reference_motif_registry,
    select_active_vocabulary,
    strict_rewrite_registry_hash,
)
from equivariant_nas.dsl.serialization import (
    dumps_program,
    load_program,
    load_task_contract,
    loads_program,
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def _append_jsonl(path, payload):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _ensure_compiler_manifest(output, payload, *, database_has_programs):
    path = Path(output) / "compiler_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                "compiler manifest mismatch; resume requires the exact task, language, rewrite registry, compiler, and backend semantics"
            )
        return path
    if database_has_programs:
        raise RuntimeError(
            "existing OpenEvolve database has no compiler_manifest.json; refuse to reinterpret a legacy search under new semantics"
        )
    _write_json(path, payload)
    return path


def _measured_evidence(program, inspirations):
    items = []
    for candidate in (program,) + tuple(inspirations):
        metrics = dict(candidate.metrics or {})
        if metrics.get("test_evaluated"):
            continue
        items.append(
            EvidenceItem(
                "openevolve:{}".format(candidate.id),
                "validation",
                "candidate_evaluation",
                {
                    "architecture_id": metrics.get("architecture_id"),
                    "fidelity_steps": metrics.get("fidelity_steps", 0),
                    "validation_alpha_mae": metrics.get("validation_alpha_mae"),
                    "parameter_count": metrics.get("parameter_count"),
                    "step_time_ms": metrics.get("step_time_ms"),
                    "valid": bool(metrics.get("valid")),
                },
            )
        )
    return tuple(items)


def _evaluate(evaluator, candidate_path, expected_id):
    metrics = dict(evaluator.evaluate(str(candidate_path)))
    reported = metrics.get("architecture_id")
    if reported and reported != expected_id:
        return {
            "valid": False,
            "combined_score": -1.0e9,
            "failure_stage": "identity_audit",
            "architecture_id": expected_id,
            "reported_architecture_id": reported,
            "error": "evaluator and compiler architecture IDs disagree",
            "test_evaluated": bool(metrics.get("test_evaluated", False)),
        }
    metrics["architecture_id"] = expected_id
    if metrics.get("test_evaluated"):
        metrics.update({
            "valid": False,
            "combined_score": -1.0e9,
            "failure_stage": "test_leakage",
            "error": "candidate-generation evaluation accessed the test split",
        })
    return metrics


async def run(args):
    sys.path.insert(0, args.openevolve_root)
    sys.path.insert(0, str(Path(args.evaluator_file).resolve().parent))
    from openevolve.config import load_config
    from openevolve.database import Program, ProgramDatabase
    from openevolve.llm.ensemble import LLMEnsemble
    import evaluator as evaluator_adapter

    output = Path(args.output).resolve()
    candidates = output / "candidates"
    output.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(exist_ok=True)
    task_path = Path(args.task_contract).resolve()
    task = load_task_contract(str(task_path))
    initial_program = load_program(args.initial_program)
    if initial_program.task_contract != task.task_id:
        raise ValueError("initial program and task contract IDs disagree")

    primitives = core_registry()
    motifs = reference_motif_registry()
    compiler = Compiler(primitives, motifs)
    language = LanguageVersion.from_registries(
        initial_program.language_version,
        "",
        primitives,
        motifs,
        _now(),
        {"search": "open-evolve-typed-dsl"},
    )
    vocabulary = select_active_vocabulary(language, task.group, primitives, motifs)
    store = EvidenceStore(str(output / "evidence.sqlite"))
    store.register_language(language)

    config = load_config(args.config)
    config.database.db_path = str(output / "database")
    config.database.in_memory = True
    config.database.random_seed = args.seed
    for model in config.llm.models:
        model.random_seed = args.seed
    database = ProgramDatabase(config.database)
    compiler_manifest = {
        "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
        "backend_semantics_version": BACKEND_SEMANTICS_VERSION,
        "rewrite_registry_hash": strict_rewrite_registry_hash(),
        "language_registry_hash": language.registry_hash(),
        "task_contract_hash": task.content_hash(),
    }
    _ensure_compiler_manifest(output, compiler_manifest, database_has_programs=bool(database.programs))
    ensemble = LLMEnsemble(config.llm.models)
    engine = DSLGenerationEngine(
        compiler,
        task,
        vocabulary,
        store,
        model_name=args.model_label,
        repair_attempts=args.repair_attempts,
    )

    os.environ["DSL_TASK_CONTRACT"] = str(task_path)
    os.environ["NAS_MAX_STEPS"] = str(args.max_steps)
    os.environ["NAS_BATCH_SIZE"] = str(args.batch_size)
    os.environ["NAS_SKIP_SYMMETRY"] = "1" if args.skip_symmetry else "0"
    if args.equiformer_v2_root:
        os.environ["EQUIFORMER_V2_ROOT"] = args.equiformer_v2_root

    if not database.programs:
        artifact = compiler.analyze(initial_program, task)
        store.add_compiled_candidate(artifact, task)
        store.add_compiler_run(
            artifact.architecture_id,
            COMPILER_SEMANTICS_VERSION,
            "success",
            inference=artifact.inference,
            rewrite_trace=artifact.rewrite_trace,
            rewrite_registry_hash=artifact.rewrite_registry_hash,
        )
        initial_path = candidates / "iteration_0000.dsl.json"
        initial_path.write_text(dumps_program(initial_program), encoding="utf-8")
        metrics = (
            json.loads(Path(args.initial_metrics).read_text(encoding="utf-8"))
            if args.initial_metrics
            else _evaluate(evaluator_adapter, initial_path, artifact.architecture_id)
        )
        metrics["architecture_id"] = artifact.architecture_id
        initial = Program(
            id=str(uuid.uuid4()),
            code=dumps_program(initial_program),
            language="json",
            metrics=metrics,
            iteration_found=0,
            metadata={"architecture_id": artifact.architecture_id, "language_version": initial_program.language_version},
        )
        database.add(initial, iteration=0)
        database.save(str(output / "database"), iteration=0)
        store.add_evaluation(
            artifact.architecture_id,
            split="validation",
            fidelity_steps=int(metrics.get("fidelity_steps", args.max_steps)),
            seed=args.seed,
            metrics=metrics,
            resources={"charged_gpu_seconds": metrics.get("charged_gpu_seconds", 0.0)},
        )
        _append_jsonl(output / "evolution.jsonl", {"iteration": 0, "kind": "initial", "metrics": metrics})

    start = int(database.last_iteration) + 1
    stop_file = Path(args.stop_file).resolve() if args.stop_file else output / "STOP"
    completed = start - 1
    for iteration in range(start, start + args.iterations):
        if stop_file.exists():
            break
        completed = iteration
        _write_json(output / "heartbeat.json", {
            "status": "proposing",
            "updated_at": _now(),
            "iteration": iteration,
            "last_completed_iteration": iteration - 1,
            "stop_file": str(stop_file),
        })
        parent, inspirations = database.sample(num_inspirations=args.inspirations)
        parent_program = loads_program(parent.code)
        scope = tuple(node.id for node in parent_program.nodes) + tuple(
            "output:{}".format(output.name) for output in parent_program.outputs
        )
        record = {
            "iteration": iteration,
            "parent_program_id": parent.id,
            "parent_architecture_id": parent.metrics.get("architecture_id"),
            "inspiration_ids": [item.id for item in inspirations],
        }
        try:
            generated = await engine.generate(
                ensemble,
                parent_program,
                _measured_evidence(parent, inspirations),
                scope,
            )
            child_id = generated.child.architecture_id
            child_path = candidates / "iteration_{:04d}_{}.dsl.json".format(iteration, child_id)
            child_path.write_text(dumps_program(generated.child.source_program), encoding="utf-8")
            _write_json(output / "current_candidate.json", {
                "status": "evaluating",
                "updated_at": _now(),
                "iteration": iteration,
                "architecture_id": child_id,
                "candidate_path": str(child_path),
            })
            metrics = _evaluate(evaluator_adapter, child_path, child_id)
            record.update({
                "architecture_id": child_id,
                "patch": generated.patch.to_dict(),
                "planner_response": dict(generated.planner_response),
                "repair_count": generated.repair_count,
                "planner_repair_count": generated.planner_repair_count,
                "metrics": metrics,
            })
            store.add_evaluation(
                child_id,
                split="validation",
                fidelity_steps=int(metrics.get("fidelity_steps", args.max_steps)),
                seed=args.seed,
                metrics=metrics,
                resources={"charged_gpu_seconds": metrics.get("charged_gpu_seconds", 0.0)},
            )
            if metrics.get("valid"):
                child = Program(
                    id=str(uuid.uuid4()),
                    code=dumps_program(generated.child.source_program),
                    language="json",
                    parent_id=parent.id,
                    generation=parent.generation + 1,
                    metrics=metrics,
                    iteration_found=iteration,
                    metadata={
                        "architecture_id": child_id,
                        "patch": generated.patch.to_dict(),
                        "repair_count": generated.repair_count,
                        "planner_repair_count": generated.planner_repair_count,
                    },
                )
                database.add(child, iteration=iteration)
        except Exception as exc:
            record.update({"error_type": type(exc).__name__, "error": str(exc)[:4000]})
        _append_jsonl(output / "evolution.jsonl", record)
        database.save(str(output / "database"), iteration=iteration)
        current = output / "current_candidate.json"
        if current.exists():
            current.unlink()
        best = database.get_best_program()
        summary = {
            "last_completed_iteration": iteration,
            "program_count": len(database.programs),
            "best_program_id": best.id if best else None,
            "best_metrics": best.metrics if best else None,
            "task_contract_hash": task.content_hash(),
            "language_registry_hash": language.registry_hash(),
            "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
            "rewrite_registry_hash": strict_rewrite_registry_hash(),
            "updated_at": _now(),
        }
        _write_json(output / "summary.json", summary)
        _write_json(output / "heartbeat.json", dict(summary, status="running", stop_file=str(stop_file)))

    best = database.get_best_program()
    database.save(str(output / "database"), iteration=completed)
    final = {
        "last_completed_iteration": completed,
        "program_count": len(database.programs),
        "best_program_id": best.id if best else None,
        "best_metrics": best.metrics if best else None,
        "task_contract_hash": task.content_hash(),
        "language_registry_hash": language.registry_hash(),
        "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
        "rewrite_registry_hash": strict_rewrite_registry_hash(),
        "status": "stopped" if stop_file.exists() else "completed",
        "updated_at": _now(),
    }
    _write_json(output / "summary.json", final)
    _write_json(output / "heartbeat.json", dict(final, stop_file=str(stop_file)))
    print(json.dumps(final, ensure_ascii=False, sort_keys=True))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-program", required=True)
    parser.add_argument("--task-contract", required=True)
    parser.add_argument("--evaluator-file", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--openevolve-root", default="/home/20262202788/openevolve")
    parser.add_argument("--equiformer-v2-root", default="")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--inspirations", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--model-label", default="openevolve-ensemble")
    parser.add_argument("--initial-metrics", default="")
    parser.add_argument("--skip-symmetry", action="store_true")
    parser.add_argument("--stop-file", default="")
    return parser


def main():
    asyncio.run(run(get_parser().parse_args()))


if __name__ == "__main__":
    main()
