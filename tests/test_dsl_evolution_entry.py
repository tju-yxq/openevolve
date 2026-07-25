from scripts.run_dsl_evolution import get_parser


def test_dsl_evolution_entry_requires_program_task_evaluator_and_config():
    args = get_parser().parse_args([
        "--initial-program", "initial.json",
        "--task-contract", "task.json",
        "--evaluator-file", "evaluator.py",
        "--config", "openevolve.yaml",
        "--output", "run",
        "--iterations", "3",
    ])
    assert args.initial_program == "initial.json"
    assert args.task_contract == "task.json"
    assert args.iterations == 3
    assert args.max_steps == 0
