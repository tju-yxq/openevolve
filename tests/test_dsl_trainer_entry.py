import pytest


pytest.importorskip("timm")

from equivariant_nas.training.fixed_step_trainer import get_parser


def test_fixed_step_trainer_accepts_a_dsl_program_path():
    parser = get_parser()
    args = parser.parse_args([
        "--dsl-program",
        "candidate.json",
        "--equiformer-v2-root",
        "/pinned/equiformer_v2",
        "--dsl-task-contract",
        "qm9-alpha-task.json",
    ])
    assert args.dsl_program == "candidate.json"
    assert args.architecture_spec is None
    assert args.equiformer_v2_root == "/pinned/equiformer_v2"
    assert args.dsl_task_contract == "qm9-alpha-task.json"
