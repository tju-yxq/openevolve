from pathlib import Path


def test_formal_v1_launcher_exposes_safe_preflight_only_mode():
    root = Path(__file__).resolve().parents[1]
    content = (root / "scripts" / "launch_dsl_formal_v1.sh").read_text(encoding="utf-8")
    assert 'MODE="${2:-}"' in content
    assert 'if [ "$MODE" = "--preflight-only" ]' in content
    assert content.index("preflight_dsl_formal_v1.py") < content.index("--preflight-only")
    assert 'NAS_BUDGET_LEDGER="${NAS_BUDGET_LEDGER:-$RUN_ROOT/budget_ledger.jsonl}"' in content
    assert "--valid-per-factor-target 2" in content
    assert '--parent-program "$RUN_ROOT/initial_program.dsl.json"' in content
    assert '--train-subset-file "$PROJECT/data_splits/qm9_train_quarter_seed201.npz"' in content
    assert "--eval-interval-epochs 10" in content


def test_formal_v1_supervisor_stops_after_completion_and_supports_one_shot_validation():
    root = Path(__file__).resolve().parents[1]
    content = (root / "scripts" / "supervise_dsl_formal_v1.sh").read_text(encoding="utf-8")
    assert "final_completed" in content
    assert 'SUPERVISOR_ONCE="${SUPERVISOR_ONCE:-0}"' in content
    assert 'LAUNCH_MODE="${LAUNCH_MODE:-}"' in content
    assert 'touch "$RUN_ROOT/COMPLETED"' in content
    assert 'flock -n 9' in content
    assert "pgrep" not in content
