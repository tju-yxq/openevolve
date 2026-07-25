from pathlib import Path


def test_formal_v1_launcher_exposes_safe_preflight_only_mode():
    root = Path(__file__).resolve().parents[1]
    content = (root / "scripts" / "launch_dsl_formal_v1.sh").read_text(encoding="utf-8")
    assert 'MODE="${2:-}"' in content
    assert 'if [ "$MODE" = "--preflight-only" ]' in content
    assert content.index("preflight_dsl_formal_v1.py") < content.index("--preflight-only")


def test_formal_v1_supervisor_stops_after_completion_and_supports_one_shot_validation():
    root = Path(__file__).resolve().parents[1]
    content = (root / "scripts" / "supervise_dsl_formal_v1.sh").read_text(encoding="utf-8")
    assert "completed_validation_selection" in content
    assert 'SUPERVISOR_ONCE="${SUPERVISOR_ONCE:-0}"' in content
    assert 'LAUNCH_MODE="${LAUNCH_MODE:-}"' in content
    assert 'touch "$RUN_ROOT/COMPLETED"' in content
    assert 'flock -n 9' in content
    assert "pgrep" not in content
