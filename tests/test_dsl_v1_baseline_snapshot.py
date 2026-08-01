from pathlib import Path


def test_v1_baseline_snapshot_covers_the_complete_current_language():
    from scripts.export_dsl_v1_baseline_snapshot import build_snapshot

    project = Path(__file__).resolve().parents[1]
    payload = build_snapshot(project, pytest_summary="test")

    assert payload["snapshot_version"] == "evoequilang-v1-baseline-snapshot@1"
    assert payload["primitive_count"] == 33
    assert payload["motif_count"] == 8
    assert payload["lowering"]["rule_count"] == 33
    assert len(payload["semantic_snapshot_id"]) == 64
    assert payload["tests"]["summary"] == "test"
    assert "equivariant_nas/dsl/registry.py" in payload["semantic_file_sha256"]
    assert "equivariant_nas/dsl/backends/e3nn_backend.py" in payload["semantic_file_sha256"]


def test_v1_semantic_snapshot_id_does_not_depend_on_capture_time_or_test_summary():
    from scripts.export_dsl_v1_baseline_snapshot import build_snapshot

    project = Path(__file__).resolve().parents[1]
    first = build_snapshot(project, pytest_summary="first")
    second = build_snapshot(project, pytest_summary="second")

    assert first["captured_at"] != second["captured_at"] or first["tests"] != second["tests"]
    assert first["semantic_snapshot_id"] == second["semantic_snapshot_id"]
