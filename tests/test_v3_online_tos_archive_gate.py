import json
import pytest

from scripts.archive_online_v3_to_tos import assert_archive_ready, build_manifest


def test_archive_gate_rejects_incomplete_run(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "state.json").write_text(json.dumps({"stage": "train_250k", "test_evaluated": False}))
    with pytest.raises(RuntimeError):
        assert_archive_ready(tmp_path)


def test_archive_manifest_after_ready_gate(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "state.json").write_text(json.dumps({"stage": "ready_for_tos_archive", "test_evaluated": True}))
    (tmp_path / "evidence.txt").write_text("complete")
    assert_archive_ready(tmp_path)
    manifest = build_manifest(tmp_path)
    assert manifest["file_count"] == 2
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])

