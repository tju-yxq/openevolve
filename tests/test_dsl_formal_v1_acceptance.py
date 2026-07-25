import hashlib
import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "preflight_dsl_formal_v1.py"
    spec = importlib.util.spec_from_file_location("preflight_dsl_formal_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_acceptance_evidence_binds_commit_critical_code_and_artifacts(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    critical = project / "critical.py"
    artifact = tmp_path / "pytest.log"
    critical.write_text("trusted = True\n", encoding="utf-8")
    artifact.write_text("158 passed, 1 skipped\n", encoding="utf-8")
    evidence_path = tmp_path / "acceptance.json"
    evidence_path.write_text(
        json.dumps(
            {
                "ready": True,
                "test_evaluated": False,
                "project_commit": "abc123",
                "critical_files": {"critical.py": _sha256(critical)},
                "artifacts": {str(artifact): _sha256(artifact)},
                "checks": [{"name": "a100_pytest", "passed": True}],
            }
        ),
        encoding="utf-8",
    )
    module = _module()
    monkeypatch.setattr(module, "command_ok", lambda _command: (True, "abc123"))
    checks, evidence = module.validate_acceptance_evidence(project, evidence_path)
    assert evidence["project_commit"] == "abc123"
    assert checks
    assert all(item["passed"] for item in checks)


def test_acceptance_evidence_rejects_stale_critical_code(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    critical = project / "critical.py"
    critical.write_text("changed = True\n", encoding="utf-8")
    evidence_path = tmp_path / "acceptance.json"
    evidence_path.write_text(
        json.dumps(
            {
                "ready": True,
                "test_evaluated": False,
                "project_commit": "abc123",
                "critical_files": {"critical.py": "0" * 64},
                "artifacts": {},
                "checks": [],
            }
        ),
        encoding="utf-8",
    )
    module = _module()
    monkeypatch.setattr(module, "command_ok", lambda _command: (True, "abc123"))
    checks, _evidence = module.validate_acceptance_evidence(project, evidence_path)
    assert any(item["name"] == "acceptance_critical_file:critical.py" and not item["passed"] for item in checks)


def test_preflight_discovers_llm_environment_contract_without_reading_secret(tmp_path):
    config = tmp_path / "llm.yaml"
    config.write_text(
        "api_key: ${GLM_API_KEY}\nsecondary: ${DEEPSEEK_API_KEY}\nrepeat: ${GLM_API_KEY}\n",
        encoding="utf-8",
    )
    assert _module().required_environment_variables(config) == ["DEEPSEEK_API_KEY", "GLM_API_KEY"]


def test_preflight_allows_audited_protocol_correction_only_before_training(tmp_path):
    module = _module()
    run_root = tmp_path / "run"
    run_root.mkdir()
    protocol = run_root / "protocol.json"
    protocol.write_text('{"version": 1}', encoding="utf-8")
    assert module.write_frozen_material(protocol, '{"version": 2}', run_root, "protocol") == "migrated"
    assert (run_root / "protocol_migrations" / "migrations.jsonl").is_file()
    evolution = run_root / "search" / "evolution.jsonl"
    evolution.parent.mkdir()
    evolution.write_text(
        json.dumps({"metrics": {"charged_gpu_seconds": 1.0, "checkpoint_last": ""}}) + "\n",
        encoding="utf-8",
    )
    with __import__("pytest").raises(RuntimeError, match="refuse protocol drift"):
        module.write_frozen_material(protocol, '{"version": 3}', run_root, "protocol")
