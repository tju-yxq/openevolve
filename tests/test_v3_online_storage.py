import json
from pathlib import Path

from equivariant_nas.dsl.online_protocol import OnlineV3Protocol
from equivariant_nas.dsl.pipeline import _checkpoint_interval_steps, _resolve_pipeline_run_root


def test_protocol_uses_portable_run_root_environment():
    path = Path(__file__).parents[1] / "configs" / "dsl_v3_online_20k_60_protocol.json"
    protocol = OnlineV3Protocol.load(path)
    assert protocol.raw["storage"]["run_root_env"] == "EQUINAS_RUN_ROOT"
    assert protocol.raw["storage"]["default_run_root"].startswith("/mlplatform/equiNAS/")
    assert "/home/20262202788" not in json.dumps(protocol.raw)


def test_pipeline_artifacts_follow_run_root(monkeypatch, tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "mlplatform-cache"
    monkeypatch.setenv("EQUINAS_RUN_ROOT", str(cache))
    assert _resolve_pipeline_run_root(source) == cache.resolve()


def test_explicit_pipeline_root_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("EQUINAS_RUN_ROOT", str(tmp_path / "experiment"))
    monkeypatch.setenv("EQUINAS_PIPELINE_RUN_ROOT", str(tmp_path / "pipeline"))
    assert _resolve_pipeline_run_root(tmp_path / "source") == (tmp_path / "pipeline").resolve()


def test_v3_uses_periodic_resumable_checkpoints(monkeypatch):
    monkeypatch.delenv("NAS_V3_CHECKPOINT_INTERVAL_STEPS", raising=False)
    assert _checkpoint_interval_steps(is_v3_program=True, max_steps=20000) == 1000
    assert _checkpoint_interval_steps(is_v3_program=True, max_steps=10) == 10


def test_v3_checkpoint_interval_is_configurable(monkeypatch):
    monkeypatch.setenv("NAS_V3_CHECKPOINT_INTERVAL_STEPS", "500")
    assert _checkpoint_interval_steps(is_v3_program=True, max_steps=20000) == 500
