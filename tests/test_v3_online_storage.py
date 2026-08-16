import json
from pathlib import Path

from equivariant_nas.dsl.online_protocol import OnlineV3Protocol


def test_protocol_uses_portable_run_root_environment():
    path = Path(__file__).parents[1] / "configs" / "dsl_v3_online_20k_60_protocol.json"
    protocol = OnlineV3Protocol.load(path)
    assert protocol.raw["storage"]["run_root_env"] == "EQUINAS_RUN_ROOT"
    assert protocol.raw["storage"]["default_run_root"].startswith("/mlplatform/equiNAS/")
    assert "/home/20262202788" not in json.dumps(protocol.raw)

