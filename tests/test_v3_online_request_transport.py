import importlib.util
import json
from pathlib import Path


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "run_dsl_v3_online_multifidelity.py"
    spec = importlib.util.spec_from_file_location("online_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_large_request_uses_file_transport(tmp_path):
    runner = _load_runner()
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "assert 'EQUINAS_REQUEST_JSON' not in os.environ\n"
        "p = Path(os.environ['EQUINAS_REQUEST_JSON_PATH'])\n"
        "request = json.loads(p.read_text(encoding='utf-8'))\n"
        "print(json.dumps({'size': len(request['large'])}))\n",
        encoding="utf-8",
    )
    payload = {"large": "x" * 500000}
    result = runner._invoke(f'python "{adapter}"', payload, run_root=tmp_path)
    assert result == {"size": 500000}
    assert list((tmp_path / "requests").glob("request_*.json")) == []
