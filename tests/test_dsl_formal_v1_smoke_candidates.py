import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_dsl_formal_v1_smoke.py"
    spec = importlib.util.spec_from_file_location("run_dsl_formal_v1_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_v1_smoke_materializes_every_certified_factor_option(tmp_path):
    candidates = _module().materialize_candidates(tmp_path)
    assert [item["factor_id"] for item in candidates] == [
        "", "F2.2", "F2.2", "F2.2", "F4.4", "F4.4", "F5.3", "F5.3", "F6.3"
    ]
    assert [item["lowering_plan"]["mode"] for item in candidates] == [
        "exact_reference",
        "exact_constructor",
        "exact_constructor",
        "exact_constructor",
        "exact_constructor",
        "exact_constructor",
        "exact_constructor",
        "exact_constructor",
        "exact_hybrid",
    ]
    assert len({item["program_id"] for item in candidates}) == 9
    assert all(Path(item["program_path"]).is_file() for item in candidates)
