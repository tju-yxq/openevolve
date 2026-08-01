import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_generic_lowering_v1_audit.py"
    spec = importlib.util.spec_from_file_location("run_generic_lowering_v1_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generic_lowering_v1_audit_proves_registry_driven_first_version(tmp_path):
    audit = _module().run_audit(tmp_path, run_tests=False)

    assert audit["first_version_achieved"]
    assert audit["generic_rule_count"] == 103
    assert audit["core_primitive_count"] == 103
    assert audit["implementation"]["registry_driven"]
    assert audit["implementation"]["large_op_dispatch_removed"]
    assert audit["completion_criteria"]["generic_qm9_path_is_explicitly_reachable"]
    assert audit["completion_criteria"]["compiler_plan_consumes_backend_support"]
    assert audit["fusion_only_primitives"] == []
    assert (tmp_path / "generic_lowering_v1_audit.json").is_file()
    assert (tmp_path / "通用Lowering第一版实现与审计报告.md").is_file()
