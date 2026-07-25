from equivariant_nas.dsl import import_equiformer_v1
from equivariant_nas.dsl.serialization import save_program
from equivariant_nas.pipeline import evaluate_candidate_pipeline
from equivariant_nas.spec import baseline_spec


def test_shared_pipeline_dispatches_dsl_json_without_parsing_legacy_python(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.dsl.json"
    save_program(import_equiformer_v1(baseline_spec()), str(candidate))
    sentinel = {"valid": True, "candidate_format": "evoequilang"}
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "equivariant_nas.dsl.pipeline.evaluate_dsl_candidate_pipeline",
        fake_pipeline,
    )
    result = evaluate_candidate_pipeline(
        str(candidate),
        str(tmp_path),
        "/equiformer",
        "/qm9",
        max_steps=17,
        run_symmetry=False,
        equiformer_v2_root="/equiformer_v2",
        dsl_task_contract="/contracts/qm9-alpha.json",
    )
    assert result is sentinel
    assert captured["program_path"] == str(candidate)
    assert captured["max_steps"] == 17
    assert captured["equiformer_v2_root"] == "/equiformer_v2"
    assert captured["task_contract_path"] == "/contracts/qm9-alpha.json"
