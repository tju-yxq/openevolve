import json

import pytest

from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    Compiler,
    Diagnostic,
    DSLValidationError,
    EquivariantType,
    EvidenceItem,
    EvidenceStore,
    GroupSpec,
    HoleSink,
    InputPort,
    Irreps,
    LanguageVersion,
    Node,
    OutputPort,
    ResourceContract,
    TaskContract,
    TypedPatch,
    TypedHole,
    AvailableValue,
    architecture_id,
    core_registry,
    complete_typed_hole,
    materialize_completion_patch,
    planner_prompt,
    repair_prompt,
    reference_motif_registry,
    select_active_vocabulary,
    synthesizer_prompt,
    task_reasoning_context,
    validate_task_reasoning,
)
from equivariant_nas.dsl.serialization import load_task_contract, save_task_contract


def setup_objects():
    group = GroupSpec.o3()
    node_scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0e", "O3"))
    graph_scalar = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0e", "O3"))
    program = ArchitectureProgram(
        "1.0.0",
        "qm9_alpha",
        (InputPort("x", node_scalar),),
        (Node("pool", "core.global_pool", {"x": ("input:x",)}),),
        (OutputPort("prediction", "pool", graph_scalar),),
    )
    task = TaskContract(
        "qm9_alpha",
        group,
        graph_scalar,
        "fixed-training-protocol",
        ResourceContract(10_000_000, 10_000_000_000, 1.5),
    )
    primitives = core_registry()
    motifs = reference_motif_registry()
    language = LanguageVersion("1.0.0", "", primitives.names(), motifs.names(), "now", {})
    vocabulary = select_active_vocabulary(language, group, primitives, motifs)
    return program, task, primitives, language, vocabulary


def test_planner_prompt_rejects_test_evidence():
    program, task, _, _, vocabulary = setup_objects()
    evidence = (EvidenceItem("test:1", "test", "mae", {"value": 0.1}),)
    with pytest.raises(DSLValidationError) as error:
        planner_prompt(task, program, evidence, vocabulary, ("pool",))
    assert error.value.diagnostics[0].code == "E_TASK_004"


def test_evidence_store_is_append_only_and_hides_test_from_generation(tmp_path):
    program, task, primitives, language, vocabulary = setup_objects()
    store = EvidenceStore(str(tmp_path / "evidence.sqlite"))
    store.register_language(language, reference_motif_registry())
    candidate_id = store.add_candidate(program, task, primitives)
    store.add_evaluation(candidate_id, split="validation", fidelity_steps=8000, seed=1, metrics={"mae": 0.2}, resources={})
    store.add_evaluation(candidate_id, split="test", fidelity_steps=250000, seed=1, metrics={"mae": 0.1}, resources={}, final_audit=True)
    visible = store.candidate_generation_evidence(candidate_id)
    assert len(visible) == 1
    assert visible[0]["split"] == "validation"
    with pytest.raises(DSLValidationError) as error:
        store.add_prompt_run(
            candidate_id,
            role="planner",
            model="test-model",
            prompt={"system": "x", "user": "y"},
            response_text="{}",
            vocabulary=vocabulary,
            evidence_ids=(),
            visible_splits=("validation", "test"),
            token_usage={},
        )
    assert error.value.diagnostics[0].code == "E_STORE_003"


def test_language_snapshot_rejects_missing_motif_definitions(tmp_path):
    _, _, _, language, _ = setup_objects()
    store = EvidenceStore(str(tmp_path / "incomplete-language.sqlite"))
    with pytest.raises(DSLValidationError) as error:
        store.register_language(language)
    assert error.value.diagnostics[0].code == "E_STORE_009"


def test_unflagged_test_evaluation_is_rejected(tmp_path):
    program, task, primitives, _, _ = setup_objects()
    store = EvidenceStore(str(tmp_path / "evidence.sqlite"))
    candidate_id = store.add_candidate(program, task, primitives)
    with pytest.raises(DSLValidationError) as error:
        store.add_evaluation(candidate_id, split="test", fidelity_steps=1, seed=1, metrics={}, resources={})
    assert error.value.diagnostics[0].code == "E_STORE_002"


def test_completion_runs_are_content_addressed_and_recoverable(tmp_path):
    program, task, primitives, language, _ = setup_objects()
    store = EvidenceStore(str(tmp_path / "evidence.sqlite"))
    candidate_id = store.add_candidate(program, task, primitives)
    node_type = program.inputs[0].value_type
    graph_type = program.outputs[0].expected_type
    hole = TypedHole("stored_readout", graph_type, (AvailableValue("input:x", node_type),), max_steps=1)
    result = complete_typed_hole(hole, primitives)
    patch = materialize_completion_patch(
        program,
        hole,
        result,
        HoleSink.program_output("prediction"),
        primitives,
        task_contract_hash=task.content_hash(),
    )
    completion_id = store.add_completion_run(
        candidate_id,
        language.registry_hash(),
        hole,
        result,
        sink=HoleSink.program_output("prediction"),
        materialized_patch=patch,
        status="materialized",
    )
    repeated_id = store.add_completion_run(
        candidate_id,
        language.registry_hash(),
        hole,
        result,
        sink=HoleSink.program_output("prediction"),
        materialized_patch=patch,
        status="applied",
    )
    assert repeated_id == completion_id
    restored = store.get_completion_run(completion_id)
    assert restored["status"] == "applied"
    assert restored["request"]["hole"]["hole_id"] == "stored_readout"
    assert restored["materialized_patch"]["expected_effects"]["test_evaluated"] is False


def test_rewrite_proof_trace_is_persisted_with_the_compiler_run(tmp_path):
    program, task, primitives, _, _ = setup_objects()
    identity = Node("identity", "core.identity", {"x": ("input:x",)})
    rewritten_source = ArchitectureProgram(
        program.language_version,
        program.task_contract,
        program.inputs,
        (identity, Node("pool", "core.global_pool", {"x": ("identity",)})),
        program.outputs,
    )
    compiler = Compiler(primitives, reference_motif_registry())
    artifact = compiler.analyze(rewritten_source, task)
    store = EvidenceStore(str(tmp_path / "rewrite.sqlite"))
    candidate_id = store.add_compiled_candidate(artifact, task)
    store.add_compiler_run(
        candidate_id,
        "evoequilang-2",
        "success",
        inference=artifact.inference,
        rewrite_trace=artifact.rewrite_trace,
        rewrite_registry_hash=artifact.rewrite_registry_hash,
    )
    import sqlite3
    with sqlite3.connect(str(tmp_path / "rewrite.sqlite")) as connection:
        registry_hash, trace_json = connection.execute(
            "SELECT rewrite_registry_hash, trace_json FROM rewrite_runs"
        ).fetchone()
    trace = json.loads(trace_json)
    assert registry_hash == artifact.rewrite_registry_hash
    assert trace[0]["rule_id"] == "core.eliminate_identity"
    assert trace[0]["before_fingerprint"] != trace[0]["after_fingerprint"]


def test_task_contract_roundtrip_preserves_the_cross_process_architecture_id(tmp_path):
    program, task, primitives, _, _ = setup_objects()
    path = tmp_path / "task.json"
    save_task_contract(task, str(path))
    loaded = load_task_contract(str(path))
    assert loaded == task
    compiler = Compiler(primitives, reference_motif_registry())
    assert compiler.analyze(program, loaded).architecture_id == compiler.analyze(program, task).architecture_id
    assert compiler.analyze(program, task).architecture_id != compiler.analyze(program).architecture_id


def test_llm_patch_prompts_expose_source_ids_and_the_exact_edit_wire_contract():
    program, task, primitives, _, vocabulary = setup_objects()
    compiler = Compiler(primitives, reference_motif_registry())
    parent_id = compiler.analyze(program, task).architecture_id
    plan = {
        "claim": "insert a typed operation before pooling",
        "scope": ["pool"],
        "abstract_goals": ["change representation flow"],
        "evidence_refs": [],
        "uncertainty": "untrained",
        "risk": "type mismatch",
    }
    motifs = reference_motif_registry()
    prompt = synthesizer_prompt(task, program, plan, vocabulary, parent_id, primitives, motifs)
    payload = json.loads(prompt["user"])
    assert payload["parent_program"]["nodes"][0]["id"] == "pool"
    assert payload["authoritative_patch_schema"]["properties"]["parent_architecture_id"]["const"] == parent_id
    edit_schema = payload["authoritative_patch_schema"]["properties"]["edits"]["items"]
    assert payload["authoritative_patch_schema"]["properties"]["scope"]["const"] == ["pool"]
    assert edit_schema["additionalProperties"] is False
    assert set(edit_schema["required"]) == {"kind", "target", "payload"}
    assert "insert_node" not in edit_schema["properties"]["kind"]["enum"]
    assert "rewire" not in edit_schema["properties"]["kind"]["enum"]
    v1_residual = next(
        item for item in payload["vocabulary_contracts"]
        if item["name"] == "motif.v1_residual_message@1"
    )
    assert any("residual add requires" in item for item in v1_residual["semantic_constraints"])
    assert any("adapter" in item for item in v1_residual["edit_guidance"])

    failed = TypedPatch("1.0", parent_id, program.language_version, plan, ("pool",), ())
    repaired = repair_prompt(
        failed,
        (Diagnostic("E_PATCH_002", "unknown patch edit fields", details={"fields": ["anchor", "mode"]}),),
        vocabulary,
        parent=program,
        parent_architecture_id=parent_id,
        rejected_response='{"edits":[{"anchor":"pool","mode":"insert_after"}]}',
        primitives=primitives,
        motifs=motifs,
    )
    repair_payload = json.loads(repaired["user"])
    assert repair_payload["parent_program"]["nodes"][0]["id"] == "pool"
    assert "anchor, mode, new_subgraph" in repaired["system"]
    assert repair_payload["rejected_response"].startswith('{"edits"')


def test_patch_parser_rejects_alternate_edit_dialects_with_actionable_fields():
    from equivariant_nas.dsl import parse_patch_response

    malformed = json.dumps({
        "patch_version": "1.0",
        "parent_architecture_id": "parent",
        "language_version": "1.0.0",
        "hypothesis": {},
        "scope": ["pool"],
        "edits": [{"anchor": "pool", "mode": "insert_after", "new_subgraph": {}}],
    })
    with pytest.raises(DSLValidationError) as error:
        parse_patch_response(malformed)
    diagnostic = error.value.diagnostics[0]
    assert diagnostic.code == "E_PATCH_002"
    assert diagnostic.details["unknown_fields"] == ["anchor", "mode", "new_subgraph"]
    assert diagnostic.details["missing_fields"] == ["kind", "payload", "target"]


def test_patch_parser_normalizes_exactly_one_fenced_json_object_with_auditable_prose():
    from equivariant_nas.dsl import parse_patch_response

    payload = {
        "patch_version": "1.0",
        "parent_architecture_id": "parent",
        "language_version": "1.0.0",
        "hypothesis": {},
        "scope": ["pool"],
        "edits": [{"kind": "change_attrs", "target": "pool", "payload": {"attrs": {}}}],
    }
    response = "The fix is below.\n```json\n{}\n```".format(json.dumps(payload))
    assert parse_patch_response(response).parent_architecture_id == "parent"
    with pytest.raises(DSLValidationError) as error:
        parse_patch_response(response + "\n```json\n{}\n```")
    assert error.value.diagnostics[0].code == "E_LLM_007"


def test_qm9_alpha_task_semantics_are_explicit_and_tensor_target_claims_are_rejected():
    _, task, _, _, _ = setup_objects()
    context = task_reasoning_context(task)
    assert "isotropic scalar" in context["target_semantics"]
    with pytest.raises(DSLValidationError) as error:
        validate_task_reasoning(task, {"claim": "QM9 alpha is a rank-2 tensor target."})
    assert error.value.diagnostics[0].code == "E_SCIENCE_001"
    validate_task_reasoning(
        task,
        {"claim": "Hidden l>0 irreps may couple into l=0 to improve the invariant scalar alpha prediction."},
    )
