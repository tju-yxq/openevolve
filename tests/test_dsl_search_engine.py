import asyncio
import json

from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    Compiler,
    DSLGenerationEngine,
    DSLValidationError,
    EquivariantType,
    EvidenceStore,
    GroupSpec,
    InputPort,
    Irreps,
    LanguageVersion,
    Node,
    OutputPort,
    ResourceContract,
    TaskContract,
    core_registry,
    reference_motif_registry,
    select_active_vocabulary,
)


class FakeEnsemble:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate_with_context(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_planner_synthesizer_repairer_loop_compiles_and_persists(tmp_path):
    group = GroupSpec.o3()
    input_type = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0e", "O3"))
    output_type = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0e", "O3"))
    parent = ArchitectureProgram(
        "1.0.0",
        "task",
        (InputPort("x", input_type),),
        (
            Node("project", "core.irrep_linear", {"x": ("input:x",)}, {"out_irreps": "2x0e"}),
            Node("readout", "core.irrep_linear", {"x": ("project",)}, {"out_irreps": "1x0e"}),
            Node("pool", "core.global_pool", {"x": ("readout",)}),
        ),
        (OutputPort("prediction", "pool", output_type),),
    )
    task = TaskContract("task", group, output_type, "protocol", ResourceContract(1_000_000, 1_000_000_000, 2.0))
    primitives = core_registry()
    motifs = reference_motif_registry()
    compiler = Compiler(primitives, motifs)
    language = LanguageVersion("1.0.0", "", primitives.names(), motifs.names(), "now", {})
    vocabulary = select_active_vocabulary(language, group, primitives, motifs)
    store = EvidenceStore(str(tmp_path / "search.sqlite"))
    plan = json.dumps({
        "claim": "change scalar channel flow",
        "scope": ["project"],
        "abstract_goals": ["change multiplicity without changing output type"],
        "evidence_refs": [],
        "uncertainty": "untrained",
        "risk": "capacity change",
    })

    parent_id = compiler.analyze(parent, task).architecture_id
    def patch(out_irreps):
        return json.dumps({
            "patch_version": "1.0",
            "parent_architecture_id": parent_id,
            "language_version": "1.0.0",
            "hypothesis": {"claim": "change scalar channel flow"},
            "scope": ["project"],
            "edits": [{"kind": "change_attrs", "target": "project", "payload": {"attrs": {"out_irreps": out_irreps}}}],
            "preconditions": [],
            "postconditions": [],
            "expected_effects": {"accuracy": "hypothesis_only"},
        })

    ensemble = FakeEnsemble([plan, patch("1x1o"), patch("3x0e")])
    engine = DSLGenerationEngine(compiler, task, vocabulary, store, model_name="fake", repair_attempts=1)
    result = asyncio.run(engine.generate(ensemble, parent, (), ("project",)))
    assert result.repair_count == 1
    assert result.child.architecture_id != result.parent.architecture_id
    assert store.candidate_exists(result.child.architecture_id)


def test_llm_protocol_accepts_one_json_fence_and_audits_failed_patch_responses(tmp_path):
    group = GroupSpec.o3()
    scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0e", "O3"))
    graph_scalar = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0e", "O3"))
    parent = ArchitectureProgram(
        "1.0.0",
        "task",
        (InputPort("x", scalar),),
        (Node("pool", "core.global_pool", {"x": ("input:x",)}),),
        (OutputPort("prediction", "pool", graph_scalar),),
    )
    task = TaskContract("task", group, graph_scalar, "protocol", ResourceContract(1000, 1000000, 2.0))
    primitives = core_registry()
    motifs = reference_motif_registry()
    compiler = Compiler(primitives, motifs)
    language = LanguageVersion("1.0.0", "", primitives.names(), motifs.names(), "now", {})
    vocabulary = select_active_vocabulary(language, group, primitives, motifs)
    store = EvidenceStore(str(tmp_path / "audit.sqlite"))
    plan = """```json
{"claim":"change pooling","scope":["pool"],"abstract_goals":["test"],"evidence_refs":[],"uncertainty":"high","risk":"invalid"}
```"""
    ensemble = FakeEnsemble([plan, "not json"])
    engine = DSLGenerationEngine(compiler, task, vocabulary, store, model_name="fake", repair_attempts=0)
    import pytest
    with pytest.raises(Exception):
        asyncio.run(engine.generate(ensemble, parent, (), ("pool",)))
    import sqlite3
    with sqlite3.connect(str(tmp_path / "audit.sqlite")) as connection:
        roles = [row[0] for row in connection.execute("SELECT role FROM prompt_runs ORDER BY created_at")]
    assert roles == ["planner", "synthesizer"]


def test_markdown_planner_response_is_repaired_before_patch_generation(tmp_path):
    group = GroupSpec.o3()
    scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0e", "O3"))
    graph_scalar = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0e", "O3"))
    parent = ArchitectureProgram(
        "1.0.0",
        "task",
        (InputPort("x", scalar),),
        (
            Node("project", "core.irrep_linear", {"x": ("input:x",)}, {"out_irreps": "2x0e"}),
            Node("readout", "core.irrep_linear", {"x": ("project",)}, {"out_irreps": "1x0e"}),
            Node("pool", "core.global_pool", {"x": ("readout",)}),
        ),
        (OutputPort("prediction", "pool", graph_scalar),),
    )
    task = TaskContract("task", group, graph_scalar, "protocol", ResourceContract(10000, 1000000, 2.0))
    primitives = core_registry()
    motifs = reference_motif_registry()
    compiler = Compiler(primitives, motifs)
    language = LanguageVersion("1.0.0", "", primitives.names(), motifs.names(), "now", {})
    vocabulary = select_active_vocabulary(language, group, primitives, motifs)
    store = EvidenceStore(str(tmp_path / "planner_repair.sqlite"))
    parent_id = compiler.analyze(parent, task).architecture_id
    repaired_plan = json.dumps({
        "claim": "change scalar multiplicity",
        "scope": ["project"],
        "abstract_goals": ["test capacity"],
        "evidence_refs": [],
        "uncertainty": "high",
        "risk": "capacity",
    })
    patch = json.dumps({
        "patch_version": "1.0",
        "parent_architecture_id": parent_id,
        "language_version": "1.0.0",
        "hypothesis": {"claim": "change scalar multiplicity"},
        "scope": ["project"],
        "edits": [{"kind": "change_attrs", "target": "project", "payload": {"attrs": {"out_irreps": "3x0e"}}}],
        "preconditions": [],
        "postconditions": [],
        "expected_effects": {"accuracy": "hypothesis_only"},
    })
    ensemble = FakeEnsemble(["# Markdown plan", repaired_plan, patch])
    result = asyncio.run(DSLGenerationEngine(compiler, task, vocabulary, store, model_name="fake", repair_attempts=1).generate(ensemble, parent, (), ("project",)))
    assert result.planner_repair_count == 1
    import sqlite3
    with sqlite3.connect(str(tmp_path / "planner_repair.sqlite")) as connection:
        roles = [row[0] for row in connection.execute("SELECT role FROM prompt_runs ORDER BY created_at")]
    assert roles == ["planner", "planner_repairer", "synthesizer"]


def test_generation_rejects_repairer_scope_expansion(tmp_path):
    import pytest

    group = GroupSpec.o3()
    scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0e", "O3"))
    graph_scalar = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0e", "O3"))
    parent = ArchitectureProgram(
        "1.0.0",
        "task",
        (InputPort("x", scalar),),
        (
            Node("project", "core.identity", {"x": ("input:x",)}),
            Node("pool", "core.global_pool", {"x": ("project",)}),
        ),
        (OutputPort("prediction", "pool", graph_scalar),),
    )
    task = TaskContract("task", group, graph_scalar, "protocol", ResourceContract(1000, 1000000, 2.0))
    primitives = core_registry()
    motifs = reference_motif_registry()
    compiler = Compiler(primitives, motifs)
    language = LanguageVersion("1.0.0", "", primitives.names(), motifs.names(), "now", {})
    vocabulary = select_active_vocabulary(language, group, primitives, motifs)
    store = EvidenceStore(str(tmp_path / "scope.sqlite"))
    parent_id = compiler.analyze(parent, task).architecture_id
    plan = json.dumps({
        "claim": "change projection",
        "scope": ["project"],
        "abstract_goals": ["test"],
        "evidence_refs": [],
        "uncertainty": "high",
        "risk": "invalid",
    })
    expanded = json.dumps({
        "patch_version": "1.0",
        "parent_architecture_id": parent_id,
        "language_version": "1.0.0",
        "hypothesis": {},
        "scope": ["project", "pool"],
        "edits": [{"kind": "change_attrs", "target": "project", "payload": {"attrs": {}}}],
    })
    ensemble = FakeEnsemble([plan, expanded])
    engine = DSLGenerationEngine(compiler, task, vocabulary, store, model_name="fake", repair_attempts=0)
    with pytest.raises(DSLValidationError) as error:
        asyncio.run(engine.generate(ensemble, parent, (), ("project", "pool")))
    assert error.value.diagnostics[0].code == "E_LLM_010"


def test_repairer_receives_trusted_completion_for_an_output_type_error(tmp_path):
    group = GroupSpec.o3()
    scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0e", "O3"))
    graph_scalar = scalar.with_carrier(Carrier.GRAPH)
    parent = ArchitectureProgram(
        "1.0.0",
        "task",
        (InputPort("x", scalar),),
        (
            Node("project", "core.identity", {"x": ("input:x",)}),
            Node("pool", "core.global_pool", {"x": ("project",)}),
        ),
        (OutputPort("prediction", "pool", graph_scalar),),
    )
    task = TaskContract("task", group, graph_scalar, "protocol", ResourceContract(1000, 1000000, 2.0))
    primitives = core_registry()
    motifs = reference_motif_registry()
    compiler = Compiler(primitives, motifs)
    language = LanguageVersion("1.0.0", "", primitives.names(), motifs.names(), "now", {})
    vocabulary = select_active_vocabulary(language, group, primitives, motifs)
    store = EvidenceStore(str(tmp_path / "completion_repair.sqlite"))
    parent_id = compiler.analyze(parent, task).architecture_id
    plan = json.dumps({
        "claim": "replace the graph readout path",
        "scope": ["pool", "output:prediction"],
        "abstract_goals": ["preserve graph scalar output"],
        "evidence_refs": [],
        "uncertainty": "high",
        "risk": "carrier mismatch",
    })
    invalid = json.dumps({
        "patch_version": "1.0",
        "parent_architecture_id": parent_id,
        "language_version": "1.0.0",
        "hypothesis": {"claim": "replace the graph readout path"},
        "scope": ["pool", "output:prediction"],
        "edits": [
            {
                "kind": "insert_after",
                "target": "pool",
                "payload": {
                    "node": {
                        "id": "wrong_projection",
                        "op": "core.irrep_linear",
                        "inputs": {"x": ["pool"]},
                        "attrs": {"out_irreps": "2x0e"},
                    }
                },
            },
            {"kind": "rewire_output", "target": "output:prediction", "payload": {"reference": "wrong_projection"}},
        ],
    })
    repaired = json.dumps({
        "patch_version": "1.0",
        "parent_architecture_id": parent_id,
        "language_version": "1.0.0",
        "hypothesis": {"claim": "replace the graph readout path"},
        "scope": ["pool", "output:prediction"],
        "edits": [
            {
                "kind": "insert_after",
                "target": "pool",
                "payload": {
                    "node": {
                        "id": "final_projection",
                        "op": "core.irrep_linear",
                        "inputs": {"x": ["pool"]},
                        "attrs": {"out_irreps": "1x0e"},
                    }
                },
            },
            {"kind": "rewire_output", "target": "output:prediction", "payload": {"reference": "final_projection"}},
        ],
    })
    ensemble = FakeEnsemble([plan, invalid, repaired])
    result = asyncio.run(
        DSLGenerationEngine(compiler, task, vocabulary, store, model_name="fake", repair_attempts=1).generate(
            ensemble,
            parent,
            (),
            ("pool", "output:prediction"),
        )
    )
    repair_payload = json.loads(ensemble.calls[2]["messages"][0]["content"])
    suggestions = repair_payload["trusted_completion_suggestions"]
    assert result.repair_count == 1
    assert len(suggestions) == 1
    assert suggestions[0]["diagnostic_code"] == "E_OUTPUT_001"
    assert suggestions[0]["scope_compatible"] is True
    assert suggestions[0]["completion"]["path"][0]["op"] in {
        "core.irrep_linear@1",
        "core.irrep_slice@1",
        "core.change_multiplicity@1",
    }
    import sqlite3
    with sqlite3.connect(str(tmp_path / "completion_repair.sqlite")) as connection:
        stored_prompt = connection.execute(
            "SELECT prompt_json FROM prompt_runs WHERE role='repairer'"
        ).fetchone()[0]
    stored_payload = json.loads(json.loads(stored_prompt)["user"])
    assert stored_payload["trusted_completion_suggestions"] == suggestions
