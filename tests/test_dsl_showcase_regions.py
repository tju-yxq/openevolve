import asyncio
import json
from dataclasses import replace

from equivariant_nas.dsl import (
    Compiler,
    DSLGenerationEngine,
    EvidenceStore,
    LanguageVersion,
    Node,
    ResourceContract,
    TaskContract,
    core_registry,
    import_equiformer_v1,
    reference_motif_registry,
    select_active_vocabulary,
    validate_region_transition,
    v1_region_registry,
)
from equivariant_nas.spec import baseline_spec


class FakeEnsemble:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate_with_context(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _hybrid(parent, auxiliary="block3"):
    pool = parent.nodes[-1]
    changed = replace(
        pool,
        op="motif.v1_multilevel_readout",
        inputs={"terminal": ("scalar_readout",), "aux": (auxiliary,)},
        attrs={},
    )
    return replace(parent, nodes=parent.nodes[:-1] + (changed,))


def _objects(tmp_path):
    parent = import_equiformer_v1(baseline_spec())
    primitives = core_registry()
    motifs = reference_motif_registry()
    compiler = Compiler(primitives, motifs)
    task = TaskContract(
        "qm9_alpha",
        parent.outputs[0].expected_type.group,
        parent.outputs[0].expected_type,
        "quarter8k-validation-only",
        ResourceContract(10_000_000, 80_000_000_000, 8.0),
    )
    language = LanguageVersion("1.0.0", "", primitives.names(), motifs.names(), "now", {})
    vocabulary = select_active_vocabulary(language, task.group, primitives, motifs)
    store = EvidenceStore(str(tmp_path / "evidence.sqlite"))
    return parent, compiler, task, vocabulary, store


def test_lowering_plan_distinguishes_reference_hybrid_and_untrusted_v1(tmp_path):
    parent, compiler, task, _vocabulary, _store = _objects(tmp_path)
    assert compiler.plan_lowering(parent, task).mode == "exact_reference"
    child = _hybrid(parent)
    plan = compiler.plan_lowering(child, task)
    assert plan.mode == "exact_hybrid"
    assert plan.details["auxiliary_block"] == 3

    changed_block = Node(
        "block3",
        "core.identity",
        {"x": ("block2",)},
        declared_types=parent.nodes[3].declared_types,
    )
    untrusted = replace(parent, nodes=parent.nodes[:3] + (changed_block,) + parent.nodes[4:])
    assert compiler.plan_lowering(untrusted, task).mode == "representation_only"


def test_region_audit_hashes_the_frozen_v1_complement(tmp_path):
    parent, _compiler, _task, _vocabulary, _store = _objects(tmp_path)
    region = next(item for item in v1_region_registry(parent) if item.region_id == "v1_readout")
    audit = validate_region_transition(parent, _hybrid(parent, "block2"), region)
    assert audit["region_id"] == "v1_readout"
    assert len(audit["frozen_complement_hash"]) == 64

    changed_block = replace(parent.nodes[1], annotations={"tampered": True})
    tampered = replace(_hybrid(parent), nodes=(parent.nodes[0], changed_block) + _hybrid(parent).nodes[2:])
    import pytest

    with pytest.raises(Exception):
        validate_region_transition(parent, tampered, region)


def test_fixed_router_critic_synthesizer_flow_produces_certified_hybrid(tmp_path):
    parent, compiler, task, vocabulary, store = _objects(tmp_path)
    parent_id = compiler.analyze(parent, task).architecture_id
    router = json.dumps({
        "factor_id": "F6.3",
        "region_id": "v1_readout",
        "rationale": "the readout is the lowest-risk exact hybrid boundary",
        "evidence_refs": [],
        "expected_value": "test whether an intermediate invariant improves validation MAE",
        "risk": "the auxiliary feature may add no useful signal",
    })
    critic = json.dumps({
        "factor_id": "F6.3",
        "region_id": "v1_readout",
        "claim": "adding one block3 invariant readout can improve endpoint validation MAE",
        "mechanism": "block3 can retain complementary local-environment information",
        "edit_plan": ["replace graph_pool with the certified multi-level readout motif", "tap block3"],
        "preserved_invariants": ["all attention blocks are frozen", "output remains one graph scalar"],
        "evidence_refs": [],
        "uncertainty": "single-seed short training may be noisy",
        "risk": "extra readout capacity may overfit",
        "acceptance_metrics": ["endpoint validation MAE", "equivariance error", "frozen complement hash"],
    })
    replacement = replace(
        parent.nodes[-1],
        op="motif.v1_multilevel_readout",
        inputs={"terminal": ("scalar_readout",), "aux": ("block3",)},
        attrs={},
    )
    patch = json.dumps({
        "patch_version": "1.0",
        "parent_architecture_id": parent_id,
        "language_version": parent.language_version,
        "hypothesis": {"claim": "add a block3 invariant readout"},
        "scope": ["graph_pool", "output:prediction"],
        "edits": [{"kind": "replace_node", "target": "graph_pool", "payload": {"node": replacement.to_dict()}}],
        "preconditions": [{"kind": "node_op_is", "node_id": "graph_pool", "op": "core.global_pool"}],
        "postconditions": [{"kind": "node_op_is", "node_id": "graph_pool", "op": "motif.v1_multilevel_readout"}],
        "expected_effects": {"accuracy": "hypothesis_only"},
    })
    ensemble = FakeEnsemble([router, critic, patch])
    engine = DSLGenerationEngine(compiler, task, vocabulary, store, model_name="fake", repair_attempts=0)
    result = asyncio.run(
        engine.generate_region_candidate(ensemble, parent, (), v1_region_registry(parent))
    )
    assert len(ensemble.calls) == 3
    assert result.region_audit["region_id"] == "v1_readout"
    assert result.region_audit["lowering_plan"]["mode"] == "exact_hybrid"
    assert result.router_response["region_id"] == "v1_readout"
    assert result.critic_response["region_id"] == "v1_readout"


def test_region_critic_schema_failure_is_repaired_without_changing_factor(tmp_path):
    parent, compiler, task, vocabulary, store = _objects(tmp_path)
    parent_id = compiler.analyze(parent, task).architecture_id
    router = json.dumps({
        "factor_id": "F6.3",
        "region_id": "v1_readout",
        "rationale": "readout intervention",
        "evidence_refs": [],
        "expected_value": "measure a local readout change",
        "risk": "no improvement",
    })
    malformed_critic = json.dumps({
        "factor_id": "F6.3",
        "region_id": "v1_readout",
        "claim": "tap block3",
        "mechanism": "intermediate invariants may help",
        "edit_plan": ["replace readout"],
        "preserved_invariants": ["keep output scalar"],
        "evidence_refs": [],
        "uncertainty": "high",
        "risk": "overfit",
        "acceptance_metrics": ["validation MAE"],
        "commentary": "extra key",
    })
    repaired_critic = json.dumps({
        "factor_id": "F6.3",
        "region_id": "v1_readout",
        "claim": "tap block3",
        "mechanism": "intermediate invariants may help",
        "edit_plan": ["replace readout"],
        "preserved_invariants": ["keep output scalar"],
        "evidence_refs": [],
        "uncertainty": "high",
        "risk": "overfit",
        "acceptance_metrics": ["validation MAE"],
    })
    replacement = replace(
        parent.nodes[-1],
        op="motif.v1_multilevel_readout",
        inputs={"terminal": ("scalar_readout",), "aux": ("block3",)},
        attrs={},
    )
    patch = json.dumps({
        "patch_version": "1.0",
        "parent_architecture_id": parent_id,
        "language_version": parent.language_version,
        "hypothesis": {"factor_id": "F6.3", "claim": "tap block3"},
        "scope": ["graph_pool", "output:prediction"],
        "edits": [{"kind": "replace_node", "target": "graph_pool", "payload": {"node": replacement.to_dict()}}],
        "preconditions": [],
        "postconditions": [],
        "expected_effects": {"accuracy": "hypothesis_only"},
    })
    ensemble = FakeEnsemble([router, malformed_critic, repaired_critic, patch])
    engine = DSLGenerationEngine(compiler, task, vocabulary, store, model_name="fake", repair_attempts=1)
    result = asyncio.run(engine.generate_region_candidate(ensemble, parent, (), v1_region_registry(parent)))
    assert result.critic_repair_count == 1
    assert result.critic_response["factor_id"] == "F6.3"
    repair_payload = json.loads(ensemble.calls[2]["messages"][0]["content"])
    assert repair_payload["immutable_factor_id"] == "F6.3"
    assert repair_payload["immutable_region_id"] == "v1_readout"

    import sqlite3
    with sqlite3.connect(str(tmp_path / "evidence.sqlite")) as connection:
        roles = [row[0] for row in connection.execute("SELECT role FROM prompt_runs ORDER BY created_at")]
    assert roles == ["factor_router", "region_critic", "region_critic_repairer", "patch_synthesizer"]


def test_forced_factor_coverage_repairs_an_already_evaluated_constructor_option(tmp_path):
    parent, compiler, task, vocabulary, store = _objects(tmp_path)
    parent_id = compiler.analyze(parent, task).architecture_id
    critic = json.dumps({
        "factor_id": "F4.4",
        "region_id": "v1_attention_heads",
        "claim": "measure the remaining certified attention-head organization",
        "mechanism": "head count changes invariant attention organization without changing irreps",
        "edit_plan": ["select one capability-admitted num_heads value not already evaluated"],
        "preserved_invariants": ["all other constructor fields remain frozen"],
        "evidence_refs": [],
        "uncertainty": "the remaining option may not improve validation MAE",
        "risk": "attention partitioning may reduce optimization quality",
        "acceptance_metrics": ["endpoint validation MAE", "equivariance error"],
    })

    def constructor_patch(value):
        return json.dumps({
            "patch_version": "1.0",
            "parent_architecture_id": parent_id,
            "language_version": parent.language_version,
            "hypothesis": {"factor_id": "F4.4", "claim": "measure unused head count"},
            "scope": ["constructor.operator.num_heads"],
            "edits": [{
                "kind": "change_parameters",
                "target": "constructor.operator.num_heads",
                "payload": {"value": value},
            }],
            "preconditions": [],
            "postconditions": [],
            "expected_effects": {"validation_alpha_mae": "hypothesis_only"},
        })

    ensemble = FakeEnsemble([critic, constructor_patch(2), constructor_patch(8)])
    result = asyncio.run(
        DSLGenerationEngine(compiler, task, vocabulary, store, model_name="fake", repair_attempts=1).generate_region_candidate(
            ensemble,
            parent,
            (),
            v1_region_registry(parent),
            forced_factor_id="F4.4",
            excluded_patch_signatures=[{
                "architecture_id": "previous-heads-2",
                "edits": [{"target": "constructor.operator.num_heads", "value": 2}],
            }],
        )
    )
    assert result.repair_count == 1
    assert result.patch.edits[0].payload == {"value": 8}
    assert result.planner_response["excluded_patch_signatures"][0]["architecture_id"] == "previous-heads-2"
    repair_payload = json.loads(ensemble.calls[2]["messages"][0]["content"])
    assert repair_payload["diagnostics"][0]["code"] == "E_SEARCH_002"
    assert repair_payload["failed_patch"]["edits"][0]["payload"] == {"value": 2}
