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
