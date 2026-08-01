import json
import sqlite3

import pytest

from equivariant_nas.dsl import (
    ArchitectureProgram,
    CandidateLineageEvidence,
    Carrier,
    Compiler,
    DSLValidationError,
    EquivariantType,
    EvidenceStore,
    GroupSpec,
    InputPort,
    Irreps,
    LanguageEvolutionBoundary,
    LanguageEvolutionPreregistration,
    LanguageVersion,
    MotifDiscoveryPolicy,
    MotifRegistry,
    Node,
    OutputPort,
    ResourceContract,
    TaskContract,
    core_registry,
    discover_motif_proposals,
    replay_motif_proposal,
    run_language_evolution_boundary,
)
from equivariant_nas.dsl.serialization import save_program, save_task_contract
from scripts.run_dsl_language_evolution import get_parser as get_language_parser, run as run_language_entry


def _objects(attribute_name="activation", values=("silu", "relu", "gelu")):
    group = GroupSpec.so3()
    node_type = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0", "SO3"))
    graph_type = node_type.with_carrier(Carrier.GRAPH)
    task = TaskContract(
        "motif-task",
        group,
        graph_type,
        "fixed-protocol",
        ResourceContract(1000000, 1000000000, 2.0),
    )
    primitives = core_registry()
    motifs = MotifRegistry()
    language = LanguageVersion.from_registries("1.0.0", "", primitives, motifs, "frozen", {})
    compiler = Compiler(primitives, motifs)
    candidates = []
    for index, value in enumerate(values):
        activation_attrs = {attribute_name: value}
        program = ArchitectureProgram(
            language.version,
            task.task_id,
            (InputPort("x", node_type),),
            (
                Node("act", "core.scalar_activation", {"x": ("input:x",)}, activation_attrs),
                Node("norm", "core.equivariant_norm", {"x": ("act",)}),
                Node("pool", "core.global_pool", {"x": ("norm",)}),
            ),
            (OutputPort("prediction", "pool", graph_type),),
        )
        artifact = compiler.analyze(program, task)
        candidates.append(
            CandidateLineageEvidence(
                artifact,
                "lineage-{}".format(index),
                task.task_id,
                ("train", "validation"),
                language.registry_hash(),
                artifact.rewrite_registry_hash,
                discovery_partition="heldout_replay" if index == len(values) - 1 else "support",
            )
        )
    return task, primitives, motifs, language, tuple(candidates)


def test_typed_discovery_anti_unifies_only_registered_safe_attributes_and_replays():
    _, primitives, motifs, _, candidates = _objects()
    report = discover_motif_proposals(candidates, primitives)
    learned = next(
        item for item in report.proposals
        if item.motif.required_attrs and len(item.motif.template_nodes) == 2
    )
    assert learned.motif.required_attrs == ("n0_activation",)
    assert learned.independent_lineages == ("lineage-0", "lineage-1")
    replay = replay_motif_proposal(learned, candidates, primitives, motifs)
    assert all(item.passed for item in replay)
    assert any(item.discovery_partition == "heldout_replay" for item in replay)


def test_undeclared_attribute_difference_is_not_promoted_into_the_language():
    with pytest.raises(DSLValidationError) as error:
        _objects("implementation_hint", ("a", "b", "c"))
    assert any(item.code == "E_ATTR_005" for item in error.value.diagnostics)


def test_description_length_never_double_counts_overlapping_occurrences():
    group = GroupSpec.so3()
    node_type = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0", "SO3"))
    graph_type = node_type.with_carrier(Carrier.GRAPH)
    program = ArchitectureProgram(
        "1.0.0",
        "overlap",
        (InputPort("x", node_type),),
        (
            Node("n0", "core.equivariant_norm", {"x": ("input:x",)}),
            Node("n1", "core.equivariant_norm", {"x": ("n0",)}),
            Node("n2", "core.equivariant_norm", {"x": ("n1",)}),
            Node("n3", "core.equivariant_norm", {"x": ("n2",)}),
            Node("pool", "core.global_pool", {"x": ("n3",)}),
        ),
        (OutputPort("prediction", "pool", graph_type),),
    )
    primitives = core_registry()
    artifact = Compiler(primitives, MotifRegistry()).analyze(program)
    candidate = CandidateLineageEvidence(
        artifact,
        "one-lineage",
        "overlap",
        ("validation",),
        "language-hash",
        artifact.rewrite_registry_hash,
    )
    report = discover_motif_proposals((candidate,), primitives)
    repeated_pair = next(
        item for item in report.proposals
        if len(item.motif.template_nodes) == 2
        and all(node.op == "core.equivariant_norm@1" for node in item.motif.template_nodes)
    )
    seen = set()
    for occurrence in repeated_pair.occurrences:
        assert not seen.intersection(occurrence.node_ids)
        seen.update(occurrence.node_ids)


def test_test_visible_candidate_is_rejected_before_language_discovery():
    _, _, _, _, candidates = _objects()
    source = candidates[0]
    with pytest.raises(DSLValidationError) as error:
        CandidateLineageEvidence(
            source.artifact,
            source.lineage_id,
            source.task_id,
            ("validation", "test"),
            source.language_registry_hash,
            source.rewrite_registry_hash,
        )
    assert error.value.diagnostics[0].code == "E_DISCOVERY_006"


def test_closed_language_boundary_publishes_one_replayed_motif_and_persists_evidence(tmp_path):
    task, primitives, motifs, language, candidates = _objects()
    preview = discover_motif_proposals(candidates, primitives)
    assert preview.proposals
    deltas = {item.proposal_id: 0.05 for item in preview.proposals}
    experiment_ids = {item.proposal_id: ("matched-generation:{}".format(item.proposal_id),) for item in preview.proposals}
    policy = MotifDiscoveryPolicy()
    preregistration = LanguageEvolutionPreregistration(
        "cycle-1-close",
        language.version,
        1,
        "all compiled and validation-only candidates in cycle 1",
        "two support lineages plus one heldout replay lineage",
        policy.content_hash(),
        "before-cycle",
    )
    boundary = preregistration.close(tuple(item.architecture_id for item in candidates), "after-cycle")
    store = EvidenceStore(str(tmp_path / "language.sqlite"))
    store.register_language(language)
    for candidate in candidates:
        store.add_compiled_candidate(candidate.artifact, task)
    result = run_language_evolution_boundary(
        candidates,
        language,
        primitives,
        motifs,
        boundary,
        new_version="1.1.0",
        frozen_at="after-admission",
        discovery_policy=policy,
        regression_passed=True,
        regression_artifact_ids=("pytest:dsl-regression",),
        valid_generation_rate_deltas=deltas,
        generation_experiment_ids=experiment_ids,
        evidence_store=store,
    )
    assert result.published
    assert result.child_language.parent_version == language.version
    assert result.selected_proposal_id
    assert len(result.child_language.motif_names) == 1
    assert all(record.replay_results for record in result.admissions)
    assert all(all(item.passed for item in record.replay_results) for record in result.admissions)
    with sqlite3.connect(str(tmp_path / "language.sqlite")) as connection:
        occurrence_count = connection.execute("SELECT COUNT(*) FROM motif_occurrences").fetchone()[0]
        proposal_count = connection.execute("SELECT COUNT(*) FROM motif_proposals").fetchone()[0]
        replay_count = connection.execute("SELECT COUNT(*) FROM language_replay_runs").fetchone()[0]
        admission_count = connection.execute("SELECT COUNT(*) FROM motif_admission_runs").fetchone()[0]
        published_count = connection.execute("SELECT COUNT(*) FROM motif_proposals WHERE status='published'").fetchone()[0]
    assert occurrence_count == len(result.discovery.occurrences)
    assert proposal_count == len(result.discovery.proposals)
    assert replay_count == sum(len(item.replay_results) for item in result.admissions)
    assert admission_count == len(result.admissions)
    assert published_count == 1
    restored_language, restored_motifs = store.get_language_snapshot("1.1.0")
    assert restored_language.registry_hash() == result.child_language.registry_hash()
    assert restored_motifs.names() == result.child_language.motif_names
    assert restored_motifs.resolve(restored_motifs.names()[0]).content_hash() == result.child_language.motif_hashes[restored_motifs.names()[0]]


def test_open_boundary_cannot_change_the_frozen_language():
    _, primitives, motifs, language, candidates = _objects()
    policy = MotifDiscoveryPolicy()
    preregistration = LanguageEvolutionPreregistration(
        "cycle-open",
        language.version,
        1,
        "all compiled and validation-only candidates in cycle 1",
        "two support lineages plus one heldout replay lineage",
        policy.content_hash(),
        "before-cycle",
    )
    boundary = preregistration.close(tuple(item.architecture_id for item in candidates), "")
    with pytest.raises(DSLValidationError) as error:
        run_language_evolution_boundary(
            candidates,
            language,
            primitives,
            motifs,
            boundary,
            new_version="1.1.0",
            frozen_at="never",
            discovery_policy=policy,
            regression_passed=True,
            regression_artifact_ids=("pytest",),
        )
    assert error.value.diagnostics[0].code == "E_LANGUAGE_EVOLUTION_009"


def test_language_evolution_entry_consumes_an_openevolve_cycle_snapshot(tmp_path):
    task, primitives, motifs, language, candidates = _objects()
    policy = MotifDiscoveryPolicy()
    preregistration = LanguageEvolutionPreregistration(
        "entry-cycle",
        language.version,
        1,
        "all valid compiled test-hidden programs",
        "fixed support and heldout replay partition",
        policy.content_hash(),
        "before-cycle",
    )
    database_path = tmp_path / "evidence.sqlite"
    store = EvidenceStore(str(database_path))
    store.register_language(language, motifs)
    records = []
    for candidate in candidates:
        store.add_compiled_candidate(candidate.artifact, task)
        program_path = tmp_path / "{}.json".format(candidate.architecture_id)
        save_program(candidate.artifact.source_program, str(program_path))
        records.append({
            "architecture_id": candidate.architecture_id,
            "program_id": candidate.architecture_id,
            "program_path": str(program_path),
            "lineage_id": candidate.lineage_id,
            "task_id": candidate.task_id,
            "visible_splits": list(candidate.visible_splits),
            "test_evaluated": False,
            "discovery_partition": candidate.discovery_partition,
            "language_registry_hash": candidate.language_registry_hash,
            "rewrite_registry_hash": candidate.rewrite_registry_hash,
        })
    snapshot = {
        "boundary_preregistration": dict(
            preregistration.to_dict(),
            preregistration_hash=preregistration.content_hash(),
        ),
        "closed_at": "after-cycle",
        "selection_rule": preregistration.selection_rule,
        "partition_rule": preregistration.partition_rule,
        "candidates": records,
    }
    snapshot_path = tmp_path / "language_cycle_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    task_path = tmp_path / "task.json"
    save_task_contract(task, str(task_path))
    output_path = tmp_path / "discovery.json"
    args = get_language_parser().parse_args([
        "--mode", "discover",
        "--cycle-snapshot", str(snapshot_path),
        "--task-contract", str(task_path),
        "--evidence-store", str(database_path),
        "--output", str(output_path),
    ])
    result = run_language_entry(args)
    assert result["status"] == "discovered"
    assert result["test_evaluated"] is False
    assert result["discovery"]["proposals"]
    assert output_path.exists()
