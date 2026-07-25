import pytest

from equivariant_nas.dsl import DSLValidationError, LanguageVersion, core_registry, reference_motif_registry
from equivariant_nas.dsl.admission import MotifAdmissionEvidence, evaluate_motif_admission, publish_motif_version
from equivariant_nas.dsl.ast import Node
from equivariant_nas.dsl.motifs import MotifDefinition


def new_motif():
    return MotifDefinition(
        "motif.certified_identity_pair",
        1,
        ("x",),
        {"out": "second"},
        (
            Node("first", "core.identity", {"x": ("$input:x",)}),
            Node("second", "core.identity", {"x": ("first",)}),
        ),
        provenance={"discovery": "two independent lineages"},
    )


def test_motif_admission_requires_evidence_not_only_a_definition():
    registry = reference_motif_registry()
    weak = MotifAdmissionEvidence((), (), 0.0, 0.0, False, False, (), {})
    decision = evaluate_motif_admission(new_motif(), weak, registry)
    assert not decision.accepted
    assert len(decision.reasons) >= 4


def test_evidence_gated_motif_publishes_a_new_language_version():
    primitives = core_registry()
    registry = reference_motif_registry()
    parent = LanguageVersion("1.0.0", "", primitives.names(), registry.names(), "old", {})
    evidence = MotifAdmissionEvidence(
        ("lineage-a", "lineage-b"),
        (),
        12.0,
        0.05,
        True,
        True,
        ("proof:1",),
        {},
    )
    child = publish_motif_version(parent, new_motif(), evidence, registry, new_version="1.1.0", frozen_at="new")
    assert child.parent_version == "1.0.0"
    assert "motif.certified_identity_pair@1" in child.motif_names
