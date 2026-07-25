from dataclasses import replace

import pytest

from equivariant_nas.dsl import (
    DSLValidationError,
    TypeChecker,
    core_registry,
    expand_motifs,
    import_equiformer_v1,
    reference_motif_registry,
)
from equivariant_nas.dsl.compiler import Compiler
from equivariant_nas.spec import baseline_spec


def test_v1_reference_program_expands_and_typechecks():
    source = import_equiformer_v1(baseline_spec())
    expanded = expand_motifs(source, reference_motif_registry())
    result = TypeChecker(core_registry()).check(expanded)
    assert len(source.nodes) == baseline_spec().macro.num_layers + 2
    assert len(expanded.nodes) == 4 + 5 * (baseline_spec().macro.num_layers - 1) + 2
    assert not result.open_obligations


def test_legacy_lowering_lock_detects_semantic_ast_change_before_importing_backend():
    source = import_equiformer_v1(baseline_spec())
    changed_node = replace(source.nodes[-2], attrs={"multiplicity": 2})
    changed = replace(source, nodes=source.nodes[:-2] + (changed_node,) + source.nodes[-1:])
    compiler = Compiler(core_registry(), reference_motif_registry())
    with pytest.raises(DSLValidationError) as error:
        compiler.lower_legacy_equiformer_v1(changed, "unused")
    assert error.value.diagnostics[0].code in ("E_TYPE_005", "E_BACKEND_002")


def test_unknown_motif_fails_with_structured_diagnostic():
    source = import_equiformer_v1(baseline_spec())
    changed = replace(source, nodes=(replace(source.nodes[0], op="motif.unknown"),) + source.nodes[1:])
    with pytest.raises(DSLValidationError) as error:
        expand_motifs(changed, reference_motif_registry())
    assert error.value.diagnostics[0].code == "E_MOTIF_005"
