import pytest

from equivariant_nas.dsl import (
    ArchitectureProgram,
    AxisSpec,
    Carrier,
    Compiler,
    DSLValidationError,
    EquivariantTensorType,
    EquivariantType,
    FeatureRole,
    Frame,
    GroupSpec,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    RecordType,
    RepresentationLayout,
    ResourceContract,
    ResolutionSpec,
    TaskContract,
    TupleType,
    VALUE_TYPE_SCHEMA_VERSION,
    architecture_id,
    core_registry,
    import_equiformer_v1,
    import_equiformer_v3,
    legacy_equivariant_view,
    migrate_program_v1_to_v2,
    migrate_task_contract_v1_to_v2,
    reference_motif_registry,
    value_type_from_dict,
)
from equivariant_nas.dsl.serialization import dumps_program, loads_program
from equivariant_nas.dsl.backends.equiformer_v1_spec import ArchitectureSpec
from equivariant_nas.dsl.backends.equiformer_v3_spec import baseline_v3_spec


def _v2_tensor(*, storage="m_primary", mmax=1):
    group = GroupSpec.so3()
    return EquivariantTensorType(
        group=group,
        carrier=Carrier.EDGE,
        irreps=Irreps.parse("4x0+2x1", group.family),
        frame=Frame("edge", "edge-frame-0"),
        axes=("head", "channel"),
        axis_specs=(
            AxisSpec("head", 4, FeatureRole.HEAD, "independent", 0, False),
            AxisSpec("channel", 8, FeatureRole.CHANNEL, "per_head", 1, False),
        ),
        layout=RepresentationLayout(
            storage=storage,
            coefficient_order="degree_then_order",
            resolution_specs=(ResolutionSpec("r0", 2, mmax),),
            truncation_state="m_truncated" if mmax < 2 else "full",
            channel_order=("alpha", "value"),
        ),
    )


def _v2_mechanism_program():
    group = GroupSpec.so3()
    irreps = Irreps.parse("2x0+2x1+2x2", group.family)
    value_type = EquivariantType(group, Carrier.NODE, irreps)
    return ArchitectureProgram(
        "1.0.0",
        "v2-mechanism",
        (InputPort("x", value_type),),
        (
            Node(
                "v2",
                "motif.v2_so2_residual_message",
                {"x": ("input:x",)},
                {"hidden_irreps": str(irreps), "frame_id": "v2-edge"},
            ),
        ),
        (OutputPort("prediction", "v2", value_type),),
    )


def test_equivariant_tensor_v2_roundtrip_preserves_axes_layout_and_schema():
    value = _v2_tensor()
    payload = value.to_dict()
    assert payload["kind"] == "equivariant_tensor"
    assert payload["schema_version"] == VALUE_TYPE_SCHEMA_VERSION
    assert value_type_from_dict(payload) == value
    assert value.with_carrier(Carrier.NODE).axis_specs == value.axis_specs
    assert value.with_carrier(Carrier.NODE).layout == value.layout


def test_axis_and_layout_contracts_reject_ambiguous_or_invalid_specs():
    with pytest.raises(DSLValidationError) as duplicate_order:
        EquivariantTensorType(
            GroupSpec.so3(),
            Carrier.NODE,
            Irreps.parse("2x0", "SO3"),
            axes=("head", "channel"),
            axis_specs=(
                AxisSpec("head", 2, FeatureRole.HEAD, order=0),
                AxisSpec("channel", 4, FeatureRole.CHANNEL, order=0),
            ),
        )
    assert any(item.code == "E_AXIS_007" for item in duplicate_order.value.diagnostics)

    with pytest.raises(DSLValidationError) as invalid_resolution:
        ResolutionSpec("r0", lmax=1, mmax=2)
    assert any(item.code == "E_LAYOUT_002" for item in invalid_resolution.value.diagnostics)


def test_invariant_record_and_tuple_types_are_tagged_and_structural():
    scalar = InvariantTensorType(
        GroupSpec.so3(),
        Carrier.EDGE,
        Irreps.parse("3x0", "SO3"),
        frame=Frame("invariant"),
        axes=("head",),
        axis_specs=(AxisSpec("head", 3, FeatureRole.HEAD),),
        feature_role=FeatureRole.ALPHA,
    )
    record = RecordType({"weights": scalar, "values": _v2_tensor()})
    structured = TupleType((scalar, record))
    assert value_type_from_dict(record.to_dict()) == record
    assert value_type_from_dict(structured.to_dict()) == structured
    assert tuple(record.field_map) == ("values", "weights")

    with pytest.raises(DSLValidationError) as nontrivial:
        InvariantTensorType(
            GroupSpec.so3(),
            Carrier.NODE,
            Irreps.parse("1x1", "SO3"),
        )
    assert any(item.code == "E_TYPE_006" for item in nontrivial.value.diagnostics)


def test_ast_accepts_structured_value_types_without_guessing_group_fields():
    record = RecordType({"out": _v2_tensor(storage="irrep_major", mmax=2)})
    program = ArchitectureProgram(
        "2.0.0",
        "record-task",
        (InputPort("x", record),),
        (),
        (OutputPort("prediction", "input:x", record),),
    )
    assert loads_program(dumps_program(program)) == program
    assert Compiler(core_registry()).analyze(program).inference.value_types["input:x"] == record


def test_v1_program_migrates_deterministically_and_keeps_legacy_tensor_contracts():
    group = GroupSpec.so3()
    node_type = EquivariantType(
        group,
        Carrier.NODE,
        Irreps.parse("2x0", group.family),
        axes=("channel",),
    )
    graph_type = node_type.with_carrier(Carrier.GRAPH)
    program = ArchitectureProgram(
        "1.0.0",
        "migration-task",
        (InputPort("x", node_type),),
        (
            Node("act", "core.scalar_activation", {"x": ("input:x",)}, {"activation": "silu"}),
            Node("pool", "core.global_pool", {"x": ("act",)}),
        ),
        (OutputPort("prediction", "pool", graph_type),),
    )
    registry = core_registry()
    first = migrate_program_v1_to_v2(program, registry)
    second = migrate_program_v1_to_v2(program, registry)

    assert first == second
    assert first.program.language_version == "2.0.0"
    assert isinstance(first.program.inputs[0].value_type, EquivariantTensorType)
    assert legacy_equivariant_view(first.program.inputs[0].value_type) == node_type
    assert first.manifest.source_architecture_id != first.manifest.target_architecture_id
    assert first.manifest.value_type_schema_version == VALUE_TYPE_SCHEMA_VERSION
    assert len(first.manifest.content_hash()) == 64
    assert Compiler(registry).analyze(first.program).architecture_id == first.manifest.target_architecture_id
    assert loads_program(dumps_program(first.program)) == first.program


def test_program_and_task_contract_migrate_as_one_identity_chain():
    group = GroupSpec.so3()
    node_type = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0", group.family))
    graph_type = node_type.with_carrier(Carrier.GRAPH)
    task = TaskContract(
        "migration-task",
        group,
        graph_type,
        "fixed-protocol",
        ResourceContract(1000, 1000000, 2.0),
    )
    program = ArchitectureProgram(
        "1.0.0",
        task.task_id,
        (InputPort("x", node_type),),
        (Node("pool", "core.global_pool", {"x": ("input:x",)}),),
        (OutputPort("prediction", "pool", graph_type),),
    )
    migrated_task = migrate_task_contract_v1_to_v2(task)
    migrated = migrate_program_v1_to_v2(
        program,
        core_registry(),
        source_task_contract_hash=task.content_hash(),
        target_task_contract_hash=migrated_task.content_hash(),
    )

    artifact = Compiler(core_registry()).analyze(migrated.program, migrated_task)
    assert artifact.architecture_id == migrated.manifest.target_architecture_id
    assert isinstance(migrated_task.output_type, EquivariantTensorType)
    assert TaskContract.from_dict(migrated_task.to_dict()) == migrated_task
    assert migrated_task.content_hash() != task.content_hash()


def test_layout_and_schema_version_participate_in_architecture_identity():
    left_type = _v2_tensor(storage="l_primary")
    right_type = _v2_tensor(storage="m_primary")
    left = ArchitectureProgram(
        "2.0.0",
        "layout-task",
        (InputPort("x", left_type),),
        (),
        (OutputPort("prediction", "input:x", left_type),),
    )
    right = ArchitectureProgram(
        "2.0.0",
        "layout-task",
        (InputPort("x", right_type),),
        (),
        (OutputPort("prediction", "input:x", right_type),),
    )
    assert architecture_id(left) != architecture_id(right)
    assert architecture_id(left, value_type_schema_version=None) != architecture_id(left)


def test_tagged_types_reject_unknown_kind_or_missing_schema_identity():
    with pytest.raises(DSLValidationError) as unknown:
        value_type_from_dict({"kind": "mystery", "schema_version": VALUE_TYPE_SCHEMA_VERSION})
    assert any(item.code == "E_SCHEMA_011" for item in unknown.value.diagnostics)

    with pytest.raises(DSLValidationError) as missing_version:
        value_type_from_dict({"kind": "record", "fields": {"x": EquivariantType(GroupSpec.so3(), Carrier.NODE, Irreps.parse("1x0", "SO3")).to_dict()}})
    assert any(item.code == "E_SCHEMA_008" for item in missing_version.value.diagnostics)


@pytest.mark.parametrize(
    "program",
    (
        import_equiformer_v1(ArchitectureSpec()),
        _v2_mechanism_program(),
        import_equiformer_v3(baseline_v3_spec()),
    ),
)
def test_current_v1_and_v3_reference_programs_migrate_and_compile_under_v2_types(program):
    primitives = core_registry()
    motifs = reference_motif_registry()
    migrated = migrate_program_v1_to_v2(program, primitives, motifs=motifs)
    artifact = Compiler(primitives, motifs).analyze(migrated.program)

    assert artifact.architecture_id == migrated.manifest.target_architecture_id
    assert artifact.expanded_program.language_version == "2.0.0"
    assert all(isinstance(value, EquivariantTensorType) for value in artifact.inference.value_types.values())


def test_m2_type_migration_evidence_exports_all_current_reference_flows(tmp_path):
    from scripts.export_dsl_v2_type_migration_evidence import build_evidence

    summary = build_evidence(tmp_path, pytest_summary="test")
    assert summary["reference_flow_count"] == 3
    assert summary["pytest_summary"] == "test"
    assert all(item["all_inferred_values_use_v2_equivariant_tensor_type"] for item in summary["records"])
    assert all(not item["claims"]["official_full_model_reproduction"] for item in summary["records"])
    assert (tmp_path / "summary.json").is_file()
