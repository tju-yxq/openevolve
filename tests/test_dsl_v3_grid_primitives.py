import pytest

from equivariant_nas.dsl import (
    EquivarianceLevel,
    GridSpec,
    GridTensorType,
    GroupSpec,
    Irreps,
    TypeChecker,
    canonicalize,
    core_registry,
    equiformer_v3_feed_forward_program,
    value_type_from_dict,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend
from equivariant_nas.dsl.backends.equiformer_v3_spec import EquiformerV3Spec
from equivariant_nas.dsl.backends.lowering import RuntimeValueKind
from equivariant_nas.dsl.types import Carrier, Frame


def _spec():
    return EquiformerV3Spec(
        use_pbc=False,
        num_channels=3,
        ffn_hidden_channels=4,
        lmax=2,
        mmax=1,
        ffn_grid_resolution=(8, 8),
        ffn_activation="sep-merge_gates2_swiglu",
        use_grid_mlp=True,
        ffn_drop=0.0,
    )


def test_grid_type_roundtrip_retains_sampling_and_aliasing_contract():
    group = GroupSpec.so3()
    grid = GridSpec(8, 9, 2, 2, normalization="component")
    value = GridTensorType(
        group=group,
        carrier=Carrier.NODE,
        source_irreps=Irreps.parse("4x0+4x1+4x2", "SO3"),
        grid=grid,
        channels=4,
        frame=Frame("global"),
        level=EquivarianceLevel.EMPIRICAL,
    )
    restored = value_type_from_dict(value.to_dict())
    assert restored == value
    assert restored.grid == grid


def test_v3_ffn_is_fully_typed_and_uses_explicit_grid_nodes():
    registry = core_registry()
    program = equiformer_v3_feed_forward_program(_spec())
    inference = TypeChecker(registry).check(program)
    canonical = canonicalize(program, registry)
    TypeChecker(registry).check(canonical)

    assert len(program.nodes) == 23
    assert all("s2_gated_swiglu_merge" not in node.op for node in program.nodes)
    assert {node.op for node in program.nodes} >= {
        "core.grid_project@1",
        "core.grid_unproject@1",
        "core.grid_split@1",
        "core.grid_pointwise_product@1",
        "core.grid_channel_linear@1",
    }
    assert inference.value_types["grid_project"].kind == "grid_tensor"
    assert inference.value_types["grid_unproject"].kind == "equivariant_tensor"


def test_v3_ffn_support_report_has_no_unlowered_nodes():
    registry = core_registry()
    program = equiformer_v3_feed_forward_program(_spec())
    report = E3NNGraphBackend(registry).support_report(program)
    assert report.unsupported_nodes == ()
    runtime = dict(report.runtime_kinds)
    assert runtime["grid_project"] == RuntimeValueKind.GRID_TENSOR
    assert runtime["grid_unproject"] == RuntimeValueKind.DENSE_TENSOR


def test_grid_product_rejects_non_grid_values_at_type_boundary():
    registry = core_registry()
    program = equiformer_v3_feed_forward_program(_spec())
    broken = program.__class__(
        language_version=program.language_version,
        task_contract=program.task_contract,
        inputs=program.inputs,
        nodes=tuple(
            node if node.id != "grid_gate_product" else node.__class__(
                node.id,
                node.op,
                {"left": node.inputs["left"], "right": ("scalar_input",)},
                node.attrs,
                node.outputs,
                node.declared_types,
                node.annotations,
            )
            for node in program.nodes
        ),
        outputs=program.outputs,
        parameters=program.parameters,
        program_id=program.program_id,
        annotations=program.annotations,
    )
    with pytest.raises(Exception):
        TypeChecker(registry).check(broken)
