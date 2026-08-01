from copy import deepcopy
from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
pytest.importorskip("e3nn")
from e3nn import o3

from equivariant_nas.dsl import (
    ArchitectureProgram,
    AxisSpec,
    Carrier,
    Compiler,
    DSLValidationError,
    EquivariantTensorType,
    FeatureRole,
    GroupSpec,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    RepresentationLayout,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend


PATH_BLOCKS = (
    {"multiplicity": 4, "irrep": "0e"},
    {"multiplicity": 2, "irrep": "0e"},
    {"multiplicity": 1, "irrep": "0e"},
    {"multiplicity": 4, "irrep": "1e"},
    {"multiplicity": 2, "irrep": "1e"},
    {"multiplicity": 2, "irrep": "1e"},
    {"multiplicity": 2, "irrep": "1e"},
    {"multiplicity": 1, "irrep": "1e"},
    {"multiplicity": 1, "irrep": "1e"},
    {"multiplicity": 4, "irrep": "2e"},
    {"multiplicity": 2, "irrep": "2e"},
    {"multiplicity": 2, "irrep": "2e"},
    {"multiplicity": 1, "irrep": "2e"},
    {"multiplicity": 1, "irrep": "2e"},
    {"multiplicity": 1, "irrep": "2e"},
)


INSTRUCTION_INDICES = (
    (0, 0, 0),
    (0, 1, 3),
    (0, 2, 9),
    (1, 0, 4),
    (1, 1, 1),
    (1, 1, 5),
    (1, 1, 10),
    (1, 2, 6),
    (1, 2, 11),
    (2, 0, 12),
    (2, 1, 7),
    (2, 1, 13),
    (2, 2, 2),
    (2, 2, 8),
    (2, 2, 14),
)


def _attrs():
    return {
        "path_blocks": list(deepcopy(PATH_BLOCKS)),
        "instructions": [
            {
                "left": left,
                "right": right,
                "out": output,
                "mode": "uvu",
                "has_weight": True,
                "path_weight": 1.0,
            }
            for left, right, output in INSTRUCTION_INDICES
        ],
    }


def _types(*, weight_numel=30, left_layout=None):
    group = GroupSpec.o3()
    left = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("4x0e+2x1e+1x2e", group.family),
        dtype="float64",
        measure="dimensionless",
        layout=left_layout or RepresentationLayout(),
    )
    right = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x0e+1x1e+1x2e", group.family),
        dtype="float64",
        measure="dimensionless",
    )
    weight = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("{}x0e".format(weight_numel), group.family),
        axes=("tp_path",),
        axis_specs=(AxisSpec("tp_path", weight_numel, FeatureRole.TP_PATH, "independent", 0),),
        dtype="float64",
        measure="dimensionless",
        feature_role=FeatureRole.RADIAL_WEIGHT,
    )
    output = replace(left, irreps=Irreps.parse("7x0e+12x1e+11x2e", group.family))
    return left, right, weight, output


def _program(*, attrs=None, weight_numel=30, left_layout=None):
    left, right, weight, output = _types(weight_numel=weight_numel, left_layout=left_layout)
    return ArchitectureProgram(
        "2.6.0",
        "official-v1-depthwise-tensor-product",
        (InputPort("left", left), InputPort("right", right), InputPort("weight", weight)),
        (
            Node(
                "tp",
                "core.tensor_product@3",
                {"left": ("input:left",), "right": ("input:right",), "weight": ("input:weight",)},
                attrs or _attrs(),
            ),
        ),
        (OutputPort("out", "tp", output),),
    )


def _direct_tensor_product():
    path_irreps = o3.Irreps(
        "+".join("{}x{}".format(block["multiplicity"], block["irrep"]) for block in PATH_BLOCKS)
    )
    instructions = [
        (left, right, output, "uvu", True, 1.0)
        for left, right, output in INSTRUCTION_INDICES
    ]
    return o3.TensorProduct(
        o3.Irreps("4x0e+2x1e+1x2e"),
        o3.Irreps("1x0e+1x1e+1x2e"),
        path_irreps,
        instructions,
        normalization=None,
        internal_weights=False,
        shared_weights=False,
        path_normalization="none",
    ).double()


def _assert_rejected(attrs, code, **program_kwargs):
    with pytest.raises(DSLValidationError) as error:
        Compiler(core_registry()).analyze(_program(attrs=attrs, **program_kwargs))
    assert any(item.code == code for item in error.value.diagnostics)


def test_tensor_product_v3_freezes_official_v1_uvu_path_and_external_parameter_contract():
    registry = core_registry()
    artifact = Compiler(registry).analyze(_program())
    output = artifact.inference.value_types["tp"]
    contract = artifact.inference.parameter_contracts["tp"][0]
    path_obligation = next(
        item for item in artifact.inference.obligations if item.kind.value == "IRREP_PATH_EXISTS"
    )

    assert str(output.irreps) == "7x0e+12x1e+11x2e"
    assert output.irreps.dimension == 98
    assert contract.storage == "external"
    assert contract.external_port == "weight"
    assert contract.shape == (30,)
    assert contract.trainable is False
    assert path_obligation.details == {
        "instruction_count": 15,
        "weight_numel": 30,
        "connection_mode": "uvu",
        "path_block_irreps": (
            "4x0e+2x0e+1x0e+4x1e+2x1e+2x1e+2x1e+1x1e+1x1e+"
            "4x2e+2x2e+2x2e+1x2e+1x2e+1x2e"
        ),
    }


def test_tensor_product_v3_lowering_matches_direct_e3nn_forward_and_all_three_gradients():
    registry = core_registry()
    program = _program()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()
    module = model.node_modules["tp"]
    direct = _direct_tensor_product()

    assert str(module.irreps_out) == str(direct.irreps_out)
    assert str(module.irreps_out) != str(module.irreps_out.simplify())
    assert int(module.weight_numel) == 30
    assert dict(module.named_parameters()) == {}
    assert module.internal_weights is False
    assert module.shared_weights is False

    left = torch.randn(9, 15, dtype=torch.float64, requires_grad=True)
    right = torch.randn(9, 9, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(9, 30, dtype=torch.float64, requires_grad=True)
    direct_left = left.detach().clone().requires_grad_(True)
    direct_right = right.detach().clone().requires_grad_(True)
    direct_weight = weight.detach().clone().requires_grad_(True)

    lowered_output = model({"left": left, "right": right, "weight": weight}, {})["out"]
    direct_output = direct(direct_left, direct_right, direct_weight)
    assert torch.allclose(lowered_output, direct_output, atol=1e-12, rtol=1e-12)

    probe = torch.randn_like(lowered_output)
    lowered_gradients = torch.autograd.grad((lowered_output * probe).sum(), (left, right, weight))
    direct_gradients = torch.autograd.grad(
        (direct_output * probe).sum(),
        (direct_left, direct_right, direct_weight),
    )
    for lowered, expected in zip(lowered_gradients, direct_gradients):
        assert torch.isfinite(lowered).all()
        assert torch.allclose(lowered, expected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("matrix_kind", ("rotation", "inversion"))
def test_tensor_product_v3_is_o3_equivariant(matrix_kind):
    registry = core_registry()
    program = _program()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()
    left = torch.randn(8, 15, dtype=torch.float64)
    right = torch.randn(8, 9, dtype=torch.float64)
    weight = torch.randn(8, 30, dtype=torch.float64)
    output = model({"left": left, "right": right, "weight": weight}, {})["out"]

    matrix = o3.rand_matrix(dtype=torch.float64) if matrix_kind == "rotation" else -torch.eye(3, dtype=torch.float64)
    left_action = o3.Irreps("4x0e+2x1e+1x2e").D_from_matrix(matrix)
    right_action = o3.Irreps("1x0e+1x1e+1x2e").D_from_matrix(matrix)
    output_action = o3.Irreps("7x0e+12x1e+11x2e").D_from_matrix(matrix)
    transformed = model(
        {
            "left": left @ left_action.transpose(0, 1),
            "right": right @ right_action.transpose(0, 1),
            "weight": weight,
        },
        {},
    )["out"]
    expected = output @ output_action.transpose(0, 1)
    relative_error = (transformed - expected).norm() / expected.norm().clamp_min(1e-12)
    assert float(relative_error) < 1e-7


def test_tensor_product_v3_is_edge_permutation_safe():
    registry = core_registry()
    program = _program()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()
    values = {
        "left": torch.randn(8, 15, dtype=torch.float64),
        "right": torch.randn(8, 9, dtype=torch.float64),
        "weight": torch.randn(8, 30, dtype=torch.float64),
    }
    reference = model(values, {})["out"]
    permutation = torch.tensor([7, 2, 0, 5, 3, 1, 6, 4], dtype=torch.long)
    permuted = model(
        {name: value.index_select(0, permutation) for name, value in values.items()},
        {},
    )["out"]
    assert torch.allclose(permuted, reference.index_select(0, permutation), atol=1e-12, rtol=1e-12)


def test_tensor_product_v3_rejects_invalid_index_and_clebsch_gordan_path():
    invalid_index = _attrs()
    invalid_index["instructions"][0]["left"] = 9
    _assert_rejected(invalid_index, "E_TP_V3_010")

    invalid_path = _attrs()
    invalid_path["instructions"][0]["out"] = 3
    _assert_rejected(invalid_path, "E_TP_V3_014")


def test_tensor_product_v3_rejects_uvu_multiplicity_mode_and_weight_size_errors():
    invalid_multiplicity = _attrs()
    invalid_multiplicity["path_blocks"][0]["multiplicity"] = 3
    _assert_rejected(invalid_multiplicity, "E_TP_V3_015")

    invalid_mode = _attrs()
    invalid_mode["instructions"][0]["mode"] = "uvw"
    _assert_rejected(invalid_mode, "E_TP_V3_011")

    _assert_rejected(_attrs(), "E_TP_V3_023", weight_numel=29)


def test_tensor_product_v3_rejects_unreferenced_noncanonical_and_non_irrep_major_paths():
    unreferenced = _attrs()
    del unreferenced["instructions"][-1]
    _assert_rejected(unreferenced, "E_TP_V3_016")

    noncanonical = _attrs()
    noncanonical["path_blocks"][2], noncanonical["path_blocks"][3] = (
        noncanonical["path_blocks"][3],
        noncanonical["path_blocks"][2],
    )
    _assert_rejected(noncanonical, "E_TP_V3_006")

    invalid_layout = RepresentationLayout(storage="grid", truncation_state="grid_projected")
    _assert_rejected(_attrs(), "E_TP_V3_019", left_layout=invalid_layout)
