from dataclasses import replace
import ast
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
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
    IndexMapType,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    RepresentationLayout,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend, to_e3nn_irreps
from equivariant_nas.dsl.backends.lowering import RuntimeValueKind


def _equivariant_type(irreps="6x0e+4x1o+2x2e"):
    group = GroupSpec.o3()
    return EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse(irreps, group.family),
        dtype="float64",
    )


def _head_type(source, heads=2):
    return replace(
        source,
        irreps=Irreps(
            tuple((multiplicity // heads, irrep) for multiplicity, irrep in source.irreps)
        ).simplify(),
        axes=("head",),
        axis_specs=(AxisSpec("head", heads, FeatureRole.HEAD, "independent", 0),),
    )


def _official_blockwise_split(value, irreps, heads):
    pieces = []
    offset = 0
    for multiplicity, irrep in irreps:
        width = multiplicity * irrep.dimension
        pieces.append(
            value[..., offset : offset + width].reshape(
                *value.shape[:-1], heads, (multiplicity // heads) * irrep.dimension
            )
        )
        offset += width
    return torch.cat(pieces, dim=-1)


def _official_blockwise_merge(value, per_head_irreps):
    pieces = []
    offset = 0
    heads = value.shape[-2]
    for multiplicity, irrep in per_head_irreps:
        width = multiplicity * irrep.dimension
        pieces.append(
            value[..., :, offset : offset + width].reshape(*value.shape[:-2], heads * width)
        )
        offset += width
    return torch.cat(pieces, dim=-1)


def _load_official_v1_head_classes():
    source_path = Path(__file__).resolve().parents[2] / "equiformer" / "nets" / "graph_attention_transformer.py"
    if not source_path.exists():
        pytest.skip("local official Equiformer V1 source is unavailable")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"Vec2AttnHeads", "AttnHeads2Vec"}
    ]
    if len(selected) != 2:
        raise RuntimeError("official V1 source no longer contains both head conversion classes")
    namespace = {"torch": torch, "o3": o3}
    from e3nn.util.jit import compile_mode

    namespace["compile_mode"] = compile_mode
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["Vec2AttnHeads"], namespace["AttnHeads2Vec"]


def _split_merge_program(source):
    head = _head_type(source)
    return ArchitectureProgram(
        "2.5.0",
        "equivariant-head-roundtrip",
        (InputPort("x", source),),
        (
            Node("split", "core.head_split@2", {"x": ("input:x",)}, {"head_axis": "head", "num_heads": 2}),
            Node("merge", "core.head_merge@2", {"x": ("split",)}, {"head_axis": "head"}),
        ),
        (OutputPort("heads", "split", head), OutputPort("out", "merge", source)),
    )


@pytest.mark.parametrize("matrix_kind", ("rotation", "inversion"))
def test_equivariant_head_split_matches_official_blockwise_layout_and_o3_action(matrix_kind):
    source = _equivariant_type()
    program = _split_merge_program(source)
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()

    value = torch.arange(4 * source.irreps.dimension, dtype=torch.float64).reshape(4, -1)
    value.requires_grad_(True)
    outputs = model({"x": value}, {})
    expected_heads = _official_blockwise_split(value, source.irreps, 2)

    assert torch.equal(outputs["heads"], expected_heads)
    assert torch.equal(outputs["out"], value)
    assert not torch.equal(outputs["heads"], value.reshape(4, 2, -1))
    assert _official_blockwise_merge(outputs["heads"], artifact.inference.value_types["split"].irreps).equal(value)

    matrix = o3.rand_matrix(dtype=torch.float64)
    if matrix_kind == "inversion":
        matrix = -matrix
    input_action = o3.Irreps(to_e3nn_irreps(source.irreps)).D_from_matrix(matrix)
    per_head_type = artifact.inference.value_types["split"]
    head_action = o3.Irreps(to_e3nn_irreps(per_head_type.irreps)).D_from_matrix(matrix)
    transformed = model({"x": value @ input_action.transpose(0, 1)}, {})["heads"]
    expected_transformed = outputs["heads"] @ head_action.transpose(0, 1)
    assert torch.allclose(transformed, expected_transformed, atol=1e-10, rtol=1e-10)

    outputs["out"].square().sum().backward()
    assert torch.equal(value.grad, 2.0 * value.detach())

    report = E3NNGraphBackend(registry).support_report(program)
    runtime_kinds = dict(report.runtime_kinds)
    assert runtime_kinds["split"] == RuntimeValueKind.EQUIVARIANT_HEADS
    assert runtime_kinds["merge"] == RuntimeValueKind.DENSE_TENSOR


def test_equivariant_head_lowering_matches_classes_extracted_from_official_v1_source():
    Vec2AttnHeads, AttnHeads2Vec = _load_official_v1_head_classes()
    source = _equivariant_type()
    program = _split_merge_program(source)
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()
    per_head = artifact.inference.value_types["split"]
    official_split = Vec2AttnHeads(o3.Irreps(to_e3nn_irreps(per_head.irreps)), 2).double()
    official_merge = AttnHeads2Vec(o3.Irreps(to_e3nn_irreps(per_head.irreps))).double()

    dsl_input = torch.randn(7, source.irreps.dimension, dtype=torch.float64, requires_grad=True)
    oracle_input = dsl_input.detach().clone().requires_grad_(True)
    dsl_heads = model({"x": dsl_input}, {})["heads"]
    oracle_heads = official_split(oracle_input)
    assert torch.equal(dsl_heads, oracle_heads)
    assert torch.equal(model({"x": dsl_input}, {})["out"], official_merge(oracle_heads))

    cotangent = torch.randn_like(dsl_heads)
    (dsl_heads * cotangent).sum().backward()
    (oracle_heads * cotangent).sum().backward()
    assert torch.equal(dsl_input.grad, oracle_input.grad)


def test_irrep_select_v2_can_separate_alpha_and_value_ranges_inside_one_scalar_block():
    source = _equivariant_type("6x0e+2x1o")
    head = _head_type(source)
    alpha = replace(head, irreps=Irreps.parse("2x0e", source.group.family))
    value_type = replace(head, irreps=Irreps.parse("1x0e+1x1o", source.group.family))
    program = ArchitectureProgram(
        "2.5.0",
        "head-alpha-value-partition",
        (InputPort("x", source),),
        (
            Node("split", "core.head_split@2", {"x": ("input:x",)}, {"head_axis": "head", "num_heads": 2}),
            Node(
                "alpha",
                "core.irrep_select@2",
                {"x": ("split",)},
                {"selections": [{"irrep": "0e", "start": 0, "multiplicity": 2}]},
            ),
            Node(
                "value",
                "core.irrep_select@2",
                {"x": ("split",)},
                {
                    "selections": [
                        {"irrep": "0e", "start": 2, "multiplicity": 1},
                        {"irrep": "1o", "start": 0, "multiplicity": 1},
                    ]
                },
            ),
        ),
        (OutputPort("alpha", "alpha", alpha), OutputPort("value", "value", value_type)),
    )
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()
    raw = torch.arange(3 * source.irreps.dimension, dtype=torch.float64).reshape(3, -1)
    split = _official_blockwise_split(raw, source.irreps, 2)
    outputs = model({"x": raw}, {})

    assert torch.equal(outputs["alpha"], split[..., :2])
    assert torch.equal(outputs["value"], torch.cat((split[..., 2:3], split[..., 3:6]), dim=-1))
    kinds = dict(E3NNGraphBackend(registry).support_report(program).runtime_kinds)
    assert kinds["alpha"] == RuntimeValueKind.EQUIVARIANT_HEADS
    assert kinds["value"] == RuntimeValueKind.EQUIVARIANT_HEADS


def _attention_program():
    message = _equivariant_type("6x0e+2x1o")
    head = _head_type(message)
    alpha_head = replace(head, irreps=Irreps.parse("2x0e", message.group.family))
    value_head = replace(head, irreps=Irreps.parse("1x0e+1x1o", message.group.family))
    alpha = InvariantTensorType(
        message.group,
        Carrier.EDGE,
        Irreps.parse("2x0e", message.group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", 2, FeatureRole.HEAD, "independent", 0),),
        dtype="float64",
        feature_role=FeatureRole.ALPHA,
    )
    segment = IndexMapType(message.group, Carrier.EDGE, Carrier.NODE, "segment", target_size=3)
    output = replace(
        message,
        carrier=Carrier.NODE,
        irreps=Irreps.parse("2x0e+2x1o", message.group.family),
    )
    nodes = (
        Node("split", "core.head_split@2", {"x": ("input:message",)}, {"head_axis": "head", "num_heads": 2}),
        Node(
            "alpha_value",
            "core.irrep_select@2",
            {"x": ("split",)},
            {"selections": [{"irrep": "0e", "start": 0, "multiplicity": 2}]},
        ),
        Node(
            "value",
            "core.irrep_select@2",
            {"x": ("split",)},
            {
                "selections": [
                    {"irrep": "0e", "start": 2, "multiplicity": 1},
                    {"irrep": "1o", "start": 0, "multiplicity": 1},
                ]
            },
        ),
        Node(
            "activated",
            "core.scalar_activation@1",
            {"x": ("alpha_value",)},
            {
                "activation": "smooth_leaky_relu",
                "negative_slope": 0.2,
                "normalization": "second_moment",
            },
        ),
        Node(
            "logits",
            "core.headwise_scalar_contraction@2",
            {"x": ("activated",)},
            {"head_axis": "head", "bias": False},
        ),
        Node(
            "softmax",
            "core.segment_softmax@2",
            {"logits": ("logits",), "index": ("input:segment_index",)},
            {},
        ),
        Node("weighted", "core.invariant_scale@1", {"weight": ("softmax",), "value": ("value",)}, {}),
        Node(
            "aggregate",
            "core.segment_reduce@1",
            {"x": ("weighted",), "index": ("input:segment_index",)},
            {"reduce": "sum", "normalization": "none"},
        ),
        Node("merge", "core.head_merge@2", {"x": ("aggregate",)}, {"head_axis": "head"}),
    )
    program = ArchitectureProgram(
        "2.5.0",
        "v1-attention-core-minimal",
        (InputPort("message", message), InputPort("segment_index", segment)),
        nodes,
        (OutputPort("out", "merge", output), OutputPort("alpha", "softmax", alpha)),
    )
    return program, message, alpha_head, value_head, output


def _segment_softmax_reference(logits, index):
    output = torch.empty_like(logits)
    for target in torch.unique(index):
        mask = index == target
        output[mask] = torch.softmax(logits[mask], dim=0)
    return output


def test_minimal_v1_attention_alpha_value_core_matches_reference_and_is_equivariant():
    program, message_type, _alpha_head, value_head, output_type = _attention_program()
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()
    contraction = model.node_modules["logits"]
    with torch.no_grad():
        contraction.weight.copy_(torch.tensor([[[0.7, -0.2], [-0.4, 0.9]]], dtype=torch.float64))

    edge_dst = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long)
    segment_payload = {"indices": edge_dst, "target_size": 3}
    message = torch.randn(edge_dst.numel(), message_type.irreps.dimension, dtype=torch.float64, requires_grad=True)
    actual = model({"message": message, "segment_index": segment_payload}, {})

    split = _official_blockwise_split(message, message_type.irreps, 2)
    alpha_value = split[..., :2]
    value = torch.cat((split[..., 2:3], split[..., 3:6]), dim=-1)
    activated = 0.6 * alpha_value + 0.4 * alpha_value * (2.0 * torch.sigmoid(alpha_value) - 1.0)
    generator = torch.Generator(device="cpu").manual_seed(0)
    samples = torch.randn(1_000_000, generator=generator, dtype=torch.float64)
    sample_activation = 0.6 * samples + 0.4 * samples * (2.0 * torch.sigmoid(samples) - 1.0)
    activated = activated * sample_activation.square().mean().pow(-0.5)
    logits = (activated * contraction.weight).sum(dim=-1)
    alpha = _segment_softmax_reference(logits, edge_dst)
    weighted = value * alpha.unsqueeze(-1)
    aggregated = weighted.new_zeros((3,) + weighted.shape[1:]).index_add_(0, edge_dst, weighted)
    expected = _official_blockwise_merge(aggregated, value_head.irreps)

    assert torch.allclose(actual["alpha"], alpha, atol=1e-12, rtol=1e-12)
    assert torch.allclose(actual["out"], expected, atol=1e-12, rtol=1e-12)

    rotation = o3.rand_matrix(dtype=torch.float64)
    input_action = o3.Irreps(to_e3nn_irreps(message_type.irreps)).D_from_matrix(rotation)
    output_action = o3.Irreps(to_e3nn_irreps(output_type.irreps)).D_from_matrix(rotation)
    rotated = model(
        {
            "message": message @ input_action.transpose(0, 1),
            "segment_index": segment_payload,
        },
        {},
    )
    assert torch.allclose(rotated["alpha"], actual["alpha"], atol=1e-9, rtol=1e-9)
    assert torch.allclose(
        rotated["out"],
        actual["out"] @ output_action.transpose(0, 1),
        atol=1e-9,
        rtol=1e-9,
    )

    permutation = torch.tensor([5, 0, 7, 3, 1, 6, 2, 4], dtype=torch.long)
    permuted_index = edge_dst.index_select(0, permutation)
    permuted = model(
        {
            "message": message.index_select(0, permutation),
            "segment_index": {"indices": permuted_index, "target_size": 3},
        },
        {},
    )
    assert torch.allclose(permuted["alpha"], actual["alpha"].index_select(0, permutation), atol=1e-12, rtol=1e-12)
    assert torch.allclose(permuted["out"], actual["out"], atol=1e-12, rtol=1e-12)

    actual["out"].square().sum().backward()
    assert message.grad is not None and torch.isfinite(message.grad).all()
    assert contraction.weight.grad is not None and torch.isfinite(contraction.weight.grad).all()

    contracts = artifact.inference.parameter_contracts["logits"]
    assert [contract.shape for contract in contracts] == [(1, 2, 2)]
    kinds = dict(E3NNGraphBackend(registry).support_report(program).runtime_kinds)
    assert kinds["split"] == RuntimeValueKind.EQUIVARIANT_HEADS
    assert kinds["alpha_value"] == RuntimeValueKind.EQUIVARIANT_HEADS
    assert kinds["activated"] == RuntimeValueKind.EQUIVARIANT_HEADS
    assert kinds["logits"] == RuntimeValueKind.DENSE_TENSOR
    assert kinds["weighted"] == RuntimeValueKind.EQUIVARIANT_HEADS
    assert kinds["aggregate"] == RuntimeValueKind.EQUIVARIANT_HEADS
    assert kinds["merge"] == RuntimeValueKind.DENSE_TENSOR


def test_equivariant_head_contracts_reject_invalid_multiplicity_ranges_and_runtime_entry():
    source = _equivariant_type("3x0e+2x1o")
    invalid_split = ArchitectureProgram(
        "2.5.0",
        "invalid-head-multiplicity",
        (InputPort("x", source),),
        (Node("split", "core.head_split@2", {"x": ("input:x",)}, {"head_axis": "head", "num_heads": 2}),),
        (OutputPort("out", "split", source),),
    )
    with pytest.raises(DSLValidationError) as split_error:
        Compiler(core_registry()).analyze(invalid_split)
    assert any(item.code == "E_HEAD_V2_006" for item in split_error.value.diagnostics)

    wrong_layout = replace(
        _equivariant_type("4x0e+2x1o"),
        layout=RepresentationLayout(storage="m_primary", coefficient_order="canonical"),
    )
    invalid_layout_split = ArchitectureProgram(
        "2.5.0",
        "invalid-head-layout",
        (InputPort("x", wrong_layout),),
        (Node("split", "core.head_split@2", {"x": ("input:x",)}, {"head_axis": "head", "num_heads": 2}),),
        (OutputPort("out", "split", wrong_layout),),
    )
    with pytest.raises(DSLValidationError) as layout_error:
        Compiler(core_registry()).analyze(invalid_layout_split)
    assert any(item.code == "E_HEAD_V2_014" for item in layout_error.value.diagnostics)

    head_source = _head_type(_equivariant_type("4x0e+2x1o"))
    invalid_select = ArchitectureProgram(
        "2.5.0",
        "invalid-irrep-range",
        (InputPort("x", head_source),),
        (
            Node(
                "select",
                "core.irrep_select@2",
                {"x": ("input:x",)},
                {"selections": [{"irrep": "0e", "start": 1, "multiplicity": 2}]},
            ),
        ),
        (OutputPort("out", "select", head_source),),
    )
    with pytest.raises(DSLValidationError) as select_error:
        Compiler(core_registry()).analyze(invalid_select)
    assert any(item.code == "E_SELECT_V2_009" for item in select_error.value.diagnostics)

    invalid_select_layout_type = replace(
        head_source,
        layout=RepresentationLayout(storage="m_primary", coefficient_order="canonical"),
    )
    invalid_select_layout = ArchitectureProgram(
        "2.5.0",
        "invalid-irrep-select-layout",
        (InputPort("x", invalid_select_layout_type),),
        (
            Node(
                "select",
                "core.irrep_select@2",
                {"x": ("input:x",)},
                {"selections": [{"irrep": "0e", "start": 0, "multiplicity": 1}]},
            ),
        ),
        (OutputPort("out", "select", invalid_select_layout_type),),
    )
    with pytest.raises(DSLValidationError) as select_layout_error:
        Compiler(core_registry()).analyze(invalid_select_layout)
    assert any(item.code == "E_SELECT_V2_012" for item in select_layout_error.value.diagnostics)

    direct_merge = ArchitectureProgram(
        "2.5.0",
        "invalid-runtime-entry",
        (InputPort("x", head_source),),
        (Node("merge", "core.head_merge@2", {"x": ("input:x",)}, {"head_axis": "head"}),),
        (OutputPort("out", "merge", _equivariant_type("4x0e+2x1o")),),
    )
    report = E3NNGraphBackend(core_registry()).support_report(direct_merge)
    assert not report.supported
    assert any(node_id == "merge" and "equivariant-head" in message for node_id, message in report.composition_errors)


def test_attention_v2_contracts_reject_nontrivial_logits_mismatched_heads_and_wrong_segment_index():
    group = GroupSpec.o3()
    nontrivial_head = _head_type(_equivariant_type("2x0e+2x1o"))
    contraction = ArchitectureProgram(
        "2.5.0",
        "invalid-headwise-logits",
        (InputPort("x", nontrivial_head),),
        (
            Node(
                "logits",
                "core.headwise_scalar_contraction@2",
                {"x": ("input:x",)},
                {"head_axis": "head", "bias": False},
            ),
        ),
        (OutputPort("out", "logits", nontrivial_head),),
    )
    with pytest.raises(DSLValidationError) as contraction_error:
        Compiler(core_registry()).analyze(contraction)
    assert any(item.code == "E_HEAD_V2_010" for item in contraction_error.value.diagnostics)

    value = _head_type(_equivariant_type("4x0e+2x1o"))
    weight = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("3x0e", group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", 3, FeatureRole.HEAD, "independent", 0),),
        dtype="float64",
        feature_role=FeatureRole.ALPHA,
    )
    scaling = ArchitectureProgram(
        "2.5.0",
        "invalid-headwise-scale",
        (InputPort("weight", weight), InputPort("value", value)),
        (Node("scale", "core.invariant_scale@1", {"weight": ("input:weight",), "value": ("input:value",)}, {}),),
        (OutputPort("out", "scale", value),),
    )
    with pytest.raises(DSLValidationError) as scale_error:
        Compiler(core_registry()).analyze(scaling)
    assert any(item.code == "E_SCALE_007" for item in scale_error.value.diagnostics)

    logits = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("2x0e", group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", 2, FeatureRole.HEAD, "independent", 0),),
        dtype="float64",
        feature_role=FeatureRole.ALPHA,
    )
    wrong_index = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "target", target_size=4)
    softmax = ArchitectureProgram(
        "2.5.0",
        "invalid-softmax-index",
        (InputPort("logits", logits), InputPort("index", wrong_index)),
        (
            Node(
                "softmax",
                "core.segment_softmax@2",
                {"logits": ("input:logits",), "index": ("input:index",)},
                {},
            ),
        ),
        (OutputPort("out", "softmax", logits),),
    )
    with pytest.raises(DSLValidationError) as softmax_error:
        Compiler(core_registry()).analyze(softmax)
    assert any(item.code == "E_SOFTMAX_V2_004" for item in softmax_error.value.diagnostics)


def test_smooth_leaky_relu_rejects_invalid_negative_slope_during_lowering_validation():
    logits = InvariantTensorType(
        GroupSpec.o3(),
        Carrier.EDGE,
        Irreps.parse("2x0e", "O3"),
        axes=("head",),
        axis_specs=(AxisSpec("head", 2, FeatureRole.HEAD, "independent", 0),),
        dtype="float64",
        feature_role=FeatureRole.ALPHA,
    )
    program = ArchitectureProgram(
        "2.5.0",
        "invalid-smooth-leaky-relu",
        (InputPort("x", logits),),
        (
            Node(
                "activation",
                "core.scalar_activation@1",
                {"x": ("input:x",)},
                {"activation": "smooth_leaky_relu", "negative_slope": 1.5},
            ),
        ),
        (OutputPort("out", "activation", logits),),
    )
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    with pytest.raises(DSLValidationError) as error:
        E3NNGraphBackend(registry).build(program, artifact.inference)
    assert any(item.code == "E_BACKEND_019" for item in error.value.diagnostics)


def test_scalar_activation_rejects_unknown_normalization_during_lowering_validation():
    logits = InvariantTensorType(
        GroupSpec.o3(),
        Carrier.EDGE,
        Irreps.parse("2x0e", "O3"),
        axes=("head",),
        axis_specs=(AxisSpec("head", 2, FeatureRole.HEAD, "independent", 0),),
        dtype="float64",
        feature_role=FeatureRole.ALPHA,
    )
    program = ArchitectureProgram(
        "2.5.0",
        "invalid-activation-normalization",
        (InputPort("x", logits),),
        (
            Node(
                "activation",
                "core.scalar_activation@1",
                {"x": ("input:x",)},
                {"activation": "smooth_leaky_relu", "normalization": "batch"},
            ),
        ),
        (OutputPort("out", "activation", logits),),
    )
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    with pytest.raises(DSLValidationError) as error:
        E3NNGraphBackend(registry).build(program, artifact.inference)
    assert any(item.code == "E_BACKEND_020" for item in error.value.diagnostics)
