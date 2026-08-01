"""Rule-registry-driven e3nn/PyTorch lowering for backend-neutral DSL graphs.

The front end remains importable without numerical dependencies.  Torch and
e3nn are imported only when a graph is built, while the lowering registry is
also the single source of truth used by backend support reporting.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence, Tuple

from ..ast import ArchitectureProgram
from ..diagnostics import DSLValidationError, Diagnostic
from ..inference import InferenceResult, TypeChecker
from ..irreps import Irreps
from ..registry import PrimitiveRegistry
from .lowering import (
    BackendSupportReport,
    DependencyRequirement,
    LoweringRule,
    LoweringRuleRegistry,
    ModuleBuildContext,
    RuntimeExecutionContext,
    RuntimeValueKind,
)
from .v2_runtime import (
    SO3RuntimeValue,
    build_s2_activation_module,
    build_so2_convolution_module,
    embedding_tensor_to_flat,
    equiformer_v2_source_available,
    flat_to_embedding_tensor,
    from_edge_frame_value,
    load_equiformer_v2_modules,
    select_runtime_scalars,
    to_edge_frame_value,
    uniform_so3_layout,
)
from .v3_runtime import (
    GridRuntimeValue,
    build_v3_axisymmetric_spherical_lift_module,
    build_v3_edge_frame_gate_activation_module,
    build_v3_edge_frame_inverse_module,
    build_v3_edge_frame_rotation_module,
    build_v3_gated_swiglu_merge_module,
    build_v3_grid_channel_linear_module,
    build_v3_grid_project_module,
    build_v3_merge_norm_module,
    build_v3_s2_swiglu_module,
    build_v3_so3_linear_module,
    build_v3_so2_linear_module,
    equiformer_v3_source_available,
    load_equiformer_v3_modules,
    unproject_v3_grid_value,
)


_SECOND_MOMENT_SCALE_CACHE: dict[tuple[str, float], float] = {}


def to_e3nn_irreps(irreps) -> str:
    """Materialize SO(3) types in e3nn's O(3)-labelled storage format."""

    if irreps.family == "SO3":
        return "+".join("{}x{}e".format(mul, ir.degree) for mul, ir in irreps)
    return str(irreps)


@dataclass(frozen=True)
class FusedSubgraph:
    """A verified backend fusion over nodes that have no external consumers."""

    end_node: str
    node_ids: Tuple[str, ...]
    input_reference: str
    module: Any
    context_key: str
    backend_semantics: str


def _runtime_preserve(port: str):
    def infer(node, resolved):
        kinds = resolved.get(port, ())
        if len(kinds) != 1:
            raise ValueError("{} requires one runtime value on {}".format(node.op, port))
        return {output: kinds[0] for output in node.outputs}

    return infer


def _runtime_categorical_remap(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.CATEGORICAL_TENSOR,):
        raise ValueError("{} requires one categorical tensor".format(node.op))
    return {output: RuntimeValueKind.CATEGORICAL_TENSOR for output in node.outputs}


def _runtime_categorical_one_hot(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.CATEGORICAL_TENSOR,):
        raise ValueError("{} requires one categorical tensor".format(node.op))
    return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}


def _runtime_categorical_embedding(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.CATEGORICAL_TENSOR,):
        raise ValueError("{} requires one categorical tensor".format(node.op))
    return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}


def _runtime_same_inputs(node, resolved):
    kinds = tuple(kind for values in resolved.values() for kind in values)
    if not kinds or any(kind != kinds[0] for kind in kinds[1:]):
        raise ValueError("{} requires equal runtime value kinds".format(node.op))
    return {output: kinds[0] for output in node.outputs}


def _runtime_dense_inputs(node, resolved):
    kinds = tuple(kind for values in resolved.values() for kind in values)
    if not kinds or any(kind != RuntimeValueKind.DENSE_TENSOR for kind in kinds):
        raise ValueError("{} requires dense tensor inputs".format(node.op))
    return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}


def _runtime_dense_variadic(port: str):
    def infer(node, resolved):
        kinds = resolved.get(port, ())
        if not kinds or any(kind != RuntimeValueKind.DENSE_TENSOR for kind in kinds):
            raise ValueError("{} requires dense tensors on {}".format(node.op, port))
        return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}

    return infer


def _runtime_invariant_scale(node, resolved):
    weights = resolved.get("weight", ())
    values = resolved.get("value", ())
    if weights != (RuntimeValueKind.DENSE_TENSOR,) or len(values) != 1:
        raise ValueError("{} requires dense invariant weights and one value".format(node.op))
    return {output: values[0] for output in node.outputs}


def _runtime_to_equivariant_heads(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.DENSE_TENSOR,):
        raise ValueError("{} requires a flattened dense equivariant tensor".format(node.op))
    return {output: RuntimeValueKind.EQUIVARIANT_HEADS for output in node.outputs}


def _runtime_from_equivariant_heads(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.EQUIVARIANT_HEADS,):
        raise ValueError("{} requires an equivariant-head runtime value".format(node.op))
    return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}


def _runtime_contract_equivariant_heads(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.EQUIVARIANT_HEADS,):
        raise ValueError("{} requires an equivariant-head runtime value".format(node.op))
    return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}


def _runtime_to_edge_frame(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.DENSE_TENSOR,):
        raise ValueError("{} requires a dense global-frame tensor".format(node.op))
    return {output: RuntimeValueKind.SO3_EDGE_FRAME for output in node.outputs}


def _runtime_from_edge_frame(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.SO3_EDGE_FRAME,):
        raise ValueError("{} requires an SO(3) edge-frame runtime value".format(node.op))
    return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}


def _runtime_to_edge_frame_v3(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.DENSE_TENSOR,) or resolved.get("direction") != (
        RuntimeValueKind.DENSE_TENSOR,
    ):
        raise ValueError("{} requires dense coefficients and an explicit direction".format(node.op))
    return {output: RuntimeValueKind.SO3_EDGE_FRAME for output in node.outputs}


def _runtime_require_edge_frame(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.SO3_EDGE_FRAME,):
        raise ValueError("{} requires an SO(3) edge-frame runtime value".format(node.op))
    return {output: RuntimeValueKind.SO3_EDGE_FRAME for output in node.outputs}


def _runtime_edge_frame_with_scalars(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.SO3_EDGE_FRAME,):
        raise ValueError("{} requires an SO(3) edge-frame value".format(node.op))
    if resolved.get("scalars") != (RuntimeValueKind.DENSE_TENSOR,):
        raise ValueError("{} requires one dense invariant scalar side path".format(node.op))
    return {output: RuntimeValueKind.SO3_EDGE_FRAME for output in node.outputs}


def _runtime_to_grid(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.DENSE_TENSOR,):
        raise ValueError("{} requires canonical dense SO(3) coefficients".format(node.op))
    return {output: RuntimeValueKind.GRID_TENSOR for output in node.outputs}


def _runtime_from_grid(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.GRID_TENSOR,):
        raise ValueError("{} requires a finite-grid runtime value".format(node.op))
    return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}


def _runtime_grid_preserve(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.GRID_TENSOR,):
        raise ValueError("{} requires a finite-grid runtime value".format(node.op))
    return {output: RuntimeValueKind.GRID_TENSOR for output in node.outputs}


def _runtime_grid_variadic(node, resolved):
    kinds = resolved.get("xs", ())
    if not kinds or any(kind != RuntimeValueKind.GRID_TENSOR for kind in kinds):
        raise ValueError("{} requires finite-grid values".format(node.op))
    return {output: RuntimeValueKind.GRID_TENSOR for output in node.outputs}


def _runtime_grid_product(node, resolved):
    kinds = tuple(kind for values in resolved.values() for kind in values)
    if len(kinds) != 2 or RuntimeValueKind.GRID_TENSOR not in kinds:
        raise ValueError("{} requires one grid and one grid or dense invariant value".format(node.op))
    if any(kind not in (RuntimeValueKind.GRID_TENSOR, RuntimeValueKind.DENSE_TENSOR) for kind in kinds):
        raise ValueError("{} received an unsupported grid-product runtime kind".format(node.op))
    return {output: RuntimeValueKind.GRID_TENSOR for output in node.outputs}


def _runtime_v3_so2_linear(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.SO3_EDGE_FRAME,):
        raise ValueError("{} requires an SO(3) edge-frame runtime value".format(node.op))
    if tuple(node.outputs) == ("out",):
        return {"out": RuntimeValueKind.SO3_EDGE_FRAME}
    if set(node.outputs) == {"out", "extra_m0"}:
        return {
            "out": RuntimeValueKind.SO3_EDGE_FRAME,
            "extra_m0": RuntimeValueKind.DENSE_TENSOR,
        }
    raise ValueError("{} declares an invalid SO2Linear output signature".format(node.op))


def _runtime_select_scalars(node, resolved):
    kinds = resolved.get("x", ())
    if len(kinds) != 1 or kinds[0] not in (
        RuntimeValueKind.DENSE_TENSOR,
        RuntimeValueKind.SO3_EDGE_FRAME,
    ):
        raise ValueError("{} received an unsupported runtime value".format(node.op))
    return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}


def _runtime_separable_s2(node, resolved):
    if resolved.get("scalars") != (RuntimeValueKind.DENSE_TENSOR,):
        raise ValueError("{} requires dense scalar side inputs".format(node.op))
    values = resolved.get("x", ())
    if len(values) != 1 or values[0] not in (
        RuntimeValueKind.DENSE_TENSOR,
        RuntimeValueKind.SO3_EDGE_FRAME,
    ):
        raise ValueError("{} received an unsupported value path".format(node.op))
    return {output: values[0] for output in node.outputs}


def _runtime_explicit_index(node, resolved):
    values = resolved.get("x", ())
    if len(values) != 1 or values[0] not in (
        RuntimeValueKind.DENSE_TENSOR,
        RuntimeValueKind.EQUIVARIANT_HEADS,
    ):
        raise ValueError("{} requires one dense or equivariant-head tensor value".format(node.op))
    if resolved.get("index") != (RuntimeValueKind.INDEX_MAP,):
        raise ValueError("{} requires one explicit index-map runtime value".format(node.op))
    return {output: values[0] for output in node.outputs}


def _runtime_endpoint_gather_v2(node, resolved):
    values = resolved.get("x", ())
    if len(values) != 1 or values[0] not in (
        RuntimeValueKind.DENSE_TENSOR,
        RuntimeValueKind.CATEGORICAL_TENSOR,
    ):
        raise ValueError("{} requires one dense or categorical tensor value".format(node.op))
    if resolved.get("index") != (RuntimeValueKind.INDEX_MAP,):
        raise ValueError("{} requires one explicit endpoint index map".format(node.op))
    return {output: values[0] for output in node.outputs}


def _runtime_explicit_segment_softmax(node, resolved):
    if resolved.get("logits") != (RuntimeValueKind.DENSE_TENSOR,):
        raise ValueError("{} requires dense invariant logits".format(node.op))
    if resolved.get("index") != (RuntimeValueKind.INDEX_MAP,):
        raise ValueError("{} requires one explicit index-map runtime value".format(node.op))
    return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}


def _runtime_v3_graph_softmax(node, resolved):
    outputs = _runtime_explicit_segment_softmax(node, resolved)
    if resolved.get("exp_rescale") != (RuntimeValueKind.DENSE_TENSOR,):
        raise ValueError("{} requires one dense invariant exponent rescale".format(node.op))
    return outputs


def _runtime_graph_stochastic_depth(node, resolved):
    if resolved.get("x") != (RuntimeValueKind.DENSE_TENSOR,):
        raise ValueError("{} requires one dense equivariant value".format(node.op))
    if resolved.get("batch") != (RuntimeValueKind.INDEX_MAP,):
        raise ValueError("{} requires one explicit batch index map".format(node.op))
    return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}


def _runtime_relative_displacement(node, resolved):
    if resolved.get("positions") != (RuntimeValueKind.AFFINE_POINT,):
        raise ValueError("{} requires affine-point positions".format(node.op))
    if resolved.get("source_index") != (RuntimeValueKind.INDEX_MAP,) or resolved.get("target_index") != (RuntimeValueKind.INDEX_MAP,):
        raise ValueError("{} requires explicit source and target index maps".format(node.op))
    return {output: RuntimeValueKind.DENSE_TENSOR for output in node.outputs}


def _runtime_periodic_displacement(node, resolved):
    outputs = _runtime_relative_displacement(node, resolved)
    if resolved.get("lattice") != (RuntimeValueKind.LATTICE,):
        raise ValueError("{} requires a lattice runtime value".format(node.op))
    if resolved.get("lattice_shift") != (RuntimeValueKind.LATTICE_SHIFT,):
        raise ValueError("{} requires a lattice-shift runtime value".format(node.op))
    return outputs


def _activation(torch, name: str, negative_slope: float = 0.2):
    functions = {
        "silu": torch.nn.functional.silu,
        "relu": torch.relu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
    }
    if name == "smooth_leaky_relu":
        alpha = float(negative_slope)

        def smooth_leaky_relu(value):
            linear = ((1.0 + alpha) / 2.0) * value
            smooth = ((1.0 - alpha) / 2.0) * value * (2.0 * torch.sigmoid(value) - 1.0)
            return linear + smooth

        return smooth_leaky_relu
    try:
        return functions[name]
    except KeyError:
        raise DSLValidationError([
            Diagnostic("E_BACKEND_005", "unsupported activation", actual=name)
        ])


def _activation_name(node) -> str:
    """Return the documented canonical activation attribute."""

    return str(node.attrs.get("activation", node.attrs.get("function", "silu")))


def _activation_normalization(node) -> str:
    return str(node.attrs.get("normalization", "none"))


def _second_moment_activation_scale(torch, name: str, negative_slope: float) -> float:
    """Match e3nn.math.normalize2mom's deterministic N(0, 1) estimate."""

    key = (name, float(negative_slope))
    if key not in _SECOND_MOMENT_SCALE_CACHE:
        generator = torch.Generator(device="cpu").manual_seed(0)
        samples = torch.randn(1_000_000, generator=generator, dtype=torch.float64, device="cpu")
        activated = _activation(torch, name, negative_slope)(samples)
        _SECOND_MOMENT_SCALE_CACHE[key] = float(activated.square().mean().pow(-0.5).item())
    return _SECOND_MOMENT_SCALE_CACHE[key]


def _validate_activation(context: ModuleBuildContext) -> None:
    slope = float(context.node.attrs.get("negative_slope", 0.2))
    if slope < 0.0 or slope > 1.0:
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_019",
                "smooth_leaky_relu negative_slope must satisfy 0 <= slope <= 1",
                node_id=context.node.id,
                actual=str(slope),
            )
        ])
    _activation(context.libraries["torch"], _activation_name(context.node), slope)
    normalization = _activation_normalization(context.node)
    if normalization not in ("none", "second_moment"):
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_020",
                "scalar activation normalization must be none or second_moment",
                node_id=context.node.id,
                actual=normalization,
            )
        ])


def _validate_radial_basis(context: ModuleBuildContext) -> None:
    cutoff = float(context.node.attrs.get("cutoff", 5.0))
    width = context.node.attrs.get("width")
    if cutoff <= 0.0:
        raise DSLValidationError([
            Diagnostic("E_BACKEND_008", "radial basis cutoff must be positive", node_id=context.node.id)
        ])
    if width is not None and float(width) <= 0.0:
        raise DSLValidationError([
            Diagnostic("E_BACKEND_009", "radial basis width must be positive", node_id=context.node.id)
        ])


def _validate_cutoff(context: ModuleBuildContext) -> None:
    cutoff = float(context.node.attrs.get("cutoff", 5.0))
    order = int(context.node.attrs.get("order", 5))
    if cutoff <= 0.0:
        raise DSLValidationError([
            Diagnostic("E_BACKEND_010", "cutoff envelope cutoff must be positive", node_id=context.node.id)
        ])
    if order < 1:
        raise DSLValidationError([
            Diagnostic("E_BACKEND_011", "cutoff envelope order must be at least one", node_id=context.node.id)
        ])


def _v2_root(libraries: Mapping[str, Any]) -> str:
    return str(libraries.get("equiformer_v2_root") or "")


def _v2_modules(libraries: Mapping[str, Any]):
    modules = libraries.get("equiformer_v2_modules")
    if modules is not None:
        return modules
    return load_equiformer_v2_modules(_v2_root(libraries))


def _v2_dependency(root: str) -> DependencyRequirement:
    return DependencyRequirement(
        "equiformer_v2_reference",
        "equiformer_v2_reference",
        checker=lambda: equiformer_v2_source_available(root),
    )


def _v3_root(libraries: Mapping[str, Any]) -> str:
    return str(libraries.get("equiformer_v3_root") or "")


def _v3_modules(libraries: Mapping[str, Any]):
    modules = libraries.get("equiformer_v3_modules")
    if modules is not None:
        return modules
    return load_equiformer_v3_modules(_v3_root(libraries))


def _v3_dependency(root: str) -> DependencyRequirement:
    return DependencyRequirement(
        "equiformer_v3_reference",
        "equiformer_v3_reference",
        checker=lambda: equiformer_v3_source_available(root),
    )


def _validate_v2_layout(context: ModuleBuildContext) -> None:
    uniform_so3_layout(context.output_type.irreps)


def _validate_to_edge_frame(context: ModuleBuildContext) -> None:
    source = context.value_type(context.node.inputs["x"][0])
    uniform_so3_layout(source.irreps)


def _validate_so2(context: ModuleBuildContext) -> None:
    source = context.value_type(context.node.inputs["x"][0])
    lmax, _channels = uniform_so3_layout(source.irreps)
    output_lmax, _output_channels = uniform_so3_layout(context.output_type.irreps)
    if output_lmax != lmax:
        raise DSLValidationError([
            Diagnostic(
                "E_V2_BACKEND_005",
                "independent SO(2) lowering requires equal input and output lmax",
                node_id=context.node.id,
            )
        ])
    mmax = int(context.node.attrs.get("mmax", lmax))
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic("E_V2_BACKEND_006", "mmax must satisfy 0 <= mmax <= lmax", node_id=context.node.id)
        ])


def _validate_s2(context: ModuleBuildContext) -> None:
    lmax, _channels = uniform_so3_layout(context.output_type.irreps)
    mmax = int(context.node.attrs.get("mmax", lmax))
    raw_resolution = context.node.attrs.get("grid_resolution", 18)
    resolutions = (
        tuple(int(value) for value in raw_resolution)
        if isinstance(raw_resolution, (list, tuple))
        else (int(raw_resolution),)
    )
    normalization = str(context.node.attrs.get("normalization", "component"))
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic("E_V2_BACKEND_006", "mmax must satisfy 0 <= mmax <= lmax", node_id=context.node.id)
        ])
    if not resolutions or len(resolutions) > 2 or any(value < 2 for value in resolutions):
        raise DSLValidationError([
            Diagnostic(
                "E_V2_BACKEND_012",
                "S2 grid_resolution must contain one or two integers >= 2",
                node_id=context.node.id,
            )
        ])
    if normalization not in ("component", "integral", "norm"):
        raise DSLValidationError([
            Diagnostic("E_V2_BACKEND_013", "unsupported S2 grid normalization", node_id=context.node.id, actual=normalization)
        ])


def _build_linear(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    return context.libraries["o3"].Linear(
        to_e3nn_irreps(source.irreps),
        to_e3nn_irreps(context.output_type.irreps),
    )


def _build_gaussian_radial_basis(context: ModuleBuildContext):
    torch = context.libraries["torch"]
    num_basis = int(context.node.attrs["num_basis"])
    cutoff = float(context.node.attrs["cutoff"])

    class LearnableGaussianRadialBasis(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_basis = num_basis
            self.cutoff = cutoff
            self.mean = torch.nn.Parameter(torch.zeros(1, num_basis))
            self.std = torch.nn.Parameter(torch.zeros(1, num_basis))
            self.weight = torch.nn.Parameter(torch.ones(1, 1))
            self.bias = torch.nn.Parameter(torch.zeros(1, 1))
            torch.nn.init.uniform_(self.mean, 0.0, 1.0)
            torch.nn.init.uniform_(self.std, 1.0 / float(num_basis), 1.0)
            torch.nn.init.constant_(self.weight, 1.0)
            torch.nn.init.constant_(self.bias, 0.0)

        def forward(self, distance):
            if distance.dim() < 1 or distance.shape[-1] != 1:
                raise RuntimeError(
                    "gaussian_radial_basis expected one scalar per edge with trailing size 1, received {}".format(
                        tuple(distance.shape)
                    )
                )
            value = distance / self.cutoff
            value = self.weight * value + self.bias
            value = value.expand(*value.shape[:-1], self.num_basis)
            std = self.std.abs() + 1.0e-5
            normalization = (2.0 * 3.14159) ** 0.5
            return torch.exp(-0.5 * (((value - self.mean) / std) ** 2)) / (normalization * std)

    return LearnableGaussianRadialBasis()


def _build_linear_rs(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    torch = context.libraries["torch"]
    o3 = context.libraries["o3"]
    irreps_in = o3.Irreps(to_e3nn_irreps(source.irreps))
    irreps_out = o3.Irreps(to_e3nn_irreps(context.output_type.irreps))
    scalar = o3.Irreps("1x0e")
    instructions = [
        (input_index, 0, output_index, "uvw", True, 1.0)
        for input_index, (_input_mul, input_irrep) in enumerate(irreps_in)
        for output_index, (_output_mul, output_irrep) in enumerate(irreps_out)
        if input_irrep == output_irrep
    ]
    use_bias = bool(context.node.attrs.get("bias", True))
    use_rescale = bool(context.node.attrs.get("rescale", True))
    initializer_scale = float(context.node.attrs.get("initializer_scale", 1.0))

    class LinearRS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.irreps_in = irreps_in
            self.irreps_out = irreps_out
            self.tp = o3.TensorProduct(
                irreps_in,
                scalar,
                irreps_out,
                instructions,
                normalization=None,
                internal_weights=True,
                shared_weights=True,
                path_normalization="none",
            )
            self.bias_slices = []
            biases = []
            if use_bias:
                for (multiplicity, irrep), output_slice in zip(irreps_out, irreps_out.slices()):
                    if irrep.l == 0 and irrep.p == 1:
                        biases.append(
                            torch.nn.Parameter(torch.zeros(multiplicity, dtype=self.tp.weight.dtype))
                        )
                        self.bias_slices.append(output_slice)
            self.bias = torch.nn.ParameterList(biases)
            fan_in_by_output = {}
            for instruction in self.tp.instructions:
                output_index = int(instruction.i_out)
                fan_in = (
                    int(self.tp.irreps_in1[instruction.i_in1].mul)
                    * int(self.tp.irreps_in2[instruction.i_in2].mul)
                )
                fan_in_by_output[output_index] = fan_in_by_output.get(output_index, 0) + fan_in
            self.slices_sqrt_k = {
                output_index: (
                    irreps_out.slices()[output_index],
                    1.0 / math.sqrt(float(fan_in)) if use_rescale else 1.0,
                )
                for output_index, fan_in in fan_in_by_output.items()
            }
            if use_rescale:
                with torch.no_grad():
                    for weight_view, instruction in zip(self.tp.weight_views(), self.tp.instructions):
                        weight_view.mul_(self.slices_sqrt_k[int(instruction.i_out)][1])
            if initializer_scale != 1.0:
                with torch.no_grad():
                    self.tp.weight.mul_(initializer_scale)

        def forward(self, value):
            invariant = torch.ones_like(value[..., 0:1])
            output = self.tp(value, invariant)
            for output_slice, bias in zip(self.bias_slices, self.bias):
                output.narrow(-1, output_slice.start, output_slice.stop - output_slice.start).add_(bias)
            return output

    module = LinearRS()
    expected_weight_numel = next(
        contract.concrete_shape[0]
        for contract in context.parameter_contracts
        if contract.name == "weight" and contract.concrete_shape is not None
    )
    if int(module.tp.weight_numel) != int(expected_weight_numel):
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_024",
                "LinearRS tensor-product weight_numel disagrees with inferred ParameterContract",
                node_id=context.node.id,
                expected=str(expected_weight_numel),
                actual=str(int(module.tp.weight_numel)),
            )
        ])
    return module


def _build_scalar_linear(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    axis_name = str(context.node.attrs["axis"])
    axis_sizes = tuple(int(axis.size) for axis in source.axis_specs)
    axis_index = next(index for index, axis in enumerate(source.axis_specs) if axis.name == axis_name)
    out_features = int(context.node.attrs["out_features"])
    bias = bool(context.node.attrs.get("bias", True))
    torch = context.libraries["torch"]

    class ScalarAxisLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.axis_sizes = axis_sizes
            self.axis_index = axis_index
            self.output_axis_sizes = tuple(
                out_features if index == axis_index else size
                for index, size in enumerate(axis_sizes)
            )
            self.linear = torch.nn.Linear(axis_sizes[axis_index], out_features, bias=bias)

        def forward(self, value):
            if value.shape[-1] != math.prod(self.axis_sizes):
                raise RuntimeError(
                    "scalar_linear expected flattened invariant feature size {} but received {}".format(
                        math.prod(self.axis_sizes),
                        value.shape[-1],
                    )
                )
            leading_shape = tuple(value.shape[:-1])
            expanded = value.reshape(*leading_shape, *self.axis_sizes)
            absolute_axis = len(leading_shape) + self.axis_index
            projected = self.linear(expanded.movedim(absolute_axis, -1))
            restored = projected.movedim(-1, absolute_axis)
            return restored.reshape(*leading_shape, math.prod(self.output_axis_sizes))

    return ScalarAxisLinear()


def _build_categorical_embedding(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    torch = context.libraries["torch"]
    module = torch.nn.Embedding(
        int(source.vocabulary_size),
        int(context.node.attrs["embedding_dim"]),
    )
    if str(context.node.attrs.get("initializer", "normal_0_1")) == "normal_then_uniform":
        torch.nn.init.uniform_(
            module.weight.data,
            float(context.node.attrs["init_min"]),
            float(context.node.attrs["init_max"]),
        )
    return module


def _initialize_categorical_embedding_v2(module, context: ModuleBuildContext):
    context.libraries["torch"].nn.init.uniform_(
        module.weight,
        float(context.node.attrs["init_min"]),
        float(context.node.attrs["init_max"]),
    )


def _build_fixed_gaussian_radial_basis(context: ModuleBuildContext):
    torch = context.libraries["torch"]
    start = float(context.node.attrs.get("start", 0.0))
    stop = float(context.node.attrs["stop"])
    count = int(context.node.attrs["num_basis"])
    width_scalar = float(context.node.attrs.get("basis_width_scalar", 1.0))
    construction_dtype = getattr(
        torch,
        str(context.node.attrs.get("construction_dtype", "float32")),
    )

    class FixedGaussianRadialBasis(torch.nn.Module):
        def __init__(self):
            super().__init__()
            offset = torch.linspace(start, stop, count, dtype=construction_dtype)
            self.coefficient = -0.5 / (width_scalar * (offset[1] - offset[0])).item() ** 2
            self.register_buffer("offset", offset)

        def forward(self, distance):
            centered = distance.reshape(-1, 1) - self.offset.reshape(1, -1)
            return torch.exp(self.coefficient * centered.pow(2))

    return FixedGaussianRadialBasis()


def _build_scalar_linear_v2(context: ModuleBuildContext):
    module = _build_scalar_linear(context)
    raw_scales = context.node.attrs.get("initializer_scales")
    if raw_scales is not None:
        torch = context.libraries["torch"]
        scales = torch.as_tensor(
            tuple(float(value) for value in raw_scales),
            dtype=module.linear.weight.dtype,
            device=module.linear.weight.device,
        )
        with torch.no_grad():
            module.linear.weight.mul_(scales.reshape(-1, 1))
            if module.linear.bias is not None:
                module.linear.bias.mul_(scales)
    return module


def _initialize_scalar_linear_v3(module, context: ModuleBuildContext):
    torch = context.libraries["torch"]
    bound = 1.0 / math.sqrt(float(module.linear.in_features))
    torch.nn.init.uniform_(module.linear.weight, -bound, bound)
    if module.linear.bias is not None:
        torch.nn.init.zeros_(module.linear.bias)


def _initialize_scalar_linear_v4(module, context: ModuleBuildContext):
    """Mirror the official V3 global initializer without redrawing weights."""

    if module.linear.bias is not None:
        context.libraries["torch"].nn.init.zeros_(module.linear.bias)


def _build_v3_axisymmetric_spherical_lift(context: ModuleBuildContext):
    modules = context.libraries.get("equiformer_v3_modules")
    if modules is None:
        raise RuntimeError("V3 axisymmetric spherical lift requires Equiformer V3 operator modules")
    source = context.output_type
    lmax, _channels = uniform_so3_layout(source.irreps)
    return build_v3_axisymmetric_spherical_lift_module(
        source.irreps,
        mmax=int(context.node.attrs.get("mmax", lmax)),
        use_rotation_mask=bool(context.node.attrs.get("use_rotation_mask", False)),
        modules=modules,
    )


def _build_v3_grid_project(context: ModuleBuildContext):
    modules = context.libraries.get("equiformer_v3_modules")
    if modules is None:
        modules = _v3_modules(context.libraries)
    source = context.value_type(context.node.inputs["x"][0])
    return build_v3_grid_project_module(
        source.irreps,
        grid_spec=context.output_type.grid,
        modules=modules,
    )


def _build_grid_channel_linear(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    return build_v3_grid_channel_linear_module(
        source.channels,
        context.output_type.channels,
        bias=bool(context.node.attrs.get("bias", True)),
    )


def _build_scalar_layer_norm(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    axis_name = str(context.node.attrs["axis"])
    axis_sizes = tuple(int(axis.size) for axis in source.axis_specs)
    axis_index = next(index for index, axis in enumerate(source.axis_specs) if axis.name == axis_name)
    torch = context.libraries["torch"]

    class ScalarAxisLayerNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.axis_sizes = axis_sizes
            self.axis_index = axis_index
            self.layer_norm = torch.nn.LayerNorm(
                axis_sizes[axis_index],
                eps=float(context.node.attrs.get("epsilon", 1.0e-5)),
                elementwise_affine=bool(context.node.attrs.get("affine", True)),
                bias=bool(context.node.attrs.get("bias", True)),
            )

        def forward(self, value):
            if value.shape[-1] != math.prod(self.axis_sizes):
                raise RuntimeError(
                    "scalar_layer_norm expected flattened invariant feature size {} but received {}".format(
                        math.prod(self.axis_sizes),
                        value.shape[-1],
                    )
                )
            leading_shape = tuple(value.shape[:-1])
            expanded = value.reshape(*leading_shape, *self.axis_sizes)
            absolute_axis = len(leading_shape) + self.axis_index
            normalized = self.layer_norm(expanded.movedim(absolute_axis, -1))
            restored = normalized.movedim(-1, absolute_axis)
            return restored.reshape(*leading_shape, math.prod(self.axis_sizes))

    return ScalarAxisLayerNorm()


def _build_scalar_offset(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    axis_name = str(context.node.attrs["axis"])
    axis_sizes = tuple(int(axis.size) for axis in source.axis_specs)
    axis_index = next(index for index, axis in enumerate(source.axis_specs) if axis.name == axis_name)
    axis_size = axis_sizes[axis_index]
    fan_in = int(context.node.attrs["fan_in"])
    torch = context.libraries["torch"]

    class ScalarAxisOffset(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.axis_sizes = axis_sizes
            self.axis_index = axis_index
            self.offset = torch.nn.Parameter(torch.empty(axis_size))
            bound = 1.0 / math.sqrt(float(fan_in))
            torch.nn.init.uniform_(self.offset, -bound, bound)
            raw_scales = context.node.attrs.get("initializer_scales")
            if raw_scales is not None:
                scales = torch.as_tensor(
                    tuple(float(value) for value in raw_scales),
                    dtype=self.offset.dtype,
                    device=self.offset.device,
                )
                with torch.no_grad():
                    self.offset.mul_(scales)

        def forward(self, value):
            if value.shape[-1] != math.prod(self.axis_sizes):
                raise RuntimeError(
                    "scalar_offset expected flattened invariant feature size {} but received {}".format(
                        math.prod(self.axis_sizes),
                        value.shape[-1],
                    )
                )
            leading_shape = tuple(value.shape[:-1])
            expanded = value.reshape(*leading_shape, *self.axis_sizes)
            absolute_axis = len(leading_shape) + self.axis_index
            view_shape = [1] * expanded.dim()
            view_shape[absolute_axis] = axis_size
            shifted = expanded + self.offset.reshape(view_shape)
            return shifted.reshape(*leading_shape, math.prod(self.axis_sizes))

    return ScalarAxisOffset()


def _build_headwise_scalar_contraction(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    axis_map = {axis.name: axis for axis in source.axis_specs}
    head_size = int(axis_map[str(context.node.attrs["head_axis"])].size)
    channel_size = int(axis_map[str(context.node.attrs["channel_axis"])].size)
    use_bias = bool(context.node.attrs.get("bias", True))
    torch = context.libraries["torch"]

    class HeadwiseScalarContraction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head_size = head_size
            self.channel_size = channel_size
            self.weight = torch.nn.Parameter(torch.empty(head_size, channel_size))
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
            if use_bias:
                self.bias = torch.nn.Parameter(torch.empty(head_size))
                bound = 1.0 / math.sqrt(channel_size)
                torch.nn.init.uniform_(self.bias, -bound, bound)
            else:
                self.register_parameter("bias", None)

        def forward(self, value):
            expected = self.head_size * self.channel_size
            if value.shape[-1] != expected:
                raise RuntimeError(
                    "headwise_scalar_contraction expected flattened feature size {} but received {}".format(
                        expected,
                        value.shape[-1],
                    )
                )
            expanded = value.reshape(*value.shape[:-1], self.head_size, self.channel_size)
            output = (expanded * self.weight).sum(dim=-1)
            return output + self.bias if self.bias is not None else output

    return HeadwiseScalarContraction()


def _build_headwise_scalar_contraction_v2(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    head_size = int(source.axis_specs[0].size)
    channel_size = int(source.irreps.dimension)
    use_bias = bool(context.node.attrs.get("bias", False))
    torch = context.libraries["torch"]

    class EquivariantHeadScalarContraction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head_size = head_size
            self.channel_size = channel_size
            # V1 GraphAttention first allocates ``alpha_dot`` with randn and
            # then overwrites it with the GATv2 Glorot uniform initializer.
            # Preserving both draws is required for seed-stable initialization
            # of this parameter and every module constructed after it.
            self.weight = torch.nn.Parameter(torch.randn(1, head_size, channel_size))
            bound = math.sqrt(6.0 / float(head_size + channel_size))
            torch.nn.init.uniform_(self.weight, -bound, bound)
            if use_bias:
                self.bias = torch.nn.Parameter(torch.zeros(head_size))
            else:
                self.register_parameter("bias", None)

        def forward(self, value):
            expected_tail = (self.head_size, self.channel_size)
            if value.dim() < 2 or tuple(value.shape[-2:]) != expected_tail:
                raise RuntimeError(
                    "headwise_scalar_contraction@2 expected runtime tail {} but received {}".format(
                        expected_tail,
                        tuple(value.shape[-2:]),
                    )
                )
            # Match Equiformer V1's exact einsum, including its non-contiguous
            # head-major output stride.  The stride is observable in training:
            # PyTorch dropout traverses the physical layout when assigning a
            # fixed-seed mask, so a numerically equal contiguous contraction
            # would diverge once alpha dropout is enabled.
            if value.dim() != 3:
                raise RuntimeError(
                    "headwise_scalar_contraction@2 exact V1 lowering requires [item, head, channel]"
                )
            output = torch.einsum("bik,aik->bi", value, self.weight)
            return output + self.bias if self.bias is not None else output

    return EquivariantHeadScalarContraction()


def _build_headwise_scalar_contraction_v3(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    axis_map = {axis.name: axis for axis in source.axis_specs}
    head_size = int(axis_map[str(context.node.attrs["head_axis"])].size)
    channel_size = int(axis_map[str(context.node.attrs["channel_axis"])].size)
    torch = context.libraries["torch"]

    class V3HeadwiseScalarContraction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(head_size, channel_size))
            bound = 1.0 / math.sqrt(float(channel_size))
            torch.nn.init.uniform_(self.weight, -bound, bound)

        def forward(self, value):
            if value.dim() != 2 or int(value.shape[-1]) != head_size * channel_size:
                raise RuntimeError(
                    "V3 headwise contraction expected [edge, {}] flattened alpha channels".format(
                        head_size * channel_size
                    )
                )
            expanded = value.view(-1, head_size, channel_size)
            return torch.einsum("bik,ik->bi", expanded, self.weight)

    return V3HeadwiseScalarContraction()


def _build_v3_graph_softmax(context: ModuleBuildContext):
    torch = context.libraries["torch"]
    epsilon = float(context.node.attrs.get("epsilon", 1.0e-16))
    exp_dropout = float(context.node.attrs.get("exp_dropout", 0.0))
    raw_softcap = context.node.attrs.get("softcap")
    softcap = None if raw_softcap is None else float(raw_softcap)

    class V3GraphSoftmax(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(exp_dropout) if exp_dropout > 0.0 else torch.nn.Identity()

        def forward(self, src, index, num_nodes, exp_rescale):
            if softcap is not None:
                src = torch.tanh(src / softcap) * softcap
            if src.shape[0] == 0:
                return src
            index_shape = (index.shape[0],) + (1,) * (src.dim() - 1)
            expanded_index = index.reshape(index_shape).expand_as(src)
            maximum = src.new_full((int(num_nodes),) + src.shape[1:], float("-inf"))
            maximum.scatter_reduce_(0, expanded_index, src.detach(), reduce="amax", include_self=True)
            out = torch.exp(src - maximum.index_select(0, index))
            out = out * exp_rescale
            out = self.dropout(out)
            denominator = src.new_zeros((int(num_nodes),) + src.shape[1:])
            denominator.index_add_(0, index, out)
            denominator = denominator + epsilon
            return out / denominator.index_select(0, index)

    return V3GraphSoftmax()


def _build_v3_edge_frame_rotation(context: ModuleBuildContext):
    modules = context.libraries.get("equiformer_v3_modules")
    if modules is None:
        modules = _v3_modules(context.libraries)
    source = context.value_type(context.node.inputs["x"][0])
    lmax, _channels = uniform_so3_layout(source.irreps)
    return build_v3_edge_frame_rotation_module(
        source.irreps,
        mmax=int(context.node.attrs.get("mmax", lmax)),
        use_rotation_mask=bool(context.node.attrs.get("use_rotation_mask", False)),
        modules=modules,
    )


def _build_v3_edge_frame_inverse(context: ModuleBuildContext):
    modules = context.libraries.get("equiformer_v3_modules")
    if modules is None:
        modules = _v3_modules(context.libraries)
    source = context.value_type(context.node.inputs["x"][0])
    lmax, _channels = uniform_so3_layout(source.irreps)
    return build_v3_edge_frame_inverse_module(
        source.irreps,
        mmax=int(context.node.attrs.get("mmax", lmax)),
        use_rotation_mask=bool(context.node.attrs.get("use_rotation_mask", False)),
        modules=modules,
    )


def _build_v3_gated_swiglu_merge(context: ModuleBuildContext):
    modules = context.libraries.get("equiformer_v3_modules")
    if modules is None:
        modules = _v3_modules(context.libraries)
    source = context.value_type(context.node.inputs["x"][0])
    lmax, _channels = uniform_so3_layout(source.irreps)
    return build_v3_gated_swiglu_merge_module(
        source.irreps,
        context.output_type.irreps,
        mmax=int(context.node.attrs.get("mmax", lmax)),
        grid_resolution=context.node.attrs.get("grid_resolution"),
        dropout=float(context.node.attrs.get("dropout", 0.0)),
        modules=modules,
    )


def _build_v3_edge_frame_gate_activation(context: ModuleBuildContext):
    modules = context.libraries.get("equiformer_v3_modules")
    if modules is None:
        modules = _v3_modules(context.libraries)
    source = context.value_type(context.node.inputs["x"][0])
    lmax, _channels = uniform_so3_layout(source.irreps)
    return build_v3_edge_frame_gate_activation_module(
        source.irreps,
        mmax=int(context.node.attrs.get("mmax", lmax)),
        modules=modules,
    )


def _build_v3_so3_linear(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    return build_v3_so3_linear_module(
        source.irreps,
        context.output_type.irreps,
        bias=bool(context.node.attrs.get("bias", True)),
        l0_weight_scale=float(context.node.attrs.get("l0_weight_scale", 1.0)),
    )


def _validate_built_module_parameters(node, module, contracts) -> None:
    if not contracts:
        return
    internal = tuple(contract for contract in contracts if contract.storage == "internal")
    if internal and module is None:
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_013",
                "parameter contracts require a constructed module",
                node_id=node.id,
                details={"parameters": [contract.name for contract in internal]},
            )
        ])
    actual = dict(module.named_parameters()) if module is not None else {}
    expected_names = {contract.backend_parameter_name for contract in internal}
    if set(actual) != expected_names:
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_014",
                "constructed module parameters do not match inferred ParameterContract names",
                node_id=node.id,
                expected=str(sorted(expected_names)),
                actual=str(sorted(actual)),
            )
        ])
    for contract in internal:
        expected_shape = contract.concrete_shape
        if expected_shape is None:
            raise DSLValidationError([
                Diagnostic(
                    "E_BACKEND_015",
                    "internal module parameter contract must have a concrete shape",
                    node_id=node.id,
                    actual=contract.name,
                )
            ])
        parameter = actual[contract.backend_parameter_name]
        if tuple(parameter.shape) != expected_shape:
            raise DSLValidationError([
                Diagnostic(
                    "E_BACKEND_016",
                    "constructed module parameter shape disagrees with inferred ParameterContract",
                    node_id=node.id,
                    port=contract.name,
                    expected=str(expected_shape),
                    actual=str(tuple(parameter.shape)),
                )
            ])
        if bool(parameter.requires_grad) != contract.trainable:
            raise DSLValidationError([
                Diagnostic(
                    "E_BACKEND_017",
                    "constructed module trainability disagrees with inferred ParameterContract",
                    node_id=node.id,
                    port=contract.name,
                    expected=str(contract.trainable),
                    actual=str(bool(parameter.requires_grad)),
                )
            ])


def _build_tensor_product(context: ModuleBuildContext):
    left = context.value_type(context.node.inputs["left"][0])
    right = context.value_type(context.node.inputs["right"][0])
    return context.libraries["o3"].FullyConnectedTensorProduct(
        to_e3nn_irreps(left.irreps),
        to_e3nn_irreps(right.irreps),
        to_e3nn_irreps(context.output_type.irreps),
        internal_weights=True,
        shared_weights=True,
    )


def _build_tensor_product_v2(context: ModuleBuildContext):
    left = context.value_type(context.node.inputs["left"][0])
    right = context.value_type(context.node.inputs["right"][0])
    module = context.libraries["o3"].FullyConnectedTensorProduct(
        to_e3nn_irreps(left.irreps),
        to_e3nn_irreps(right.irreps),
        to_e3nn_irreps(context.output_type.irreps),
        internal_weights=False,
        shared_weights=False,
    )
    external = tuple(contract for contract in context.parameter_contracts if contract.storage == "external")
    expected = external[0].concrete_shape if len(external) == 1 else None
    if expected is None or expected != (int(module.weight_numel),):
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_018",
                "e3nn external tensor-product weight_numel disagrees with inferred ParameterContract",
                node_id=context.node.id,
                expected=str(expected),
                actual=str((int(module.weight_numel),)),
            )
        ])
    return module


def _build_tensor_product_v3(context: ModuleBuildContext):
    """Lower an explicit weighted ``uvu`` instruction graph to e3nn.

    ``context.output_type.irreps`` is intentionally not used as ``irreps_out``
    here: the front end simplifies adjacent equal irreps for its public value
    type, while an e3nn instruction addresses the original unsimplified path
    blocks by index.  Reconstructing those blocks is therefore part of the
    numerical contract rather than backend-specific metadata.
    """

    left = context.value_type(context.node.inputs["left"][0])
    right = context.value_type(context.node.inputs["right"][0])
    o3 = context.libraries["o3"]
    path_irreps = o3.Irreps(
        "+".join(
            "{}x{}".format(int(block["multiplicity"]), str(block["irrep"]))
            for block in context.node.attrs["path_blocks"]
        )
    )
    instructions = [
        (
            int(instruction["left"]),
            int(instruction["right"]),
            int(instruction["out"]),
            str(instruction["mode"]),
            bool(instruction.get("has_weight", True)),
            float(instruction.get("path_weight", 1.0)),
        )
        for instruction in context.node.attrs["instructions"]
    ]
    module = o3.TensorProduct(
        to_e3nn_irreps(left.irreps),
        to_e3nn_irreps(right.irreps),
        path_irreps,
        instructions,
        normalization=None,
        internal_weights=False,
        shared_weights=False,
        path_normalization="none",
    )
    external = tuple(contract for contract in context.parameter_contracts if contract.storage == "external")
    expected = external[0].concrete_shape if len(external) == 1 else None
    actual = (int(module.weight_numel),)
    if expected is None or expected != actual:
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_020",
                "e3nn instruction tensor-product weight_numel disagrees with inferred ParameterContract",
                node_id=context.node.id,
                expected=str(expected),
                actual=str(actual),
            )
        ])
    if module.irreps_out != path_irreps:
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_021",
                "e3nn instruction tensor product changed the declared unsimplified path-block layout",
                node_id=context.node.id,
                expected=str(path_irreps),
                actual=str(module.irreps_out),
            )
        ])
    if bool(module.internal_weights) or bool(module.shared_weights) or tuple(module.named_parameters()):
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_022",
                "tensor_product@3 lowering must not create internal or shared parameters",
                node_id=context.node.id,
            )
        ])
    actual_instructions = tuple(
        (
            int(item.i_in1),
            int(item.i_in2),
            int(item.i_out),
            str(item.connection_mode),
            bool(item.has_weight),
        )
        for item in module.instructions
    )
    expected_instructions = tuple(item[:5] for item in instructions)
    if actual_instructions != expected_instructions:
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_023",
                "e3nn instruction tensor product changed the indexed path graph",
                node_id=context.node.id,
                expected=str(expected_instructions),
                actual=str(actual_instructions),
            )
        ])
    return module


def _build_tensor_product_v4(context: ModuleBuildContext):
    left = context.value_type(context.node.inputs["left"][0])
    right = context.value_type(context.node.inputs["right"][0])
    torch = context.libraries["torch"]
    o3 = context.libraries["o3"]
    irreps_left = o3.Irreps(to_e3nn_irreps(left.irreps))
    irreps_right = o3.Irreps(to_e3nn_irreps(right.irreps))
    irreps_out = o3.Irreps(to_e3nn_irreps(context.output_type.irreps))
    instructions = [
        (left_index, right_index, output_index, "uvw", True, 1.0)
        for left_index, (_left_mul, left_irrep) in enumerate(irreps_left)
        for right_index, (_right_mul, right_irrep) in enumerate(irreps_right)
        for output_index, (_output_mul, output_irrep) in enumerate(irreps_out)
        if output_irrep in left_irrep * right_irrep
    ]
    use_bias = bool(context.node.attrs.get("bias", True))
    use_rescale = bool(context.node.attrs.get("rescale", True))

    class FullyConnectedTensorProductRescale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.irreps_in1 = irreps_left
            self.irreps_in2 = irreps_right
            self.irreps_out = irreps_out
            self.tp = o3.TensorProduct(
                irreps_left,
                irreps_right,
                irreps_out,
                instructions,
                normalization=None,
                internal_weights=True,
                shared_weights=True,
                path_normalization="none",
            )
            self.bias_slices = []
            biases = []
            if use_bias:
                for (multiplicity, irrep), output_slice in zip(irreps_out, irreps_out.slices()):
                    if irrep.l == 0 and irrep.p == 1:
                        biases.append(torch.nn.Parameter(torch.zeros(multiplicity, dtype=self.tp.weight.dtype)))
                        self.bias_slices.append(output_slice)
            self.bias = torch.nn.ParameterList(biases)
            fan_in_by_output = {}
            for instruction in self.tp.instructions:
                output_index = int(instruction.i_out)
                fan_in = (
                    int(self.tp.irreps_in1[instruction.i_in1].mul)
                    * int(self.tp.irreps_in2[instruction.i_in2].mul)
                )
                fan_in_by_output[output_index] = fan_in_by_output.get(output_index, 0) + fan_in
            self.slices_sqrt_k = {
                output_index: (
                    irreps_out.slices()[output_index],
                    1.0 / math.sqrt(float(fan_in)) if use_rescale else 1.0,
                )
                for output_index, fan_in in fan_in_by_output.items()
            }
            if use_rescale:
                with torch.no_grad():
                    for weight_view, instruction in zip(self.tp.weight_views(), self.tp.instructions):
                        weight_view.mul_(self.slices_sqrt_k[int(instruction.i_out)][1])

        def forward(self, left_value, right_value):
            output = self.tp(left_value, right_value)
            for output_slice, bias in zip(self.bias_slices, self.bias):
                output.narrow(-1, output_slice.start, output_slice.stop - output_slice.start).add_(bias)
            return output

    module = FullyConnectedTensorProductRescale()
    expected_weight_numel = next(
        contract.concrete_shape[0]
        for contract in context.parameter_contracts
        if contract.name == "weight" and contract.concrete_shape is not None
    )
    if int(module.tp.weight_numel) != int(expected_weight_numel):
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_025",
                "tensor_product@4 weight_numel disagrees with inferred ParameterContract",
                node_id=context.node.id,
                expected=str(expected_weight_numel),
                actual=str(int(module.tp.weight_numel)),
            )
        ])
    return module


def _build_tensor_product_v5(context: ModuleBuildContext):
    left = context.value_type(context.node.inputs["left"][0])
    right = context.value_type(context.node.inputs["right"][0])
    torch = context.libraries["torch"]
    o3 = context.libraries["o3"]
    irreps_left = o3.Irreps(to_e3nn_irreps(left.irreps))
    irreps_right = o3.Irreps(to_e3nn_irreps(right.irreps))
    path_irreps = o3.Irreps(
        "+".join(
            "{}x{}".format(int(block["multiplicity"]), str(block["irrep"]))
            for block in context.node.attrs["path_blocks"]
        )
    )
    instructions = [
        (
            int(instruction["left"]),
            int(instruction["right"]),
            int(instruction["out"]),
            str(instruction["mode"]),
            bool(instruction.get("has_weight", True)),
            float(instruction.get("path_weight", 1.0)),
        )
        for instruction in context.node.attrs["instructions"]
    ]
    use_rescale = bool(context.node.attrs.get("rescale", True))

    class InternalSharedInstructionTensorProduct(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.irreps_in1 = irreps_left
            self.irreps_in2 = irreps_right
            self.irreps_out = path_irreps
            self.tp = o3.TensorProduct(
                irreps_left,
                irreps_right,
                path_irreps,
                instructions,
                normalization=None,
                internal_weights=True,
                shared_weights=True,
                path_normalization="none",
            )
            fan_in_by_output = {}
            for instruction in self.tp.instructions:
                output_index = int(instruction.i_out)
                fan_in = int(self.tp.irreps_in2[instruction.i_in2].mul)
                fan_in_by_output[output_index] = fan_in_by_output.get(output_index, 0) + fan_in
            self.slices_sqrt_k = {
                output_index: (
                    path_irreps.slices()[output_index],
                    1.0 / math.sqrt(float(fan_in)) if use_rescale else 1.0,
                )
                for output_index, fan_in in fan_in_by_output.items()
            }
            if use_rescale:
                with torch.no_grad():
                    for weight_view, instruction in zip(self.tp.weight_views(), self.tp.instructions):
                        weight_view.mul_(self.slices_sqrt_k[int(instruction.i_out)][1])

        def forward(self, left_value, right_value):
            return self.tp(left_value, right_value)

    module = InternalSharedInstructionTensorProduct()
    expected_weight_numel = next(
        contract.concrete_shape[0]
        for contract in context.parameter_contracts
        if contract.name == "weight" and contract.concrete_shape is not None
    )
    if int(module.tp.weight_numel) != int(expected_weight_numel):
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_026",
                "tensor_product@5 weight_numel disagrees with inferred ParameterContract",
                node_id=context.node.id,
                expected=str(expected_weight_numel),
                actual=str(int(module.tp.weight_numel)),
            )
        ])
    if not bool(module.tp.internal_weights) or not bool(module.tp.shared_weights):
        raise DSLValidationError([
            Diagnostic(
                "E_BACKEND_027",
                "tensor_product@5 lowering must use internal shared weights",
                node_id=context.node.id,
            )
        ])
    return module


def _build_norm_activation(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    torch = context.libraries["torch"]
    return context.libraries["NormActivation"](
        to_e3nn_irreps(source.irreps),
        _activation(torch, _activation_name(context.node), float(context.node.attrs.get("negative_slope", 0.2))),
        normalize=bool(context.node.attrs.get("normalize", True)),
        epsilon=float(context.node.attrs.get("epsilon", 1.0e-8)),
        bias=bool(context.node.attrs.get("bias", False)),
    )


def _build_equivariant_norm(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    return context.libraries["BatchNorm"](
        to_e3nn_irreps(source.irreps),
        eps=float(context.node.attrs.get("epsilon", 1.0e-5)),
        affine=bool(context.node.attrs.get("affine", True)),
        instance=bool(context.node.attrs.get("instance", False)),
        normalization=str(context.node.attrs.get("normalization", "component")),
    )


def _build_irrep_layer_norm(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    torch = context.libraries["torch"]
    irreps = tuple(source.irreps)
    epsilon = float(context.node.attrs.get("epsilon", 1.0e-5))
    affine = bool(context.node.attrs.get("affine", True))
    normalization = str(context.node.attrs.get("normalization", "component"))
    feature_count = sum(multiplicity for multiplicity, _irrep in irreps)
    scalar_count = sum(
        multiplicity
        for multiplicity, irrep in irreps
        if irrep.degree == 0 and irrep.parity == 1
    )

    class IrrepLayerNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if affine:
                self.affine_weight = torch.nn.Parameter(torch.ones(feature_count))
                if scalar_count:
                    self.affine_bias = torch.nn.Parameter(torch.zeros(scalar_count))
                else:
                    self.register_parameter("affine_bias", None)
            else:
                self.register_parameter("affine_weight", None)
                self.register_parameter("affine_bias", None)

        def forward(self, node_input):
            if node_input.dim() != 2 or node_input.shape[-1] != source.irreps.dimension:
                raise RuntimeError(
                    "irrep_layer_norm@1 expected [node, {}] but received {}".format(
                        source.irreps.dimension,
                        tuple(node_input.shape),
                    )
                )
            fields = []
            input_offset = 0
            weight_offset = 0
            bias_offset = 0
            for multiplicity, irrep in irreps:
                irrep_dimension = irrep.dimension
                width = multiplicity * irrep_dimension
                field = node_input.narrow(1, input_offset, width).reshape(-1, multiplicity, irrep_dimension)
                input_offset += width
                if irrep.degree == 0 and irrep.parity == 1:
                    field = field - field.mean(dim=1, keepdim=True)
                if normalization == "norm":
                    field_norm = field.square().sum(dim=-1)
                else:
                    field_norm = field.square().mean(dim=-1)
                field_norm = field_norm.mean(dim=1, keepdim=True).add(epsilon).pow(-0.5)
                if self.affine_weight is not None:
                    weight = self.affine_weight[None, weight_offset : weight_offset + multiplicity]
                    weight_offset += multiplicity
                    field_norm = field_norm * weight
                field = field * field_norm.reshape(-1, multiplicity, 1)
                if self.affine_bias is not None and irrep.dimension == 1 and irrep.parity == 1:
                    bias = self.affine_bias[bias_offset : bias_offset + multiplicity]
                    bias_offset += multiplicity
                    field = field + bias.reshape(multiplicity, 1)
                fields.append(field.reshape(-1, width))
            return torch.cat(fields, dim=-1)

    return IrrepLayerNorm()


def _build_so2_convolution(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    lmax, _channels = uniform_so3_layout(source.irreps)
    return build_so2_convolution_module(
        source.irreps,
        context.output_type.irreps,
        int(context.node.attrs.get("mmax", lmax)),
        _v2_modules(context.libraries),
    )


def _build_s2_activation(context: ModuleBuildContext, *, separable: bool):
    lmax, _channels = uniform_so3_layout(context.output_type.irreps)
    return build_s2_activation_module(
        context.output_type.irreps,
        mmax=int(context.node.attrs.get("mmax", lmax)),
        resolution=int(context.node.attrs.get("grid_resolution", 18)),
        normalization=str(context.node.attrs.get("normalization", "component")),
        separable=separable,
        modules=_v2_modules(context.libraries),
    )


def _build_plain_s2_activation(context: ModuleBuildContext):
    return _build_s2_activation(context, separable=False)


def _build_separable_s2_activation(context: ModuleBuildContext):
    return _build_s2_activation(context, separable=True)


def _build_v3_s2_swiglu(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    lmax, _channels = uniform_so3_layout(source.irreps)
    return build_v3_s2_swiglu_module(
        source.irreps,
        context.output_type.irreps,
        mmax=int(context.node.attrs.get("mmax", lmax)),
        grid_resolution=context.node.attrs.get("grid_resolution"),
        edge_frame=source.frame.kind == "edge",
        modules=_v3_modules(context.libraries),
    )


def _build_v3_so2_linear(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    lmax, _channels = uniform_so3_layout(source.irreps)
    return build_v3_so2_linear_module(
        source.irreps,
        context.output_types["out"].irreps,
        mmax=int(context.node.attrs.get("mmax", lmax)),
        extra_m0_channels=int(context.node.attrs.get("extra_m0_channels", 0)),
        m0_prefix_rows=int(context.node.attrs.get("m0_prefix_rows", 0)),
        m0_prefix_scale=float(context.node.attrs.get("m0_prefix_scale", 1.0)),
        zero_bias=bool(context.node.attrs.get("zero_bias", False)),
    )


def _build_v3_merge_norm(context: ModuleBuildContext):
    source = context.value_type(context.node.inputs["x"][0])
    return build_v3_merge_norm_module(
        source.irreps,
        modules=_v3_modules(context.libraries),
        epsilon=float(context.node.attrs.get("epsilon", 1.0e-5)),
        affine=bool(context.node.attrs.get("affine", True)),
        normalization=str(context.node.attrs.get("normalization", "component")),
        centering=bool(context.node.attrs.get("centering", True)),
    )


def _index_select_irreps(tensor, source_irreps, target_irreps, torch):
    offsets = {}
    start = 0
    for multiplicity, irrep in source_irreps:
        width = multiplicity * irrep.dimension
        offsets.setdefault(irrep, []).append((start, multiplicity, irrep.dimension))
        start += width
    selected = []
    consumed = {}
    for multiplicity, irrep in target_irreps:
        available = offsets.get(irrep, [])
        remaining = multiplicity
        for start, count, dimension in available:
            already = consumed.get((irrep, start), 0)
            take = min(remaining, count - already)
            if take > 0:
                begin = start + already * dimension
                selected.append(tensor[..., begin : begin + take * dimension])
                consumed[(irrep, start)] = already + take
                remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError("static irrep_slice invariant was violated")
    return torch.cat(selected, dim=-1) if selected else tensor[..., :0]


def _segment_reduce(
    values,
    indices,
    count: int,
    torch,
    mean: bool = False,
    target_cardinality: bool = False,
):
    output = values.new_zeros((count,) + values.shape[1:])
    output.index_add_(0, indices, values)
    if mean or target_cardinality:
        denominator = values.new_zeros(count)
        denominator.index_add_(0, indices, values.new_ones(indices.shape[0]))
        shape = (count,) + (1,) * (values.dim() - 1)
        if mean:
            output = output / denominator.clamp_min_(1).reshape(shape)
        else:
            output = output * denominator.reshape(shape)
    return output


def _segment_softmax(values, indices, torch):
    """Dependency-free, differentiable softmax over destination segments."""

    if values.shape[0] == 0:
        return values
    count = int(indices.max().item()) + 1
    index_shape = (indices.shape[0],) + (1,) * (values.dim() - 1)
    expanded = indices.reshape(index_shape).expand_as(values)
    if hasattr(torch.Tensor, "scatter_reduce_"):
        maximum = values.new_full((count,) + values.shape[1:], float("-inf"))
        maximum.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
        shifted = values - maximum.index_select(0, indices)
        numerator = torch.exp(shifted)
        denominator = values.new_zeros((count,) + values.shape[1:])
        denominator.index_add_(0, indices, numerator)
        return numerator / denominator.index_select(0, indices).clamp_min(1.0e-12)

    # Compatibility fallback for old torch releases. CopySlices keeps the
    # selected softmax values connected to autograd, albeit less efficiently.
    output = torch.empty_like(values)
    for segment in torch.unique(indices):
        mask = indices == segment
        output[mask] = torch.softmax(values[mask], dim=0)
    return output


def _gate(gates, values, irreps, torch):
    if gates.shape[-1] == 1:
        return values * gates
    pieces = []
    value_offset = 0
    gate_offset = 0
    for multiplicity, irrep in irreps:
        width = multiplicity * irrep.dimension
        block = values[..., value_offset : value_offset + width]
        if irrep.degree == 0:
            pieces.append(block)
        else:
            block = block.reshape(block.shape[:-1] + (multiplicity, irrep.dimension))
            block_gates = gates[..., gate_offset : gate_offset + multiplicity].unsqueeze(-1)
            pieces.append((block * block_gates).reshape(block.shape[:-2] + (width,)))
            gate_offset += multiplicity
        value_offset += width
    return torch.cat(pieces, dim=-1)


def _irrep_dropout(values, irreps, probability, training, torch, whole_value=False):
    if not training or probability <= 0.0:
        return values
    if whole_value:
        shape = values.shape[:-1] + (1,)
        mask = (torch.rand(shape, device=values.device) >= probability).to(values.dtype) / (1.0 - probability)
        return values * mask
    pieces = []
    offset = 0
    for multiplicity, irrep in irreps:
        width = multiplicity * irrep.dimension
        block = values[..., offset : offset + width].reshape(values.shape[:-1] + (multiplicity, irrep.dimension))
        mask_shape = values.shape[:-1] + (multiplicity, 1)
        mask = (torch.rand(mask_shape, device=values.device) >= probability).to(values.dtype) / (1.0 - probability)
        pieces.append((block * mask).reshape(values.shape[:-1] + (width,)))
        offset += width
    return torch.cat(pieces, dim=-1)


def _execute_identity(context: RuntimeExecutionContext):
    return context.resolved["x"][0]


def _execute_categorical_remap(context: RuntimeExecutionContext):
    torch = context.libraries["torch"]
    value = context.resolved["x"][0]
    if value.dim() != 1:
        raise RuntimeError(
            "categorical_remap expects one category index per carrier item, received shape {}".format(
                tuple(value.shape)
            )
        )
    source_type = context.value_type(context.node.inputs["x"][0])
    if value.numel() and (bool((value < 0).any()) or bool((value >= int(source_type.vocabulary_size)).any())):
        raise RuntimeError("categorical_remap input contains an out-of-vocabulary category")
    table = torch.tensor(
        tuple(int(item) for item in context.node.attrs["mapping"]),
        dtype=torch.long,
        device=value.device,
    )
    output = table.index_select(0, value.to(dtype=torch.long))
    if output.numel() and bool((output < 0).any()):
        rejected = sorted(set(int(item) for item in value[output < 0].detach().cpu().tolist()))
        raise RuntimeError("categorical_remap rejected unsupported input categories {}".format(rejected))
    return output.to(dtype=value.dtype)


def _execute_categorical_one_hot(context: RuntimeExecutionContext):
    torch = context.libraries["torch"]
    value = context.resolved["x"][0]
    source_type = context.value_type(context.node.inputs["x"][0])
    if value.dim() != 1:
        raise RuntimeError(
            "categorical_one_hot expects one category index per carrier item, received shape {}".format(
                tuple(value.shape)
            )
        )
    if value.numel() and (bool((value < 0).any()) or bool((value >= int(source_type.vocabulary_size)).any())):
        raise RuntimeError("categorical_one_hot input contains an out-of-vocabulary category")
    dtype_name = str(context.node.attrs.get("dtype", "float32"))
    dtype = getattr(torch, dtype_name)
    return torch.nn.functional.one_hot(
        value.to(dtype=torch.long),
        num_classes=int(source_type.vocabulary_size),
    ).to(dtype=dtype)


def _execute_categorical_embedding(context: RuntimeExecutionContext):
    value = context.resolved["x"][0]
    source_type = context.value_type(context.node.inputs["x"][0])
    if value.dim() != 1:
        raise RuntimeError(
            "categorical_embedding expects one category index per carrier item, received shape {}".format(
                tuple(value.shape)
            )
        )
    if value.numel() and (bool((value < 0).any()) or bool((value >= int(source_type.vocabulary_size)).any())):
        raise RuntimeError("categorical_embedding input contains an out-of-vocabulary category")
    return context.module(value.to(dtype=context.libraries["torch"].long))


def _execute_invariant_concat(context: RuntimeExecutionContext):
    return context.libraries["torch"].cat(tuple(context.resolved["xs"]), dim=-1)


def _execute_invariant_slice(context: RuntimeExecutionContext):
    value = context.resolved["x"][0]
    return value.narrow(-1, int(context.node.attrs["start"]), int(context.node.attrs["length"]))


def _execute_invariant_product(context: RuntimeExecutionContext):
    return context.resolved["left"][0] * context.resolved["right"][0]


def _execute_equivariant_channel_concat(context: RuntimeExecutionContext):
    torch = context.libraries["torch"]
    values = tuple(context.resolved["xs"])
    references = context.node.inputs["xs"]
    embeddings = [
        flat_to_embedding_tensor(value, context.value_type(reference).irreps)
        for value, reference in zip(values, references)
    ]
    output = torch.cat(embeddings, dim=-1)
    return embedding_tensor_to_flat(output, context.output_type.irreps)


def _execute_degreewise_invariant_scale(context: RuntimeExecutionContext):
    torch = context.libraries["torch"]
    weight = context.resolved["weight"][0]
    value = context.resolved["value"][0]
    value_type = context.value_type(context.node.inputs["value"][0])
    lmax, channels = uniform_so3_layout(value_type.irreps)
    embedding = flat_to_embedding_tensor(value, value_type.irreps)
    degree_weight = weight.reshape(weight.shape[0], lmax + 1, channels)
    expand_index = torch.zeros((lmax + 1) ** 2, dtype=torch.long, device=weight.device)
    for degree in range(lmax + 1):
        expand_index[degree ** 2 : (degree + 1) ** 2] = degree
    output = embedding * degree_weight.index_select(1, expand_index)
    return embedding_tensor_to_flat(output, context.output_type.irreps)


def _execute_fixed_gaussian_radial_basis(context: RuntimeExecutionContext):
    distance = context.resolved["distance"][0]
    return context.module(distance)


def _execute_axisymmetric_spherical_lift(context: RuntimeExecutionContext):
    return context.module(
        context.resolved["amplitudes"][0],
        context.resolved["direction"][0],
    )


def _execute_constant_scale(context: RuntimeExecutionContext):
    return context.resolved["x"][0] * float(context.node.attrs["factor"])


def _execute_equivariant_head_split(context: RuntimeExecutionContext):
    value = context.resolved["x"][0]
    source = context.value_type(context.node.inputs["x"][0])
    num_heads = int(context.node.attrs["num_heads"])
    if value.shape[-1] != source.irreps.dimension:
        raise RuntimeError(
            "head_split@2 expected flattened irrep dimension {} but received {}".format(
                source.irreps.dimension,
                value.shape[-1],
            )
        )
    pieces = []
    offset = 0
    leading = tuple(value.shape[:-1])
    for multiplicity, irrep in source.irreps:
        width = multiplicity * irrep.dimension
        per_head_width = (multiplicity // num_heads) * irrep.dimension
        block = value[..., offset : offset + width]
        pieces.append(block.reshape(*leading, num_heads, per_head_width))
        offset += width
    return context.libraries["torch"].cat(pieces, dim=-1)


def _execute_equivariant_head_merge(context: RuntimeExecutionContext):
    value = context.resolved["x"][0]
    source = context.value_type(context.node.inputs["x"][0])
    num_heads = int(source.axis_specs[0].size)
    if value.dim() < 2 or tuple(value.shape[-2:]) != (num_heads, source.irreps.dimension):
        raise RuntimeError(
            "head_merge@2 expected runtime tail {} but received {}".format(
                (num_heads, source.irreps.dimension),
                tuple(value.shape[-2:]),
            )
        )
    pieces = []
    offset = 0
    leading = tuple(value.shape[:-2])
    for multiplicity, irrep in source.irreps:
        width = multiplicity * irrep.dimension
        block = value[..., :, offset : offset + width]
        pieces.append(block.reshape(*leading, num_heads * width))
        offset += width
    return context.libraries["torch"].cat(pieces, dim=-1)


def _execute_module_x(context: RuntimeExecutionContext):
    return context.module(context.resolved["x"][0])


def _execute_gaussian_radial_basis(context: RuntimeExecutionContext):
    return context.module(context.resolved["distance"][0])


def _execute_tensor_product(context: RuntimeExecutionContext):
    return context.module(context.resolved["left"][0], context.resolved["right"][0])


def _execute_tensor_product_v2(context: RuntimeExecutionContext):
    return context.module(
        context.resolved["left"][0],
        context.resolved["right"][0],
        context.resolved["weight"][0],
    )


def _execute_concat(context: RuntimeExecutionContext):
    return context.libraries["torch"].cat(context.resolved["xs"], dim=-1)


def _execute_select(context: RuntimeExecutionContext):
    if isinstance(context.resolved["x"][0], SO3RuntimeValue):
        multiplicity = int(context.node.attrs.get("multiplicity", context.output_type.irreps.dimension))
        return select_runtime_scalars(context.resolved["x"][0], multiplicity)
    source = context.value_type(context.node.inputs["x"][0])
    return _index_select_irreps(
        context.resolved["x"][0],
        source.irreps,
        context.output_type.irreps,
        context.libraries["torch"],
    )


def _execute_irrep_select_v2(context: RuntimeExecutionContext):
    value = context.resolved["x"][0]
    source = context.value_type(context.node.inputs["x"][0])
    if value.shape[-1] != source.irreps.dimension:
        raise RuntimeError(
            "irrep_select@2 expected coefficient dimension {} but received {}".format(
                source.irreps.dimension,
                value.shape[-1],
            )
        )
    offsets = {}
    offset = 0
    for multiplicity, irrep in source.irreps:
        offsets[irrep] = (offset, multiplicity)
        offset += multiplicity * irrep.dimension
    pieces = []
    for raw in context.node.attrs["selections"]:
        irrep = Irreps.parse("1x{}".format(str(raw["irrep"])), source.group.family).terms[0][1]
        start = int(raw["start"])
        multiplicity = int(raw["multiplicity"])
        block_offset, _available = offsets[irrep]
        begin = block_offset + start * irrep.dimension
        end = begin + multiplicity * irrep.dimension
        pieces.append(value[..., begin:end])
    return context.libraries["torch"].cat(pieces, dim=-1)


def _execute_irrep_pad(context: RuntimeExecutionContext):
    torch = context.libraries["torch"]
    value = context.resolved["x"][0]
    source = context.value_type(context.node.inputs["x"][0])
    target = context.output_type
    if value.shape[-1] != source.irreps.dimension:
        raise RuntimeError(
            "irrep_pad expected coefficient dimension {} but received {}".format(
                source.irreps.dimension,
                value.shape[-1],
            )
        )
    source_blocks = {}
    source_offset = 0
    for multiplicity, irrep in source.irreps:
        width = multiplicity * irrep.dimension
        source_blocks[irrep] = (source_offset, multiplicity, width)
        source_offset += width
    pieces = []
    for target_multiplicity, irrep in target.irreps:
        source_block = source_blocks.get(irrep)
        if source_block is None:
            pieces.append(value.new_zeros(value.shape[:-1] + (target_multiplicity * irrep.dimension,)))
            continue
        offset, source_multiplicity, width = source_block
        copied = value[..., offset : offset + width]
        if source_multiplicity == target_multiplicity:
            pieces.append(copied)
            continue
        padding = value.new_zeros(
            value.shape[:-1] + ((target_multiplicity - source_multiplicity) * irrep.dimension,)
        )
        pieces.append(torch.cat((copied, padding), dim=-1))
    return torch.cat(pieces, dim=-1)


def _execute_residual(context: RuntimeExecutionContext):
    left = context.resolved["left"][0]
    right = context.resolved["right"][0]
    if isinstance(left, SO3RuntimeValue) or isinstance(right, SO3RuntimeValue):
        if not isinstance(left, SO3RuntimeValue) or not isinstance(right, SO3RuntimeValue):
            raise RuntimeError("residual addition cannot mix wrapped and flat SO(3) values")
        metadata_left = (left.irreps, left.lmax, left.mmax, left.channels, left.frame_id)
        metadata_right = (right.irreps, right.lmax, right.mmax, right.channels, right.frame_id)
        if metadata_left != metadata_right:
            raise RuntimeError("residual SO(3) runtime values have incompatible layouts")
        return left.replace_embedding(left.embedding + right.embedding)
    return left + right


def _execute_scalar_activation(context: RuntimeExecutionContext):
    torch = context.libraries["torch"]
    name = _activation_name(context.node)
    slope = float(context.node.attrs.get("negative_slope", 0.2))
    function = _activation(
        torch,
        name,
        slope,
    )
    output = function(context.resolved["x"][0])
    if _activation_normalization(context.node) == "second_moment":
        output = output * _second_moment_activation_scale(torch, name, slope)
    return output


def _execute_gate(context: RuntimeExecutionContext):
    return _gate(
        context.resolved["gates"][0],
        context.resolved["value"][0],
        context.output_type.irreps,
        context.libraries["torch"],
    )


def _execute_invariant_weight(context: RuntimeExecutionContext):
    weight = context.resolved["weight"][0]
    value = context.resolved["value"][0]
    if isinstance(value, SO3RuntimeValue):
        if weight.dim() != 2 or weight.shape[-1] != 1:
            raise RuntimeError("wrapped invariant weighting requires one scalar per item")
        return value.replace_embedding(value.embedding * weight.unsqueeze(1))
    return weight * value


def _execute_invariant_scale(context: RuntimeExecutionContext):
    weight = context.resolved["weight"][0]
    value = context.resolved["value"][0]
    weight_type = context.value_type(context.node.inputs["weight"][0])
    value_type = context.value_type(context.node.inputs["value"][0])
    if isinstance(value, SO3RuntimeValue):
        if len(weight_type.axis_specs) != 1 or weight_type.axis_specs[0].role.value != "head":
            if weight.numel() != weight.shape[0]:
                raise RuntimeError("SO3 invariant scaling requires one scalar per carrier item")
            return value.replace_embedding(value.embedding * weight.reshape(weight.shape[0], 1, 1))
        head_count = int(weight_type.axis_specs[0].size)
        if value.channels % head_count != 0 or weight.shape[-1] != head_count:
            raise RuntimeError("SO3 head scaling received incompatible head and channel sizes")
        per_head = value.channels // head_count
        expanded = value.embedding.view(
            value.embedding.shape[0], value.embedding.shape[1], head_count, per_head
        )
        scaled = expanded * weight.view(weight.shape[0], 1, head_count, 1)
        return value.replace_embedding(scaled.reshape_as(value.embedding))
    if weight_type.axis_specs:
        if value.dim() < 2 or weight.shape[-1] != value.shape[-2]:
            raise RuntimeError(
                "headwise invariant_scale expected weight [..., H] and value [..., H, D], received {} and {}".format(
                    tuple(weight.shape),
                    tuple(value.shape),
                )
            )
        return value * weight.unsqueeze(-1)
    if value_type.axis_specs and value_type.axis_specs[0].role.value == "head":
        return value * weight.unsqueeze(-1)
    return value * weight


def _execute_compatibility(context: RuntimeExecutionContext):
    return (context.resolved["query"][0] * context.resolved["key"][0]).sum(dim=-1, keepdim=True)


def _execute_segment_softmax(context: RuntimeExecutionContext):
    return _segment_softmax(
        context.resolved["logits"][0],
        context.graph_context["edge_dst"],
        context.libraries["torch"],
    )


def _execute_explicit_segment_softmax(context: RuntimeExecutionContext):
    indices, _target_size = _index_map_payload(context.resolved["index"][0], context.node.op)
    return _segment_softmax(
        context.resolved["logits"][0],
        indices,
        context.libraries["torch"],
    )


def _execute_v3_graph_softmax(context: RuntimeExecutionContext):
    indices, target_size = _index_map_payload(context.resolved["index"][0], context.node.op)
    return context.module(
        context.resolved["logits"][0],
        indices,
        target_size,
        context.resolved["exp_rescale"][0],
    )


def _execute_edge_lift(context: RuntimeExecutionContext):
    endpoint = str(context.node.attrs.get("endpoint", "source"))
    if endpoint not in ("source", "target"):
        raise RuntimeError("edge_lift endpoint must be source or target")
    key = "edge_src" if endpoint == "source" else "edge_dst"
    context.require_context(key)
    return context.resolved["x"][0].index_select(0, context.graph_context[key])


def _index_map_payload(value, op: str):
    if not isinstance(value, MappingABC):
        raise RuntimeError("{} index input must be a mapping with indices and target_size".format(op))
    if "indices" not in value or "target_size" not in value:
        raise RuntimeError("{} index input requires indices and target_size".format(op))
    return value["indices"], int(value["target_size"])


def _execute_endpoint_gather(context: RuntimeExecutionContext):
    indices, _target_size = _index_map_payload(context.resolved["index"][0], context.node.op)
    return context.resolved["x"][0].index_select(0, indices)


def _execute_segment_reduce(context: RuntimeExecutionContext):
    indices = context.graph_context["edge_dst"]
    count = context.graph_context.get("num_nodes")
    if count is None:
        count = int(indices.max().item()) + 1 if indices.numel() else 0
    return _segment_reduce(
        context.resolved["x"][0],
        indices,
        int(count),
        context.libraries["torch"],
        mean=context.node.op.endswith("segment_mean") or context.node.op.endswith("segment_mean@1"),
    )


def _execute_explicit_segment_reduce(context: RuntimeExecutionContext):
    indices, target_size = _index_map_payload(context.resolved["index"][0], context.node.op)
    normalization = str(context.node.attrs.get("normalization", "none"))
    return _segment_reduce(
        context.resolved["x"][0],
        indices,
        target_size,
        context.libraries["torch"],
        mean=str(context.node.attrs.get("reduce", "sum")) == "mean",
        target_cardinality=normalization == "target_cardinality",
    )


def _execute_global_pool(context: RuntimeExecutionContext):
    batch = context.graph_context["batch"]
    count = context.graph_context.get("num_graphs")
    if count is None:
        count = int(batch.max().item()) + 1 if batch.numel() else 0
    return _segment_reduce(
        context.resolved["x"][0],
        batch,
        int(count),
        context.libraries["torch"],
        mean=str(context.node.attrs.get("reduce", "sum")) == "mean",
    )


def _execute_relative_position(context: RuntimeExecutionContext):
    source = context.resolved["source"][0].index_select(0, context.graph_context["edge_src"])
    target = context.resolved["target"][0].index_select(0, context.graph_context["edge_dst"])
    return target - source


def _gather_displacement_endpoints(context: RuntimeExecutionContext):
    source_index, _ = _index_map_payload(context.resolved["source_index"][0], context.node.op)
    target_index, _ = _index_map_payload(context.resolved["target_index"][0], context.node.op)
    positions = context.resolved["positions"][0]
    return positions.index_select(0, source_index), positions.index_select(0, target_index)


def _execute_relative_displacement_v2(context: RuntimeExecutionContext):
    source, target = _gather_displacement_endpoints(context)
    return target - source


def _execute_relative_displacement_v3(context: RuntimeExecutionContext):
    source, target = _gather_displacement_endpoints(context)
    return source - target


def _execute_periodic_displacement(context: RuntimeExecutionContext):
    source, target = _gather_displacement_endpoints(context)
    lattice = context.resolved["lattice"][0]
    shift = context.resolved["lattice_shift"][0].to(dtype=lattice.dtype)
    if lattice.dim() == 2:
        offset = shift @ lattice
    elif lattice.dim() == 3 and lattice.shape[0] == shift.shape[0]:
        offset = context.libraries["torch"].bmm(shift.unsqueeze(1), lattice).squeeze(1)
    else:
        raise RuntimeError("periodic_displacement lattice must have shape [3,3] or [edge,3,3]")
    return target + offset - source


def _execute_periodic_displacement_v2(context: RuntimeExecutionContext):
    source, target = _gather_displacement_endpoints(context)
    lattice = context.resolved["lattice"][0]
    shift = context.resolved["lattice_shift"][0].to(dtype=lattice.dtype)
    if lattice.dim() == 2:
        offset = shift @ lattice
    elif lattice.dim() == 3 and lattice.shape[0] == shift.shape[0]:
        offset = context.libraries["torch"].bmm(shift.unsqueeze(1), lattice).squeeze(1)
    else:
        raise RuntimeError("periodic_displacement@2 lattice must have shape [3,3] or [edge,3,3]")
    return source - target + offset


def _execute_distance(context: RuntimeExecutionContext):
    return context.libraries["torch"].linalg.vector_norm(
        context.resolved["vector"][0], dim=-1, keepdim=True
    )


def _execute_radial_basis(context: RuntimeExecutionContext):
    torch = context.libraries["torch"]
    distance = context.resolved["distance"][0]
    count = int(context.node.attrs["num_basis"])
    cutoff = float(context.node.attrs.get("cutoff", 5.0))
    centers = torch.linspace(0.0, cutoff, count, dtype=distance.dtype, device=distance.device)
    width = float(context.node.attrs.get("width", cutoff / max(count - 1, 1)))
    return torch.exp(-((distance - centers) / width) ** 2)


def _execute_cutoff_envelope(context: RuntimeExecutionContext):
    """Apply the smooth polynomial envelope used by directional GNNs."""

    torch = context.libraries["torch"]
    values = context.resolved["x"][0]
    cutoff = float(context.node.attrs.get("cutoff", 5.0))
    order = int(context.node.attrs.get("order", 5))
    scaled = (values / cutoff).clamp_min(0.0)
    coefficient_a = (order + 1) * (order + 2) / 2.0
    coefficient_b = order * (order + 2)
    coefficient_c = order * (order + 1) / 2.0
    envelope = (
        1.0
        - coefficient_a * scaled.pow(order)
        + coefficient_b * scaled.pow(order + 1)
        - coefficient_c * scaled.pow(order + 2)
    )
    return torch.where(scaled < 1.0, envelope, torch.zeros_like(envelope))


def _execute_spherical_harmonics(context: RuntimeExecutionContext):
    return context.libraries["o3"].spherical_harmonics(
        to_e3nn_irreps(context.output_type.irreps),
        context.resolved["direction"][0],
        normalize=True,
        normalization=str(context.node.attrs.get("normalization", "component")),
    )


def _execute_to_edge_frame(context: RuntimeExecutionContext):
    source = context.value_type(context.node.inputs["x"][0])
    lmax, _channels = uniform_so3_layout(source.irreps)
    return to_edge_frame_value(
        context.resolved["x"][0],
        source.irreps,
        context.graph_context["edge_vectors"],
        str(context.node.attrs.get("frame_id", context.node.id)),
        _v2_modules(context.libraries),
        int(context.node.attrs.get("mmax", lmax)),
    )


def _execute_v3_to_edge_frame(context: RuntimeExecutionContext):
    return context.module(
        context.resolved["x"][0],
        context.resolved["direction"][0],
        str(context.node.attrs.get("frame_id", context.node.id)),
    )


def _execute_from_edge_frame(context: RuntimeExecutionContext):
    return from_edge_frame_value(
        context.resolved["x"][0],
        context.output_type.irreps,
        str(context.node.attrs.get("frame_id", "")),
        _v2_modules(context.libraries),
    )


def _execute_v3_from_edge_frame(context: RuntimeExecutionContext):
    return context.module(
        context.resolved["x"][0],
        str(context.node.attrs.get("frame_id", "")),
    )


def _execute_v3_gated_swiglu_merge(context: RuntimeExecutionContext):
    return context.module(
        context.resolved["x"][0],
        context.resolved["scalars"][0],
    )


def _grid_runtime(context: RuntimeExecutionContext, port: str) -> GridRuntimeValue:
    value = context.resolved[port][0]
    if not isinstance(value, GridRuntimeValue):
        raise RuntimeError("{} expected GridRuntimeValue on {}".format(context.node.op, port))
    return value


def _grid_runtime_output(value: GridRuntimeValue, tensor, output_type):
    return value.replace_values(tensor, source_irreps=output_type.source_irreps)


def _execute_grid_unproject(context: RuntimeExecutionContext):
    return unproject_v3_grid_value(_grid_runtime(context, "x"), context.output_type.irreps)


def _execute_grid_split(context: RuntimeExecutionContext):
    value = _grid_runtime(context, "x")
    left_channels = int(context.output_types["left"].channels)
    left, right = context.libraries["torch"].split(
        value.values,
        [left_channels, int(context.output_types["right"].channels)],
        dim=-1,
    )
    return {
        "left": _grid_runtime_output(value, left, context.output_types["left"]),
        "right": _grid_runtime_output(value, right, context.output_types["right"]),
    }


def _execute_grid_concat(context: RuntimeExecutionContext):
    values = tuple(context.resolved["xs"])
    if not values or any(not isinstance(value, GridRuntimeValue) for value in values):
        raise RuntimeError("grid_concat requires GridRuntimeValue inputs")
    tensor = context.libraries["torch"].cat(tuple(value.values for value in values), dim=-1)
    return _grid_runtime_output(values[0], tensor, context.output_type)


def _execute_grid_pointwise_activation(context: RuntimeExecutionContext):
    value = _grid_runtime(context, "x")
    activation = str(context.node.attrs.get("activation", "silu"))
    functional = context.libraries["torch"].nn.functional
    if activation == "identity":
        tensor = value.values
    elif activation == "silu":
        tensor = functional.silu(value.values)
    elif activation == "sigmoid":
        tensor = value.values.sigmoid()
    elif activation == "square":
        tensor = value.values.square()
    else:
        raise RuntimeError("unsupported grid activation {}".format(activation))
    return _grid_runtime_output(value, tensor, context.output_type)


def _execute_grid_pointwise_product(context: RuntimeExecutionContext):
    left = context.resolved["left"][0]
    right = context.resolved["right"][0]
    base = left if isinstance(left, GridRuntimeValue) else right
    if not isinstance(base, GridRuntimeValue):
        raise RuntimeError("grid product requires one GridRuntimeValue")
    left_tensor = left.values if isinstance(left, GridRuntimeValue) else left
    right_tensor = right.values if isinstance(right, GridRuntimeValue) else right

    # Official SO3Grid values are stored as [carrier, grid_point, channel].
    # A typed invariant gate is stored as [carrier, *feature_axes].  PyTorch
    # would align [carrier, channel] with the final two grid dimensions and
    # therefore compare carrier against grid_point.  Flatten only the declared
    # invariant feature axes and insert singleton sampling axes explicitly.
    def broadcast_invariant(tensor):
        if tensor.ndim < 1 or int(tensor.shape[0]) != int(base.values.shape[0]):
            raise RuntimeError("grid broadcast invariant has an incompatible carrier dimension")
        feature_count = int(tensor.numel() // max(1, int(tensor.shape[0])))
        if feature_count not in (1, base.channels):
            raise RuntimeError(
                "grid broadcast invariant must contain one or {} values per carrier".format(
                    base.channels
                )
            )
        shape = [int(tensor.shape[0])]
        shape.extend([1] * (base.values.ndim - 2))
        shape.append(feature_count)
        return tensor.reshape(shape)

    if not isinstance(left, GridRuntimeValue):
        left_tensor = broadcast_invariant(left_tensor)
    if not isinstance(right, GridRuntimeValue):
        right_tensor = broadcast_invariant(right_tensor)
    tensor = left_tensor * right_tensor
    return _grid_runtime_output(base, tensor, context.output_type)


def _execute_grid_channel_linear(context: RuntimeExecutionContext):
    value = _grid_runtime(context, "x")
    return _grid_runtime_output(value, context.module(value.values), context.output_type)


def _execute_grid_dropout(context: RuntimeExecutionContext):
    value = _grid_runtime(context, "x")
    probability = float(context.node.attrs.get("probability", 0.0))
    tensor = context.libraries["torch"].nn.functional.dropout(
        value.values,
        p=probability,
        training=context.training,
    )
    return _grid_runtime_output(value, tensor, context.output_type)


def _execute_so2_convolution(context: RuntimeExecutionContext):
    return context.module(context.resolved["x"][0])


def _execute_s2_activation(context: RuntimeExecutionContext):
    return context.module(context.resolved["x"][0])


def _execute_separable_s2_activation(context: RuntimeExecutionContext):
    return context.module(
        context.resolved["x"][0],
        context.resolved["scalars"][0],
    )


def _execute_dropout(context: RuntimeExecutionContext):
    probability = float(context.node.attrs.get("p", 0.0))
    op = LoweringRuleRegistry.qualify(context.node.op)
    return _irrep_dropout(
        context.resolved["x"][0],
        context.output_type.irreps,
        probability,
        context.training,
        context.libraries["torch"],
        whole_value=op == "core.stochastic_depth@1",
    )


def _execute_graph_stochastic_depth(context: RuntimeExecutionContext):
    values = context.resolved["x"][0]
    probability = float(context.node.attrs.get("p", 0.0))
    if not context.training or probability <= 0.0:
        return values
    indices, target_size = _index_map_payload(context.resolved["batch"][0], context.node.op)
    if indices.dim() != 1 or int(indices.shape[0]) != int(values.shape[0]):
        raise RuntimeError(
            "{} batch indices must be rank-1 with one entry per carrier item".format(context.node.op)
        )
    if indices.numel() == 0:
        raise RuntimeError("{} does not admit empty carrier batches".format(context.node.op))
    if int(indices.min().item()) != 0 or int(indices.max().item()) + 1 != int(target_size):
        raise RuntimeError(
            "{} requires contiguous graph ids covering target_size".format(context.node.op)
        )
    if int(context.libraries["torch"].unique(indices).numel()) != int(target_size):
        raise RuntimeError("{} batch payload contains an empty graph target".format(context.node.op))
    keep_probability = 1.0 - probability
    shape = (int(target_size),) + (1,) * (values.dim() - 1)
    random_tensor = keep_probability + context.libraries["torch"].rand(
        shape,
        dtype=values.dtype,
        device=values.device,
    )
    random_tensor.floor_()
    graph_scale = random_tensor.div(keep_probability)
    return values * graph_scale.index_select(0, indices)


def _execute_scalar_dropout(context: RuntimeExecutionContext):
    values = context.resolved["x"][0]
    probability = float(context.node.attrs.get("p", 0.0))
    return context.libraries["torch"].nn.functional.dropout(
        values,
        p=probability,
        training=context.training,
        inplace=False,
    )


def _execute_equivariant_dropout(context: RuntimeExecutionContext):
    values = context.resolved["x"][0]
    probability = float(context.node.attrs.get("p", 0.0))
    if not context.training or probability <= 0.0:
        return values
    source = context.value_type(context.node.inputs["x"][0])
    torch = context.libraries["torch"]
    mask = torch.ones(
        (values.shape[0], sum(multiplicity for multiplicity, _irrep in source.irreps)),
        dtype=values.dtype,
        device=values.device,
    )
    mask = torch.nn.functional.dropout(mask, p=probability, training=True, inplace=True)
    pieces = []
    value_offset = 0
    mask_offset = 0
    for multiplicity, irrep in source.irreps:
        width = multiplicity * irrep.dimension
        block = values.narrow(-1, value_offset, width).reshape(
            values.shape[:-1] + (multiplicity, irrep.dimension)
        )
        block_mask = mask.narrow(-1, mask_offset, multiplicity).unsqueeze(-1)
        pieces.append((block * block_mask).reshape(values.shape[:-1] + (width,)))
        value_offset += width
        mask_offset += multiplicity
    return torch.cat(pieces, dim=-1)


def build_e3nn_lowering_registry(
    equiformer_v2_root: str = "",
    equiformer_v3_root: str = "",
) -> LoweringRuleRegistry:
    """Return the first compositional lowering rule set for core primitives."""

    registry = LoweringRuleRegistry(
        "e3nn_graph",
        (
            DependencyRequirement("torch"),
            DependencyRequirement("e3nn"),
        ),
    )

    def add(
        primitive,
        executor,
        module_builder=None,
        validator=None,
        required_context_keys=(),
        dependencies=(),
        exactness="numerically_tested",
        description="",
        runtime_kind_rule=None,
        post_build_initializer=None,
    ):
        registry.register(
            LoweringRule(
                primitive=primitive,
                executor=executor,
                module_builder=module_builder,
                validator=validator,
                dependencies=tuple(dependencies),
                required_context_keys=tuple(required_context_keys),
                exactness=exactness,
                description=description,
                runtime_kind_rule=runtime_kind_rule,
                post_build_initializer=post_build_initializer,
            )
        )

    add("core.identity@1", _execute_identity, exactness="constructive_exact", runtime_kind_rule=_runtime_preserve("x"))
    add(
        "core.categorical_remap@1",
        _execute_categorical_remap,
        exactness="constructive_exact",
        runtime_kind_rule=_runtime_categorical_remap,
    )
    add(
        "core.categorical_one_hot@1",
        _execute_categorical_one_hot,
        exactness="constructive_exact",
        runtime_kind_rule=_runtime_categorical_one_hot,
    )
    add(
        "core.categorical_embedding@1",
        _execute_categorical_embedding,
        _build_categorical_embedding,
        exactness="constructive_exact",
        description="Typed categorical lookup with a checkpoint-compatible [vocabulary, channel] table.",
        runtime_kind_rule=_runtime_categorical_embedding,
    )
    add(
        "core.categorical_embedding@2",
        _execute_categorical_embedding,
        _build_categorical_embedding,
        exactness="constructive_exact",
        description="Categorical embedding with an explicit scheduled uniform-overwrite initialization barrier.",
        runtime_kind_rule=_runtime_categorical_embedding,
        post_build_initializer=_initialize_categorical_embedding_v2,
    )
    add(
        "core.invariant_concat@1",
        _execute_invariant_concat,
        exactness="constructive_exact",
        description="Feature-axis concatenation of complete invariant values.",
    )
    add("core.invariant_slice@1", _execute_invariant_slice, exactness="constructive_exact", runtime_kind_rule=_runtime_preserve("x"))
    add("core.invariant_product@1", _execute_invariant_product, exactness="constructive_exact", runtime_kind_rule=_runtime_dense_inputs)
    add("core.equivariant_channel_concat@1", _execute_equivariant_channel_concat, exactness="constructive_exact", runtime_kind_rule=_runtime_dense_variadic("xs"))
    add("core.degreewise_invariant_scale@1", _execute_degreewise_invariant_scale, exactness="constructive_exact", runtime_kind_rule=_runtime_dense_inputs)
    add("core.constant_scale@1", _execute_constant_scale, exactness="constructive_exact", runtime_kind_rule=_runtime_preserve("x"))
    add("core.flatten_invariant_axes@1", _execute_identity, exactness="constructive_exact", runtime_kind_rule=_runtime_preserve("x"))
    add("core.irrep_linear@1", _execute_module_x, _build_linear, exactness="library_exact")
    add("core.irrep_linear@2", _execute_module_x, _build_linear_rs, exactness="library_exact")
    add("core.irrep_pad@1", _execute_irrep_pad, exactness="constructive_exact", runtime_kind_rule=_runtime_preserve("x"))
    add("core.scalar_linear@1", _execute_module_x, _build_scalar_linear, exactness="constructive_exact")
    add("core.scalar_linear@2", _execute_module_x, _build_scalar_linear_v2, exactness="constructive_exact")
    add(
        "core.scalar_linear@3",
        _execute_module_x,
        _build_scalar_linear,
        exactness="constructive_exact",
        description="Torch Linear with graph-level deferred official V3 uniform fan-in initialization.",
        post_build_initializer=_initialize_scalar_linear_v3,
    )
    add(
        "core.scalar_linear@4",
        _execute_module_x,
        _build_scalar_linear,
        exactness="constructive_exact",
        description="Torch Linear preserving its default weight while the official V3 initializer zeroes only bias.",
        post_build_initializer=_initialize_scalar_linear_v4,
    )
    add("core.scalar_layer_norm@1", _execute_module_x, _build_scalar_layer_norm, exactness="constructive_exact")
    add("core.scalar_offset@1", _execute_module_x, _build_scalar_offset, exactness="constructive_exact")
    add("core.head_split@1", _execute_identity, exactness="constructive_exact", runtime_kind_rule=_runtime_preserve("x"))
    add("core.head_merge@1", _execute_identity, exactness="constructive_exact", runtime_kind_rule=_runtime_preserve("x"))
    add(
        "core.head_split@2",
        _execute_equivariant_head_split,
        exactness="constructive_exact",
        runtime_kind_rule=_runtime_to_equivariant_heads,
    )
    add(
        "core.head_merge@2",
        _execute_equivariant_head_merge,
        exactness="constructive_exact",
        runtime_kind_rule=_runtime_from_equivariant_heads,
    )
    add(
        "core.headwise_scalar_contraction@1",
        _execute_module_x,
        _build_headwise_scalar_contraction,
        exactness="constructive_exact",
    )
    add(
        "core.headwise_scalar_contraction@2",
        _execute_module_x,
        _build_headwise_scalar_contraction_v2,
        exactness="constructive_exact",
        runtime_kind_rule=_runtime_contract_equivariant_heads,
    )
    add(
        "core.headwise_scalar_contraction@3",
        _execute_module_x,
        _build_headwise_scalar_contraction_v3,
        exactness="constructive_exact",
    )
    add("core.change_multiplicity@1", _execute_module_x, _build_linear, exactness="library_exact")
    add("core.irrep_concat@1", _execute_concat, exactness="constructive_exact")
    add("core.irrep_slice@1", _execute_select, exactness="constructive_exact")
    add("core.irrep_select@2", _execute_irrep_select_v2, exactness="constructive_exact", runtime_kind_rule=_runtime_preserve("x"))
    add("core.residual_add@1", _execute_residual, exactness="constructive_exact", runtime_kind_rule=_runtime_same_inputs)
    add("core.residual_add@2", _execute_residual, exactness="constructive_exact", runtime_kind_rule=_runtime_same_inputs)
    add("core.tensor_product@1", _execute_tensor_product, _build_tensor_product, exactness="library_exact")
    add("core.tensor_product@2", _execute_tensor_product_v2, _build_tensor_product_v2, exactness="library_exact")
    add("core.tensor_product@3", _execute_tensor_product_v2, _build_tensor_product_v3, exactness="library_exact")
    add("core.tensor_product@4", _execute_tensor_product, _build_tensor_product_v4, exactness="library_exact")
    add("core.tensor_product@5", _execute_tensor_product, _build_tensor_product_v5, exactness="library_exact")
    add(
        "core.scalar_activation@1",
        _execute_scalar_activation,
        validator=_validate_activation,
        runtime_kind_rule=_runtime_preserve("x"),
    )
    add("core.norm_activation@1", _execute_module_x, _build_norm_activation, _validate_activation)
    add("core.gate@1", _execute_gate)
    add("core.equivariant_norm@1", _execute_module_x, _build_equivariant_norm)
    add(
        "core.irrep_layer_norm@1",
        _execute_module_x,
        _build_irrep_layer_norm,
        exactness="constructive_exact",
    )
    add("core.invariant_weight@1", _execute_invariant_weight, exactness="constructive_exact", runtime_kind_rule=_runtime_invariant_scale)
    add("core.invariant_scale@1", _execute_invariant_scale, exactness="constructive_exact", runtime_kind_rule=_runtime_invariant_scale)
    add("core.invariant_scale@2", _execute_invariant_scale, exactness="constructive_exact", runtime_kind_rule=_runtime_invariant_scale)
    add("core.invariant_compatibility@1", _execute_compatibility, exactness="constructive_exact")
    add(
        "core.segment_softmax@1",
        _execute_segment_softmax,
        required_context_keys=("edge_dst",),
        description="Pure-PyTorch segmented softmax; no torch_geometric dependency.",
    )
    add(
        "core.segment_softmax@2",
        _execute_explicit_segment_softmax,
        exactness="constructive_exact",
        description="Explicit-index segmented softmax; no hidden edge_dst context.",
        runtime_kind_rule=_runtime_explicit_segment_softmax,
    )
    add(
        "core.segment_softmax@3",
        _execute_v3_graph_softmax,
        _build_v3_graph_softmax,
        exactness="constructive_exact",
        description="Official V3 detached-max GraphSoftmax with envelope rescaling, mask dropout, softcap, and epsilon.",
        runtime_kind_rule=_runtime_v3_graph_softmax,
    )
    add("core.edge_lift@1", _execute_edge_lift, exactness="constructive_exact")
    add(
        "core.endpoint_gather@1",
        _execute_endpoint_gather,
        exactness="constructive_exact",
        description="Explicit typed index-map gather; does not read edge_src or edge_dst from graph context.",
        runtime_kind_rule=_runtime_explicit_index,
    )
    add(
        "core.endpoint_gather@2",
        _execute_endpoint_gather,
        exactness="constructive_exact",
        description="Explicit endpoint gather preserving dense or categorical runtime kind.",
        runtime_kind_rule=_runtime_endpoint_gather_v2,
    )
    add(
        "core.segment_sum@1",
        _execute_segment_reduce,
        required_context_keys=("edge_dst",),
        exactness="constructive_exact",
    )
    add(
        "core.segment_mean@1",
        _execute_segment_reduce,
        required_context_keys=("edge_dst",),
        exactness="constructive_exact",
    )
    add(
        "core.segment_reduce@1",
        _execute_explicit_segment_reduce,
        exactness="constructive_exact",
        description="Explicit typed segment reduction with runtime indices and target_size.",
        runtime_kind_rule=_runtime_explicit_index,
    )
    add(
        "core.global_pool@1",
        _execute_global_pool,
        required_context_keys=("batch",),
        exactness="constructive_exact",
    )
    add("core.select_scalars@1", _execute_select, exactness="constructive_exact", runtime_kind_rule=_runtime_select_scalars)
    add("core.select_scalars@2", _execute_select, exactness="constructive_exact", runtime_kind_rule=_runtime_select_scalars)
    add(
        "core.relative_position@1",
        _execute_relative_position,
        required_context_keys=("edge_src", "edge_dst"),
        exactness="constructive_exact",
    )
    add(
        "core.relative_displacement@2",
        _execute_relative_displacement_v2,
        exactness="constructive_exact",
        description="Explicit target-source displacement from affine points and endpoint maps.",
        runtime_kind_rule=_runtime_relative_displacement,
    )
    add(
        "core.relative_displacement@3",
        _execute_relative_displacement_v3,
        exactness="constructive_exact",
        description="Explicit source-target displacement matching official fairchem V3.",
        runtime_kind_rule=_runtime_relative_displacement,
    )
    add(
        "core.periodic_displacement@1",
        _execute_periodic_displacement,
        exactness="constructive_exact",
        description="Explicit target + shift @ lattice - source periodic displacement.",
        runtime_kind_rule=_runtime_periodic_displacement,
    )
    add(
        "core.periodic_displacement@2",
        _execute_periodic_displacement_v2,
        exactness="constructive_exact",
        description="Explicit source + source-image shift @ lattice - target fairchem displacement.",
        runtime_kind_rule=_runtime_periodic_displacement,
    )
    add("core.distance@1", _execute_distance, exactness="constructive_exact")
    add("core.distance@2", _execute_distance, exactness="constructive_exact")
    add("core.radial_basis@1", _execute_radial_basis, validator=_validate_radial_basis, exactness="experimental")
    add(
        "core.fixed_gaussian_radial_basis@1",
        _execute_fixed_gaussian_radial_basis,
        _build_fixed_gaussian_radial_basis,
        exactness="constructive_exact",
        description="Fixed linspace Gaussian smearing with explicit construction-dtype and width-scalar semantics.",
    )
    add(
        "core.gaussian_radial_basis@1",
        _execute_gaussian_radial_basis,
        _build_gaussian_radial_basis,
        exactness="constructive_exact",
    )
    add("core.cutoff_envelope@1", _execute_cutoff_envelope, validator=_validate_cutoff, exactness="experimental")
    add(
        "core.cutoff_envelope@2",
        _execute_cutoff_envelope,
        validator=_validate_cutoff,
        exactness="constructive_exact",
        description="Dimensionless official V3 polynomial distance envelope.",
    )
    add("core.spherical_harmonics@1", _execute_spherical_harmonics, exactness="library_exact")
    add("core.stochastic_depth@1", _execute_dropout)
    add(
        "core.graph_stochastic_depth@1",
        _execute_graph_stochastic_depth,
        exactness="constructive_exact",
        description="Explicit batch-indexed graph-shared stochastic depth.",
        runtime_kind_rule=_runtime_graph_stochastic_depth,
    )
    add("core.scalar_dropout@1", _execute_scalar_dropout, exactness="constructive_exact")
    add("core.equivariant_dropout@1", _execute_equivariant_dropout, exactness="constructive_exact")
    add("core.invariant_dropout@1", _execute_dropout)
    v2_dependency = (_v2_dependency(equiformer_v2_root),)
    add(
        "core.to_edge_frame@1",
        _execute_to_edge_frame,
        validator=_validate_to_edge_frame,
        required_context_keys=("edge_vectors",),
        dependencies=v2_dependency,
        exactness="library_exact",
        description="Independent official Wigner-D rotation into an edge-local frame.",
        runtime_kind_rule=_runtime_to_edge_frame,
    )
    add(
        "core.from_edge_frame@1",
        _execute_from_edge_frame,
        validator=_validate_v2_layout,
        dependencies=v2_dependency,
        exactness="library_exact",
        description="Inverse of the exact edge-frame rotation carried by the runtime value.",
        runtime_kind_rule=_runtime_from_edge_frame,
    )
    add(
        "core.so2_convolution@1",
        _execute_so2_convolution,
        _build_so2_convolution,
        _validate_so2,
        dependencies=v2_dependency,
        exactness="library_exact",
        runtime_kind_rule=_runtime_require_edge_frame,
    )
    add(
        "core.so2_linear@1",
        _execute_module_x,
        _build_v3_so2_linear,
        _validate_so2,
        exactness="constructive_exact",
        description="Pure PyTorch real-form SO(2) linear intertwiner in m-primary edge-frame storage.",
        runtime_kind_rule=_runtime_v3_so2_linear,
    )
    add(
        "core.so2_linear@2",
        _execute_module_x,
        _build_v3_so2_linear,
        _validate_so2,
        exactness="constructive_exact",
        description="SO(2) linear intertwiner with a separately typed invariant extra-m0 output.",
        runtime_kind_rule=_runtime_v3_so2_linear,
    )
    add(
        "core.s2_activation@1",
        _execute_s2_activation,
        _build_plain_s2_activation,
        _validate_s2,
        dependencies=v2_dependency,
        exactness="library_exact",
        runtime_kind_rule=_runtime_preserve("x"),
    )
    add(
        "core.separable_s2_activation@1",
        _execute_separable_s2_activation,
        _build_separable_s2_activation,
        _validate_s2,
        dependencies=v2_dependency,
        exactness="library_exact",
        runtime_kind_rule=_runtime_separable_s2,
    )
    v3_dependency = (_v3_dependency(equiformer_v3_root),)
    add(
        "core.grid_project@1",
        _execute_module_x,
        _build_v3_grid_project,
        dependencies=v3_dependency,
        exactness="library_exact",
        description="Low-level SO3Grid projection into an explicit GridRuntimeValue.",
        runtime_kind_rule=_runtime_to_grid,
    )
    add(
        "core.grid_unproject@1",
        _execute_grid_unproject,
        dependencies=v3_dependency,
        exactness="library_exact",
        description="Unproject through the inverse matrix created at the typed grid boundary.",
        runtime_kind_rule=_runtime_from_grid,
    )
    add("core.grid_split@1", _execute_grid_split, exactness="constructive_exact", runtime_kind_rule=_runtime_grid_preserve)
    add("core.grid_concat@1", _execute_grid_concat, exactness="constructive_exact", runtime_kind_rule=_runtime_grid_variadic)
    add(
        "core.grid_pointwise_activation@1",
        _execute_grid_pointwise_activation,
        exactness="empirical_finite_grid",
        runtime_kind_rule=_runtime_grid_preserve,
    )
    add(
        "core.grid_pointwise_product@1",
        _execute_grid_pointwise_product,
        exactness="empirical_finite_grid",
        runtime_kind_rule=_runtime_grid_product,
    )
    add(
        "core.grid_channel_linear@1",
        _execute_grid_channel_linear,
        _build_grid_channel_linear,
        exactness="constructive_exact",
        runtime_kind_rule=_runtime_grid_preserve,
    )
    add(
        "core.grid_dropout@1",
        _execute_grid_dropout,
        exactness="stochastic_empirical",
        runtime_kind_rule=_runtime_grid_preserve,
    )
    add(
        "core.to_edge_frame@2",
        _execute_v3_to_edge_frame,
        _build_v3_edge_frame_rotation,
        dependencies=v3_dependency,
        exactness="library_exact",
        description="Official V3 SO3Rotation into explicit m-primary edge-frame storage.",
        runtime_kind_rule=_runtime_to_edge_frame_v3,
    )
    add(
        "core.from_edge_frame@2",
        _execute_v3_from_edge_frame,
        _build_v3_edge_frame_inverse,
        dependencies=v3_dependency,
        exactness="library_exact",
        description="Official V3 inverse Wigner rotation from m-primary edge-frame storage.",
        runtime_kind_rule=_runtime_from_edge_frame,
    )
    add(
        "core.axisymmetric_spherical_lift@1",
        _execute_axisymmetric_spherical_lift,
        _build_v3_axisymmetric_spherical_lift,
        dependencies=v3_dependency,
        exactness="library_exact",
        description="Official V3 inverse-Wigner lift of per-degree m=0 edge amplitudes.",
    )
    add(
        "core.s2_swiglu@1",
        _execute_module_x,
        _build_v3_s2_swiglu,
        dependencies=v3_dependency,
        exactness="library_exact",
        description="Official Equiformer V3 S2 projection/SwiGLU/projection operator.",
        runtime_kind_rule=_runtime_preserve("x"),
    )
    add(
        "core.s2_gated_swiglu_merge@1",
        _execute_v3_gated_swiglu_merge,
        _build_v3_gated_swiglu_merge,
        dependencies=v3_dependency,
        exactness="library_exact",
        description="Official Equiformer V3 sep-merge_gates2_swiglu edge-frame activation.",
        runtime_kind_rule=_runtime_require_edge_frame,
    )
    add(
        "core.edge_frame_gate_activation@1",
        _execute_v3_gated_swiglu_merge,
        _build_v3_edge_frame_gate_activation,
        dependencies=v3_dependency,
        exactness="library_exact",
        description="Official V3 GateActivation over an m-primary edge-frame value.",
        runtime_kind_rule=_runtime_edge_frame_with_scalars,
    )
    add(
        "core.so3_linear@1",
        _execute_module_x,
        _build_v3_so3_linear,
        exactness="constructive_exact",
        description="Official degree-wise V3 SO3Linear parameterization and l=0 bias.",
    )
    add(
        "core.so3_linear@2",
        _execute_module_x,
        _build_v3_so3_linear,
        exactness="constructive_exact",
        description="Official degree-wise V3 SO3Linear with explicit l=0 initialization rescale.",
        runtime_kind_rule=_runtime_preserve("x"),
    )
    add(
        "core.equivariant_merge_norm@1",
        _execute_module_x,
        _build_v3_merge_norm,
        dependencies=v3_dependency,
        exactness="library_exact",
        description="Official Equiformer V3 merged equivariant layer normalization.",
    )
    return registry


_DEFAULT_LOWERING_RULES = build_e3nn_lowering_registry()
# Compatibility for audits and callers that previously consumed the static set.
# It is derived from the rule registry and is not a second support declaration.
_SUPPORTED = frozenset(_DEFAULT_LOWERING_RULES.names())


class E3NNGraphBackend:
    semantic_version = "e3nn-graph-lowering-registry-v23"

    def __init__(
        self,
        registry: PrimitiveRegistry,
        lowering_rules: LoweringRuleRegistry = None,
        equiformer_v2_root: str = "",
        equiformer_v3_root: str = "",
    ):
        self.registry = registry
        self.equiformer_v2_root = str(equiformer_v2_root or "")
        self.equiformer_v3_root = str(equiformer_v3_root or "")
        self.lowering_rules = lowering_rules or (
            build_e3nn_lowering_registry(
                self.equiformer_v2_root,
                self.equiformer_v3_root,
            )
            if self.equiformer_v2_root or self.equiformer_v3_root
            else _DEFAULT_LOWERING_RULES
        )

    def _support_report(
        self,
        program: ArchitectureProgram,
        ignored_nodes: Sequence[str] = (),
    ) -> BackendSupportReport:
        return self.lowering_rules.support_report(program, ignored_nodes)

    def support_report(self, program: ArchitectureProgram) -> BackendSupportReport:
        return self._support_report(program)

    def lowering_manifest(self) -> Mapping[str, Any]:
        return self.lowering_rules.audit()

    def build(
        self,
        program: ArchitectureProgram,
        inference: InferenceResult = None,
        *,
        fused_subgraphs: Sequence[FusedSubgraph] = (),
    ):
        if inference is None:
            inference = TypeChecker(self.registry).check(program)
        fusion_by_end = {}
        fused_node_ids = set()
        for fusion in fused_subgraphs:
            if fusion.end_node not in fusion.node_ids:
                raise DSLValidationError([
                    Diagnostic("E_BACKEND_006", "fused subgraph end node is outside its node set", node_id=fusion.end_node)
                ])
            overlap = fused_node_ids.intersection(fusion.node_ids)
            if fusion.end_node in fusion_by_end or overlap:
                raise DSLValidationError([
                    Diagnostic("E_BACKEND_007", "fused subgraphs overlap", details={"nodes": sorted(overlap)})
                ])
            fusion_by_end[fusion.end_node] = fusion
            fused_node_ids.update(fusion.node_ids)

        report = self._support_report(program, fused_node_ids)
        if report.missing_dependencies:
            raise DSLValidationError([
                Diagnostic(
                    "E_BACKEND_003",
                    "e3nn backend dependencies are unavailable",
                    details={"missing": list(report.missing_dependencies)},
                )
            ])
        if report.unsupported_nodes:
            raise DSLValidationError([
                Diagnostic(
                    "E_BACKEND_004",
                    "program contains operations unsupported by the e3nn graph backend",
                    details={"nodes": list(report.unsupported_nodes)},
                )
            ])
        if report.composition_errors:
            raise DSLValidationError([
                Diagnostic(
                    "E_BACKEND_012",
                    "program contains runtime value representations that cannot be composed",
                    details={"errors": [list(item) for item in report.composition_errors]},
                )
            ])

        import torch
        # e3nn<=0.5 stores a built-in ``slice`` object in its trusted local
        # Wigner constants.  PyTorch 2.6 switched ``torch.load`` to the
        # restricted weights-only loader, so explicitly allow this harmless
        # built-in before importing e3nn.
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([slice])
        from e3nn import o3
        from e3nn.nn import BatchNorm, NormActivation

        libraries = {
            "torch": torch,
            "o3": o3,
            "BatchNorm": BatchNorm,
            "NormActivation": NormActivation,
            "equiformer_v2_root": self.equiformer_v2_root,
            "equiformer_v2_modules": (
                load_equiformer_v2_modules(self.equiformer_v2_root)
                if self.equiformer_v2_root
                else None
            ),
            "equiformer_v3_root": self.equiformer_v3_root,
            "equiformer_v3_modules": (
                load_equiformer_v3_modules(self.equiformer_v3_root)
                if self.equiformer_v3_root
                else None
            ),
        }
        node_by_id = {node.id: node for node in program.nodes}
        ordered_nodes = tuple(node_by_id[item] for item in inference.node_order)
        # Runtime evaluation follows the dependency-derived topological order,
        # but module construction follows the canonical program order.  The
        # latter is part of the parameter-initialization contract: independent
        # nodes may be freely reordered by the type checker, while doing so
        # during construction would silently change global-RNG parameter draws.
        build_nodes = tuple(program.nodes)
        lowering_contract = program.parameters.get("lowering_contract", {})
        if isinstance(lowering_contract, MappingABC):
            raw_construction_order = lowering_contract.get("module_construction_order")
        else:
            raw_construction_order = None
        if raw_construction_order is not None:
            construction_order = tuple(str(node_id) for node_id in raw_construction_order)
            duplicate_ids = sorted({node_id for node_id in construction_order if construction_order.count(node_id) > 1})
            unknown_ids = sorted(set(construction_order) - set(node_by_id))
            missing_ids = sorted(set(node_by_id) - set(construction_order))
            if duplicate_ids or unknown_ids or missing_ids:
                raise DSLValidationError([
                    Diagnostic(
                        "E_BACKEND_013",
                        "module_construction_order must list every program node exactly once",
                        details={
                            "duplicates": duplicate_ids,
                            "unknown": unknown_ids,
                            "missing": missing_ids,
                        },
                    )
                ])
            build_nodes = tuple(node_by_id[node_id] for node_id in construction_order)
        initializer_events = {}
        if isinstance(lowering_contract, MappingABC):
            raw_initializer_schedule = lowering_contract.get("initializer_schedule", ())
        else:
            raw_initializer_schedule = ()
        scheduled_initializer_nodes = set()
        for raw_event in raw_initializer_schedule:
            if not isinstance(raw_event, MappingABC):
                raise DSLValidationError([
                    Diagnostic("E_BACKEND_014", "initializer_schedule entries must be mappings")
                ])
            after_node = str(raw_event.get("after_node", ""))
            event_nodes = tuple(str(node_id) for node_id in raw_event.get("nodes", ()))
            invalid = sorted(({after_node} | set(event_nodes)) - set(node_by_id))
            duplicates = sorted(set(event_nodes).intersection(scheduled_initializer_nodes))
            if not after_node or not event_nodes or invalid or duplicates:
                raise DSLValidationError([
                    Diagnostic(
                        "E_BACKEND_014",
                        "initializer_schedule must name valid, uniquely initialized nodes after a valid construction node",
                        details={
                            "after_node": after_node,
                            "nodes": list(event_nodes),
                            "invalid": invalid,
                            "duplicates": duplicates,
                        },
                    )
                ])
            initializer_events.setdefault(after_node, []).extend(event_nodes)
            scheduled_initializer_nodes.update(event_nodes)

        def value_type(reference: str):
            return inference.value_types.get(reference)

        def node_output_types(node):
            return {
                port: inference.value_types["{}:{}".format(node.id, port)]
                for port in node.outputs
            }

        modules = {}
        build_contexts = {}
        for node in build_nodes:
            if node.id in fused_node_ids:
                if node.id in fusion_by_end:
                    modules[node.id] = fusion_by_end[node.id].module
                continue
            rule = self.lowering_rules.resolve(node.op)
            context = ModuleBuildContext(
                node,
                node_output_types(node),
                value_type,
                libraries,
                inference.parameter_contracts.get(node.id, ()),
            )
            rule.validate(context)
            module = rule.build_module(context)
            _validate_built_module_parameters(node, module, context.parameter_contracts)
            build_contexts[node.id] = context
            if module is not None:
                modules[node.id] = module

            for initializer_node_id in initializer_events.get(node.id, ()):
                if initializer_node_id not in build_contexts:
                    raise DSLValidationError([
                        Diagnostic(
                            "E_BACKEND_015",
                            "initializer barrier ran before its target module was constructed",
                            node_id=initializer_node_id,
                            details={"after_node": node.id},
                        )
                    ])
                initializer_node = node_by_id[initializer_node_id]
                initializer_rule = self.lowering_rules.resolve(initializer_node.op)
                initializer_rule.initialize(
                    modules.get(initializer_node_id),
                    build_contexts[initializer_node_id],
                )

        for node in build_nodes:
            if node.id in fused_node_ids or node.id not in build_contexts:
                continue
            if node.id in scheduled_initializer_nodes:
                continue
            rule = self.lowering_rules.resolve(node.op)
            module = modules.get(node.id)
            rule.initialize(module, build_contexts[node.id])

        lowering_rules = self.lowering_rules

        class CompiledE3NNGraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.node_modules = torch.nn.ModuleDict(modules)
                self.architecture_program = program
                semantics = [E3NNGraphBackend.semantic_version]
                semantics.extend(sorted(item.backend_semantics for item in fused_subgraphs))
                self.backend_semantics_version = "+".join(semantics)
                self.fused_subgraphs = tuple(fused_subgraphs)
                self.lowering_rule_manifest = lowering_rules.audit()
                self.parameter_contract_manifest = {
                    node_id: [contract.to_dict() for contract in contracts]
                    for node_id, contracts in sorted(inference.parameter_contracts.items())
                    if contracts
                }

            @staticmethod
            def _resolve(reference, values):
                if reference in values:
                    return values[reference]
                raise KeyError(reference)

            @staticmethod
            def _bind_outputs(node, output, values):
                if len(node.outputs) == 1:
                    port = node.outputs[0]
                    if isinstance(output, MappingABC):
                        if set(output) != {port}:
                            raise RuntimeError(
                                "lowering of {} returned output ports {} but {} declares {}".format(
                                    node.op,
                                    sorted(output),
                                    node.id,
                                    list(node.outputs),
                                )
                            )
                        value = output[port]
                    else:
                        value = output
                    values["{}:{}".format(node.id, port)] = value
                    values[node.id] = value
                    return
                if not isinstance(output, MappingABC):
                    raise RuntimeError(
                        "multi-output lowering of {} must return a mapping keyed by {}".format(
                            node.op,
                            list(node.outputs),
                        )
                    )
                if set(output) != set(node.outputs):
                    raise RuntimeError(
                        "lowering of {} returned output ports {} but {} declares {}".format(
                            node.op,
                            sorted(output),
                            node.id,
                            list(node.outputs),
                        )
                    )
                for port in node.outputs:
                    values["{}:{}".format(node.id, port)] = output[port]

            def forward(self, inputs: Mapping[str, Any], context: Mapping[str, Any]):
                values = {"input:{}".format(name): value for name, value in inputs.items()}
                for node in ordered_nodes:
                    if node.id in fused_node_ids:
                        fusion = fusion_by_end.get(node.id)
                        if fusion is None:
                            continue
                        if fusion.context_key not in context:
                            raise RuntimeError(
                                "fused backend {} requires context value {}".format(
                                    fusion.backend_semantics,
                                    fusion.context_key,
                                )
                            )
                        output = self.node_modules[node.id](
                            self._resolve(fusion.input_reference, values),
                            context[fusion.context_key],
                        )
                    else:
                        resolved = {
                            port: tuple(self._resolve(ref, values) for ref in refs)
                            for port, refs in node.inputs.items()
                        }
                        rule = lowering_rules.resolve(node.op)
                        module = self.node_modules[node.id] if node.id in self.node_modules else None
                        output = rule.execute(
                            RuntimeExecutionContext(
                                node=node,
                                resolved=resolved,
                                output_types=node_output_types(node),
                                value_type=value_type,
                                module=module,
                                graph_context=context,
                                program_inputs=inputs,
                                training=self.training,
                                libraries=libraries,
                            )
                        )
                    self._bind_outputs(node, output, values)
                return {
                    item.name: self._resolve(item.source, values)
                    for item in program.outputs
                }

        return CompiledE3NNGraph()
