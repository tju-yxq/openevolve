"""Independent numerical support for Equiformer V3 primitive semantics.

Only the operator modules are imported.  The full fairchem model constructor
is intentionally not used, so a successful lowering remains a node-by-node
execution of the typed program.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ..diagnostics import DSLValidationError, Diagnostic
from .equiformer_v3_spec import V3_REFERENCE_COMMIT
from .v2_runtime import (
    SO3RuntimeValue,
    embedding_tensor_to_flat,
    flat_to_embedding_tensor,
    uniform_so3_layout,
)


V3_OPERATOR_SEMANTICS = "equiformer-v3-operators@{}".format(V3_REFERENCE_COMMIT)


@dataclass(frozen=True)
class GridRuntimeValue:
    """Finite S2 samples plus the exact inverse projection used at the boundary."""

    values: object
    source_irreps: object
    lmax: int
    mmax: int
    latitude: int
    longitude: int
    normalization: str
    use_m_primary: bool
    from_grid_mat: object

    @property
    def channels(self) -> int:
        return int(self.values.shape[-1])

    def replace_values(self, values, *, source_irreps=None):
        return GridRuntimeValue(
            values=values,
            source_irreps=self.source_irreps if source_irreps is None else source_irreps,
            lmax=self.lmax,
            mmax=self.mmax,
            latitude=self.latitude,
            longitude=self.longitude,
            normalization=self.normalization,
            use_m_primary=self.use_m_primary,
            from_grid_mat=self.from_grid_mat,
        )


def resolve_equiformer_v3_package_path(root: str) -> Optional[Path]:
    if not root:
        return None
    source = Path(root).expanduser().resolve()
    candidates = (
        source / "experimental" / "models" / "equiformer_v3",
        source / "models" / "equiformer_v3",
        source,
    )
    required = (
        "activation.py",
        "layer_norm.py",
        "so2_ops.py",
        "so3.py",
        "wigner.py",
        "Jd.pt",
    )
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate
    return None


def equiformer_v3_source_available(root: str) -> bool:
    return resolve_equiformer_v3_package_path(root) is not None


def load_equiformer_v3_modules(root: str):
    """Load the official V3 operator package without importing fairchem."""

    package_path = resolve_equiformer_v3_package_path(root)
    if package_path is None:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_BACKEND_001",
                "Equiformer V3 reference operator source is incomplete",
                details={"root": str(root)},
            )
        ])
    import torch

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    identity = hashlib.sha256(str(package_path).encode("utf-8")).hexdigest()[:12]
    package_name = "evoequilang_equiformer_v3_ref_{}".format(identity)
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return (
        importlib.import_module(package_name + ".so3"),
        importlib.import_module(package_name + ".so2_ops"),
        importlib.import_module(package_name + ".activation"),
        importlib.import_module(package_name + ".layer_norm"),
    )


def _resolution_list(value) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value), int(value)]
    result = [int(item) for item in value]
    if len(result) != 2 or any(item < 2 for item in result):
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_002", "V3 grid resolution must contain two integers >= 2")
        ])
    return result


def build_v3_grid_project_module(input_irreps, *, grid_spec, modules):
    """Build one low-level SO3Grid boundary without importing a V3 block."""

    import torch

    so3, _so2_ops, _activation, _layer_norm = modules
    lmax, channels = uniform_so3_layout(input_irreps)
    if lmax != int(grid_spec.lmax):
        raise DSLValidationError([
            Diagnostic("E_V3_GRID_001", "grid project bandlimit disagrees with input irreps")
        ])

    class V3GridProject(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.grid = so3.SO3Grid(
                lmax=lmax,
                mmax=int(grid_spec.mmax),
                normalization=str(grid_spec.normalization),
                resolution_list=[int(grid_spec.latitude), int(grid_spec.longitude)],
                use_m_primary=bool(grid_spec.use_m_primary),
            )

        def forward(self, flat):
            embedding = flat_to_embedding_tensor(flat, input_irreps)
            if int(embedding.shape[-1]) != channels:
                raise RuntimeError("grid project runtime channel count disagrees with its static contract")
            values = self.grid.to_grid(embedding)
            return GridRuntimeValue(
                values=values,
                source_irreps=input_irreps,
                lmax=lmax,
                mmax=int(grid_spec.mmax),
                latitude=int(grid_spec.latitude),
                longitude=int(grid_spec.longitude),
                normalization=str(grid_spec.normalization),
                use_m_primary=bool(grid_spec.use_m_primary),
                from_grid_mat=self.grid.get_from_grid_mat(),
            )

    return V3GridProject()


def unproject_v3_grid_value(value, output_irreps):
    import torch

    if not isinstance(value, GridRuntimeValue):
        raise RuntimeError("grid unproject requires GridRuntimeValue")
    output_lmax, output_channels = uniform_so3_layout(output_irreps)
    if output_lmax != value.lmax or output_channels != value.channels:
        raise RuntimeError("grid unproject runtime layout disagrees with its output contract")
    embedding = torch.einsum("ja,nac->njc", value.from_grid_mat, value.values)
    return embedding_tensor_to_flat(embedding, output_irreps)


def build_v3_grid_channel_linear_module(in_channels: int, out_channels: int, *, bias: bool):
    import torch

    return torch.nn.Linear(int(in_channels), int(out_channels), bias=bool(bias))


def build_v3_s2_swiglu_module(
    input_irreps,
    output_irreps,
    *,
    mmax: int,
    grid_resolution,
    edge_frame: bool,
    modules,
):
    """Build the official S2 projection/SwiGLU/projection operator."""

    import torch

    _so3, _so2_ops, activation, _layer_norm = modules
    input_lmax, input_channels = uniform_so3_layout(input_irreps)
    output_lmax, output_channels = uniform_so3_layout(output_irreps)
    if input_lmax != output_lmax or input_channels != 2 * output_channels:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_BACKEND_003",
                "S2 SwiGLU requires equal lmax and exactly twice as many input channels",
                details={
                    "input_channels": input_channels,
                    "output_channels": output_channels,
                    "input_lmax": input_lmax,
                    "output_lmax": output_lmax,
                },
            )
        ])
    if mmax < 0 or mmax > input_lmax:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_004", "mmax must satisfy 0 <= mmax <= lmax")
        ])

    class V3S2SwiGLU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = activation.S2Activation_SwiGLU(
                input_lmax,
                mmax,
                grid_resolution_list=_resolution_list(grid_resolution),
                use_m_primary=False,
            )

        def forward(self, value):
            wrapped = isinstance(value, SO3RuntimeValue)
            if wrapped:
                if value.lmax != input_lmax or value.mmax != mmax or value.channels != input_channels:
                    raise RuntimeError("S2 SwiGLU runtime layout does not match its static contract")
                embedding = value.embedding
            else:
                if edge_frame or mmax != input_lmax:
                    raise RuntimeError("truncated or edge-frame S2 SwiGLU requires an SO3RuntimeValue")
                embedding = flat_to_embedding_tensor(value, input_irreps)
            output = self.activation(embedding)
            if wrapped:
                return value.replace_embedding(
                    output,
                    irreps=output_irreps,
                    channels=output_channels,
                )
            return embedding_tensor_to_flat(output, output_irreps)

    return V3S2SwiGLU()


def build_v3_edge_frame_rotation_module(
    irreps,
    *,
    mmax: int,
    use_rotation_mask: bool,
    modules,
):
    """Rotate canonical dense SO(3) coefficients into the official V3 m-primary frame."""

    import torch

    so3, _so2_ops, _activation, _layer_norm = modules
    edge_rot_mat = importlib.import_module(so3.__package__ + ".edge_rot_mat")
    lmax, channels = uniform_so3_layout(irreps)
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_011", "V3 edge-frame mmax must satisfy 0 <= mmax <= lmax")
        ])

    class V3EdgeFrameRotation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotation = so3.SO3Rotation(
                lmax,
                int(mmax),
                use_rotation_mask=bool(use_rotation_mask),
            )

        def forward(self, flat, direction, frame_id=""):
            if flat.dim() != 2 or int(flat.shape[-1]) != irreps.dimension:
                raise RuntimeError(
                    "V3 edge-frame rotation expected [edge, {}] canonical coefficients".format(
                        irreps.dimension
                    )
                )
            if direction.dim() != 2 or int(direction.shape[-1]) != 3:
                raise RuntimeError("V3 edge-frame rotation expected [edge, 3] directions")
            if flat.shape[0] != direction.shape[0] or direction.shape[0] == 0:
                raise RuntimeError("V3 edge-frame rotation requires the same nonzero edge count")
            rotation_matrix = edge_rot_mat.init_edge_rot_mat(
                direction,
                use_rotation_mask=bool(use_rotation_mask),
            )
            self.rotation.set_wigner(rotation_matrix)
            embedding = flat_to_embedding_tensor(flat, irreps)
            rotated = self.rotation.rotate(embedding)
            return SO3RuntimeValue(
                rotated,
                irreps,
                lmax,
                int(mmax),
                channels,
                str(frame_id),
                rotation_matrix,
            )

    return V3EdgeFrameRotation()


def build_v3_edge_frame_inverse_module(
    irreps,
    *,
    mmax: int,
    use_rotation_mask: bool,
    modules,
):
    """Apply the official V3 inverse Wigner map stored by the forward rotation."""

    import torch

    so3, _so2_ops, _activation, _layer_norm = modules
    lmax, channels = uniform_so3_layout(irreps)
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_012", "V3 inverse edge-frame mmax must satisfy 0 <= mmax <= lmax")
        ])

    class V3EdgeFrameInverse(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotation = so3.SO3Rotation(
                lmax,
                int(mmax),
                use_rotation_mask=bool(use_rotation_mask),
            )

        def forward(self, value, frame_id=""):
            if not isinstance(value, SO3RuntimeValue) or value.edge_rotation_matrix is None:
                raise RuntimeError("V3 inverse edge-frame requires a rotated SO3RuntimeValue")
            if (
                value.irreps != irreps
                or value.lmax != lmax
                or value.mmax != int(mmax)
                or value.channels != channels
            ):
                raise RuntimeError("V3 inverse edge-frame runtime layout does not match its static contract")
            if str(frame_id) and value.frame_id != str(frame_id):
                raise RuntimeError(
                    "V3 inverse edge-frame expected frame {} but received {}".format(
                        frame_id, value.frame_id
                    )
                )
            self.rotation.set_wigner(value.edge_rotation_matrix)
            embedding = self.rotation.rotate_inv(value.embedding)
            return embedding_tensor_to_flat(embedding, irreps)

    return V3EdgeFrameInverse()


def build_v3_gated_swiglu_merge_module(
    input_irreps,
    output_irreps,
    *,
    mmax: int,
    grid_resolution,
    dropout: float,
    modules,
):
    """Build the official ``sep-merge_gates2_swiglu`` activation operator."""

    import torch

    _so3, _so2_ops, activation, _layer_norm = modules
    input_lmax, input_channels = uniform_so3_layout(input_irreps)
    output_lmax, output_channels = uniform_so3_layout(output_irreps)
    if input_lmax != output_lmax or input_channels != 2 * output_channels:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_BACKEND_013",
                "V3 gated S2 SwiGLU requires equal lmax and twice as many input channels",
            )
        ])
    if mmax < 0 or mmax > input_lmax:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_014", "V3 gated S2 SwiGLU mmax is outside 0..lmax")
        ])
    if not math.isfinite(float(dropout)) or float(dropout) < 0.0 or float(dropout) >= 1.0:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_015", "V3 gated S2 SwiGLU dropout must satisfy 0 <= p < 1")
        ])

    class V3GatedSwiGLUMerge(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = activation.SeparableGateS2Activation_SwiGLU_Merge(
                input_lmax,
                int(mmax),
                grid_resolution_list=_resolution_list(grid_resolution),
                use_m_primary=True,
            )
            if float(dropout) > 0.0:
                self.activation.grid_drop = torch.nn.Dropout(float(dropout))

        def forward(self, value, scalars):
            if not isinstance(value, SO3RuntimeValue):
                raise RuntimeError("V3 gated S2 SwiGLU requires an SO3RuntimeValue")
            if (
                value.irreps != input_irreps
                or value.lmax != input_lmax
                or value.mmax != int(mmax)
                or value.channels != input_channels
            ):
                raise RuntimeError("V3 gated S2 SwiGLU runtime layout does not match its static contract")
            expected_scalars = input_channels + output_channels
            if scalars.dim() != 2 or int(scalars.shape[-1]) != expected_scalars:
                raise RuntimeError(
                    "V3 gated S2 SwiGLU expected [edge, {}] scalar side input".format(
                        expected_scalars
                    )
                )
            output = self.activation(value.embedding, scalars)
            return value.replace_embedding(
                output,
                irreps=output_irreps,
                channels=output_channels,
            )

    return V3GatedSwiGLUMerge()


def build_v3_edge_frame_gate_activation_module(
    irreps,
    *,
    mmax: int,
    modules,
):
    """Build the official V3 degree-wise GateActivation in m-primary storage."""

    import torch

    _so3, _so2_ops, activation, _layer_norm = modules
    lmax, channels = uniform_so3_layout(irreps)
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_016", "V3 edge-frame gate mmax is outside 0..lmax")
        ])

    class V3EdgeFrameGateActivation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = activation.GateActivation(
                lmax,
                int(mmax),
                use_m_primary=True,
            )

        def forward(self, value, scalars):
            if not isinstance(value, SO3RuntimeValue):
                raise RuntimeError("V3 edge-frame gate requires an SO3RuntimeValue")
            if (
                value.irreps != irreps
                or value.lmax != lmax
                or value.mmax != int(mmax)
                or value.channels != channels
            ):
                raise RuntimeError("V3 edge-frame gate runtime layout does not match its static contract")
            expected_scalars = lmax * channels
            if scalars.dim() != 2 or int(scalars.shape[-1]) != expected_scalars:
                raise RuntimeError(
                    "V3 edge-frame gate expected [edge, {}] scalar gates".format(
                        expected_scalars
                    )
                )
            output = self.activation(value.embedding, scalars)
            return value.replace_embedding(output)

    return V3EdgeFrameGateActivation()


def build_v3_so3_linear_module(
    input_irreps,
    output_irreps,
    *,
    bias: bool,
    l0_weight_scale: float = 1.0,
):
    """Construct the official degree-wise V3 ``SO3Linear`` parameterization."""

    import torch

    input_lmax, input_channels = uniform_so3_layout(input_irreps)
    output_lmax, output_channels = uniform_so3_layout(output_irreps)
    if input_lmax != output_lmax:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_016", "V3 SO3Linear requires equal input and output lmax")
        ])

    class V3SO3Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(input_lmax + 1, output_channels, input_channels)
            )
            bound = 1.0 / math.sqrt(float(input_channels))
            torch.nn.init.uniform_(self.weight, -bound, bound)
            if float(l0_weight_scale) != 1.0:
                with torch.no_grad():
                    self.weight[0].mul_(float(l0_weight_scale))
            if bias:
                self.bias = torch.nn.Parameter(torch.zeros(1, 1, output_channels))
            else:
                self.register_parameter("bias", None)
            expand_index = torch.zeros((input_lmax + 1) ** 2, dtype=torch.long)
            for degree in range(input_lmax + 1):
                expand_index[degree ** 2 : (degree + 1) ** 2] = degree
            self.register_buffer("expand_index", expand_index)

        def forward(self, flat):
            embedding = flat_to_embedding_tensor(flat, input_irreps)
            weight = torch.index_select(self.weight, 0, self.expand_index)
            output = torch.einsum("bmi,moi->bmo", embedding, weight)
            if self.bias is not None:
                output[:, 0:1, :] = output.narrow(1, 0, 1) + self.bias
            return embedding_tensor_to_flat(output, output_irreps)

    return V3SO3Linear()


def build_v3_so2_linear_module(
    input_irreps,
    output_irreps,
    *,
    mmax: int,
    extra_m0_channels: int = 0,
    m0_prefix_rows: int = 0,
    m0_prefix_scale: float = 1.0,
    zero_bias: bool = False,
):
    """Construct the V3 SO(2) intertwiner without importing a V3 Block.

    The implementation is the explicit real form of a complex SO(2) linear
    map in m-primary coefficient order.  It intentionally mirrors the
    parameter tree used by the official low-level ``SO2Linear`` operator so a
    checkpoint can be mapped parameter by parameter, while remaining a plain
    PyTorch lowering owned by the DSL backend.
    """

    import torch

    input_lmax, input_channels = uniform_so3_layout(input_irreps)
    output_lmax, output_channels = uniform_so3_layout(output_irreps)
    if input_lmax != output_lmax:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_005", "V3 SO2Linear requires equal input and output lmax")
        ])
    if mmax < 0 or mmax > input_lmax:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_006", "mmax must satisfy 0 <= mmax <= lmax")
        ])
    if extra_m0_channels < 0:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_007", "extra_m0_channels must be nonnegative")
        ])
    total_m0_rows = (input_lmax + 1) * output_channels + int(extra_m0_channels)
    if m0_prefix_rows < 0 or m0_prefix_rows > total_m0_rows:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_008", "m0_prefix_rows is outside the m=0 output range")
        ])
    if not math.isfinite(float(m0_prefix_scale)) or float(m0_prefix_scale) <= 0.0:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_009", "m0_prefix_scale must be finite and positive")
        ])

    class SO2MLinear(torch.nn.Module):
        def __init__(self, m_value: int):
            super().__init__()
            self.m = int(m_value)
            component_count = input_lmax - self.m + 1
            self.in_features = component_count * input_channels
            self.out_features = component_count * output_channels
            self.fc = torch.nn.Linear(
                self.in_features,
                2 * self.out_features,
                bias=False,
            )
            self.fc.weight.data.mul_(1.0 / math.sqrt(2.0))

        def forward(self, x_m):
            mixed = self.fc(x_m)
            real = mixed.narrow(2, 0, self.out_features)
            imag = mixed.narrow(2, self.out_features, self.out_features)
            positive = real.narrow(1, 0, 1) - imag.narrow(1, 1, 1)
            negative = real.narrow(1, 1, 1) + imag.narrow(1, 0, 1)
            return positive, negative

    class V3SO2Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_in_channels = input_channels
            self.num_out_channels = output_channels
            self.lmax = input_lmax
            self.mmax = int(mmax)
            self.extra_m0_out_channels = int(extra_m0_channels) or None
            input_m0 = (self.lmax + 1) * self.num_in_channels
            output_m0 = (self.lmax + 1) * self.num_out_channels + int(extra_m0_channels)
            self.fc_m0 = torch.nn.Linear(input_m0, output_m0)
            if m0_prefix_rows:
                self.fc_m0.weight.data[: int(m0_prefix_rows)].mul_(float(m0_prefix_scale))
            self.so2_m_linear = torch.nn.ModuleList(
                SO2MLinear(m_value) for m_value in range(1, self.mmax + 1)
            )
            if zero_bias:
                torch.nn.init.zeros_(self.fc_m0.bias)

        def forward(self, value):
            wrapped = isinstance(value, SO3RuntimeValue)
            if wrapped:
                if (
                    value.lmax != self.lmax
                    or value.mmax != self.mmax
                    or value.channels != self.num_in_channels
                ):
                    raise RuntimeError("V3 SO2Linear runtime layout does not match its static contract")
                embedding = value.embedding
            else:
                embedding = value
            expected_coefficients = (self.lmax + 1) + sum(
                2 * (self.lmax + 1 - order) for order in range(1, self.mmax + 1)
            )
            if embedding.dim() != 3 or int(embedding.shape[1]) != expected_coefficients:
                raise RuntimeError(
                    "V3 SO2Linear expected [edge, {}, channel] m-primary coefficients".format(
                        expected_coefficients
                    )
                )
            num_edges = embedding.shape[0]
            outputs = []
            m0 = embedding.narrow(1, 0, self.lmax + 1).reshape(num_edges, -1)
            m0 = self.fc_m0(m0)
            extra = None
            if extra_m0_channels:
                extra, m0 = torch.split(
                    m0,
                    [int(extra_m0_channels), (self.lmax + 1) * self.num_out_channels],
                    dim=1,
                )
            outputs.append(m0.view(num_edges, self.lmax + 1, self.num_out_channels))
            offset = self.lmax + 1
            for order, module in enumerate(self.so2_m_linear, start=1):
                count = self.lmax + 1 - order
                x_m = embedding.narrow(1, offset, 2 * count).reshape(num_edges, 2, -1)
                offset += 2 * count
                positive, negative = module(x_m)
                outputs.append(positive.view(num_edges, count, self.num_out_channels))
                outputs.append(negative.view(num_edges, count, self.num_out_channels))
            output_embedding = torch.cat(outputs, dim=1)
            output = (
                value.replace_embedding(
                    output_embedding,
                    irreps=output_irreps,
                    channels=self.num_out_channels,
                )
                if wrapped
                else output_embedding
            )
            if extra_m0_channels:
                return {"out": output, "extra_m0": extra}
            return output

    return V3SO2Linear()


def build_v3_axisymmetric_spherical_lift_module(
    output_irreps,
    *,
    mmax: int,
    use_rotation_mask: bool,
    modules,
):
    """Build the official V3 inverse-Wigner m=0 spherical lift.

    This is the low-level operation used by ``EdgeDegreeEmbedding`` after its
    radial MLP.  It accepts invariant amplitudes rather than an official
    embedding object and returns canonical flattened DSL storage, so no V3
    Block or input-block constructor is part of the lowering path.
    """

    import torch

    so3, _so2_ops, _activation, _layer_norm = modules
    edge_rot_mat = importlib.import_module(so3.__package__ + ".edge_rot_mat")
    lmax, channels = uniform_so3_layout(output_irreps)
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic("E_V3_BACKEND_010", "axisymmetric lift mmax must satisfy 0 <= mmax <= lmax")
        ])

    class V3AxisymmetricSphericalLift(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotation = so3.SO3Rotation(
                lmax,
                int(mmax),
                use_rotation_mask=bool(use_rotation_mask),
            )

        def forward(self, amplitudes, direction):
            if amplitudes.dim() != 2 or int(amplitudes.shape[-1]) != (lmax + 1) * channels:
                raise RuntimeError(
                    "axisymmetric spherical lift expected [edge, {}] amplitudes".format(
                        (lmax + 1) * channels
                    )
                )
            if direction.dim() != 2 or int(direction.shape[-1]) != 3:
                raise RuntimeError("axisymmetric spherical lift expected [edge, 3] directions")
            if amplitudes.shape[0] != direction.shape[0] or direction.shape[0] == 0:
                raise RuntimeError("axisymmetric spherical lift requires the same nonzero edge count")
            rotation_matrix = edge_rot_mat.init_edge_rot_mat(
                direction,
                use_rotation_mask=bool(use_rotation_mask),
            )
            self.rotation.set_wigner(rotation_matrix)
            m0 = amplitudes.reshape(amplitudes.shape[0], lmax + 1, channels)
            embedding = torch.bmm(
                self.rotation.wigner_inv.narrow(2, 0, lmax + 1),
                m0,
            )
            return embedding_tensor_to_flat(embedding, output_irreps)

    return V3AxisymmetricSphericalLift()


def build_v3_merge_norm_module(irreps, *, modules, epsilon, affine, normalization, centering):
    import torch

    _so3, _so2_ops, _activation, layer_norm = modules
    lmax, channels = uniform_so3_layout(irreps)

    class V3MergeNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = layer_norm.EquivariantMergeLayerNorm(
                lmax=lmax,
                num_channels=channels,
                eps=float(epsilon),
                affine=bool(affine),
                normalization=str(normalization),
                centering=bool(centering),
            )

        def forward(self, value):
            wrapped = isinstance(value, SO3RuntimeValue)
            if wrapped:
                if value.mmax != value.lmax:
                    raise RuntimeError("V3 merged normalization requires complete SO(3) coefficient blocks")
                output = self.norm(value.embedding)
                return value.replace_embedding(output)
            embedding = flat_to_embedding_tensor(value, irreps)
            return embedding_tensor_to_flat(self.norm(embedding), irreps)

    return V3MergeNorm()
