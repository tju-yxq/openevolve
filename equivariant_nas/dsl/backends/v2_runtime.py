"""Independent Equiformer V2 primitive runtime support.

The official V2 frame constructor chooses an arbitrary SO(2) gauge around
each edge.  A local-frame value therefore cannot be represented by a naked
tensor: the inverse transform must reuse the exact rotation chosen by the
forward transform.  ``SO3RuntimeValue`` carries that state between otherwise
independent DSL nodes while keeping the public DSL type backend-neutral.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Tuple

from ..diagnostics import DSLValidationError, Diagnostic


V2_REFERENCE_COMMIT = "d5ad4be729b56f74012ebb7f097f77c5b00a1004"
V2_FUSION_SEMANTICS = "equiformer-v2-so2-fusion@{}".format(V2_REFERENCE_COMMIT)


def resolve_equiformer_v2_package_path(root: str) -> Optional[Path]:
    """Resolve both the original Equiformer V2 and current fairchem layouts."""

    if not root:
        return None
    source = Path(root).expanduser().resolve()
    candidates = (
        source / "nets" / "equiformer_v2",
        source / "src" / "fairchem" / "core" / "models" / "equiformer_v2",
        source,
    )
    required = ("so3.py", "so2_ops.py", "edge_rot_mat.py", "activation.py")
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate
    return None


def equiformer_v2_source_available(root: str) -> bool:
    return resolve_equiformer_v2_package_path(root) is not None


def load_equiformer_v2_modules(root: str):
    """Load the operator package without importing a top-level training model."""

    package_path = resolve_equiformer_v2_package_path(root)
    if package_path is None:
        raise DSLValidationError([
            Diagnostic(
                "E_V2_BACKEND_001",
                "Equiformer V2 reference source is incomplete",
                details={"root": str(root)},
            )
        ])
    identity = hashlib.sha256(str(package_path).encode("utf-8")).hexdigest()[:12]
    package_name = "evoequilang_equiformer_v2_ref_{}".format(identity)
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return (
        importlib.import_module(package_name + ".so3"),
        importlib.import_module(package_name + ".so2_ops"),
        importlib.import_module(package_name + ".edge_rot_mat"),
        importlib.import_module(package_name + ".activation"),
    )


def uniform_so3_layout(irreps) -> Tuple[int, int]:
    """Return ``(lmax, channels)`` for the dense V2 coefficient layout."""

    if irreps.family != "SO3" or not irreps.terms:
        raise DSLValidationError([
            Diagnostic("E_V2_BACKEND_002", "V2 primitives require nonempty SO(3) irreps")
        ])
    degrees = [ir.degree for _, ir in irreps]
    lmax = max(degrees)
    if degrees != list(range(lmax + 1)):
        raise DSLValidationError([
            Diagnostic(
                "E_V2_BACKEND_003",
                "V2 coefficient layout requires every degree from zero through lmax",
                actual=str(irreps),
            )
        ])
    multiplicities = {mul for mul, _ in irreps}
    if len(multiplicities) != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_V2_BACKEND_004",
                "V2 coefficient layout requires uniform multiplicity across degrees",
                actual=str(irreps),
            )
        ])
    return lmax, next(iter(multiplicities))


def flat_to_embedding_tensor(flat, irreps):
    """Convert e3nn-style flattened irreps to ``[N, coefficient, channel]``."""

    import torch

    _lmax, channels = uniform_so3_layout(irreps)
    if flat.shape[-1] != irreps.dimension:
        raise RuntimeError(
            "flat SO(3) width {} does not match irreps dimension {}".format(
                flat.shape[-1], irreps.dimension
            )
        )
    pieces = []
    offset = 0
    for multiplicity, irrep in irreps:
        width = multiplicity * irrep.dimension
        block = flat[..., offset : offset + width]
        block = block.reshape(flat.shape[0], multiplicity, irrep.dimension).transpose(1, 2)
        pieces.append(block)
        offset += width
    return flat.new_zeros((flat.shape[0], 0, channels)) if not pieces else torch.cat(pieces, dim=1)


def embedding_tensor_to_flat(embedding, irreps):
    """Convert a full ``mmax=lmax`` V2 coefficient tensor back to flat irreps."""

    import torch

    lmax, channels = uniform_so3_layout(irreps)
    expected = (lmax + 1) ** 2
    if embedding.shape[1] != expected or embedding.shape[2] != channels:
        raise RuntimeError(
            "SO(3) embedding shape {} is incompatible with {}".format(
                tuple(embedding.shape), irreps
            )
        )
    pieces = []
    offset = 0
    for multiplicity, irrep in irreps:
        width = irrep.dimension
        block = embedding[:, offset : offset + width]
        pieces.append(block.transpose(1, 2).reshape(embedding.shape[0], multiplicity * width))
        offset += width
    return embedding.new_zeros((embedding.shape[0], 0)) if not pieces else torch.cat(pieces, dim=-1)


@dataclass(frozen=True)
class SO3RuntimeValue:
    """Backend value for dense or order-truncated SO(3) coefficients."""

    embedding: Any
    irreps: Any
    lmax: int
    mmax: int
    channels: int
    frame_id: str = ""
    edge_rotation_matrix: Any = None

    def replace_embedding(self, embedding, *, irreps=None, mmax=None, channels=None):
        return replace(
            self,
            embedding=embedding,
            irreps=self.irreps if irreps is None else irreps,
            mmax=self.mmax if mmax is None else int(mmax),
            channels=self.channels if channels is None else int(channels),
        )

    def to_official_embedding(self, so3):
        value = so3.SO3_Embedding(
            0,
            [self.lmax],
            self.channels,
            self.embedding.device,
            self.embedding.dtype,
        )
        value.set_embedding(self.embedding)
        value.set_lmax_mmax([self.lmax], [self.mmax])
        return value


def to_edge_frame_value(flat, irreps, edge_vectors, frame_id: str, modules, mmax=None):
    so3, _so2_ops, edge_rot_mat, _activation = modules
    lmax, channels = uniform_so3_layout(irreps)
    mmax = lmax if mmax is None else int(mmax)
    if mmax < 0 or mmax > lmax:
        raise RuntimeError("edge-frame mmax must satisfy 0 <= mmax <= lmax")
    if flat.shape[0] != edge_vectors.shape[0]:
        raise RuntimeError("edge-frame input and edge_vectors must have the same leading dimension")
    if edge_vectors.shape[0] == 0:
        raise RuntimeError("Equiformer V2 frame construction requires at least one edge")
    rotation_matrix = edge_rot_mat.init_edge_rot_mat(edge_vectors)
    rotation = so3.SO3_Rotation(lmax)
    rotation.set_wigner(rotation_matrix)
    embedding = so3.SO3_Embedding(0, [lmax], channels, flat.device, flat.dtype)
    embedding.set_embedding(flat_to_embedding_tensor(flat, irreps))
    embedding._rotate([rotation], [lmax], [mmax])
    return SO3RuntimeValue(
        embedding.embedding,
        irreps,
        lmax,
        mmax,
        channels,
        str(frame_id),
        rotation_matrix,
    )


def from_edge_frame_value(value: SO3RuntimeValue, output_irreps, frame_id: str, modules):
    if not isinstance(value, SO3RuntimeValue) or value.edge_rotation_matrix is None:
        raise RuntimeError("from_edge_frame requires a runtime value produced by to_edge_frame")
    if str(frame_id) and value.frame_id != str(frame_id):
        raise RuntimeError(
            "from_edge_frame expected frame {} but received {}".format(frame_id, value.frame_id)
        )
    so3, _so2_ops, _edge_rot_mat, _activation = modules
    rotation = so3.SO3_Rotation(value.lmax)
    rotation.set_wigner(value.edge_rotation_matrix)
    embedding = value.to_official_embedding(so3)
    mapping = so3.CoefficientMappingModule([value.lmax], [value.mmax])
    embedding._rotate_inv([rotation], mapping)
    return embedding_tensor_to_flat(embedding.embedding, output_irreps)


def select_runtime_scalars(value: SO3RuntimeValue, multiplicity: int):
    if multiplicity <= 0 or multiplicity > value.channels:
        raise RuntimeError("requested scalar multiplicity is unavailable in SO(3) runtime value")
    return value.embedding[:, 0, :multiplicity]


def build_so2_convolution_module(input_irreps, output_irreps, mmax: int, modules):
    import torch

    so3, so2_ops, _edge_rot_mat, _activation = modules
    input_lmax, input_channels = uniform_so3_layout(input_irreps)
    output_lmax, output_channels = uniform_so3_layout(output_irreps)
    if input_lmax != output_lmax:
        raise DSLValidationError([
            Diagnostic("E_V2_BACKEND_005", "independent SO(2) lowering requires equal input and output lmax")
        ])
    if mmax < 0 or mmax > input_lmax:
        raise DSLValidationError([
            Diagnostic("E_V2_BACKEND_006", "mmax must satisfy 0 <= mmax <= lmax")
        ])

    class IndependentSO2Convolution(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mapping = so3.CoefficientMappingModule([input_lmax], [mmax])
            self.convolution = so2_ops.SO2_Convolution(
                input_channels,
                output_channels,
                [input_lmax],
                [mmax],
                self.mapping,
                internal_weights=True,
                edge_channels_list=None,
            )
            self.lmax = input_lmax
            self.mmax = mmax

        def forward(self, value):
            if not isinstance(value, SO3RuntimeValue):
                raise RuntimeError("SO(2) convolution requires an edge-frame SO3RuntimeValue")
            embedding = value.to_official_embedding(so3)
            edge_scalars = value.embedding.new_zeros((value.embedding.shape[0], 1))
            output = self.convolution(embedding, edge_scalars)
            return SO3RuntimeValue(
                output.embedding,
                output_irreps,
                output_lmax,
                mmax,
                output_channels,
                value.frame_id,
                value.edge_rotation_matrix,
            )

    return IndependentSO2Convolution()


def _build_grids(torch, so3, lmax: int, resolution: int, normalization: str):
    grids = []
    for l_value in range(lmax + 1):
        grids.append(torch.nn.ModuleList([
            so3.SO3_Grid(
                l_value,
                m_value,
                resolution=resolution,
                normalization=normalization,
            )
            for m_value in range(lmax + 1)
        ]))
    return torch.nn.ModuleList(grids)


def build_s2_activation_module(
    irreps,
    *,
    mmax: int,
    resolution: int,
    normalization: str,
    separable: bool,
    modules,
):
    import torch

    so3, _so2_ops, _edge_rot_mat, activation = modules
    lmax, channels = uniform_so3_layout(irreps)
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic("E_V2_BACKEND_006", "mmax must satisfy 0 <= mmax <= lmax")
        ])

    class IndependentS2Activation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = (
                activation.SeparableS2Activation(lmax, mmax)
                if separable
                else activation.S2Activation(lmax, mmax)
            )
            self.grids = _build_grids(torch, so3, lmax, resolution, normalization)
            self.lmax = lmax
            self.mmax = mmax
            self.separable = separable

        def forward(self, value, scalars=None):
            wrapped = isinstance(value, SO3RuntimeValue)
            if wrapped:
                if value.lmax != lmax or value.mmax != mmax or value.channels != channels:
                    raise RuntimeError("S2 activation runtime layout does not match its static contract")
                embedding = value.embedding
            else:
                if mmax != lmax:
                    raise RuntimeError("order-truncated S2 activation requires an edge-frame runtime value")
                embedding = flat_to_embedding_tensor(value, irreps)
            if self.separable:
                if scalars is None:
                    raise RuntimeError("separable S2 activation requires scalar side input")
                output = self.activation(scalars, embedding, self.grids)
            else:
                output = self.activation(embedding, self.grids)
            if wrapped:
                return value.replace_embedding(output)
            return embedding_tensor_to_flat(output, irreps)

    return IndependentS2Activation()
