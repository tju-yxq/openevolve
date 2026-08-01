"""Typed values carried by EvoEquiLang graph edges."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from .diagnostics import DSLValidationError, Diagnostic
from .groups import GroupSpec
from .irreps import Irreps


VALUE_TYPE_SCHEMA_VERSION = "evoequilang-value-types-v2@1"


class Carrier(str):
    NODE = "node"
    EDGE = "edge"
    GRAPH = "graph"
    GRID = "grid"
    PAIR = "pair"

    @classmethod
    def validate(cls, value: str) -> str:
        if value not in (cls.NODE, cls.EDGE, cls.GRAPH, cls.GRID, cls.PAIR):
            raise DSLValidationError([Diagnostic("E_TYPE_001", "unsupported carrier", actual=value)])
        return value


class EquivarianceLevel(IntEnum):
    UNVERIFIED = 0
    EMPIRICAL = 1
    CONSTRUCTIVE = 2
    CORE_CERTIFIED = 3


class FeatureRole(str, Enum):
    CHANNEL = "channel"
    HEAD = "head"
    DEGREE = "degree"
    ORDER_M = "order_m"
    RESOLUTION = "resolution"
    ENDPOINT = "endpoint"
    SPECIES = "species"
    BASIS = "basis"
    TP_PATH = "tp_path"
    GRID_LATITUDE = "grid_latitude"
    GRID_LONGITUDE = "grid_longitude"
    CARTESIAN = "cartesian"
    ALPHA = "alpha"
    VALUE = "value"
    GATE = "gate"
    EXTRA_M0 = "extra_m0"
    RADIAL_WEIGHT = "radial_weight"
    NODE = "node"
    EDGE = "edge"
    GRAPH = "graph"
    BATCH = "batch"
    NEIGHBOR = "neighbor"
    MULTIPLICITY = "multiplicity"
    COEFFICIENT = "coefficient"

    @classmethod
    def parse(cls, value: Any) -> "FeatureRole":
        try:
            return value if isinstance(value, cls) else cls(str(value))
        except ValueError:
            raise DSLValidationError([
                Diagnostic("E_AXIS_001", "unsupported feature or axis role", actual=str(value))
            ])


@dataclass(frozen=True, order=True)
class AxisSpec:
    name: str
    size: Optional[int] = None
    role: FeatureRole = FeatureRole.CHANNEL
    sharing: str = "independent"
    order: int = 0
    broadcastable: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise DSLValidationError([Diagnostic("E_AXIS_002", "axis name must be nonempty")])
        if self.size is not None and int(self.size) <= 0:
            raise DSLValidationError([Diagnostic("E_AXIS_003", "axis size must be positive or dynamic", actual=str(self.size))])
        if not self.sharing:
            raise DSLValidationError([Diagnostic("E_AXIS_004", "axis sharing policy must be nonempty", actual=self.name)])
        if int(self.order) < 0:
            raise DSLValidationError([Diagnostic("E_AXIS_005", "axis order must be nonnegative", actual=str(self.order))])
        object.__setattr__(self, "role", FeatureRole.parse(self.role))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "role": self.role.value,
            "sharing": self.sharing,
            "order": self.order,
            "broadcastable": self.broadcastable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AxisSpec":
        allowed = {"name", "size", "role", "sharing", "order", "broadcastable"}
        unknown = set(data) - allowed
        if unknown:
            raise DSLValidationError([Diagnostic("E_SCHEMA_004", "unknown axis fields", details={"fields": sorted(unknown)})])
        raw_size = data.get("size")
        return cls(
            name=str(data["name"]),
            size=None if raw_size is None else int(raw_size),
            role=FeatureRole.parse(data.get("role", FeatureRole.CHANNEL.value)),
            sharing=str(data.get("sharing", "independent")),
            order=int(data.get("order", 0)),
            broadcastable=bool(data.get("broadcastable", False)),
        )


@dataclass(frozen=True, order=True)
class ResolutionSpec:
    name: str
    lmax: int
    mmax: int

    def __post_init__(self) -> None:
        if not self.name:
            raise DSLValidationError([Diagnostic("E_LAYOUT_001", "resolution name must be nonempty")])
        if self.lmax < 0 or self.mmax < 0 or self.mmax > self.lmax:
            raise DSLValidationError([
                Diagnostic(
                    "E_LAYOUT_002",
                    "resolution requires 0 <= mmax <= lmax",
                    actual="lmax={},mmax={}".format(self.lmax, self.mmax),
                )
            ])

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "lmax": self.lmax, "mmax": self.mmax}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResolutionSpec":
        allowed = {"name", "lmax", "mmax"}
        unknown = set(data) - allowed
        if unknown:
            raise DSLValidationError([Diagnostic("E_SCHEMA_005", "unknown resolution fields", details={"fields": sorted(unknown)})])
        return cls(str(data["name"]), int(data["lmax"]), int(data["mmax"]))


@dataclass(frozen=True)
class RepresentationLayout:
    storage: str = "irrep_major"
    coefficient_order: str = "canonical"
    resolution_specs: Tuple[ResolutionSpec, ...] = field(default_factory=tuple)
    truncation_state: str = "full"
    parity_convention: str = "group_native"
    channel_order: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.storage not in ("irrep_major", "l_primary", "m_primary", "grid", "dense", "unspecified"):
            raise DSLValidationError([Diagnostic("E_LAYOUT_003", "unsupported representation storage", actual=self.storage)])
        if self.truncation_state not in ("full", "m_truncated", "grid_projected", "unspecified"):
            raise DSLValidationError([Diagnostic("E_LAYOUT_004", "unsupported truncation state", actual=self.truncation_state)])
        names = [item.name for item in self.resolution_specs]
        if len(names) != len(set(names)):
            raise DSLValidationError([Diagnostic("E_LAYOUT_005", "resolution names must be unique")])
        if len(self.channel_order) != len(set(self.channel_order)):
            raise DSLValidationError([Diagnostic("E_LAYOUT_006", "channel-order labels must be unique")])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage": self.storage,
            "coefficient_order": self.coefficient_order,
            "resolution_specs": [item.to_dict() for item in self.resolution_specs],
            "truncation_state": self.truncation_state,
            "parity_convention": self.parity_convention,
            "channel_order": list(self.channel_order),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepresentationLayout":
        allowed = {
            "storage", "coefficient_order", "resolution_specs", "truncation_state",
            "parity_convention", "channel_order",
        }
        unknown = set(data) - allowed
        if unknown:
            raise DSLValidationError([Diagnostic("E_SCHEMA_006", "unknown representation-layout fields", details={"fields": sorted(unknown)})])
        return cls(
            storage=str(data.get("storage", "irrep_major")),
            coefficient_order=str(data.get("coefficient_order", "canonical")),
            resolution_specs=tuple(ResolutionSpec.from_dict(item) for item in data.get("resolution_specs", ())),
            truncation_state=str(data.get("truncation_state", "full")),
            parity_convention=str(data.get("parity_convention", "group_native")),
            channel_order=tuple(str(item) for item in data.get("channel_order", ())),
        )


@dataclass(frozen=True, order=True)
class GridSpec:
    """Finite S2 sampling and quadrature contract used by grid primitives."""

    latitude: int
    longitude: int
    lmax: int
    mmax: int
    normalization: str = "component"
    quadrature: str = "e3nn_s2grid"
    sampling: str = "equiangular"
    use_m_primary: bool = False

    def __post_init__(self) -> None:
        if int(self.latitude) < 2 or int(self.longitude) < 2:
            raise DSLValidationError([
                Diagnostic(
                    "E_GRID_TYPE_001",
                    "grid latitude and longitude resolutions must be at least two",
                    actual="{}x{}".format(self.latitude, self.longitude),
                )
            ])
        if self.lmax < 0 or self.mmax < 0 or self.mmax > self.lmax:
            raise DSLValidationError([
                Diagnostic(
                    "E_GRID_TYPE_002",
                    "grid bandlimit requires 0 <= mmax <= lmax",
                    actual="lmax={},mmax={}".format(self.lmax, self.mmax),
                )
            ])
        if self.normalization not in ("component", "norm", "integral"):
            raise DSLValidationError([
                Diagnostic(
                    "E_GRID_TYPE_003",
                    "unsupported S2 grid normalization",
                    actual=self.normalization,
                )
            ])
        if self.quadrature not in ("e3nn_s2grid",):
            raise DSLValidationError([
                Diagnostic("E_GRID_TYPE_004", "unsupported S2 quadrature contract", actual=self.quadrature)
            ])
        if self.sampling not in ("equiangular",):
            raise DSLValidationError([
                Diagnostic("E_GRID_TYPE_005", "unsupported S2 sampling domain", actual=self.sampling)
            ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": int(self.latitude),
            "longitude": int(self.longitude),
            "lmax": int(self.lmax),
            "mmax": int(self.mmax),
            "normalization": self.normalization,
            "quadrature": self.quadrature,
            "sampling": self.sampling,
            "use_m_primary": bool(self.use_m_primary),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GridSpec":
        allowed = {
            "latitude", "longitude", "lmax", "mmax", "normalization",
            "quadrature", "sampling", "use_m_primary",
        }
        unknown = set(data) - allowed
        if unknown:
            raise DSLValidationError([
                Diagnostic("E_SCHEMA_012", "unknown grid-spec fields", details={"fields": sorted(unknown)})
            ])
        return cls(
            latitude=int(data["latitude"]),
            longitude=int(data["longitude"]),
            lmax=int(data["lmax"]),
            mmax=int(data["mmax"]),
            normalization=str(data.get("normalization", "component")),
            quadrature=str(data.get("quadrature", "e3nn_s2grid")),
            sampling=str(data.get("sampling", "equiangular")),
            use_m_primary=bool(data.get("use_m_primary", False)),
        )


class ValueType:
    """Base of the versioned DSL v2 value-type union.

    Legacy ``EquivariantType`` deliberately remains an untagged wire format.
    New types carry a ``kind`` discriminator and the v2 schema version.
    """

    kind: ClassVar[str] = "abstract"
    schema_version: ClassVar[str] = VALUE_TYPE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValueType":
        return value_type_from_dict(data)

    def compatible(self, other: "ValueType", **_: Any) -> bool:
        return self == other


@dataclass(frozen=True, order=True)
class Frame:
    kind: str = "global"
    reference: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("global", "edge", "local", "invariant"):
            raise DSLValidationError([Diagnostic("E_FRAME_001", "unsupported frame", actual=self.kind)])
        if self.kind in ("edge", "local") and not self.reference:
            raise DSLValidationError([Diagnostic("E_FRAME_002", "local frames require a reference token")])
        if self.kind in ("global", "invariant") and self.reference:
            raise DSLValidationError([Diagnostic("E_FRAME_003", "global and invariant frames cannot carry references")])

    def __str__(self) -> str:
        return self.kind if not self.reference else "{}({})".format(self.kind, self.reference)


@dataclass(frozen=True)
class EquivariantType(ValueType):
    kind: ClassVar[str] = "legacy_equivariant"
    group: GroupSpec
    carrier: str
    irreps: Irreps
    frame: Frame = field(default_factory=Frame)
    axes: Tuple[str, ...] = field(default_factory=tuple)
    dtype: str = "float32"
    measure: str = "dimensionless"
    level: EquivarianceLevel = EquivarianceLevel.CONSTRUCTIVE

    def __post_init__(self) -> None:
        Carrier.validate(self.carrier)
        expected = self.group.family
        if expected in ("O3", "SO3", "O2", "SO2") and self.irreps.terms and self.irreps.family != expected:
            raise DSLValidationError([
                Diagnostic("E_TYPE_002", "irrep family does not match group", expected=expected, actual=self.irreps.family)
            ])
        if self.dtype not in ("float16", "bfloat16", "float32", "float64"):
            raise DSLValidationError([Diagnostic("E_TYPE_003", "unsupported dtype", actual=self.dtype)])

    def with_irreps(self, irreps: Irreps) -> "EquivariantType":
        return replace(self, irreps=irreps)

    def with_carrier(self, carrier: str) -> "EquivariantType":
        return replace(self, carrier=carrier)

    def with_frame(self, frame: Frame) -> "EquivariantType":
        return replace(self, frame=frame)

    def with_measure(self, measure: str) -> "EquivariantType":
        return replace(self, measure=str(measure))

    def compatible(self, other: "EquivariantType", *, exact_irreps: bool = True) -> bool:
        return (
            self.group == other.group
            and self.carrier == other.carrier
            and self.frame == other.frame
            and (self.irreps == other.irreps or not exact_irreps)
            and self.axes == other.axes
            and self.dtype == other.dtype
            and self.measure == other.measure
            and self.level == other.level
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group.to_dict(),
            "carrier": self.carrier,
            "irreps": str(self.irreps),
            "frame": {"kind": self.frame.kind, "reference": self.frame.reference},
            "axes": list(self.axes),
            "dtype": self.dtype,
            "measure": self.measure,
            "equivariance_level": int(self.level),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EquivariantType":
        allowed = {"group", "carrier", "irreps", "frame", "axes", "dtype", "measure", "equivariance_level"}
        unknown = set(data) - allowed
        if unknown:
            raise DSLValidationError([Diagnostic("E_SCHEMA_002", "unknown type fields", details={"fields": sorted(unknown)})])
        group = GroupSpec.from_dict(data["group"])
        frame_data = data.get("frame", {"kind": "global", "reference": ""})
        frame = Frame(str(frame_data.get("kind", "global")), str(frame_data.get("reference", "")))
        return cls(
            group=group,
            carrier=str(data["carrier"]),
            irreps=Irreps.parse(str(data["irreps"]), group.family),
            frame=frame,
            axes=tuple(str(item) for item in data.get("axes", [])),
            dtype=str(data.get("dtype", "float32")),
            measure=str(data.get("measure", "dimensionless")),
            level=EquivarianceLevel(int(data.get("equivariance_level", 2))),
        )


def _validate_axis_specs(axes: Tuple[str, ...], axis_specs: Tuple[AxisSpec, ...]) -> Tuple[str, ...]:
    names = tuple(item.name for item in axis_specs)
    orders = tuple(item.order for item in axis_specs)
    if len(names) != len(set(names)):
        raise DSLValidationError([Diagnostic("E_AXIS_006", "axis names must be unique")])
    if len(orders) != len(set(orders)):
        raise DSLValidationError([Diagnostic("E_AXIS_007", "axis orders must be unique")])
    if tuple(sorted(orders)) != tuple(range(len(orders))):
        raise DSLValidationError([Diagnostic("E_AXIS_008", "axis orders must form a contiguous zero-based sequence")])
    ordered_names = tuple(item.name for item in sorted(axis_specs, key=lambda item: item.order))
    if axes and axes != ordered_names:
        raise DSLValidationError([
            Diagnostic("E_AXIS_009", "legacy axis names disagree with explicit AxisSpec order", expected=str(axes), actual=str(ordered_names))
        ])
    return axes or ordered_names


@dataclass(frozen=True)
class EquivariantTensorType(EquivariantType):
    kind: ClassVar[str] = "equivariant_tensor"
    axis_specs: Tuple[AxisSpec, ...] = field(default_factory=tuple)
    layout: RepresentationLayout = field(default_factory=RepresentationLayout)

    def __post_init__(self) -> None:
        super().__post_init__()
        specs = self.axis_specs
        if not specs and self.axes:
            specs = tuple(
                AxisSpec(name=name, role=_infer_feature_role(name), order=index)
                for index, name in enumerate(self.axes)
            )
            object.__setattr__(self, "axis_specs", specs)
        object.__setattr__(self, "axes", _validate_axis_specs(self.axes, specs))

    def compatible(self, other: "ValueType", *, exact_irreps: bool = True) -> bool:
        return (
            isinstance(other, EquivariantTensorType)
            and super().compatible(other, exact_irreps=exact_irreps)
            and self.axis_specs == other.axis_specs
            and self.layout == other.layout
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload.update({
            "kind": self.kind,
            "schema_version": self.schema_version,
            "axis_specs": [item.to_dict() for item in self.axis_specs],
            "layout": self.layout.to_dict(),
        })
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EquivariantTensorType":
        return cls(**_equivariant_tensor_kwargs(data, extra_fields=()))


@dataclass(frozen=True)
class InvariantTensorType(EquivariantTensorType):
    kind: ClassVar[str] = "invariant_tensor"
    feature_role: FeatureRole = FeatureRole.CHANNEL

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "feature_role", FeatureRole.parse(self.feature_role))
        nontrivial = [str(irrep) for _, irrep in self.irreps if irrep.degree != 0 or irrep.parity != 1]
        if nontrivial:
            raise DSLValidationError([
                Diagnostic("E_TYPE_006", "InvariantTensorType requires only trivial representations", details={"irreps": nontrivial})
            ])

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["kind"] = self.kind
        payload["feature_role"] = self.feature_role.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InvariantTensorType":
        kwargs = _equivariant_tensor_kwargs(data, extra_fields=("feature_role",))
        kwargs["feature_role"] = FeatureRole.parse(data.get("feature_role", FeatureRole.CHANNEL.value))
        return cls(**kwargs)


@dataclass(frozen=True)
class GridTensorType(ValueType):
    """One finite S2 sample grid per item of a geometric carrier.

    The value is not stored as irreducible-representation coefficients.  Its
    ``source_irreps`` and ``grid`` fields retain the representation boundary
    needed to unproject it, while ``channels`` describes the pointwise feature
    axis shared over every latitude/longitude sample.
    """

    kind: ClassVar[str] = "grid_tensor"
    group: GroupSpec
    carrier: str
    source_irreps: Irreps
    grid: GridSpec
    channels: int
    frame: Frame = field(default_factory=Frame)
    dtype: str = "float32"
    measure: str = "dimensionless"
    level: EquivarianceLevel = EquivarianceLevel.EMPIRICAL
    channel_role: FeatureRole = FeatureRole.CHANNEL
    aliasing_model: str = "finite_grid_truncation"

    def __post_init__(self) -> None:
        Carrier.validate(self.carrier)
        if self.group.dimension != 3 or self.group.family not in ("O3", "SO3"):
            raise DSLValidationError([
                Diagnostic("E_GRID_TYPE_006", "GridTensorType currently requires O(3) or SO(3)")
            ])
        if self.frame.kind == "invariant":
            raise DSLValidationError([
                Diagnostic("E_GRID_TYPE_007", "S2 grid samples cannot use an invariant frame")
            ])
        if self.source_irreps.family != self.group.family:
            raise DSLValidationError([
                Diagnostic(
                    "E_GRID_TYPE_008",
                    "grid source irreps must match the geometric group",
                    expected=self.group.family,
                    actual=self.source_irreps.family,
                )
            ])
        if isinstance(self.channels, bool) or int(self.channels) <= 0:
            raise DSLValidationError([
                Diagnostic("E_GRID_TYPE_009", "grid channels must be a positive integer", actual=str(self.channels))
            ])
        degrees = tuple(irrep.degree for _multiplicity, irrep in self.source_irreps)
        multiplicities = {int(multiplicity) for multiplicity, _irrep in self.source_irreps}
        if degrees != tuple(range(self.grid.lmax + 1)) or multiplicities != {int(self.channels)}:
            raise DSLValidationError([
                Diagnostic(
                    "E_GRID_TYPE_010",
                    "grid source irreps must contain every degree through lmax with one uniform channel multiplicity",
                    expected="degrees=0..{},channels={}".format(self.grid.lmax, self.channels),
                    actual=str(self.source_irreps),
                )
            ])
        if self.dtype not in ("float16", "bfloat16", "float32", "float64"):
            raise DSLValidationError([
                Diagnostic("E_GRID_TYPE_011", "unsupported grid dtype", actual=self.dtype)
            ])
        object.__setattr__(self, "channel_role", FeatureRole.parse(self.channel_role))
        if self.aliasing_model not in ("finite_grid_truncation", "linear_projection_only"):
            raise DSLValidationError([
                Diagnostic("E_GRID_TYPE_012", "unsupported grid aliasing model", actual=self.aliasing_model)
            ])

    def with_channels(self, channels: int, source_irreps: Irreps) -> "GridTensorType":
        return replace(self, channels=int(channels), source_irreps=source_irreps)

    def with_level(self, level: EquivarianceLevel, *, aliasing_model: Optional[str] = None) -> "GridTensorType":
        return replace(
            self,
            level=EquivarianceLevel(level),
            aliasing_model=self.aliasing_model if aliasing_model is None else str(aliasing_model),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "group": self.group.to_dict(),
            "carrier": self.carrier,
            "source_irreps": str(self.source_irreps),
            "grid": self.grid.to_dict(),
            "channels": int(self.channels),
            "frame": {"kind": self.frame.kind, "reference": self.frame.reference},
            "dtype": self.dtype,
            "measure": self.measure,
            "equivariance_level": int(self.level),
            "channel_role": self.channel_role.value,
            "aliasing_model": self.aliasing_model,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GridTensorType":
        allowed = {
            "kind", "schema_version", "group", "carrier", "source_irreps", "grid",
            "channels", "frame", "dtype", "measure", "equivariance_level",
            "channel_role", "aliasing_model",
        }
        _check_tagged_fields(data, allowed, "grid tensor type")
        group = GroupSpec.from_dict(data["group"])
        frame_data = data.get("frame", {"kind": "global", "reference": ""})
        return cls(
            group=group,
            carrier=str(data["carrier"]),
            source_irreps=Irreps.parse(str(data["source_irreps"]), group.family),
            grid=GridSpec.from_dict(data["grid"]),
            channels=int(data["channels"]),
            frame=Frame(str(frame_data.get("kind", "global")), str(frame_data.get("reference", ""))),
            dtype=str(data.get("dtype", "float32")),
            measure=str(data.get("measure", "dimensionless")),
            level=EquivarianceLevel(int(data.get("equivariance_level", 1))),
            channel_role=FeatureRole.parse(data.get("channel_role", FeatureRole.CHANNEL.value)),
            aliasing_model=str(data.get("aliasing_model", "finite_grid_truncation")),
        )


@dataclass(frozen=True)
class CategoricalTensorType(ValueType):
    """One discrete category index per item of a geometric carrier.

    Category indices are invariant under the geometric group but are not
    floating-point trivial irreps.  Keeping them outside
    ``InvariantTensorType`` prevents integer species/atom identifiers from
    entering arithmetic equivariant primitives before an explicit encoding.
    """

    kind: ClassVar[str] = "categorical_tensor"
    group: GroupSpec
    carrier: str
    vocabulary_size: int
    dtype: str = "int64"
    feature_role: FeatureRole = FeatureRole.SPECIES
    labels: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        Carrier.validate(self.carrier)
        if isinstance(self.vocabulary_size, bool) or int(self.vocabulary_size) <= 0:
            raise DSLValidationError([
                Diagnostic(
                    "E_CATEGORICAL_001",
                    "categorical vocabulary_size must be a positive integer",
                    actual=str(self.vocabulary_size),
                )
            ])
        if self.dtype not in ("int32", "int64"):
            raise DSLValidationError([
                Diagnostic(
                    "E_CATEGORICAL_002",
                    "categorical tensors require int32 or int64 storage",
                    actual=self.dtype,
                )
            ])
        object.__setattr__(self, "feature_role", FeatureRole.parse(self.feature_role))
        if self.feature_role != FeatureRole.SPECIES:
            raise DSLValidationError([
                Diagnostic(
                    "E_CATEGORICAL_003",
                    "the first categorical tensor schema uses species semantics",
                    actual=self.feature_role.value,
                )
            ])
        if self.labels and len(self.labels) != int(self.vocabulary_size):
            raise DSLValidationError([
                Diagnostic(
                    "E_CATEGORICAL_004",
                    "categorical labels must be empty or match vocabulary_size",
                    expected=str(int(self.vocabulary_size)),
                    actual=str(len(self.labels)),
                )
            ])
        if len(self.labels) != len(set(self.labels)):
            raise DSLValidationError([
                Diagnostic("E_CATEGORICAL_005", "categorical labels must be unique")
            ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "group": self.group.to_dict(),
            "carrier": self.carrier,
            "vocabulary_size": int(self.vocabulary_size),
            "dtype": self.dtype,
            "feature_role": self.feature_role.value,
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CategoricalTensorType":
        allowed = {
            "kind", "schema_version", "group", "carrier", "vocabulary_size",
            "dtype", "feature_role", "labels",
        }
        _check_tagged_fields(data, allowed, "categorical tensor type")
        return cls(
            group=GroupSpec.from_dict(data["group"]),
            carrier=str(data["carrier"]),
            vocabulary_size=int(data["vocabulary_size"]),
            dtype=str(data.get("dtype", "int64")),
            feature_role=FeatureRole.parse(data.get("feature_role", FeatureRole.SPECIES.value)),
            labels=tuple(str(item) for item in data.get("labels", ())),
        )


@dataclass(frozen=True, init=False)
class RecordType(ValueType):
    kind: ClassVar[str] = "record"
    fields: Tuple[Tuple[str, ValueType], ...]

    def __init__(self, fields: Mapping[str, ValueType] | Sequence[Tuple[str, ValueType]]):
        raw = fields.items() if isinstance(fields, Mapping) else fields
        normalized = tuple(sorted(((str(name), value) for name, value in raw), key=lambda item: item[0]))
        names = [name for name, _ in normalized]
        if not normalized or any(not name for name in names) or len(names) != len(set(names)):
            raise DSLValidationError([Diagnostic("E_TYPE_007", "record fields must be nonempty, named, and unique")])
        if any(not isinstance(value, ValueType) for _, value in normalized):
            raise DSLValidationError([Diagnostic("E_TYPE_008", "record fields must contain ValueType values")])
        object.__setattr__(self, "fields", normalized)

    @property
    def field_map(self) -> Dict[str, ValueType]:
        return dict(self.fields)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "fields": {name: value.to_dict() for name, value in self.fields},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecordType":
        _check_tagged_fields(data, {"kind", "schema_version", "fields"}, "record type")
        fields = data.get("fields", {})
        if not isinstance(fields, Mapping):
            raise DSLValidationError([Diagnostic("E_SCHEMA_007", "record fields must be an object")])
        return cls({str(name): value_type_from_dict(value) for name, value in fields.items()})


@dataclass(frozen=True)
class TupleType(ValueType):
    kind: ClassVar[str] = "tuple"
    items: Tuple[ValueType, ...]

    def __post_init__(self) -> None:
        if not self.items or any(not isinstance(value, ValueType) for value in self.items):
            raise DSLValidationError([Diagnostic("E_TYPE_009", "tuple type requires one or more ValueType items")])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "items": [value.to_dict() for value in self.items],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TupleType":
        _check_tagged_fields(data, {"kind", "schema_version", "items"}, "tuple type")
        return cls(tuple(value_type_from_dict(value) for value in data.get("items", ())))


@dataclass(frozen=True)
class IndexMapType(ValueType):
    kind: ClassVar[str] = "index_map"
    group: GroupSpec
    source_carrier: str
    target_carrier: str
    endpoint: str = "segment"
    index_dtype: str = "int64"
    target_size: Optional[int] = None
    sorted_by_target: bool = False
    allows_empty_targets: bool = True

    def __post_init__(self) -> None:
        Carrier.validate(self.source_carrier)
        Carrier.validate(self.target_carrier)
        if self.source_carrier == self.target_carrier:
            raise DSLValidationError([Diagnostic("E_INDEX_001", "index maps must connect distinct carriers")])
        if self.endpoint not in ("source", "target", "segment", "batch"):
            raise DSLValidationError([Diagnostic("E_INDEX_002", "unsupported index-map role", actual=self.endpoint)])
        if self.index_dtype not in ("int32", "int64"):
            raise DSLValidationError([Diagnostic("E_INDEX_003", "index maps require int32 or int64", actual=self.index_dtype)])
        if self.target_size is not None and int(self.target_size) < 0:
            raise DSLValidationError([Diagnostic("E_INDEX_004", "target_size must be nonnegative or dynamic", actual=str(self.target_size))])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "group": self.group.to_dict(),
            "source_carrier": self.source_carrier,
            "target_carrier": self.target_carrier,
            "endpoint": self.endpoint,
            "index_dtype": self.index_dtype,
            "target_size": self.target_size,
            "sorted_by_target": self.sorted_by_target,
            "allows_empty_targets": self.allows_empty_targets,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IndexMapType":
        allowed = {
            "kind", "schema_version", "group", "source_carrier", "target_carrier",
            "endpoint", "index_dtype", "target_size", "sorted_by_target", "allows_empty_targets",
        }
        _check_tagged_fields(data, allowed, "index-map type")
        raw_size = data.get("target_size")
        return cls(
            group=GroupSpec.from_dict(data["group"]),
            source_carrier=str(data["source_carrier"]),
            target_carrier=str(data["target_carrier"]),
            endpoint=str(data.get("endpoint", "segment")),
            index_dtype=str(data.get("index_dtype", "int64")),
            target_size=None if raw_size is None else int(raw_size),
            sorted_by_target=bool(data.get("sorted_by_target", False)),
            allows_empty_targets=bool(data.get("allows_empty_targets", True)),
        )


@dataclass(frozen=True)
class GraphTopologyType(ValueType):
    kind: ClassVar[str] = "graph_topology"
    group: GroupSpec
    source_index: IndexMapType
    target_index: IndexMapType
    batch_index: Optional[IndexMapType] = None
    directed: bool = True
    adjacency: str = "edge_list"
    edge_order: str = "unspecified"
    periodic_mapping: str = "none"

    def __post_init__(self) -> None:
        for role, index in (("source", self.source_index), ("target", self.target_index)):
            if index.group != self.group:
                raise DSLValidationError([Diagnostic("E_TOPOLOGY_001", "{} index uses a different group".format(role))])
            if index.source_carrier != Carrier.NODE or index.target_carrier != Carrier.EDGE or index.endpoint != role:
                raise DSLValidationError([Diagnostic("E_TOPOLOGY_002", "{} index must map node values to edge endpoints".format(role))])
        if self.batch_index is not None:
            if self.batch_index.group != self.group or self.batch_index.source_carrier != Carrier.NODE or self.batch_index.target_carrier != Carrier.GRAPH:
                raise DSLValidationError([Diagnostic("E_TOPOLOGY_003", "batch index must map nodes to graphs under the same group")])
        if self.adjacency != "edge_list":
            raise DSLValidationError([Diagnostic("E_TOPOLOGY_004", "only explicit edge-list topology is currently supported", actual=self.adjacency)])
        if self.edge_order not in ("unspecified", "source_major", "target_major", "stable"):
            raise DSLValidationError([Diagnostic("E_TOPOLOGY_005", "unsupported edge ordering", actual=self.edge_order)])
        if self.periodic_mapping not in ("none", "lattice_shift"):
            raise DSLValidationError([Diagnostic("E_TOPOLOGY_006", "unsupported periodic mapping", actual=self.periodic_mapping)])
        if self.periodic_mapping == "lattice_shift" and self.group.periodicity != "lattice":
            raise DSLValidationError([Diagnostic("E_TOPOLOGY_007", "lattice-shift topology requires a periodic group")])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "group": self.group.to_dict(),
            "source_index": self.source_index.to_dict(),
            "target_index": self.target_index.to_dict(),
            "batch_index": self.batch_index.to_dict() if self.batch_index is not None else None,
            "directed": self.directed,
            "adjacency": self.adjacency,
            "edge_order": self.edge_order,
            "periodic_mapping": self.periodic_mapping,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphTopologyType":
        allowed = {
            "kind", "schema_version", "group", "source_index", "target_index", "batch_index",
            "directed", "adjacency", "edge_order", "periodic_mapping",
        }
        _check_tagged_fields(data, allowed, "graph-topology type")
        raw_batch = data.get("batch_index")
        return cls(
            group=GroupSpec.from_dict(data["group"]),
            source_index=IndexMapType.from_dict(data["source_index"]),
            target_index=IndexMapType.from_dict(data["target_index"]),
            batch_index=None if raw_batch is None else IndexMapType.from_dict(raw_batch),
            directed=bool(data.get("directed", True)),
            adjacency=str(data.get("adjacency", "edge_list")),
            edge_order=str(data.get("edge_order", "unspecified")),
            periodic_mapping=str(data.get("periodic_mapping", "none")),
        )


@dataclass(frozen=True)
class AffinePointType(ValueType):
    kind: ClassVar[str] = "affine_point"
    group: GroupSpec
    carrier: str = Carrier.NODE
    coordinate_dimension: int = 3
    frame: Frame = field(default_factory=Frame)
    axes: Tuple[str, ...] = field(default_factory=tuple)
    axis_specs: Tuple[AxisSpec, ...] = field(default_factory=tuple)
    dtype: str = "float32"
    measure: str = "length"

    def __post_init__(self) -> None:
        Carrier.validate(self.carrier)
        if self.carrier != Carrier.NODE:
            raise DSLValidationError([Diagnostic("E_AFFINE_001", "the first AffinePointType version is node-carried", actual=self.carrier)])
        if self.coordinate_dimension != self.group.dimension:
            raise DSLValidationError([
                Diagnostic("E_AFFINE_002", "affine coordinate dimension must equal the group dimension", expected=str(self.group.dimension), actual=str(self.coordinate_dimension))
            ])
        if self.frame.kind != "global":
            raise DSLValidationError([Diagnostic("E_AFFINE_003", "affine points must use the global frame in the first schema version", actual=str(self.frame))])
        if self.dtype not in ("float16", "bfloat16", "float32", "float64"):
            raise DSLValidationError([Diagnostic("E_TYPE_003", "unsupported dtype", actual=self.dtype)])
        if self.measure == "dimensionless" or not self.measure:
            raise DSLValidationError([Diagnostic("E_AFFINE_004", "affine points require a non-dimensionless length measure", actual=self.measure)])
        specs = self.axis_specs
        if not specs and self.axes:
            specs = tuple(AxisSpec(name=name, role=_infer_feature_role(name), order=index) for index, name in enumerate(self.axes))
            object.__setattr__(self, "axis_specs", specs)
        object.__setattr__(self, "axes", _validate_axis_specs(self.axes, specs))

    def compatible(self, other: "ValueType", **_: Any) -> bool:
        return isinstance(other, AffinePointType) and self == other

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "group": self.group.to_dict(),
            "carrier": self.carrier,
            "coordinate_dimension": self.coordinate_dimension,
            "frame": {"kind": self.frame.kind, "reference": self.frame.reference},
            "axes": list(self.axes),
            "axis_specs": [item.to_dict() for item in self.axis_specs],
            "dtype": self.dtype,
            "measure": self.measure,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AffinePointType":
        allowed = {
            "kind", "schema_version", "group", "carrier", "coordinate_dimension", "frame",
            "axes", "axis_specs", "dtype", "measure",
        }
        _check_tagged_fields(data, allowed, "affine-point type")
        group = GroupSpec.from_dict(data["group"])
        frame_data = data.get("frame", {"kind": "global", "reference": ""})
        return cls(
            group=group,
            carrier=str(data.get("carrier", Carrier.NODE)),
            coordinate_dimension=int(data.get("coordinate_dimension", group.dimension)),
            frame=Frame(str(frame_data.get("kind", "global")), str(frame_data.get("reference", ""))),
            axes=tuple(str(item) for item in data.get("axes", ())),
            axis_specs=tuple(AxisSpec.from_dict(item) for item in data.get("axis_specs", ())),
            dtype=str(data.get("dtype", "float32")),
            measure=str(data.get("measure", "length")),
        )


@dataclass(frozen=True)
class LatticeType(ValueType):
    kind: ClassVar[str] = "lattice"
    group: GroupSpec
    lattice_id: str
    carrier: str = Carrier.GRAPH
    dtype: str = "float32"
    measure: str = "length"
    matrix_convention: str = "row_vectors"
    handedness: str = "right"

    def __post_init__(self) -> None:
        if self.group.periodicity != "lattice":
            raise DSLValidationError([Diagnostic("E_LATTICE_001", "LatticeType requires group.periodicity=lattice")])
        Carrier.validate(self.carrier)
        if self.carrier != Carrier.GRAPH:
            raise DSLValidationError([Diagnostic("E_LATTICE_002", "lattice values must be graph-carried", actual=self.carrier)])
        if not self.lattice_id:
            raise DSLValidationError([Diagnostic("E_LATTICE_003", "lattice_id must be nonempty")])
        if self.dtype not in ("float32", "float64"):
            raise DSLValidationError([Diagnostic("E_LATTICE_004", "lattice matrices require float32 or float64", actual=self.dtype)])
        if self.measure == "dimensionless" or not self.measure:
            raise DSLValidationError([Diagnostic("E_LATTICE_005", "lattice matrices require a length measure", actual=self.measure)])
        if self.matrix_convention != "row_vectors":
            raise DSLValidationError([Diagnostic("E_LATTICE_006", "the first lattice contract uses row vectors", actual=self.matrix_convention)])
        if self.handedness not in ("right", "left"):
            raise DSLValidationError([Diagnostic("E_LATTICE_007", "unsupported lattice handedness", actual=self.handedness)])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "group": self.group.to_dict(),
            "lattice_id": self.lattice_id,
            "carrier": self.carrier,
            "dtype": self.dtype,
            "measure": self.measure,
            "matrix_convention": self.matrix_convention,
            "handedness": self.handedness,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LatticeType":
        allowed = {
            "kind", "schema_version", "group", "lattice_id", "carrier", "dtype",
            "measure", "matrix_convention", "handedness",
        }
        _check_tagged_fields(data, allowed, "lattice type")
        return cls(
            group=GroupSpec.from_dict(data["group"]),
            lattice_id=str(data["lattice_id"]),
            carrier=str(data.get("carrier", Carrier.GRAPH)),
            dtype=str(data.get("dtype", "float32")),
            measure=str(data.get("measure", "length")),
            matrix_convention=str(data.get("matrix_convention", "row_vectors")),
            handedness=str(data.get("handedness", "right")),
        )


@dataclass(frozen=True)
class LatticeShiftType(ValueType):
    kind: ClassVar[str] = "lattice_shift"
    group: GroupSpec
    lattice_id: str
    carrier: str = Carrier.EDGE
    dtype: str = "int64"
    convention: str = "target_image"

    def __post_init__(self) -> None:
        if self.group.periodicity != "lattice":
            raise DSLValidationError([Diagnostic("E_LATTICE_008", "LatticeShiftType requires group.periodicity=lattice")])
        Carrier.validate(self.carrier)
        if self.carrier != Carrier.EDGE:
            raise DSLValidationError([Diagnostic("E_LATTICE_009", "lattice shifts must be edge-carried", actual=self.carrier)])
        if not self.lattice_id:
            raise DSLValidationError([Diagnostic("E_LATTICE_010", "lattice shift requires a lattice_id")])
        if self.dtype not in ("int32", "int64"):
            raise DSLValidationError([Diagnostic("E_LATTICE_011", "lattice shifts require int32 or int64", actual=self.dtype)])
        if self.convention not in ("target_image", "source_image"):
            raise DSLValidationError([
                Diagnostic(
                    "E_LATTICE_012",
                    "lattice-shift convention must identify the shifted source or target image",
                    actual=self.convention,
                )
            ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "group": self.group.to_dict(),
            "lattice_id": self.lattice_id,
            "carrier": self.carrier,
            "dtype": self.dtype,
            "convention": self.convention,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LatticeShiftType":
        allowed = {"kind", "schema_version", "group", "lattice_id", "carrier", "dtype", "convention"}
        _check_tagged_fields(data, allowed, "lattice-shift type")
        return cls(
            group=GroupSpec.from_dict(data["group"]),
            lattice_id=str(data["lattice_id"]),
            carrier=str(data.get("carrier", Carrier.EDGE)),
            dtype=str(data.get("dtype", "int64")),
            convention=str(data.get("convention", "target_image")),
        )


def _infer_feature_role(name: str) -> FeatureRole:
    try:
        return FeatureRole.parse(name)
    except DSLValidationError:
        return FeatureRole.CHANNEL


def _check_schema_version(data: Mapping[str, Any]) -> None:
    version = str(data.get("schema_version", ""))
    if version != VALUE_TYPE_SCHEMA_VERSION:
        raise DSLValidationError([
            Diagnostic("E_SCHEMA_008", "unsupported value-type schema version", expected=VALUE_TYPE_SCHEMA_VERSION, actual=version)
        ])


def _check_tagged_fields(data: Mapping[str, Any], allowed: set, kind: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise DSLValidationError([Diagnostic("E_SCHEMA_009", "unknown {} fields".format(kind), details={"fields": sorted(unknown)})])
    _check_schema_version(data)


def _equivariant_tensor_kwargs(data: Mapping[str, Any], *, extra_fields: Sequence[str]) -> Dict[str, Any]:
    allowed = {
        "kind", "schema_version", "group", "carrier", "irreps", "frame", "axes",
        "dtype", "measure", "equivariance_level", "axis_specs", "layout",
        *extra_fields,
    }
    _check_tagged_fields(data, allowed, "equivariant tensor type")
    group = GroupSpec.from_dict(data["group"])
    frame_data = data.get("frame", {"kind": "global", "reference": ""})
    return {
        "group": group,
        "carrier": str(data["carrier"]),
        "irreps": Irreps.parse(str(data["irreps"]), group.family),
        "frame": Frame(str(frame_data.get("kind", "global")), str(frame_data.get("reference", ""))),
        "axes": tuple(str(item) for item in data.get("axes", ())),
        "dtype": str(data.get("dtype", "float32")),
        "measure": str(data.get("measure", "dimensionless")),
        "level": EquivarianceLevel(int(data.get("equivariance_level", 2))),
        "axis_specs": tuple(AxisSpec.from_dict(item) for item in data.get("axis_specs", ())),
        "layout": RepresentationLayout.from_dict(data.get("layout", {})),
    }


def value_type_from_dict(data: Mapping[str, Any]) -> ValueType:
    if not isinstance(data, Mapping):
        raise DSLValidationError([Diagnostic("E_SCHEMA_010", "value type must be an object")])
    kind = data.get("kind")
    if kind is None:
        return EquivariantType.from_dict(data)
    dispatch = {
        EquivariantTensorType.kind: EquivariantTensorType.from_dict,
        InvariantTensorType.kind: InvariantTensorType.from_dict,
        GridTensorType.kind: GridTensorType.from_dict,
        CategoricalTensorType.kind: CategoricalTensorType.from_dict,
        RecordType.kind: RecordType.from_dict,
        TupleType.kind: TupleType.from_dict,
        IndexMapType.kind: IndexMapType.from_dict,
        GraphTopologyType.kind: GraphTopologyType.from_dict,
        AffinePointType.kind: AffinePointType.from_dict,
        LatticeType.kind: LatticeType.from_dict,
        LatticeShiftType.kind: LatticeShiftType.from_dict,
    }
    loader = dispatch.get(str(kind))
    if loader is None:
        raise DSLValidationError([Diagnostic("E_SCHEMA_011", "unknown value-type kind", actual=str(kind))])
    return loader(data)


def value_type_group_families(value: ValueType) -> Tuple[str, ...]:
    if isinstance(value, EquivariantType):
        return (value.group.family,)
    if isinstance(value, (GridTensorType, CategoricalTensorType, IndexMapType, GraphTopologyType, AffinePointType, LatticeType, LatticeShiftType)):
        return (value.group.family,)
    if isinstance(value, RecordType):
        return tuple(sorted({family for _, item in value.fields for family in value_type_group_families(item)}))
    if isinstance(value, TupleType):
        return tuple(sorted({family for item in value.items for family in value_type_group_families(item)}))
    return ()


def value_type_groups(value: ValueType) -> Tuple[GroupSpec, ...]:
    if isinstance(value, EquivariantType):
        return (value.group,)
    if isinstance(value, (GridTensorType, CategoricalTensorType, IndexMapType, GraphTopologyType, AffinePointType, LatticeType, LatticeShiftType)):
        return (value.group,)
    if isinstance(value, RecordType):
        return tuple(sorted({group for _, item in value.fields for group in value_type_groups(item)}))
    if isinstance(value, TupleType):
        return tuple(sorted({group for item in value.items for group in value_type_groups(item)}))
    return ()
