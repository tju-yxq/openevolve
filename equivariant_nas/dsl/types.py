"""Typed values carried by EvoEquiLang graph edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Mapping, Tuple

from .diagnostics import DSLValidationError, Diagnostic
from .groups import GroupSpec
from .irreps import Irreps


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
class EquivariantType:
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
        return EquivariantType(self.group, self.carrier, irreps, self.frame, self.axes, self.dtype, self.measure, self.level)

    def with_carrier(self, carrier: str) -> "EquivariantType":
        return EquivariantType(self.group, carrier, self.irreps, self.frame, self.axes, self.dtype, self.measure, self.level)

    def with_frame(self, frame: Frame) -> "EquivariantType":
        return EquivariantType(self.group, self.carrier, self.irreps, frame, self.axes, self.dtype, self.measure, self.level)

    def compatible(self, other: "EquivariantType", *, exact_irreps: bool = True) -> bool:
        return (
            self.group == other.group
            and self.carrier == other.carrier
            and self.frame == other.frame
            and (self.irreps == other.irreps or not exact_irreps)
            and self.axes == other.axes
            and self.measure == other.measure
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
