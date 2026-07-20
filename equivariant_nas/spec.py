"""Typed architecture grammar for Equiformer V1.

The grammar is intentionally narrower than arbitrary source-code evolution.
Every field maps to a constructor argument whose symmetry semantics are known.
This makes invalid candidates cheap to reject and makes parent/child changes
auditable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


class SpecValidationError(ValueError):
    """Raised when a candidate violates the architecture grammar."""


class EvolutionFactor(str, Enum):
    REPRESENTATION = "REPRESENTATION"
    OPERATOR = "OPERATOR"
    ACTION = "ACTION"
    MACRO = "MACRO"


_CHANNEL_CHOICES = (8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256)
_SCALAR_CHOICES = (64, 96, 128, 160, 192, 256)
_FEATURE_CHOICES = (256, 384, 512, 640)


@dataclass(frozen=True)
class RepresentationSpec:
    lmax: int = 2
    scalar_channels: int = 128
    vector_channels: int = 64
    tensor_channels: int = 32
    l3_channels: int = 0
    head_scalar_channels: int = 32
    head_vector_channels: int = 16
    head_tensor_channels: int = 8
    head_l3_channels: int = 0
    mlp_multiplier: int = 3
    feature_channels: int = 512

    def validate(self) -> None:
        if self.lmax not in (1, 2, 3):
            raise SpecValidationError("lmax must be one of 1, 2, 3")
        if self.scalar_channels not in _SCALAR_CHOICES:
            raise SpecValidationError("unsupported scalar channel count")
        for name in ("vector_channels", "tensor_channels", "l3_channels"):
            value = getattr(self, name)
            if value != 0 and value not in _CHANNEL_CHOICES:
                raise SpecValidationError("unsupported {}={}".format(name, value))
        for name in (
            "head_scalar_channels",
            "head_vector_channels",
            "head_tensor_channels",
            "head_l3_channels",
        ):
            value = getattr(self, name)
            if value != 0 and value not in _CHANNEL_CHOICES:
                raise SpecValidationError("unsupported {}={}".format(name, value))
        if self.head_scalar_channels <= 0:
            raise SpecValidationError("attention requires scalar head channels")
        if self.mlp_multiplier not in (2, 3, 4):
            raise SpecValidationError("mlp_multiplier must be 2, 3, or 4")
        if self.feature_channels not in _FEATURE_CHOICES:
            raise SpecValidationError("unsupported feature channel count")

        active = {
            1: (self.vector_channels, self.head_vector_channels),
            2: (self.tensor_channels, self.head_tensor_channels),
            3: (self.l3_channels, self.head_l3_channels),
        }
        for degree, pair in active.items():
            if degree <= self.lmax and (pair[0] <= 0 or pair[1] <= 0):
                raise SpecValidationError(
                    "l={} requires positive embedding and head channels".format(degree)
                )
            if degree > self.lmax and pair != (0, 0):
                raise SpecValidationError(
                    "channels above lmax must be zero (l={})".format(degree)
                )

    def embedding_irreps(self) -> str:
        parts = ["{}x0e".format(self.scalar_channels)]
        if self.lmax >= 1:
            parts.append("{}x1e".format(self.vector_channels))
        if self.lmax >= 2:
            parts.append("{}x2e".format(self.tensor_channels))
        if self.lmax >= 3:
            parts.append("{}x3e".format(self.l3_channels))
        return "+".join(parts)

    def head_irreps(self) -> str:
        parts = ["{}x0e".format(self.head_scalar_channels)]
        if self.lmax >= 1:
            parts.append("{}x1e".format(self.head_vector_channels))
        if self.lmax >= 2:
            parts.append("{}x2e".format(self.head_tensor_channels))
        if self.lmax >= 3:
            parts.append("{}x3e".format(self.head_l3_channels))
        return "+".join(parts)

    def spherical_harmonics_irreps(self) -> str:
        return "+".join("1x{}e".format(degree) for degree in range(self.lmax + 1))

    def mlp_irreps(self) -> str:
        values = [
            (0, self.scalar_channels),
            (1, self.vector_channels),
            (2, self.tensor_channels),
            (3, self.l3_channels),
        ]
        return "+".join(
            "{}x{}e".format(channels * self.mlp_multiplier, degree)
            for degree, channels in values
            if degree <= self.lmax
        )

    def capacity_profile(self) -> Dict[str, float]:
        total = float(
            self.scalar_channels
            + self.vector_channels
            + self.tensor_channels
            + self.l3_channels
        )
        return {
            "scalar_fraction": self.scalar_channels / total,
            "vector_fraction": self.vector_channels / total,
            "higher_order_fraction": (self.tensor_channels + self.l3_channels) / total,
        }


@dataclass(frozen=True)
class OperatorSpec:
    basis_type: str = "gaussian"
    num_basis: int = 128
    radial_hidden: Tuple[int, int] = (64, 64)
    nonlinear_message: bool = True
    num_heads: int = 4

    def validate(self) -> None:
        if self.basis_type not in ("gaussian", "bessel"):
            raise SpecValidationError("basis_type must be gaussian or bessel")
        if self.num_basis not in (32, 64, 96, 128):
            raise SpecValidationError("unsupported radial basis count")
        if tuple(self.radial_hidden) not in ((32, 32), (64, 64), (96, 96), (128, 128)):
            raise SpecValidationError("unsupported radial_hidden")
        if self.num_heads not in (2, 4, 8):
            raise SpecValidationError("num_heads must be 2, 4, or 8")


@dataclass(frozen=True)
class ActionSpec:
    norm_layer: str = "layer"
    rescale_degree: bool = False
    alpha_drop: float = 0.2
    projection_drop: float = 0.0
    output_drop: float = 0.0
    drop_path: float = 0.0

    def validate(self) -> None:
        if self.norm_layer not in ("layer", "instance", "graph", "fast_layer"):
            raise SpecValidationError("unsupported norm layer")
        for name in ("alpha_drop", "projection_drop", "output_drop", "drop_path"):
            value = float(getattr(self, name))
            if value not in (0.0, 0.05, 0.1, 0.2):
                raise SpecValidationError("unsupported {}={}".format(name, value))


@dataclass(frozen=True)
class MacroSpec:
    num_layers: int = 6
    radius: float = 5.0

    def validate(self) -> None:
        if self.num_layers not in (3, 4, 5, 6, 7, 8):
            raise SpecValidationError("num_layers must be between 3 and 8")
        if float(self.radius) not in (4.0, 5.0, 6.0):
            raise SpecValidationError("radius must be 4.0, 5.0, or 6.0")


@dataclass(frozen=True)
class ArchitectureSpec:
    schema_version: int = 1
    representation: RepresentationSpec = field(default_factory=RepresentationSpec)
    operator: OperatorSpec = field(default_factory=OperatorSpec)
    action: ActionSpec = field(default_factory=ActionSpec)
    macro: MacroSpec = field(default_factory=MacroSpec)

    def validate(self) -> "ArchitectureSpec":
        if self.schema_version != 1:
            raise SpecValidationError("unsupported schema_version")
        self.representation.validate()
        self.operator.validate()
        self.action.validate()
        self.macro.validate()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def architecture_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]

    def changed_factors(self, other: "ArchitectureSpec") -> Tuple[EvolutionFactor, ...]:
        changed = []
        for factor, attr in (
            (EvolutionFactor.REPRESENTATION, "representation"),
            (EvolutionFactor.OPERATOR, "operator"),
            (EvolutionFactor.ACTION, "action"),
            (EvolutionFactor.MACRO, "macro"),
        ):
            if getattr(self, attr) != getattr(other, attr):
                changed.append(factor)
        return tuple(changed)

    def assert_factor_local_change(
        self, child: "ArchitectureSpec", selected: EvolutionFactor
    ) -> None:
        changed = self.changed_factors(child)
        if changed != (selected,):
            raise SpecValidationError(
                "expected exactly {} to change; observed {}".format(
                    selected.value, [item.value for item in changed]
                )
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArchitectureSpec":
        allowed = {"schema_version", "representation", "operator", "action", "macro"}
        unknown = set(data) - allowed
        if unknown:
            raise SpecValidationError("unknown top-level fields: {}".format(sorted(unknown)))
        representation = _strict_dataclass(RepresentationSpec, data.get("representation", {}))
        operator_data = dict(data.get("operator", {}))
        if "radial_hidden" in operator_data:
            operator_data["radial_hidden"] = tuple(operator_data["radial_hidden"])
        operator = _strict_dataclass(OperatorSpec, operator_data)
        action = _strict_dataclass(ActionSpec, data.get("action", {}))
        macro = _strict_dataclass(MacroSpec, data.get("macro", {}))
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            representation=representation,
            operator=operator,
            action=action,
            macro=macro,
        ).validate()

    @classmethod
    def from_json(cls, text: str) -> "ArchitectureSpec":
        value = json.loads(text)
        if not isinstance(value, dict):
            raise SpecValidationError("architecture JSON must contain an object")
        return cls.from_dict(value)


def _strict_dataclass(kind: Any, values: Mapping[str, Any]) -> Any:
    if not isinstance(values, Mapping):
        raise SpecValidationError("{} must be an object".format(kind.__name__))
    fields = set(kind.__dataclass_fields__)
    unknown = set(values) - fields
    if unknown:
        raise SpecValidationError(
            "unknown {} fields: {}".format(kind.__name__, sorted(unknown))
        )
    try:
        return kind(**dict(values))
    except TypeError as exc:
        raise SpecValidationError(str(exc))


def baseline_spec() -> ArchitectureSpec:
    """The official nonlinear-l2 Equiformer V1 architecture."""

    return ArchitectureSpec().validate()

