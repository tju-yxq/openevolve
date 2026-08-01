"""Typed trainable/external parameter contracts for primitive instances."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Tuple, Union

from .diagnostics import DSLValidationError, Diagnostic


PARAMETER_CONTRACT_SCHEMA_VERSION = "evoequilang-parameter-contract-v1@1"
ParameterSize = Union[int, str]


def _strict_fields(data: Mapping[str, Any], allowed, kind: str) -> None:
    unknown = set(data) - set(allowed)
    if unknown:
        raise DSLValidationError([
            Diagnostic(
                "E_PARAMETER_001",
                "unknown {} fields".format(kind),
                details={"fields": sorted(unknown)},
            )
        ])


@dataclass(frozen=True, order=True)
class ParameterAxis:
    """One logical axis of a trainable or externally supplied parameter."""

    name: str
    size: ParameterSize
    role: str
    source_axis: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.role:
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_002", "parameter axis name and role must be nonempty")
            ])
        if isinstance(self.size, bool) or not isinstance(self.size, (int, str)):
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_003", "parameter axis size must be a positive integer or symbol", actual=str(self.size))
            ])
        if isinstance(self.size, int) and self.size <= 0:
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_003", "parameter axis size must be a positive integer or symbol", actual=str(self.size))
            ])
        if isinstance(self.size, str) and not self.size.strip():
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_003", "parameter axis size symbol must be nonempty")
            ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "role": self.role,
            "source_axis": self.source_axis,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterAxis":
        _strict_fields(data, ("name", "size", "role", "source_axis"), "parameter-axis")
        raw_size = data["size"]
        size = int(raw_size) if isinstance(raw_size, int) and not isinstance(raw_size, bool) else str(raw_size)
        return cls(
            name=str(data["name"]),
            size=size,
            role=str(data["role"]),
            source_axis=str(data.get("source_axis", "")),
        )


@dataclass(frozen=True)
class ParameterContract:
    """Concrete parameter semantics inferred for one primitive node."""

    name: str
    axes: Tuple[ParameterAxis, ...]
    storage: str = "internal"
    sharing_axes: Tuple[str, ...] = field(default_factory=tuple)
    trainable: bool = True
    is_bias: bool = False
    bias_irreps: str = ""
    initializer: str = "backend_default"
    rescale: float = 1.0
    checkpoint_names: Tuple[str, ...] = field(default_factory=tuple)
    backend_parameter_name: str = ""
    external_port: str = ""
    architecture_identity: str = "contract_only"
    schema_version: str = PARAMETER_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise DSLValidationError([Diagnostic("E_PARAMETER_004", "parameter name must be nonempty")])
        axis_names = [axis.name for axis in self.axes]
        if len(axis_names) != len(set(axis_names)):
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_005", "parameter axis names must be unique", actual=self.name)
            ])
        if self.storage not in ("internal", "external"):
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_006", "parameter storage must be internal or external", actual=self.storage)
            ])
        if self.storage == "external" and not self.external_port:
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_007", "external parameter contract requires an explicit input port", actual=self.name)
            ])
        if self.storage == "internal" and self.external_port:
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_008", "internal parameter contract cannot declare an external input port", actual=self.name)
            ])
        if self.storage == "internal" and not self.backend_parameter_name:
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_009", "internal parameter contract requires a backend parameter name", actual=self.name)
            ])
        if len(self.sharing_axes) != len(set(self.sharing_axes)) or any(not item for item in self.sharing_axes):
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_010", "sharing axes must be nonempty and unique", actual=self.name)
            ])
        if self.is_bias and not self.bias_irreps:
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_011", "bias parameter contract must declare allowed bias irreps", actual=self.name)
            ])
        if not self.initializer:
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_012", "parameter initializer must be nonempty", actual=self.name)
            ])
        if not math.isfinite(float(self.rescale)) or float(self.rescale) <= 0.0:
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_013", "parameter rescale must be finite and positive", actual=str(self.rescale))
            ])
        if len(self.checkpoint_names) != len(set(self.checkpoint_names)) or any(not item for item in self.checkpoint_names):
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_014", "checkpoint names must be nonempty and unique", actual=self.name)
            ])
        if self.architecture_identity not in ("contract_only", "value", "excluded"):
            raise DSLValidationError([
                Diagnostic(
                    "E_PARAMETER_015",
                    "unsupported parameter architecture identity policy",
                    actual=self.architecture_identity,
                )
            ])
        if self.schema_version != PARAMETER_CONTRACT_SCHEMA_VERSION:
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_016", "unsupported parameter contract schema version", actual=self.schema_version)
            ])

    @property
    def shape(self) -> Tuple[ParameterSize, ...]:
        return tuple(axis.size for axis in self.axes)

    @property
    def concrete_shape(self) -> Tuple[int, ...] | None:
        if any(not isinstance(size, int) for size in self.shape):
            return None
        return tuple(int(size) for size in self.shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "parameter_contract",
            "schema_version": self.schema_version,
            "name": self.name,
            "axes": [axis.to_dict() for axis in self.axes],
            "storage": self.storage,
            "sharing_axes": list(self.sharing_axes),
            "trainable": self.trainable,
            "is_bias": self.is_bias,
            "bias_irreps": self.bias_irreps,
            "initializer": self.initializer,
            "rescale": self.rescale,
            "checkpoint_names": list(self.checkpoint_names),
            "backend_parameter_name": self.backend_parameter_name,
            "external_port": self.external_port,
            "architecture_identity": self.architecture_identity,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterContract":
        _strict_fields(
            data,
            (
                "kind", "schema_version", "name", "axes", "storage", "sharing_axes",
                "trainable", "is_bias", "bias_irreps", "initializer", "rescale",
                "checkpoint_names", "backend_parameter_name", "external_port",
                "architecture_identity",
            ),
            "parameter-contract",
        )
        if data.get("kind") != "parameter_contract":
            raise DSLValidationError([
                Diagnostic("E_PARAMETER_017", "invalid parameter contract kind", actual=str(data.get("kind")))
            ])
        return cls(
            name=str(data["name"]),
            axes=tuple(ParameterAxis.from_dict(item) for item in data.get("axes", ())),
            storage=str(data.get("storage", "internal")),
            sharing_axes=tuple(str(item) for item in data.get("sharing_axes", ())),
            trainable=bool(data.get("trainable", True)),
            is_bias=bool(data.get("is_bias", False)),
            bias_irreps=str(data.get("bias_irreps", "")),
            initializer=str(data.get("initializer", "backend_default")),
            rescale=float(data.get("rescale", 1.0)),
            checkpoint_names=tuple(str(item) for item in data.get("checkpoint_names", ())),
            backend_parameter_name=str(data.get("backend_parameter_name", "")),
            external_port=str(data.get("external_port", "")),
            architecture_identity=str(data.get("architecture_identity", "contract_only")),
            schema_version=str(data.get("schema_version", PARAMETER_CONTRACT_SCHEMA_VERSION)),
        )

    def content_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
