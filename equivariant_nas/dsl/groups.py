"""Group and symmetry contracts shared by DSL types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from .diagnostics import DSLValidationError, Diagnostic


_SUPPORTED = {
    "SO2": 2,
    "O2": 2,
    "Cyclic2D": 2,
    "Dihedral2D": 2,
    "SO3": 3,
    "O3": 3,
}


@dataclass(frozen=True, order=True)
class GroupSpec:
    family: str
    dimension: int
    order: int = 0
    translation: str = "relative_coordinates"
    permutation: str = "node_set"
    periodicity: str = "none"

    def __post_init__(self) -> None:
        expected = _SUPPORTED.get(self.family)
        if expected is None:
            raise DSLValidationError([Diagnostic("E_GROUP_001", "unsupported group family", actual=self.family)])
        if self.dimension != expected:
            raise DSLValidationError([
                Diagnostic("E_GROUP_002", "group dimension mismatch", expected=str(expected), actual=str(self.dimension))
            ])
        if self.family in ("Cyclic2D", "Dihedral2D") and self.order < 2:
            raise DSLValidationError([Diagnostic("E_GROUP_003", "finite 2D groups require order >= 2")])
        if self.family not in ("Cyclic2D", "Dihedral2D") and self.order != 0:
            raise DSLValidationError([Diagnostic("E_GROUP_004", "continuous groups must not set finite order")])
        if self.translation not in ("none", "relative_coordinates", "explicit"):
            raise DSLValidationError([Diagnostic("E_GROUP_005", "unsupported translation policy")])
        if self.permutation not in ("none", "node_set"):
            raise DSLValidationError([Diagnostic("E_GROUP_006", "unsupported permutation policy")])
        if self.periodicity not in ("none", "lattice"):
            raise DSLValidationError([Diagnostic("E_GROUP_007", "unsupported periodicity policy")])

    @classmethod
    def o3(cls) -> "GroupSpec":
        return cls("O3", 3)

    @classmethod
    def so3(cls) -> "GroupSpec":
        return cls("SO3", 3)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroupSpec":
        allowed = {"family", "dimension", "order", "translation", "permutation", "periodicity"}
        unknown = set(data) - allowed
        if unknown:
            raise DSLValidationError([Diagnostic("E_SCHEMA_001", "unknown group fields", details={"fields": sorted(unknown)})])
        return cls(
            family=str(data["family"]),
            dimension=int(data["dimension"]),
            order=int(data.get("order", 0)),
            translation=str(data.get("translation", "relative_coordinates")),
            permutation=str(data.get("permutation", "node_set")),
            periodicity=str(data.get("periodicity", "none")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "dimension": self.dimension,
            "order": self.order,
            "translation": self.translation,
            "permutation": self.permutation,
            "periodicity": self.periodicity,
        }
