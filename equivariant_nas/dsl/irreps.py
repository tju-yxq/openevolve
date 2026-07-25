"""Canonical irreducible-representation algebra for O(3), SO(3), O(2), and SO(2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from .diagnostics import DSLValidationError, Diagnostic


_O3_TERM = re.compile(r"^(?:(\d+)x)?(\d+)([eo])?$")
_SO2_TERM = re.compile(r"^(?:(\d+)x)?m(-?\d+)([eo])?$")


@dataclass(frozen=True, order=True)
class Irrep:
    degree: int
    parity: int = 1
    family: str = "O3"

    def __post_init__(self) -> None:
        if self.degree < 0 and self.family not in ("SO2", "O2"):
            raise DSLValidationError([Diagnostic("E_IRREP_001", "negative degree is only valid for 2D frequency representations")])
        if self.parity not in (-1, 1):
            raise DSLValidationError([Diagnostic("E_IRREP_002", "parity must be +1 or -1")])
        if self.family not in ("O3", "SO3", "O2", "SO2"):
            raise DSLValidationError([Diagnostic("E_IRREP_003", "unsupported irrep family", actual=self.family)])
        if self.family in ("SO3", "SO2") and self.parity != 1:
            raise DSLValidationError([Diagnostic("E_IRREP_004", "special orthogonal groups do not carry parity labels")])

    @property
    def dimension(self) -> int:
        if self.family in ("O3", "SO3"):
            return 2 * self.degree + 1
        return 1 if self.degree == 0 else 2

    def tensor_product(self, other: "Irrep") -> Tuple["Irrep", ...]:
        if self.family != other.family:
            raise DSLValidationError([Diagnostic("E_IRREP_005", "cannot couple irreps from different group families")])
        parity = self.parity * other.parity
        if self.family in ("O3", "SO3"):
            return tuple(Irrep(l, parity, self.family) for l in range(abs(self.degree - other.degree), self.degree + other.degree + 1))
        frequencies = {abs(self.degree + other.degree), abs(self.degree - other.degree)}
        return tuple(Irrep(m, parity, self.family) for m in sorted(frequencies))

    def __str__(self) -> str:
        if self.family in ("O3", "SO3"):
            suffix = "" if self.family == "SO3" else ("e" if self.parity == 1 else "o")
            return "{}{}".format(self.degree, suffix)
        suffix = "" if self.family == "SO2" else ("e" if self.parity == 1 else "o")
        return "m{}{}".format(self.degree, suffix)


@dataclass(frozen=True)
class Irreps:
    terms: Tuple[Tuple[int, Irrep], ...]

    def __post_init__(self) -> None:
        for multiplicity, _ in self.terms:
            if multiplicity <= 0:
                raise DSLValidationError([Diagnostic("E_IRREP_006", "irrep multiplicity must be positive")])
        families = {ir.family for _, ir in self.terms}
        if len(families) > 1:
            raise DSLValidationError([Diagnostic("E_IRREP_007", "an Irreps value cannot mix group families")])

    @classmethod
    def parse(cls, text: str, family: str = "O3") -> "Irreps":
        compact = text.replace(" ", "")
        if not compact:
            return cls(())
        parsed: List[Tuple[int, Irrep]] = []
        pattern = _O3_TERM if family in ("O3", "SO3") else _SO2_TERM
        for raw in compact.split("+"):
            match = pattern.match(raw)
            if not match:
                raise DSLValidationError([Diagnostic("E_IRREP_008", "invalid irrep term", actual=raw)])
            multiplicity = int(match.group(1) or 1)
            degree = int(match.group(2))
            suffix = match.group(3)
            if family in ("SO3", "SO2") and suffix:
                raise DSLValidationError([Diagnostic("E_IRREP_009", "parity suffix is invalid for special orthogonal group", actual=raw)])
            parity = -1 if suffix == "o" else 1
            parsed.append((multiplicity, Irrep(degree, parity, family)))
        return cls(tuple(parsed)).simplify()

    @property
    def family(self) -> str:
        return self.terms[0][1].family if self.terms else "O3"

    @property
    def dimension(self) -> int:
        return sum(mul * ir.dimension for mul, ir in self.terms)

    def simplify(self) -> "Irreps":
        counts: Dict[Irrep, int] = {}
        for multiplicity, irrep in self.terms:
            counts[irrep] = counts.get(irrep, 0) + multiplicity
        return Irreps(tuple((counts[ir], ir) for ir in sorted(counts)))

    def contains(self, irrep: Irrep) -> bool:
        return any(item == irrep for _, item in self.terms)

    def multiplicity(self, irrep: Irrep) -> int:
        return sum(mul for mul, item in self.terms if item == irrep)

    def direct_sum(self, other: "Irreps") -> "Irreps":
        if self.terms and other.terms and self.family != other.family:
            raise DSLValidationError([Diagnostic("E_IRREP_010", "cannot concatenate irreps from different families")])
        return Irreps(self.terms + other.terms).simplify()

    def allowed_tensor_product_outputs(self, other: "Irreps") -> Tuple[Irrep, ...]:
        outputs = set()
        for _, left in self.terms:
            for _, right in other.terms:
                outputs.update(left.tensor_product(right))
        return tuple(sorted(outputs))

    def __iter__(self) -> Iterator[Tuple[int, Irrep]]:
        return iter(self.terms)

    def __bool__(self) -> bool:
        return bool(self.terms)

    def __str__(self) -> str:
        return "+".join("{}x{}".format(mul, ir) for mul, ir in self.terms)
