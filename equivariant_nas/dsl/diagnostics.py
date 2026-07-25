"""Structured diagnostics emitted by the DSL parser and type checker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    node_id: str = ""
    port: str = ""
    expected: str = ""
    actual: str = ""
    trace: Tuple[str, ...] = field(default_factory=tuple)
    repairs: Tuple[str, ...] = field(default_factory=tuple)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "node_id": self.node_id,
            "port": self.port,
            "expected": self.expected,
            "actual": self.actual,
            "trace": list(self.trace),
            "repairs": list(self.repairs),
            "details": dict(self.details),
        }


class DSLValidationError(ValueError):
    """A validation failure carrying machine-readable diagnostics."""

    def __init__(self, diagnostics: Iterable[Diagnostic]):
        self.diagnostics = tuple(diagnostics)
        if not self.diagnostics:
            raise ValueError("DSLValidationError requires at least one diagnostic")
        super().__init__("; ".join("{}: {}".format(d.code, d.message) for d in self.diagnostics))
