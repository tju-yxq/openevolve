"""Strict JSON serialization for architecture programs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .ast import ArchitectureProgram
from .diagnostics import DSLValidationError, Diagnostic
from .task import TaskContract


def loads_program(text: str) -> ArchitectureProgram:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DSLValidationError([
            Diagnostic("E_PARSE_001", "invalid JSON architecture program", details={"line": exc.lineno, "column": exc.colno, "reason": exc.msg})
        ])
    if not isinstance(value, Mapping):
        raise DSLValidationError([Diagnostic("E_PARSE_002", "architecture program must be a JSON object")])
    return ArchitectureProgram.from_dict(value)


def dumps_program(program: ArchitectureProgram, *, indent: int = 2) -> str:
    return json.dumps(program.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent) + "\n"


def load_program(path: str) -> ArchitectureProgram:
    return loads_program(Path(path).read_text(encoding="utf-8"))


def save_program(program: ArchitectureProgram, path: str) -> None:
    Path(path).write_text(dumps_program(program), encoding="utf-8")


def load_task_contract(path: str) -> TaskContract:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DSLValidationError([
            Diagnostic("E_PARSE_003", "invalid JSON task contract", details={"line": exc.lineno, "column": exc.colno, "reason": exc.msg})
        ])
    if not isinstance(value, Mapping):
        raise DSLValidationError([Diagnostic("E_PARSE_004", "task contract must be a JSON object")])
    return TaskContract.from_dict(value)


def save_task_contract(task: TaskContract, path: str) -> None:
    Path(path).write_text(
        json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
