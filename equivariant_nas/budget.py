"""Auditable GPU budget ledger for prototype search."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path: str, limit_gpu_hours: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit_gpu_hours = float(limit_gpu_hours)

    def records(self):
        if not self.path.exists():
            return []
        output = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                output.append(json.loads(line))
        return output

    def used_gpu_hours(self) -> float:
        return sum(float(item.get("gpu_seconds", 0.0)) for item in self.records()) / 3600.0

    def remaining_gpu_hours(self) -> float:
        return max(0.0, self.limit_gpu_hours - self.used_gpu_hours())

    def require_available(self, requested_gpu_hours: float = 0.0) -> None:
        if self.used_gpu_hours() + requested_gpu_hours > self.limit_gpu_hours:
            raise BudgetExceeded(
                "GPU budget would exceed {:.3f} hours".format(self.limit_gpu_hours)
            )

    def estimate_stage_gpu_hours(
        self,
        stage: str,
        fallback_gpu_hours: float,
        safety_multiplier: float = 1.20,
    ) -> float:
        """Estimate a repeated stage from observed ledger cost.

        A median resists one-off startup outliers; the multiplier preserves a
        conservative pre-flight reservation before CUDA work begins.
        """

        samples = [
            float(item.get("gpu_seconds", 0.0)) / 3600.0
            for item in self.records()
            if item.get("stage") == stage and float(item.get("gpu_seconds", 0.0)) > 0.0
        ]
        if not samples:
            return float(fallback_gpu_hours)
        return max(0.0, statistics.median(samples) * float(safety_multiplier))

    def append(self, event: Dict[str, Any]) -> None:
        record = dict(event)
        record.setdefault("timestamp", time.time())
        next_total = self.used_gpu_hours() + float(record.get("gpu_seconds", 0.0)) / 3600.0
        if next_total > self.limit_gpu_hours + 1.0e-12:
            raise BudgetExceeded(
                "record would raise GPU use to {:.3f}/{:.3f} hours".format(
                    next_total, self.limit_gpu_hours
                )
            )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
