"""Authorization rules for fidelity promotion."""

from __future__ import annotations

from typing import Mapping, Optional


class UntrustedFidelityError(RuntimeError):
    pass


def authorize_promotion(
    purpose: str,
    trust_report: Optional[Mapping[str, object]] = None,
) -> bool:
    """Return whether promoted metrics may influence architecture selection.

    Calibration promotions are allowed solely to measure proxy reliability and
    are never selection-eligible. Selection promotions require a passed SCFTG
    report; absence of evidence is a hard failure rather than implicit trust.
    """

    normalized = str(purpose).strip().lower()
    if normalized == "calibration":
        return False
    if normalized != "selection":
        raise ValueError("purpose must be calibration or selection")
    if not trust_report:
        raise UntrustedFidelityError("selection promotion requires a trust report")
    if not bool(trust_report.get("trustworthy", False)):
        raise UntrustedFidelityError(
            "source fidelity is not trusted: {}".format(
                trust_report.get("reason", "unspecified failure")
            )
        )
    return True
