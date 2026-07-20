"""Deterministic scientific-semantic guards for QM9 alpha reasoning."""

from __future__ import annotations

import re


class ScientificSemanticsError(ValueError):
    pass


def validate_qm9_alpha_reasoning(text: str) -> None:
    """Reject recurring physical misconceptions before they enter lineage memory."""

    lowered = " ".join(str(text).lower().split())
    # Remove explicit corrections so the guard does not reject statements such
    # as "alpha is not a rank-2 tensor target".
    normalized = lowered.replace("not a rank-2 tensor target", "scalar target")
    normalized = normalized.replace("not a rank 2 tensor target", "scalar target")
    normalized = normalized.replace("not tensor-valued", "scalar")
    forbidden = (
        (r"\b(rank[- ]?2|second[- ]order) tensor target\b", "alpha mislabeled as tensor target"),
        (r"\btensor-valued target\b", "alpha mislabeled as tensor-valued"),
        (
            r"\b(irreps_head|head_tensor_channels|head_vector_channels)\b.{0,80}\boutput head\b",
            "internal attention irreps mislabeled as output head",
        ),
        (
            r"\b(higher[- ]order|l\s*[>=]+\s*1|l\s*[>=]+\s*2|tensor_channels)\b.{0,120}"
            r"\b(cannot|can't|never|no)\b.{0,50}\b(help|contribute|improve|serve|benefit)\b",
            "hidden non-scalar irreps declared intrinsically useless",
        ),
    )
    for pattern, message in forbidden:
        if re.search(pattern, normalized):
            raise ScientificSemanticsError(message)
