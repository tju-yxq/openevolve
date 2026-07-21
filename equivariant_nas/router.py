"""Evidence-calibrated factor router.

SPARK routes edits semantically.  This router adds empirical credit assignment:
factors with good parent-child gains are exploited, while an uncertainty bonus
preserves exploration and a failure penalty suppresses invalid edit regions.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from .spec import EvolutionFactor


@dataclass
class FactorStats:
    attempts: int = 0
    valid: int = 0
    total_mae_gain: float = 0.0
    total_efficiency_gain: float = 0.0

    @property
    def validity_rate(self) -> float:
        return self.valid / self.attempts if self.attempts else 1.0

    @property
    def mean_reward(self) -> float:
        if not self.valid:
            return 0.0
        return (self.total_mae_gain + 0.05 * self.total_efficiency_gain) / self.valid


class EvidenceCalibratedRouter:
    def __init__(self, seed: int = 42, exploration: float = 0.35):
        self.random = random.Random(seed)
        self.exploration = float(exploration)
        self.stats: Dict[EvolutionFactor, FactorStats] = {
            factor: FactorStats() for factor in EvolutionFactor
        }
        self.prior_provenance: Dict[str, object] = {}

    def load_prior(self, path: str) -> None:
        """Warm-start factor credit from a frozen, auditable evidence file.

        The file contains sufficient statistics, not model predictions.  This
        matters in low-budget runs: an empty UCB router otherwise spends its
        first four proposals merely touching every factor once and cannot use
        evidence gathered in an earlier, explicitly frozen stage.
        """

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported router-prior schema")
        if payload.get("dataset") != "QM9" or int(payload.get("target", -1)) != 1:
            raise ValueError("router prior is not registered for QM9 target 1")
        factor_stats = payload.get("factor_stats", {})
        if set(factor_stats) != {factor.value for factor in EvolutionFactor}:
            raise ValueError("router prior must cover every evolution factor")
        loaded = {}
        for factor in EvolutionFactor:
            raw = factor_stats[factor.value]
            stats = FactorStats(
                attempts=int(raw["attempts"]),
                valid=int(raw["valid"]),
                total_mae_gain=float(raw.get("total_mae_gain", 0.0)),
                total_efficiency_gain=float(
                    raw.get("total_efficiency_gain", 0.0)
                ),
            )
            if stats.attempts < 0 or stats.valid < 0 or stats.valid > stats.attempts:
                raise ValueError("invalid router-prior counts for {}".format(factor.value))
            loaded[factor] = stats
        self.stats = loaded
        self.prior_provenance = {
            "path": str(path),
            "name": payload.get("name", ""),
            "frozen_before_phase2": bool(payload.get("frozen_before_phase2")),
            "source_sha256": payload.get("source_sha256", {}),
        }

    def load_state(self, path: str) -> None:
        """Restore mutable router statistics from a trusted run checkpoint."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        factor_stats = payload.get("factor_stats", {})
        if set(factor_stats) != {factor.value for factor in EvolutionFactor}:
            raise ValueError("router state must cover every evolution factor")
        restored = {}
        for factor in EvolutionFactor:
            raw = factor_stats[factor.value]
            stats = FactorStats(
                attempts=int(raw["attempts"]),
                valid=int(raw["valid"]),
                total_mae_gain=float(raw.get("total_mae_gain", 0.0)),
                total_efficiency_gain=float(
                    raw.get("total_efficiency_gain", 0.0)
                ),
            )
            if stats.attempts < 0 or stats.valid < 0 or stats.valid > stats.attempts:
                raise ValueError("invalid router state for {}".format(factor.value))
            restored[factor] = stats
        self.stats = restored
        self.prior_provenance = dict(payload.get("prior_provenance", {}))

    def select(self, context: Optional[Mapping[str, float]] = None) -> EvolutionFactor:
        total = sum(item.attempts for item in self.stats.values()) + 1
        scores = {}
        for factor, stats in self.stats.items():
            if stats.attempts == 0:
                scores[factor] = float("inf")
                continue
            uncertainty = self.exploration * math.sqrt(math.log(total + 1) / stats.attempts)
            failure_penalty = 0.5 * (1.0 - stats.validity_rate)
            context_bonus = self._context_bonus(factor, context or {})
            scores[factor] = stats.mean_reward + uncertainty - failure_penalty + context_bonus
        maximum = max(scores.values())
        candidates = [factor for factor, score in scores.items() if score == maximum]
        return self.random.choice(candidates)

    @staticmethod
    def _context_bonus(factor: EvolutionFactor, context: Mapping[str, float]) -> float:
        # Layer-wise symmetry drift points toward representation/operator edits;
        # high latency points toward macro/operator edits. Bonuses are deliberately
        # small: measured parent-child rewards remain the primary signal.
        symmetry = float(context.get("symmetry_drift", 0.0))
        latency = float(context.get("relative_step_time", 1.0))
        parameter_ratio = float(context.get("parameter_ratio", 1.0))
        if symmetry > 1.0 and factor in (
            EvolutionFactor.REPRESENTATION,
            EvolutionFactor.OPERATOR,
        ):
            return 0.05
        if (latency > 1.1 or parameter_ratio > 1.05) and factor in (
            EvolutionFactor.OPERATOR,
            EvolutionFactor.MACRO,
        ):
            return 0.05
        return 0.0

    def update(
        self,
        factor: EvolutionFactor,
        valid: bool,
        mae_gain: float = 0.0,
        efficiency_gain: float = 0.0,
    ) -> None:
        stats = self.stats[factor]
        stats.attempts += 1
        if valid:
            stats.valid += 1
            stats.total_mae_gain += float(mae_gain)
            stats.total_efficiency_gain += float(efficiency_gain)

    def to_dict(self):
        return {
            "factor_stats": {
                factor.value: asdict(stats) for factor, stats in self.stats.items()
            },
            "prior_provenance": self.prior_provenance,
        }

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def prompt_for_factor(
    factor: EvolutionFactor,
    parent_json: str,
    metrics: Mapping[str, object],
    inspirations: Iterable[str],
    artifacts: Mapping[str, object],
    reflection: Optional[Mapping[str, object]] = None,
    search_memory: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    system = (
        "You are evolving an E(3)-aware Equiformer architecture for QM9 alpha. "
        "You may change exactly one router-selected factor. Never change data, target, "
        "optimizer, batch size, training steps, evaluator, or output invariance. "
        "Return strict JSON and no executable code. Do not infer accuracy, underfitting, "
        "overfitting, or convergence when fidelity_steps is zero. Separate measured evidence "
        "from hypotheses in the reasoning. The child must stay at or below 1.2 times "
        "the 3,531,715-parameter baseline. higher_order_fraction is the fraction of "
        "channel multiplicities assigned to l>=2, not a count of irreps. QM9 alpha here is "
        "the scalar isotropic polarizability, not a rank-2 tensor target; the model output "
        "must remain rotation invariant. Higher-order hidden irreps may still be hypothesized "
        "to help through equivariant couplings into l=0; scalar output does not imply l>0 "
        "hidden features are useless. irreps_head is an internal attention representation, "
        "not the output head. Never justify changes using unprovided quantitative facts "
        "about QM9 molecular sizes or geometry."
    )
    user = """Selected factor: {factor}

Parent ArchitectureSpec:
{parent}

Measured metrics:
{metrics}

Evidence rule: fidelity_steps=0 means feasibility only; combined_score=0 at that
stage is neutral and contains no accuracy information. Only validation_alpha_mae
is evidence about predictive quality.

Diagnostic artifacts:
{artifacts}

Trusted lineage/search-memory summary:
{search_memory}

Mutation rule: if plateau_status is unknown, do not claim a plateau. If it is
yes, explore a meaningfully different design inside the selected factor only;
otherwise prefer one conservative, attributable change.

Factor-local reflection produced by the RC stage:
{reflection}

Allowed values for this factor:
{allowed_values}

Inspiration specs:
{inspirations}

Return exactly:
{{
  "selected_factor": "{factor}",
  "reasoning": "short evidence-based reason",
  "patch": {{ complete replacement object for the selected factor }}
}}
""".format(
        factor=factor.value,
        parent=parent_json,
        metrics=json.dumps(dict(metrics), sort_keys=True),
        artifacts=json.dumps(dict(artifacts), sort_keys=True),
        search_memory=json.dumps(dict(search_memory or {}), sort_keys=True),
        reflection=json.dumps(dict(reflection or {}), sort_keys=True),
        allowed_values=_allowed_values(factor),
        inspirations=json.dumps(list(inspirations), sort_keys=True),
    )
    return {"system": system, "user": user}


def _allowed_values(factor: EvolutionFactor) -> str:
    if factor == EvolutionFactor.REPRESENTATION:
        return (
            "lmax in {1,2,3}; scalar_channels in {64,96,128,160,192,256}; "
            "other nonzero channels in {8,16,24,32,48,64,96,128,160,192,256}; "
            "mlp_multiplier in {2,3,4}; feature_channels in {256,384,512,640}; "
            "channels above lmax must be zero. Prefer one conservative change under the cap."
        )
    if factor == EvolutionFactor.OPERATOR:
        return (
            "basis_type in {gaussian,bessel}; num_basis in {32,64,96,128}; "
            "radial_hidden in {[32,32],[64,64],[96,96],[128,128]}; "
            "nonlinear_message boolean; num_heads in {2,4,8}."
        )
    if factor == EvolutionFactor.ACTION:
        return (
            "norm_layer in {layer,instance,graph,fast_layer}; rescale_degree boolean; "
            "alpha_drop/projection_drop/output_drop/drop_path each in {0,0.05,0.1,0.2}."
        )
    return "num_layers in {3,4,5,6,7,8}; radius in {4.0,5.0,6.0}."


def prompt_for_reflection(
    factor: EvolutionFactor,
    parent_json: str,
    metrics: Mapping[str, object],
    artifacts: Mapping[str, object],
    search_memory: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    """RC stage: diagnose one factor without proposing executable code."""

    system = (
        "You are the reflection component of an equivariant NAS system. Analyze only the "
        "router-selected factor. Distinguish measurements from hypotheses. A zero-step "
        "candidate has no accuracy evidence. QM9 alpha is scalar isotropic polarizability, "
        "not a rank-2 tensor target, and the output is rotation invariant. Hidden l>0 "
        "irreps can still contribute to scalar output through equivariant coupling; do not "
        "claim they are intrinsically useless. irreps_head is internal attention state, not "
        "an output head. Do not introduce unmeasured quantitative dataset facts. Return "
        "strict JSON, not code and not a patch."
    )
    user = """Selected factor: {factor}
Parent ArchitectureSpec: {parent}
Metrics: {metrics}
Diagnostics: {artifacts}
Trusted lineage/search-memory summary: {search_memory}
Metric definition: higher_order_fraction is the fraction of channel multiplicities
assigned to l>=2. It is not a count of representation types.

Return exactly:
{{
  "direction": "one factor-local design direction",
  "evidence": ["measured fact or explicit absence of evidence"],
  "risk": "main symmetry, optimization, or efficiency risk"
}}
""".format(
        factor=factor.value,
        parent=parent_json,
        metrics=json.dumps(dict(metrics), sort_keys=True),
        artifacts=json.dumps(dict(artifacts), sort_keys=True),
        search_memory=json.dumps(dict(search_memory or {}), sort_keys=True),
    )
    return {"system": system, "user": user}
