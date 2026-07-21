# Method: symmetry-aware factorized architecture self-evolution

## Search problem and invariants

For an architecture specification $A$ and a phase-complete training budget
$T$, the primary objective is validation MAE on QM9 isotropic polarizability:

$$
\min_A\; \mathrm{MAE}_{\mathrm{val}}(A,T)
$$

subject to $\mathrm{Params}(A) \le 1.2\,\mathrm{Params}(A_0)$, invariant scalar
output, legal irrep flow, numerical symmetry diagnostics below the catastrophic
threshold, a fixed data/optimizer protocol, and a global auditable GPU budget.
The test split is not part of the search objective. Runtime and parameter count
are secondary credit signals and quality-diversity descriptors, not silently
mixed with MAE.

The following invariants are enforced by trusted code rather than prompts:

1. exactly one factor changes between parent and child;
2. candidate text is parsed as a literal and never executed;
3. data, target, loss, optimizer, batch size, step budget, and evaluator are
   outside the genotype;
4. invalid, duplicate, over-budget, or over-parameter candidates do not enter
   the archive;
5. only candidates evaluated at the same fidelity endpoint are compared;
6. a low-fidelity cohort cannot promote candidates unless SCFTG grants trust.

## Why this is not a simple OpenEvolve + SPARK combination

OpenEvolve treats source code as the genotype. SPARK narrows source edits to
semantic regions, but its public implementation still exposes a large Python
surface and its factor router is not grounded in measured parent-child credit.
For equivariant networks, a syntactically local edit can globally invalidate an
irrep flow. The proposed method therefore changes the genotype, router, and
evaluation landscape together.

## Insight 1 — Symmetry-Preserving Architecture Grammar (SPAG)

The genotype is a typed irrep-flow specification. Every field has a known map to
a trusted Equiformer constructor. Unknown fields and executable statements are
rejected through AST literal parsing. Equivariance is preserved by construction
where possible and verified numerically before training.

## Insight 2 — Evidence-Calibrated Factor Routing (ECFR)

Each child changes one factor: representation, message operator, update action,
or macro structure. Routing is a bandit over measured factor-local rewards:
validation-MAE gain, validity rate, efficiency gain, and a small context bonus
from layer-wise symmetry drift. The LLM supplies semantic direction, but cannot
override the selected factor.

## Insight 3 — Finite-Precision Symmetry Stability

Theoretically equivariant operators can differ substantially under finite
precision, neighborhood reconstruction, normalization, and radial bases. The
evaluator records both observable scalar invariance and a layer-wise irrep
consistency profile. Rotation/translation/permutation error of the final scalar
output is the hard safety gate. The layer profile is warning-level search
feedback until hook coordinates and internal irrep conventions are independently
calibrated; stage-one auditing showed that an uncalibrated layer transform can
also flag the official Gaussian baseline.

## Insight 4 — Symmetry-Aware Progressive Fidelity (SAPF)

Candidates pass schema, construction, forward/backward gradient health, and
resource gates before optimizer steps are spent. Initial symmetry is recorded
as a warning-level numerical profile; the trained checkpoint must pass the hard
observable scalar-invariance audit before entering the archive. Training
candidates are compared only inside the same fidelity cohort. Promotion uses
validation alpha MAE; the test split remains unavailable until a final
architecture is frozen.

## Insight 5 — Learning-Rate-Phase-Aware Fidelity (LRPF)

Equal step counts are not equal information when the optimizer is in different
schedule phases. Stage-one evidence showed that a candidate can dominate at 300
steps, collapse immediately after the step-859 LR transition, and recover only
partly by the end of warmup. The evaluator therefore records an optimization
response fingerprint: pre-transition MAE, post-transition shock, warmup-end MAE,
and recovery ratio. Promotion budgets are aligned to LR phase boundaries rather
than chosen only by round step counts.

## Insight 6 — Self-Calibrating Fidelity Trust Gate (SCFTG)

Low fidelity is treated as a hypothesis, not a free speedup. For candidates
observed both before and after a complete optimizer phase, the controller
measures Spearman rank correlation, Kendall tau, top-k recall, and the
reference-fidelity regret of the proxy winner. A proxy receives promotion
authority only after all pre-registered trust thresholds pass on a minimum
cohort. Otherwise it is retained for trajectory modeling but disabled for NAS
selection. This converts misleading early convergence from hidden bias into an
auditable negative result and prevents systematic search misallocation.

## Insight 7 — Interaction-Aware Counterfactual Credit (IACC)

Single-factor edits make attribution possible but do not make factor effects
additive. When a child strongly rescues a parent that had degraded relative to
an ancestor, the framework schedules a sibling counterfactual: apply the same
new factor patch to the ancestor while removing the earlier factor change. The
resulting 2x2 contrast estimates the new factor's main effect and an explicit
difference-in-differences epistasis term. Router credit can then distinguish a
generally useful operator from an operator that only works in combination with
a particular action or representation choice.

Resolved counterfactual main effects are ingested back into the factor router
exactly once. The router never receives the larger, interaction-contaminated
rescue-chain gain when the sibling shows that the factor's standalone effect is
smaller.

## Insight 8 — Frozen Evidence Memory (FEM)

Low-budget evolution cannot afford to relearn factor utility from scratch. An
empty UCB router spends its first proposals touching every factor, so a two- or
six-candidate experiment barely tests evidence-calibrated routing at all. FEM
freezes sufficient statistics from the previous stage, includes one neutral
virtual observation for every factor, records source SHA-256 hashes, and warm-
starts only the evidence-routed method. For the observed rescue chain,
OPERATOR receives the IACC standalone gain `0.17709`, not the misleading joint
gain `0.65971`. This turns prior experiments into auditable search memory while
keeping the test split absent.

## Insight 9 — Trajectory-Conditioned Reflect–Edit (TCRE)

SPARK's reflection/edit separation is extended with a trusted lineage summary.
Before RC, deterministic code extracts same-fidelity validation history,
factor history, lineage depth, recent best improvement, and plateau status. A
plateau is `unknown` when evidence is absent or insufficient; the LLM cannot
invent one from zero-step feasibility metrics. RC sees the bounded summary and
measured risks, while SAR receives a mutation regime: conservative for normal
progress and meaningfully exploratory *inside the already selected factor*
only after a measured plateau. Prompt metrics are compacted by trusted code so
large symmetry tensors do not inflate API cost or obscure the evidence.

## OpenEvolve quality-diversity substrate

OpenEvolve supplies islands, lineage storage, sampling, and MAP-Elites. The
quality-diversity coordinates are equivariant capacity descriptors rather than
source-code descriptors: `lmax`, higher-order channel fraction, parameter
ratio, and depth. SPARK-style RC/SAR editing operates inside a router-selected
factor, while SPAG and SCFTG define which children and which evidence are
allowed to enter selection.

## Insight 10 — Irrep-Semantic Weight Transfer (ISWT)

Candidate training can be expensive even when a child changes only one typed
factor. ISWT uses a parent checkpoint only as a calibration initializer: exact
state compatibility is intersected with factor semantics. Representation edits
reset all learned states because the irrep layout changes; Gaussian/Bessel or
radial-width edits reset radial basis and radial networks; normalization edits
reset normalization parameters. Optimizer and scheduler state are never
inherited. A transfer report records every copied and blocked tensor, coverage,
and reason. It is test-split-free, selection-ineligible, and final training
always starts from scratch.

ISWT is not assumed to be a speedup. It must first pass a pre-registered paired
calibration against scratch 5,000-step training on at least eight candidates,
using rank correlation, top-k recall, and selection regret. If the gate fails,
the initializer remains diagnostic-only.

## Algorithm sketch

```text
initialize typed Equiformer baseline A0
evaluate A0 with trusted gates and selected fidelity
insert A0 into OpenEvolve islands / equivariant MAP-Elites map

while GPU budget and iteration budget remain:
    parent, inspirations <- OpenEvolve.sample()
    factor <- ECFR(parent evidence, factor posterior)
    reflection <- RC(parent, factor, measured artifacts)
    patch <- SAR(parent, factor, reflection)
    child <- SPAG.compile_and_repair(patch)
    if duplicate or static/symmetry/resource gate fails: reject or repair
    metrics <- fixed-protocol QM9 validation at the cohort fidelity
    if child rescues a degraded parent:
        keep factor MAE credit provisional and request IACC sibling
    else:
        update factor-local parent-child credit
    update OpenEvolve archive and MAP-Elites cell

when a cohort is proposed for promotion to a higher fidelity:
    if purpose is calibration:
        allow evaluation but mark results selection-ineligible
    else if SCFTG does not trust the source fidelity:
        reject promotion
    else:
        allow selection promotion
```

Compiler repair, duplicate repair, and evaluator repair are all constrained to
the originally selected factor. A repair cannot broaden the mutation surface.

## Falsifiable stage-one hypotheses

1. Typed candidates have a higher executable/valid rate than free Python edits.
2. ECFR reaches a given short-fidelity validation MAE using fewer evaluated
   candidates than random routing and free OpenEvolve edits.
3. Layer-wise symmetry stability varies across otherwise valid Equiformer
   primitives and predicts at least some training instability or rank changes.
4. Short-fidelity rankings contain enough top-k signal to justify a later,
   separately approved full-budget experiment.
