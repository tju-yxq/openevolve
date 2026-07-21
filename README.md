# Equivariant Architecture Self-Evolution

Research prototype for symmetry-preserving, LLM-driven neural architecture
search on Equiformer and QM9 polarizability (`target=1`).

The project deliberately separates trusted experiment code from evolved
architecture descriptions.  The LLM evolves a small, typed `ArchitectureSpec`;
it cannot edit the dataset, optimizer, loss, training budget, or evaluator.

## Method hypotheses

1. **Symmetry-Preserving Architecture Grammar (SPAG).** Candidates are valid
   irrep-flow programs rather than arbitrary Python rewrites. Compatibility is
   checked before a model is constructed.
2. **Irrep-Flow Factorized Evolution (IFE).** Each child changes exactly one of
   `REPRESENTATION`, `OPERATOR`, `ACTION`, or `MACRO`. Parent-to-child metric
   deltas provide factor-specific credit assignment.
3. **Symmetry-Aware Progressive Fidelity (SAPF).** Layer-wise irrep consistency,
   scalar invariance, gradient health, and cost are cheap gates before training;
   learning-curve evidence controls later promotion.
4. **Learning-Rate-Phase-Aware Fidelity (LRPF).** Fidelity endpoints align with
   optimizer phases and retain shock/recovery fingerprints instead of treating
   every short curve as the same noisy accuracy estimate.
5. **Self-Calibrating Fidelity Trust Gate (SCFTG).** A cheap proxy must pass
   cross-fidelity rank correlation, top-k recall, and selection-regret checks
   before it can influence promotion. Calibration runs are explicitly marked
   ineligible for architecture selection.
6. **Interaction-Aware Counterfactual Credit (IACC).** A factor that rescues a
   degraded parent receives provisional rather than immediate MAE credit; a
   sibling counterfactual separates its main effect from cross-factor epistasis.
7. **Frozen Evidence Memory (FEM).** A hash-provenanced factor posterior carries
   resolved stage evidence into low-budget searches; counterfactual main effects,
   rather than interaction-contaminated rescue gains, warm-start the router.
8. **Trajectory-Conditioned Reflect–Edit (TCRE).** SPARK-style RC/SAR receives a
   deterministic bounded lineage and plateau summary. Zero-step records cannot
   fabricate convergence claims, and plateau escape remains factor-local.

These are research hypotheses, not claims of final superiority. Stage one must
validate them within a hard 5 A100-hour prototype budget. The 257,700-step,
three-seed protocol is a separately approved final experiment and is never
launched by the search loop. The 300- and 1,000-step observations are retained
for proxy calibration only: current evidence shows that their ranking can
reverse after warmup. Pilot architecture comparisons therefore use a
5,000-step endpoint until a cheaper proxy is empirically validated.

OpenEvolve supplies islands, lineage and MAP-Elites; SPARK-style RC/SAR splits
evidence reflection from factor-local editing. The LLM never emits trusted
training code. MAP-Elites cells use equivariant capacity descriptors (`lmax`,
higher-order fraction, parameter ratio and depth), and compiler, duplicate and
evaluator failures share a same-factor repair loop.

The integration boundary is explicit: OpenEvolve is the population, island,
lineage and MAP-Elites runtime; SPARK contributes the structural reviewer/editor
workflow and history-conditioned search principle; this project replaces free
Python edits with a trusted Equiformer architecture compiler and adds FEM, ECFR,
SAPF/LRPF/SCFTG and IACC for symmetry-aware scientific search.

Phase 2 begins with a six-candidate micro gate (two trained-valid candidates per
method) capped at 2.8 A100-hours. It is a framework kill gate, not evidence of
search superiority. The previously estimated 8.2-hour single-seed screen is
unlocked only if the micro gate passes its preregistered checks; no later gate
runs automatically.
