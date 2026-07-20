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
