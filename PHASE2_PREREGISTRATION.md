# Phase 2 preregistration: evidence required for a publication claim

This document freezes the next experimental decision rules before any Phase 2
GPU work. Stage one is engineering and hypothesis evidence; it is not treated
as statistical proof of search superiority.

## Registered hypotheses

1. The full method reaches low validation MAE earlier than typed random search,
   measured by normalized best-so-far AUC at the same 5,000-step candidate
   fidelity and the same number of trained-valid candidates.
2. ECFR improves search efficiency relative to an otherwise identical LLM
   evolution run with uniform factor routing.
3. RC improves validity and search efficiency relative to SAR without the
   reflection stage.
4. IACC prevents misleading factor credit on rescue chains and changes later
   routing decisions relative to immediate parent-child credit.
5. SCFTG rejects proxies with poor cross-fidelity rank agreement and reduces
   selection regret relative to unconditional successive halving.

## Why five search seeds are eventually necessary

The search algorithm itself is stochastic through LLM sampling, parent and
inspiration selection, islands and factor routing. Repeating only training seeds
does not measure search-method variance. With an exact one-sided paired sign-flip
test, three search seeds cannot produce a p-value below 0.125; five consistent
paired wins can reach 0.03125. Consequently, Gate 2A uses two seeds only as a
cost-controlled screen. A publication-level superiority claim requires the
registered total of five search seeds.

## Gate 2A-micro: cheap framework kill gate

Before spending the full single-seed screen budget, run only two trained-valid
candidates per method on search seed 101: six candidates total. The observed
median cost implies about 2.25 A100-hours, with a registered hard cap of 2.8
A100-hours including a 20% reserve. This gate asks only whether the framework is
healthy enough to justify more evidence; it cannot establish search superiority.

The continuation is killed unless every method reaches two valid candidates
within ten proposals, the full method produces a novel non-duplicate candidate,
its best validation MAE is no more than 15% worse than the fixed stage-one
baseline, its normalized AUC is no more than 10% worse than typed random, and
there is no protocol or test-split violation. These are deliberately permissive
engineering checks, not a post-hoc publication claim.

## Gate 2A: sequential controlled screen

- Methods: full framework, uniform-router LLM evolution, typed random search.
- First search seed: 101 only.
- If and only if Gate 2A-micro passes, continue the same runs from two to six
  trained-valid candidates per method. The six micro-gate candidates are reused,
  not discarded or rerun.
- Candidate fidelity: 5,000 optimizer steps, trainer seed 0 as a common-random-
  numbers control.
- Primary metrics: normalized best-so-far AUC and final best validation MAE.
- Cumulative hard cap, including Gate 2A-micro: 8.2 A100-hours. This is derived
  from the observed 0.374 A100-hour median candidate cost for 18 candidates with
  a 20% reserve.
- Test split remains unavailable.

Seed 102 is not launched automatically. It unlocks only if the full framework
has at least 80% valid candidates, improves normalized AUC by at least 5%
against both controls, beats typed random in final-best MAE on seed 101, and has
no protocol violation. Seed 102 receives a separate 8.2-hour cap and must pass
the same rule before Gate 2B.

## Gate 2B: publication-scale search evidence

If Gate 2A passes, add search seeds 103, 104 and 105, preserving every setting.
Report all five paired runs, not only successful seeds. The primary statistical
test is an exact paired sign-flip test on normalized AUC; paired bootstrap
confidence intervals and time-to-threshold are secondary estimates. Invalid
candidates, repair attempts, LLM calls, wall time and A100-hours are reported.

## Component ablations

Conditional ablations use the same typed genotype and trusted evaluator:

- uniform factor router: removes ECFR while preserving RC/SAR;
- no RC: SAR receives metrics and hard constraints without reflection;
- no IACC: rescue-chain MAE credit is applied immediately;
- dynamic MAP-Elites scaling: restores the arrival-order-dependent OpenEvolve
  default in an isolated ablation;
- unconditional short fidelity: calibration-only because stage one already
  falsified its use for selection.

Ablations are not permitted to change data, optimizer, batch size, candidate
steps, parameter cap, test policy or candidate count.

## Accuracy confirmation and final protocol

Only after search evidence passes are the top frozen architectures evaluated at
20,000 steps with training seeds 0, 1 and 2. One architecture is then frozen for
the 257,700-step, three-seed final protocol. The test set is evaluated once,
after freezing, and is never used to choose between architectures.

The machine-readable source of truth is
`configs/phase2_preregistration.json`. Any deviation must be documented before
observing the affected results; post-hoc threshold changes are prohibited.
