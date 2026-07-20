#!/usr/bin/env python
"""Generate a small, factor-labelled architecture pool for Gate 1."""

from dataclasses import replace
from pathlib import Path

from equivariant_nas.candidate import render_candidate
from equivariant_nas.spec import baseline_spec


def main():
    root = Path("/home/20262202788/equivariant-nas/runs/gate1_candidates")
    root.mkdir(parents=True, exist_ok=True)
    base = baseline_spec()
    candidates = {
        "baseline": base,
        "macro_depth4": replace(base, macro=replace(base.macro, num_layers=4)),
        "macro_depth5": replace(base, macro=replace(base.macro, num_layers=5)),
        "macro_depth7": replace(base, macro=replace(base.macro, num_layers=7)),
        "operator_bessel": replace(
            base, operator=replace(base.operator, basis_type="bessel")
        ),
        "operator_linear_message": replace(
            base, operator=replace(base.operator, nonlinear_message=False)
        ),
        "action_graph_norm": replace(
            base, action=replace(base.action, norm_layer="graph")
        ),
        "action_fast_layer": replace(
            base, action=replace(base.action, norm_layer="fast_layer")
        ),
        "representation_narrow": replace(
            base,
            representation=replace(
                base.representation,
                scalar_channels=96,
                vector_channels=48,
                tensor_channels=24,
                head_scalar_channels=24,
                head_vector_channels=8,
                head_tensor_channels=8,
                feature_channels=384,
            ),
        ),
        "representation_lmax1": replace(
            base,
            representation=replace(
                base.representation,
                lmax=1,
                tensor_channels=0,
                head_tensor_channels=0,
            ),
        ),
        "representation_lmax3": replace(
            base,
            representation=replace(
                base.representation,
                lmax=3,
                l3_channels=8,
                head_l3_channels=8,
            ),
        ),
    }
    for name, spec in candidates.items():
        spec.validate()
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "candidate.py").write_text(render_candidate(spec), encoding="utf-8")
        (directory / "architecture_id.txt").write_text(
            spec.architecture_id() + "\n", encoding="utf-8"
        )
    print("generated {} candidates in {}".format(len(candidates), root))


if __name__ == "__main__":
    main()

