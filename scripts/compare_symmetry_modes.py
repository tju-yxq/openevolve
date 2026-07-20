#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from torch_geometric.loader import DataLoader

from equivariant_nas.builder import add_equiformer_to_path, build_equiformer
from equivariant_nas.candidate import extract_literal_spec
from equivariant_nas.diagnostics import symmetry_report


def maximum(report):
    return max(
        report.rotation_invariance.maximum,
        report.translation_invariance.maximum,
        report.permutation_invariance.maximum,
        *(value.maximum for value in report.layerwise_rotation_equivariance.values())
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("programs", nargs="+")
    parser.add_argument("--equiformer-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    add_equiformer_to_path(args.equiformer_root)
    from datasets.pyg.qm9 import QM9

    batch = next(iter(DataLoader(QM9(args.data_path, "valid", feature_type="one_hot"), batch_size=2))).to("cuda")
    rows = []
    for program in args.programs:
        spec = extract_literal_spec(program)
        model = build_equiformer(spec, args.equiformer_root).to("cuda")
        dynamic = symmetry_report(model, batch, rotations=3, translations=2, freeze_neighbors=False)
        fixed = symmetry_report(model, batch, rotations=3, translations=2, freeze_neighbors=True)
        row = {
            "candidate": Path(program).parent.name,
            "architecture_id": spec.architecture_id(),
            "dynamic_max": maximum(dynamic),
            "fixed_neighbor_max": maximum(fixed),
            "dynamic": dynamic.to_dict(),
            "fixed_neighbors": fixed.to_dict(),
        }
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("candidate", "dynamic_max", "fixed_neighbor_max")}, sort_keys=True), flush=True)
    Path(args.output).write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

