#!/usr/bin/env python
import argparse
import json
import sys

import torch
from torch_geometric.loader import DataLoader

from equivariant_nas.builder import add_equiformer_to_path, build_equiformer
from equivariant_nas.diagnostics import symmetry_report
from equivariant_nas.spec import ArchitectureSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--equiformer-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--freeze-neighbors", action="store_true")
    args = parser.parse_args()

    add_equiformer_to_path(args.equiformer_root)
    from datasets.pyg.qm9 import QM9

    spec = ArchitectureSpec.from_json(open(args.spec, encoding="utf-8").read())
    dataset = QM9(args.data_path, "valid", feature_type="one_hot")
    data = next(iter(DataLoader(dataset, batch_size=2))).to("cuda")
    model = build_equiformer(spec, args.equiformer_root).to("cuda")
    report = symmetry_report(
        model,
        data,
        rotations=2,
        translations=2,
        freeze_neighbors=args.freeze_neighbors,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
