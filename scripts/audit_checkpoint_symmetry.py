#!/usr/bin/env python
import argparse
import json
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from equivariant_nas.builder import build_equiformer
from equivariant_nas.candidate import extract_literal_spec
from equivariant_nas.diagnostics import symmetry_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotations", type=int, default=5)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_equiformer(
        extract_literal_spec(args.program), "/home/20262202788/equiformer"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.cuda()
    from datasets.pyg.qm9 import QM9

    batch = next(
        iter(
            DataLoader(
                QM9(
                    "/home/20262202788/equiformer/datasets/qm9",
                    "train",
                    feature_type="one_hot",
                ),
                batch_size=2,
            )
        )
    ).cuda()
    report = symmetry_report(
        model, batch, rotations=args.rotations, translations=3, freeze_neighbors=False
    ).to_dict()
    payload = {
        "program": args.program,
        "checkpoint": args.checkpoint,
        "global_step": checkpoint.get("global_step"),
        "report": report,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
