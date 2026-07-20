#!/usr/bin/env python
import argparse
import json
import random

import numpy as np
import torch
from e3nn import o3
from torch_geometric.loader import DataLoader

from equivariant_nas.builder import build_equiformer
from equivariant_nas.candidate import extract_literal_spec
from equivariant_nas.diagnostics import _model_call, _single_graph
from equivariant_nas.diagnostics import symmetry_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze-neighbors", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    spec = extract_literal_spec(args.program)
    model = build_equiformer(spec, "/home/20262202788/equiformer").cuda().eval()
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
    graph = _single_graph(batch)
    if args.freeze_neighbors:
        print(
            json.dumps(
                symmetry_report(
                    model, batch, rotations=2, translations=0, freeze_neighbors=True
                ).to_dict(),
                indent=2,
                sort_keys=True,
            )
        )
        return
    captures = []
    hooks = [
        block.register_forward_hook(lambda _m, _i, out: captures.append(out.detach()))
        for block in model.blocks
    ]
    with torch.no_grad():
        captures.clear()
        _model_call(model, graph)
        reference = [item.clone() for item in captures]
        rotation = o3.rand_matrix(dtype=torch.float64, device=graph.pos.device)
        captures.clear()
        _model_call(
            model,
            graph,
            positions=graph.pos @ rotation.to(graph.pos.dtype).transpose(0, 1),
        )
        observed = [item.clone() for item in captures]
    rows = []
    for index, (base, rotated) in enumerate(zip(reference, observed)):
        irreps = model.irreps_node_embedding if index < len(model.blocks) - 1 else model.irreps_feature
        d_rotation = irreps.D_from_matrix(rotation).to(base.dtype)
        d_inverse = irreps.D_from_matrix(rotation.transpose(0, 1)).to(base.dtype)
        expected = base @ d_rotation.transpose(0, 1)
        expected_direct = base @ d_rotation
        expected_inverse_t = base @ d_inverse.transpose(0, 1)
        expected_inverse = base @ d_inverse
        diff = torch.linalg.vector_norm((expected - rotated).reshape(-1))
        diff_direct = torch.linalg.vector_norm((expected_direct - rotated).reshape(-1))
        diff_inverse_t = torch.linalg.vector_norm((expected_inverse_t - rotated).reshape(-1))
        diff_inverse = torch.linalg.vector_norm((expected_inverse - rotated).reshape(-1))
        ref = torch.linalg.vector_norm(expected.reshape(-1))
        obs = torch.linalg.vector_norm(rotated.reshape(-1))
        rows.append(
            {
                "block": index,
                "elements": expected.numel(),
                "difference_norm": float(diff.cpu()),
                "reference_norm": float(ref.cpu()),
                "observed_norm": float(obs.cpu()),
                "relative_to_reference": float((diff / ref.clamp_min(1e-12)).cpu()),
                "relative_direct_D": float(
                    (
                        diff_direct
                        / torch.linalg.vector_norm(expected_direct.reshape(-1)).clamp_min(1e-12)
                    ).cpu()
                ),
                "relative_inverse_D_transpose": float(
                    (
                        diff_inverse_t
                        / torch.linalg.vector_norm(expected_inverse_t.reshape(-1)).clamp_min(1e-12)
                    ).cpu()
                ),
                "relative_inverse_D": float(
                    (
                        diff_inverse
                        / torch.linalg.vector_norm(expected_inverse.reshape(-1)).clamp_min(1e-12)
                    ).cpu()
                ),
                "symmetric_relative": float((diff / torch.maximum(ref, obs).clamp_min(1e-12)).cpu()),
                "rms_difference": float((diff / expected.numel() ** 0.5).cpu()),
            }
        )
    for hook in hooks:
        hook.remove()
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
