from dataclasses import replace

import torch

from equivariant_nas.inheritance import analyze_transfer, apply_transfer
from equivariant_nas.spec import baseline_spec


def tensor(*shape, value=0.0):
    return torch.full(shape, value)


def test_operator_basis_change_blocks_radial_semantics_but_transfers_safe_state():
    parent = baseline_spec()
    child = replace(
        parent,
        operator=replace(parent.operator, basis_type="bessel", num_basis=64),
    ).validate()
    parent_state = {
        "atom_embed.weight": tensor(4, 4, value=1.0),
        "rbf.rbf.frequencies": tensor(64, value=1.0),
        "blocks.0.ga.sep_act.dtp_rad.net.2.weight": tensor(4, 4, value=1.0),
        "blocks.0.ffn.weight": tensor(4, 4, value=1.0),
    }
    child_state = {
        key: tensor(*value.shape, value=0.0) for key, value in parent_state.items()
    }
    merged, report = apply_transfer(parent, child, parent_state, child_state)
    assert torch.all(merged["atom_embed.weight"] == 1.0)
    assert torch.all(merged["blocks.0.ffn.weight"] == 1.0)
    assert torch.all(merged["rbf.rbf.frequencies"] == 0.0)
    assert torch.all(merged["blocks.0.ga.sep_act.dtp_rad.net.2.weight"] == 0.0)
    assert report.blocked_reason_counts["radial_basis_semantics_changed"] == 1
    assert report.selection_eligible is False


def test_representation_change_never_uses_shape_only_transfer():
    parent = baseline_spec()
    child = replace(
        parent,
        representation=replace(parent.representation, lmax=3, l3_channels=8, head_l3_channels=8),
    ).validate()
    state = {"same.shape.weight": tensor(2, 2)}
    report = analyze_transfer(parent, child, state, state)
    assert report.transferred_tensors == 0
    assert report.blocked_reason_counts["representation_irrep_layout_changed"] == 1
