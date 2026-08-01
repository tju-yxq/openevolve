#!/usr/bin/env python
"""Freeze the official V1 nonlinear-message GraphAttention parameter contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _instruction(item) -> dict[str, Any]:
    return {
        "left": int(item.i_in1),
        "right": int(item.i_in2),
        "out": int(item.i_out),
        "mode": str(item.connection_mode),
        "has_weight": bool(item.has_weight),
        "path_weight": float(item.path_weight),
        "path_shape": list(item.path_shape),
    }


def _tp(module) -> dict[str, Any]:
    tp = module.tp
    return {
        "irreps_in1": str(tp.irreps_in1),
        "irreps_in2": str(tp.irreps_in2),
        "irreps_out": str(tp.irreps_out),
        "internal_weights": bool(tp.internal_weights),
        "shared_weights": bool(tp.shared_weights),
        "weight_numel": int(tp.weight_numel),
        "instructions": [_instruction(item) for item in tp.instructions],
        "slices_sqrt_k": {
            str(index): {
                "slice": [value[0].start, value[0].stop, value[0].step],
                "scale": float(value[1]),
            }
            for index, value in module.slices_sqrt_k.items()
        },
    }


def _module(module) -> dict[str, Any]:
    return {
        "class": "{}.{}".format(type(module).__module__, type(module).__qualname__),
        "parameters": {
            name: list(parameter.shape)
            for name, parameter in module.named_parameters()
        },
    }


def build_audit(output: Path, official_root: Path) -> dict[str, Any]:
    import torch

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    from e3nn import o3

    from scripts.audit_equiformer_v1_graph_attention_contract import _load_official_module

    official_module, source = _load_official_module(official_root, torch)
    config = {
        "irreps_node_input": o3.Irreps("4x0e+2x1e+1x2e"),
        "irreps_node_attr": o3.Irreps("1x0e"),
        "irreps_edge_attr": o3.Irreps("1x0e+1x1e+1x2e"),
        "irreps_node_output": o3.Irreps("4x0e+2x1e+1x2e"),
        "fc_neurons": [6, 8],
        "irreps_head": o3.Irreps("2x0e+1x1e+1x2e"),
        "num_heads": 2,
        "irreps_pre_attn": None,
        "rescale_degree": False,
        "nonlinear_message": True,
        "alpha_drop": 0.0,
        "proj_drop": 0.0,
    }
    torch.manual_seed(20260751)
    model = official_module.GraphAttention(**config).double().eval()
    payload = {
        "oracle": "EquiformerV1.GraphAttention(nonlinear_message=True)",
        "source_file": str(source.resolve()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_tensor_count": len(tuple(model.parameters())),
        "parameters": {
            name: list(parameter.shape)
            for name, parameter in model.named_parameters()
        },
        "irreps": {
            "sep_act.dtp_out": str(model.sep_act.dtp.irreps_out),
            "sep_act.lin_out": str(model.sep_act.lin.irreps_out),
            "sep_act.gate_in": str(model.sep_act.gate.irreps_in),
            "sep_act.gate_out": str(model.sep_act.gate.irreps_out),
            "sep_alpha_out": str(model.sep_alpha.irreps_out),
            "sep_value.dtp_out": str(model.sep_value.dtp.irreps_out),
            "sep_value.lin_out": str(model.sep_value.lin.irreps_out),
        },
        "modules": {
            "merge_src": _module(model.merge_src),
            "merge_dst": _module(model.merge_dst),
            "sep_act": _module(model.sep_act),
            "sep_alpha": _module(model.sep_alpha),
            "sep_value": _module(model.sep_value),
            "proj": _module(model.proj),
        },
        "tensor_products": {
            "sep_act.dtp": _tp(model.sep_act.dtp),
            "sep_value.dtp": _tp(model.sep_value.dtp),
        },
        "claims": {
            "sep_act_uses_external_unshared_uvu": (
                not model.sep_act.dtp.tp.internal_weights
                and not model.sep_act.dtp.tp.shared_weights
            ),
            "sep_value_uses_internal_shared_uvu": (
                model.sep_value.dtp.tp.internal_weights
                and model.sep_value.dtp.tp.shared_weights
            ),
            "official_constructor_used_for_dsl_execution": False,
            "dsl_implementation_completed": False,
        },
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("audit-equiformer-v1-nonlinear-message-contract")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_audit(Path(args.output), Path(args.official_root))
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "parameter_tensor_count": payload["parameter_tensor_count"],
                "parameter_count": payload["parameter_count"],
                "sep_act_weight_numel": payload["tensor_products"]["sep_act.dtp"]["weight_numel"],
                "sep_value_weight_numel": payload["tensor_products"]["sep_value.dtp"]["weight_numel"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
