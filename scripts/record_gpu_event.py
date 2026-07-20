#!/usr/bin/env python
import argparse

from equivariant_nas.budget import BudgetLedger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--gpu-seconds", type=float, required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    ledger = BudgetLedger(
        "/home/20262202788/equivariant-nas/runs/budget_ledger.jsonl", 5.0
    )
    ledger.append(
        {
            "architecture_id": "manual-audit",
            "stage": args.stage,
            "seed": 0,
            "gpu_seconds": args.gpu_seconds,
            "valid": True,
            "note": args.note,
        }
    )
    print("used_gpu_hours={:.6f}".format(ledger.used_gpu_hours()))


if __name__ == "__main__":
    main()
