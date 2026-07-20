#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from equivariant_nas.fidelity_trust import assess_fidelity_trust


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--minimum-cohort", type=int, default=8)
    args = parser.parse_args()
    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    proxy = {item["name"]: item["proxy_mae"] for item in pairs}
    reference = {item["name"]: item["reference_mae"] for item in pairs}
    report = assess_fidelity_trust(
        proxy,
        reference,
        top_k=args.top_k,
        minimum_cohort=args.minimum_cohort,
    ).to_dict()
    report["pairs"] = pairs
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
