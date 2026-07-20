#!/usr/bin/env python
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in Path(args.ledger).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = [
        float(item["gpu_seconds"]) / 3600.0
        for item in records
        if item.get("stage") == "steps5000"
        and item.get("valid")
        and float(item.get("gpu_seconds", 0.0)) > 300.0
    ]
    if not samples:
        raise RuntimeError("no completed 5,000-step samples in ledger")
    median = statistics.median(samples)
    methods = len(config["methods"])
    micro_seeds = len(config["gate2a_micro"]["search_seeds"])
    micro_candidates = int(
        config["gate2a_micro"]["trained_valid_candidates_per_method"]
    )
    micro_count = methods * micro_seeds * micro_candidates
    micro_estimate = median * micro_count
    full_seeds = len(config["gate2a"]["search_seeds"])
    full_candidates = int(
        config["gate2a"]["trained_valid_candidates_per_method_total"]
    )
    full_count = methods * full_seeds * full_candidates
    full_estimate = median * full_count
    payload = {
        "observed_samples": len(samples),
        "median_a100_hours_per_candidate": median,
        "micro_gate": {
            "trained_candidate_count": micro_count,
            "point_estimate_a100_hours": micro_estimate,
            "estimate_with_20_percent_reserve": micro_estimate * 1.2,
            "registered_cap": config["gate2a_micro"][
                "incremental_a100_hour_cap"
            ],
        },
        "gate2a_cumulative_if_unlocked": {
            "trained_candidate_count": full_count,
            "point_estimate_a100_hours": full_estimate,
            "estimate_with_20_percent_reserve": full_estimate * 1.2,
            "registered_cap": config["gate2a"][
                "cumulative_a100_hour_cap_including_micro_gate"
            ],
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
