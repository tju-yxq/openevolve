#!/usr/bin/env python
"""Build the baseline spec and print an auditable summary."""

import argparse
import json

from equivariant_nas.builder import build_equiformer, count_trainable_parameters
from equivariant_nas.spec import baseline_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--equiformer-root", required=True)
    args = parser.parse_args()
    spec = baseline_spec()
    model = build_equiformer(spec, args.equiformer_root)
    print(
        json.dumps(
            {
                "architecture_id": spec.architecture_id(),
                "parameter_count": count_trainable_parameters(model),
                "embedding_irreps": spec.representation.embedding_irreps(),
                "head_irreps": spec.representation.head_irreps(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

