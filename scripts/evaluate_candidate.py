#!/usr/bin/env python
import argparse
import json

from equivariant_nas.evaluation import cached_static_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--equiformer-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--parameter-ratio-limit", type=float, default=1.2)
    args = parser.parse_args()
    result = cached_static_evaluate(
        args.program,
        args.equiformer_root,
        args.cache_dir,
        args.parameter_ratio_limit,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

