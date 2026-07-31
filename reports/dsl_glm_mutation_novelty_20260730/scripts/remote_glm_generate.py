#!/usr/bin/env python
"""One-shot GLM bridge executed on the existing OpenEvolve server.

The helper reads one JSON request from stdin and returns one model response.
It does not import the NAS evaluator, construct a model, access a dataset, or
start training.  The API credential remains inside the server-side OpenEvolve
configuration and is never printed.
"""

from __future__ import print_function

import argparse
import asyncio
import base64
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser("remote-glm-generate-once")
    parser.add_argument("--openevolve-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


async def generate(args, payload):
    sys.path.insert(0, args.openevolve_root)
    from openevolve.config import load_config
    from openevolve.llm.ensemble import LLMEnsemble

    config = load_config(args.config)
    for model in config.llm.models:
        model.random_seed = int(args.seed)
    ensemble = LLMEnsemble(config.llm.models)
    return await ensemble.generate_with_context(
        system_message=str(payload["system"]),
        messages=[{"role": "user", "content": str(payload["user"])}],
    )


def main():
    args = parse_args()
    payload = json.loads(sys.stdin.read())
    if set(payload) != {"system", "user"}:
        raise ValueError("request must contain exactly system and user")
    response = asyncio.run(generate(args, payload))
    encoded = base64.b64encode(str(response).encode("utf-8")).decode("ascii")
    print("__GLM_RESPONSE_BASE64__=" + encoded)


if __name__ == "__main__":
    main()
