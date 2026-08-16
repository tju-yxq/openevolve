#!/usr/bin/env python
"""Run the frozen online V3 protocol using auditable JSON command adapters."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from equivariant_nas.dsl.online_controller import OnlineV3Controller
from equivariant_nas.dsl.online_protocol import OnlineV3Protocol


def _invoke(template: str, payload, **values):
    env = os.environ.copy()
    run_root = Path(values.get("run_root", os.environ.get("EQUINAS_RUN_ROOT", ".")))
    request_root = run_root / "requests"
    request_root.mkdir(parents=True, exist_ok=True)
    descriptor, request_name = tempfile.mkstemp(prefix="request_", suffix=".json", dir=str(request_root))
    request_path = Path(request_name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    env.pop("EQUINAS_REQUEST_JSON", None)
    env["EQUINAS_REQUEST_JSON_PATH"] = str(request_path.resolve())
    command = template.format(**{key: shlex.quote(str(value)) for key, value in values.items()})
    try:
        completed = subprocess.run(command, shell=True, text=True, capture_output=True, env=env, check=False)
    finally:
        request_path.unlink(missing_ok=True)
    if completed.returncode:
        raise RuntimeError(f"adapter exited {completed.returncode}: {completed.stderr[-4000:]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("adapter must emit one JSON object on stdout") from error


def run(args):
    protocol = OnlineV3Protocol.load(args.protocol_config)
    root = Path(args.run_root or os.environ.get("EQUINAS_RUN_ROOT", protocol.raw["storage"]["default_run_root"]))

    def generate(attempt, population, parent):
        return _invoke(args.generator_command, {"attempt": attempt, "population": population, "parent": parent, "protocol_hash": protocol.content_hash}, attempt=attempt, run_root=root)

    def train(candidate, endpoint, checkpoint):
        result = _invoke(args.trainer_command, candidate, endpoint=endpoint, checkpoint=checkpoint, run_root=root)
        expected_start = {20000: 0, 80000: 20000, 250000: 80000}[endpoint]
        if result.get("test_evaluated") is not False or int(result.get("endpoint_step", 0)) != endpoint:
            raise RuntimeError("trainer violated endpoint or Test isolation")
        if int(result.get("start_global_step", -1)) != expected_start:
            raise RuntimeError("trainer did not resume from the exact required global step")
        expected_data = "full_train" if endpoint == 250000 else "fixed_quarter"
        if result.get("training_data") != expected_data:
            raise RuntimeError("trainer used the wrong data identity")
        return result

    def test(winner):
        if not args.test_command:
            raise RuntimeError("--test-command is required only when advancing beyond the frozen winner to final Test")
        result = _invoke(args.test_command, winner, run_root=root)
        if result.get("split") != "test" or result.get("evaluation_only") is not True:
            raise RuntimeError("final Test adapter must be evaluation-only")
        return result

    controller = OnlineV3Controller(root, protocol, generator=generate, trainer=train, test_evaluator=test)
    state = controller.run(stop_after_stage=args.stop_after_stage)
    print(json.dumps({"run_root": str(root), "stage": state["stage"], "protocol_hash": protocol.content_hash}, sort_keys=True))
    return state


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", default=str(PROJECT_ROOT / "configs/dsl_v3_online_20k_60_protocol.json"))
    parser.add_argument("--run-root", default="")
    parser.add_argument("--generator-command", required=True, help="shell template; receives request in EQUINAS_REQUEST_JSON")
    parser.add_argument("--trainer-command", required=True, help="supports {endpoint}, {checkpoint}, and {run_root}")
    parser.add_argument("--test-command", default="", help="evaluation-only Test adapter; omit when stopping at evaluate_test")
    parser.add_argument("--stop-after-stage", choices=("freeze_20k_selection", "train_80k", "freeze_80k_selection", "train_250k", "freeze_winner", "evaluate_test", "ready_for_tos_archive"), default="ready_for_tos_archive")
    return parser


if __name__ == "__main__":
    run(get_parser().parse_args())
