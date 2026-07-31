#!/usr/bin/env python
"""Keep the GLM credential local while the authoritative compiler runs by SSH."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REQUEST_PREFIX = "__GLM_REQUEST_BASE64__="
RESPONSE_PREFIX = "__GLM_RESPONSE_BASE64__="


def parse_args():
    parser = argparse.ArgumentParser("run-server-glm-bridge")
    parser.add_argument("--host", default="volcano-equiformer")
    parser.add_argument(
        "--remote-python",
        default="/home/20262202788/conda-envs/equiformer/bin/python",
    )
    parser.add_argument("--remote-script", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--api-key-env", default="GLM_API_KEY")
    parser.add_argument("--api-base", default="https://glm.llm.autos/v1")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("remote_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def credential(name):
    value = os.environ.get(name, "")
    if value:
        return value
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, name)
                return str(value)
        except (FileNotFoundError, OSError):
            pass
    raise RuntimeError("environment credential {} is not configured".format(name))


def call_glm(args, key, payload):
    from openai import OpenAI

    error = None
    for attempt in range(int(args.retries) + 1):
        try:
            client = OpenAI(
                api_key=key,
                base_url=args.api_base,
                timeout=float(args.timeout),
                max_retries=0,
            )
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": str(payload["system"])},
                    {"role": "user", "content": str(payload["user"])},
                ],
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_tokens),
                seed=int(payload["seed"]) + 7919 * attempt,
            )
            content = completion.choices[0].message.content
            if content is None:
                raise RuntimeError("GLM returned an empty response")
            return str(content)
        except Exception as exc:
            error = exc
            if attempt >= int(args.retries):
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    raise error


def main():
    args = parse_args()
    key = credential(args.api_key_env)
    log_path = Path(args.log).resolve()
    remote_args = list(args.remote_args)
    if remote_args and remote_args[0] == "--":
        remote_args = remote_args[1:]
    command = [
        "ssh",
        "-T",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=20",
        args.host,
        args.remote_python,
        "-u",
        args.remote_script,
    ] + remote_args
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        for line in process.stdout:
            stripped = line.rstrip("\r\n")
            if not stripped.startswith(REQUEST_PREFIX):
                print(stripped, flush=True)
                append_jsonl(
                    log_path,
                    {"time": now(), "event": "remote_output", "text": stripped[-4000:]},
                )
                continue
            payload = json.loads(
                base64.b64decode(stripped.split("=", 1)[1]).decode("utf-8")
            )
            started = now()
            response = ""
            error = ""
            try:
                response = call_glm(args, key, payload)
                encoded = base64.b64encode(response.encode("utf-8")).decode("ascii")
                process.stdin.write(RESPONSE_PREFIX + encoded + "\n")
                process.stdin.flush()
            except Exception as exc:
                message = "{}: {}".format(type(exc).__name__, str(exc))
                if key:
                    message = message.replace(key, "<redacted>")
                error = message[-2000:]
                process.stdin.close()
                process.terminate()
                raise RuntimeError(error) from None
            finally:
                append_jsonl(
                    log_path,
                    {
                        "time": now(),
                        "event": "glm_call",
                        "started_at": started,
                        "call_index": int(payload.get("call_index", 0)),
                        "seed": int(payload.get("seed", 0)),
                        "model": args.model,
                        "api_base": args.api_base,
                        "credential_serialized": False,
                        "system_sha256": sha256_text(str(payload.get("system", ""))),
                        "user_sha256": sha256_text(str(payload.get("user", ""))),
                        "response_sha256": sha256_text(response) if response else "",
                        "error": error,
                    },
                )
    finally:
        returncode = process.wait(timeout=30)
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
