#!/usr/bin/env python
"""Generate audit-only local DSL mutations with the existing GLM workflow.

This program deliberately stops before any optimizer, dataset loader, or NAS
evaluator is constructed.  It uses the production Router -> Critic ->
Synthesizer -> compiler-guided Repair protocol, but admits children only to the
experimental node-graph backend for structural and equivariance auditing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


def parse_args():
    parser = argparse.ArgumentParser("generate-glm-local-dsl-mutants")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--task-contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transport", choices=("local", "ssh", "bridge"), default="local")
    parser.add_argument("--api-key-env", default="GLM_API_KEY")
    parser.add_argument("--api-base", default="https://glm.llm.autos/v1")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--ssh-host", default="volcano-equiformer")
    parser.add_argument("--remote-python", default="/home/20262202788/conda-envs/openevolve/bin/python")
    parser.add_argument("--remote-helper", default="")
    parser.add_argument("--remote-openevolve-root", default="/home/20262202788/openevolve")
    parser.add_argument("--remote-config", default="/home/20262202788/openevolve/configs/local_glm_5_2.yaml")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--maximum-attempts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument(
        "--protocol",
        choices=("baseline", "factor_isolated_experience"),
        default="baseline",
    )
    parser.add_argument(
        "--factor-sequence",
        default="",
        help="Optional comma-separated forced factor schedule, for example E6.1,E6.2,E6.3,E6.4.",
    )
    parser.add_argument(
        "--factor-policy",
        choices=("balanced", "annealed_epsilon_greedy"),
        default="balanced",
        help="Global factor-allocation policy. Local synthesis remains factor-isolated.",
    )
    parser.add_argument("--minimum-per-factor", type=int, default=1)
    parser.add_argument("--epsilon-start", type=float, default=0.75)
    parser.add_argument("--epsilon-end", type=float, default=0.12)
    parser.add_argument("--epsilon-decay-attempts", type=float, default=24.0)
    parser.add_argument("--stagnation-patience", type=int, default=6)
    parser.add_argument("--stagnation-boost", type=float, default=0.20)
    parser.add_argument("--maximum-factor-share", type=float, default=0.50)
    parser.add_argument("--repeat-factor-penalty", type=float, default=0.35)
    parser.add_argument(
        "--experience-file",
        default="",
        help="Optional prior experience-memory JSON used by Router, Critic, and Synthesizer prompts.",
    )
    parser.add_argument("--audit-script", default="")
    parser.add_argument("--audit-seed", type=int, default=201)
    parser.add_argument(
        "--audit-seeds",
        default="",
        help="Optional comma-separated robust-admission seeds; defaults to audit-seed only.",
    )
    parser.add_argument("--audit-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--audit-timeout", type=float, default=300.0)
    parser.add_argument("--minimum-output-norm-ratio", type=float, default=0.0)
    parser.add_argument("--maximum-output-norm-ratio", type=float, default=100.0)
    parser.add_argument("--maximum-inserted-gradient", type=float, default=1.0e8)
    return parser.parse_args()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SSHGLMEnsemble:
    """OpenEvolve-compatible ensemble whose single call runs over SSH."""

    def __init__(
        self,
        *,
        host: str,
        remote_python: str,
        remote_helper: str,
        remote_openevolve_root: str,
        remote_config: str,
        seed: int,
        log_path: Path,
    ):
        self.host = host
        self.remote_python = remote_python
        self.remote_helper = remote_helper
        self.remote_openevolve_root = remote_openevolve_root
        self.remote_config = remote_config
        self.seed = int(seed)
        self.call_index = 0
        self.log_path = log_path

    async def generate_with_context(self, *, system_message: str, messages: Sequence[Mapping[str, str]]):
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("SSH GLM bridge expects exactly one user message")
        self.call_index += 1
        call_seed = self.seed + 1009 * self.call_index
        payload = {"system": system_message, "user": str(messages[0]["content"])}
        return await asyncio.to_thread(self._call_sync, payload, call_seed)

    def _call_sync(self, payload: Mapping[str, str], call_seed: int) -> str:
        remote_command = " ".join(
            shlex.quote(item)
            for item in (
                self.remote_python,
                self.remote_helper,
                "--openevolve-root",
                self.remote_openevolve_root,
                "--config",
                self.remote_config,
                "--seed",
                str(call_seed),
            )
        )
        login_command = "bash -lc {}".format(shlex.quote(remote_command))
        started = now()
        completed = subprocess.run(
            ["ssh", self.host, login_command],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        response = None
        for line in completed.stdout.splitlines():
            if line.startswith("__GLM_RESPONSE_BASE64__="):
                encoded = line.split("=", 1)[1].strip()
                response = base64.b64decode(encoded).decode("utf-8")
        append_jsonl(
            self.log_path,
            {
                "call_index": self.call_index,
                "seed": call_seed,
                "started_at": started,
                "completed_at": now(),
                "returncode": int(completed.returncode),
                "system_sha256": sha256_text(payload["system"]),
                "user_sha256": sha256_text(payload["user"]),
                "response_sha256": sha256_text(response) if response is not None else "",
                "stderr_tail": completed.stderr[-2000:],
                "stdout_without_response_tail": "\n".join(
                    line for line in completed.stdout.splitlines()
                    if not line.startswith("__GLM_RESPONSE_BASE64__=")
                )[-2000:],
            },
        )
        if completed.returncode != 0 or response is None:
            raise RuntimeError(
                "remote GLM call failed with return code {}: {}".format(
                    completed.returncode,
                    completed.stderr[-1000:],
                )
            )
        return response


class LocalGLMEnsemble:
    """OpenEvolve-compatible ensemble using a local OpenAI-compatible client.

    The credential is read by the Python process from one named environment
    variable.  It is never added to a command line, serialized to an artifact,
    logged, or copied to the remote training server.
    """

    def __init__(
        self,
        *,
        api_key_env: str,
        api_base: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: float,
        retries: int,
        seed: int,
        log_path: Path,
    ):
        if not os.environ.get(api_key_env):
            raise ValueError("Environment variable {} is not set".format(api_key_env))
        self.api_key_env = str(api_key_env)
        self.api_base = str(api_base)
        self.model = str(model)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.seed = int(seed)
        self.call_index = 0
        self.log_path = log_path

    async def generate_with_context(self, *, system_message: str, messages: Sequence[Mapping[str, str]]):
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("Local GLM adapter expects exactly one user message")
        self.call_index += 1
        call_seed = self.seed + 1009 * self.call_index
        payload = {"system": system_message, "user": str(messages[0]["content"])}
        return await asyncio.to_thread(self._call_sync, payload, call_seed)

    def _call_sync(self, payload: Mapping[str, str], call_seed: int) -> str:
        from openai import OpenAI

        started = now()
        response = None
        error = ""
        try:
            client = OpenAI(
                api_key=os.environ[self.api_key_env],
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=self.retries,
            )
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": payload["system"]},
                    {"role": "user", "content": payload["user"]},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                seed=int(call_seed),
            )
            response = completion.choices[0].message.content
            if response is None:
                raise RuntimeError("GLM returned an empty message content")
            response = str(response)
            return response
        except Exception as exc:
            secret = os.environ.get(self.api_key_env, "")
            message = "{}: {}".format(type(exc).__name__, str(exc))
            if secret:
                message = message.replace(secret, "<redacted>")
            error = message[-2000:]
            raise RuntimeError(error) from None
        finally:
            append_jsonl(
                self.log_path,
                {
                    "transport": "local_openai_compatible",
                    "call_index": self.call_index,
                    "seed": call_seed,
                    "started_at": started,
                    "completed_at": now(),
                    "model": self.model,
                    "api_base": self.api_base,
                    "api_key_env": self.api_key_env,
                    "credential_serialized": False,
                    "system_sha256": sha256_text(payload["system"]),
                    "user_sha256": sha256_text(payload["user"]),
                    "response_sha256": sha256_text(response) if response is not None else "",
                    "error": error,
                },
            )


class BridgeGLMEnsemble:
    """Send prompts to a trusted local client while compilation runs remotely.

    The remote process writes a base64-encoded request marker to stdout and
    reads one base64-encoded response marker from stdin.  No credential is
    transmitted to, serialized on, or read by the remote server.
    """

    REQUEST_PREFIX = "__GLM_REQUEST_BASE64__="
    RESPONSE_PREFIX = "__GLM_RESPONSE_BASE64__="

    def __init__(self, *, seed: int, log_path: Path):
        self.seed = int(seed)
        self.call_index = 0
        self.log_path = log_path

    async def generate_with_context(self, *, system_message: str, messages: Sequence[Mapping[str, str]]):
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("GLM bridge expects exactly one user message")
        self.call_index += 1
        payload = {
            "call_index": self.call_index,
            "seed": self.seed + 1009 * self.call_index,
            "system": str(system_message),
            "user": str(messages[0]["content"]),
        }
        started = now()
        encoded = base64.b64encode(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        print(self.REQUEST_PREFIX + encoded, flush=True)
        loop = asyncio.get_running_loop()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            raise RuntimeError("local GLM bridge closed before returning a response")
        line = line.strip()
        if not line.startswith(self.RESPONSE_PREFIX):
            raise RuntimeError("invalid local GLM bridge response envelope")
        try:
            response = base64.b64decode(
                line.split("=", 1)[1].encode("ascii")
            ).decode("utf-8")
        except Exception as exc:
            raise RuntimeError("invalid base64 response from local GLM bridge: {}".format(exc))
        append_jsonl(
            self.log_path,
            {
                "transport": "local_key_remote_compiler_bridge",
                "call_index": self.call_index,
                "seed": payload["seed"],
                "started_at": started,
                "completed_at": now(),
                "credential_serialized": False,
                "system_sha256": sha256_text(payload["system"]),
                "user_sha256": sha256_text(payload["user"]),
                "response_sha256": sha256_text(response),
            },
        )
        return response


def build_audit_parent(task_id: str):
    try:
        from equivariant_nas.dsl.backends.equiformer_v1_spec import ArchitectureSpec
    except ImportError:
        # The frozen server experiment commit keeps ArchitectureSpec in the
        # pre-DSL compatibility module.  This fallback makes the audit bind to
        # that exact server implementation instead of requiring local files.
        from equivariant_nas.spec import ArchitectureSpec
    from equivariant_nas.dsl.reference_programs import import_equiformer_v1

    official = import_equiformer_v1(ArchitectureSpec(), task_contract=task_id)
    annotations = dict(official.annotations)
    annotations.update(
        {
            "legacy_backend": "equiformer_v1_compositional_audit",
            "reference_backend": "equiformer_v1_compositional_audit",
            "audit_only": True,
            "formal_ranking_admitted": False,
            "lineage_reference_backend": "equiformer_v1",
            "lineage_reference_lock_architecture_id": annotations.get(
                "reference_lock_architecture_id", ""
            ),
            "representation_scope": (
                "full typed node graph with experimental primitive-by-primitive lowering; "
                "not checkpoint-identical to the official constructor"
            ),
        }
    )
    return official, replace(
        official,
        program_id=official.program_id + "_generic_readout_audit",
        annotations=annotations,
    )


def factor_operation_contracts() -> Dict[str, Dict[str, Any]]:
    """Hard source-graph contracts used in addition to prompt instructions."""

    return {
        "E6.1": {
            "allowed": (
                "core.identity@1",
                "core.irrep_linear@1",
                "core.irrep_concat@1",
                "core.residual_add@1",
                "core.global_pool@1",
                "core.select_scalars@1",
                "core.scalar_activation@1",
                "motif.v1_multilevel_readout@1",
            ),
            "required_any": (
                "core.irrep_concat@1",
                "core.residual_add@1",
                "motif.v1_multilevel_readout@1",
            ),
            "forbidden": (
                "core.tensor_product@1",
                "core.invariant_compatibility@1",
                "core.invariant_weight@1",
            ),
            "purpose": "scalar-only cross-depth fusion without tensor coupling or gating",
        },
        "E6.2": {
            "allowed": (
                "core.identity@1",
                "core.irrep_linear@1",
                "core.norm_activation@1",
                "core.tensor_product@1",
                "core.select_scalars@1",
                "core.scalar_activation@1",
                "core.residual_add@1",
                "core.global_pool@1",
            ),
            "required_all": ("core.tensor_product@1", "core.norm_activation@1"),
            "minimum_counts": {"core.norm_activation@1": 3},
            "forbidden": (
                "core.invariant_compatibility@1",
                "core.invariant_weight@1",
            ),
            "purpose": "cross-depth normalized higher-order irrep coupling into l=0 without a compatibility gate",
        },
        "E6.3": {
            "allowed": (
                "core.identity@1",
                "core.irrep_linear@1",
                "core.select_scalars@1",
                "core.scalar_activation@1",
                "core.invariant_weight@1",
                "core.residual_add@1",
                "core.global_pool@1",
            ),
            "required_all": ("core.scalar_activation@1", "core.invariant_weight@1"),
            "forbidden": (
                "core.tensor_product@1",
                "core.invariant_compatibility@1",
            ),
            "purpose": "one-depth bounded invariant gate without cross-depth compatibility",
        },
        "E6.4": {
            "allowed": (
                "core.identity@1",
                "core.irrep_linear@1",
                "core.norm_activation@1",
                "core.select_scalars@1",
                "core.invariant_compatibility@1",
                "core.scalar_activation@1",
                "core.invariant_weight@1",
                "core.residual_add@1",
                "core.global_pool@1",
            ),
            "required_all": (
                "core.invariant_compatibility@1",
                "core.norm_activation@1",
                "core.scalar_activation@1",
                "core.invariant_weight@1",
            ),
            "minimum_counts": {"core.norm_activation@1": 2},
            "forbidden": ("core.tensor_product@1",),
            "purpose": "bounded cross-depth compatibility gate without higher-order tensor-product leakage",
        },
    }


def compact_factor_experience(experience: Sequence[Mapping[str, Any]], factor_id: str, limit: int = 8):
    rows = []
    for item in reversed(tuple(experience)):
        if str(item.get("factor_id", "")) != str(factor_id):
            continue
        rows.append(
            {
                "architecture_id": str(item.get("architecture_id", "")),
                "outcome": str(item.get("outcome", "unknown")),
                "operator_signature": list(item.get("operator_signature", ())),
                "equivariance_passed": item.get("equivariance_passed"),
                "path_activity": item.get("path_activity", "unknown"),
                "output_norm_ratio": item.get("output_norm_ratio"),
                "maximum_inserted_gradient": item.get("maximum_inserted_gradient"),
                "reason": str(item.get("reason", ""))[:500],
            }
        )
        if len(rows) >= int(limit):
            break
    return tuple(rows)


def build_regions(
    protocol: str = "baseline",
    experience: Sequence[Mapping[str, Any]] = (),
) -> Tuple[Any, ...]:
    from equivariant_nas.dsl.regions import RegionDefinition

    boundaries = (
        "scalar_readout",
        "block0",
        "block1",
        "block2",
        "block3",
        "block4",
        "block5",
    )
    editable = ("graph_pool", "output:prediction")
    shared_allowed = (
        "core.identity@1",
        "core.irrep_linear@1",
        "core.irrep_concat@1",
        "core.residual_add@1",
        "core.tensor_product@1",
        "core.scalar_activation@1",
        "core.invariant_weight@1",
        "core.global_pool@1",
        "core.select_scalars@1",
        "core.invariant_compatibility@1",
        "core.equivariant_norm@1",
        "core.norm_activation@1",
        "motif.v1_multilevel_readout@1",
    )
    common_invariants = (
        "The six Equiformer V1 message blocks and scalar_readout remain frozen.",
        "The task output remains exactly one graph-carried SO(3) scalar.",
        "A non-scalar irrep may affect the output only through a typed operation that produces l=0, such as tensor_product.",
        "Every inserted node must lie on the prediction path; dead branches are forbidden.",
        "No dataset, target, optimizer, evaluator, task contract, or test split is editable.",
        "core.invariant_compatibility produces an invariant scalar: its declared frame.kind must be invariant, never global.",
        "This region is admitted only to experimental_node_graph auditing and never to formal ranking.",
    )
    definitions = (
        (
            "experimental_cross_depth_scalar_fusion",
            "E6.1",
            "Construct a local cross-depth invariant readout by extracting and fusing scalar channels from one or more frozen blocks with the terminal scalar path.",
            (
                "Prefer a topology that differs from the existing one-tap multilevel readout.",
                "Keep the number of inserted nodes small and state which depths interact.",
            ),
        ),
        (
            "experimental_irrep_coupled_readout",
            "E6.2",
            "Let higher-order hidden irreps contribute to the scalar prediction through a legal self- or cross-depth tensor product whose requested output is l=0.",
            (
                "Do not merely discard all l>0 channels with select_scalars.",
                "Use only Clebsch-Gordan-legal output irreps and return one scalar prediction.",
            ),
        ),
        (
            "experimental_invariant_gated_readout",
            "E6.3",
            "Build an invariant gate from a frozen intermediate representation and use it to modulate or residually correct the terminal scalar readout.",
            (
                "The gate must be an l=0 node or graph scalar with carrier compatible with the value it weights.",
                "Avoid an inactive side branch that is disconnected from prediction.",
            ),
        ),
        (
            "experimental_cross_depth_compatibility_readout",
            "E6.4",
            "Use equal-type hidden states from two depths to form an invariant compatibility signal and integrate it into the terminal prediction.",
            (
                "invariant_compatibility inputs must have exactly equal equivariant types.",
                "The compatibility path must materially alter the prediction graph.",
            ),
        ),
    )
    contracts = factor_operation_contracts()
    regions = []
    for region_id, factor_id, description, extra in definitions:
        contract = contracts[factor_id]
        lessons = compact_factor_experience(experience, factor_id)
        if protocol == "factor_isolated_experience":
            allowed = tuple(contract["allowed"])
            dynamic = (
                "Hard factor-isolation contract: inserted source nodes may use only allowed_ops; this is audited after compilation.",
                "The mutation must implement {}.".format(contract["purpose"]),
                "Any scalar used as invariant_weight.weight must be produced directly by scalar_activation(function=tanh or sigmoid).",
                "Prefer a residual correction whose random-weight output norm stays within 100x of the parent; unbounded multiplicative gates are rejected.",
                "Prior same-factor experience visible to this attempt: {}".format(
                    json.dumps(lessons, ensure_ascii=False, sort_keys=True)
                ),
            )
            if factor_id == "E6.4":
                dynamic = dynamic + (
                    "Before invariant_compatibility, pass both depth features through distinct core.norm_activation nodes with normalize=true and function=tanh.",
                    "After invariant_compatibility, pass the scalar score through core.scalar_activation(function=tanh or sigmoid) and use that node directly as invariant_weight.weight.",
                    "Use scalar_readout directly as invariant_weight.value; residual_add must combine scalar_readout with that weighted correction; rewire the existing graph_pool.x to the residual.",
                    "Do not insert a new global_pool. Do not pool the gate or scalar_readout separately before weighting.",
                    "Type law: compatibility and its scalar activation are node-carried 1x0 with frame.kind=invariant; invariant_weight and residual_add outputs are node-carried 1x0 with frame.kind=global; graph_pool is graph-carried 1x0 global.",
                    "core.select_scalars is allowed only as a type-alignment helper before norm_activation; do not use tensor_product or an unbounded SiLU/ReLU compatibility gate in E6.4.",
                )
            elif factor_id == "E6.2":
                dynamic = dynamic + (
                    "Both tensor_product inputs must directly reference two distinct core.norm_activation(normalize=true) nodes; normalize before coupling, not only after it.",
                    "Those two norm_activation nodes must consume two different frozen block depths; duplicating one block through two differently named normalization nodes is forbidden.",
                    "Pass the tensor_product output through a third core.norm_activation(normalize=true) before any scalar projection, activation, pooling, or residual fusion.",
                    "The multi-seed archive associates a post-tensor-product irrep_linear projection with gradient explosion. Prefer the parameter-light path norm_activation(tensor_product) -> scalar_activation(tanh) -> global_pool -> residual_add, and do not reproduce rejected architecture b4568ab650d282e8.",
                )
            elif factor_id == "E6.3":
                dynamic = dynamic + (
                    "Build one bounded node-level scalar gate from exactly one frozen depth; use it directly as invariant_weight.weight and scalar_readout directly as invariant_weight.value.",
                    "Residual-add the weighted correction to scalar_readout, then rewire the existing graph_pool; insert no new global_pool.",
                )
        else:
            allowed = shared_allowed
            dynamic = ()
        regions.append(
            RegionDefinition(
                region_id=region_id,
                factor_id=factor_id,
                description=description,
                editable_targets=editable,
                boundary_sources=boundaries,
                allowed_ops=allowed,
                max_new_nodes=12,
                backend_capability="experimental_node_graph",
                invariants=common_invariants + extra + dynamic,
            )
        )
    return tuple(regions)


def visible_names() -> Tuple[str, ...]:
    return (
        "core.identity@1",
        "core.irrep_linear@1",
        "core.irrep_concat@1",
        "core.residual_add@1",
        "core.tensor_product@1",
        "core.scalar_activation@1",
        "core.invariant_weight@1",
        "core.global_pool@1",
        "core.select_scalars@1",
        "core.invariant_compatibility@1",
        "core.equivariant_norm@1",
        "core.norm_activation@1",
        "motif.v1_multilevel_readout@1",
    )


def node_edges(program) -> Tuple[Tuple[str, str, str], ...]:
    node_map = {node.id: node for node in program.nodes}
    edges = []
    for node in program.nodes:
        destination = node.op if "@" in node.op else node.op + "@1"
        for port, references in node.inputs.items():
            for reference in references:
                source_id = reference.split(":", 1)[0]
                if source_id in node_map:
                    source_op = node_map[source_id].op
                    source_op = source_op if "@" in source_op else source_op + "@1"
                else:
                    source_op = reference
                edges.append((source_op, destination, port))
    return tuple(sorted(edges))


def multiset_delta(left: Iterable[Any], right: Iterable[Any]) -> Dict[str, Any]:
    from collections import Counter

    a = Counter(left)
    b = Counter(right)
    removed = list((a - b).elements())
    inserted = list((b - a).elements())
    return {
        "removed_count": len(removed),
        "inserted_count": len(inserted),
        "removed": removed,
        "inserted": inserted,
    }


def load_history(project_root: Path, output: Path, compiler, task) -> Dict[str, Sequence[str]]:
    from equivariant_nas.dsl.serialization import load_program

    matches: Dict[str, list] = {}
    for path in project_root.rglob("*.dsl.json"):
        try:
            resolved = path.resolve()
            if output == resolved or output in resolved.parents:
                continue
            program = load_program(str(path))
            artifact = compiler.analyze(program, task)
            matches.setdefault(artifact.architecture_id, []).append(str(resolved))
        except Exception:
            continue
    return {key: tuple(sorted(value)) for key, value in matches.items()}


def candidate_audit(parent_artifact, child_artifact, child_program, lowering, history, primitives):
    from equivariant_nas.dsl.canonicalize import canonical_json

    parent_program = parent_artifact.source_program
    parent_expanded = parent_artifact.expanded_program
    child_expanded = child_artifact.expanded_program
    # Canonicalize the expanded primitive graph.  The source graph may still
    # contain legal motif nodes, which cannot be resolved by PrimitiveRegistry
    # directly and previously caused a false E_REGISTRY_002 failure here.
    child_canonical = canonical_json(child_expanded, primitives)
    digest = sha256_text(child_canonical)
    parent_ops = [node.op if "@" in node.op else node.op + "@1" for node in parent_expanded.nodes]
    child_ops = [node.op if "@" in node.op else node.op + "@1" for node in child_expanded.nodes]
    parent_edges = node_edges(parent_expanded)
    child_edges = node_edges(child_expanded)
    exact_history = list(history.get(child_artifact.architecture_id, ()))
    source_parent_ids = {node.id for node in parent_program.nodes}
    source_child_ids = {node.id for node in child_program.nodes}
    inserted_source = sorted(source_child_ids - source_parent_ids)
    deleted_source = sorted(source_parent_ids - source_child_ids)
    changed_source = sorted(
        node_id
        for node_id in source_parent_ids & source_child_ids
        if next(node for node in parent_program.nodes if node.id == node_id).to_dict()
        != next(node for node in child_program.nodes if node.id == node_id).to_dict()
    )
    graph_changed = bool(inserted_source or deleted_source or changed_source)
    topology_delta = multiset_delta(parent_edges, child_edges)
    op_delta = multiset_delta(parent_ops, child_ops)
    return {
        "architecture_id": child_artifact.architecture_id,
        "parent_architecture_id": parent_artifact.architecture_id,
        "semantic_duplicate_of_parent": child_artifact.architecture_id == parent_artifact.architecture_id,
        "canonical_sha256": digest,
        "exact_historical_matches": exact_history,
        "exact_historical_match_count": len(exact_history),
        "source_graph_changed": graph_changed,
        "inserted_source_nodes": inserted_source,
        "deleted_source_nodes": deleted_source,
        "changed_source_nodes": changed_source,
        "expanded_operator_delta": op_delta,
        "expanded_edge_delta": topology_delta,
        "lowering": lowering.to_dict(),
        "structurally_unique_within_local_archive": bool(
            graph_changed
            and child_artifact.architecture_id != parent_artifact.architecture_id
            and not exact_history
        ),
        "formal_ranking_admitted": False,
        "test_split_loaded": False,
    }


def qualified_op(node) -> str:
    return node.op if "@" in node.op else node.op + "@1"


def factor_contract_audit(parent_program, child_program, factor_id: str) -> Dict[str, Any]:
    """Audit factor isolation on every inserted source node.

    The frozen server compiler validates type, scope, and lowering, but its
    RegionDefinition.allowed_ops check is intentionally root-oriented.  This
    experiment adds a stricter post-compile audit over all inserted nodes so a
    nominal E6.4 candidate cannot silently borrow E6.2 tensor products.
    """

    contract = factor_operation_contracts()[str(factor_id)]
    parent_ids = {node.id for node in parent_program.nodes}
    inserted = [node for node in child_program.nodes if node.id not in parent_ids]
    inserted_ops = tuple(qualified_op(node) for node in inserted)
    allowed = set(contract["allowed"])
    forbidden = set(contract.get("forbidden", ()))
    outside = sorted({op for op in inserted_ops if op not in allowed})
    forbidden_used = sorted({op for op in inserted_ops if op in forbidden})
    missing_all = sorted(set(contract.get("required_all", ())) - set(inserted_ops))
    required_any = tuple(contract.get("required_any", ()))
    missing_any = bool(required_any and not set(required_any).intersection(inserted_ops))
    insufficient_counts = {
        op: {"required": int(count), "actual": int(inserted_ops.count(op))}
        for op, count in dict(contract.get("minimum_counts", {})).items()
        if inserted_ops.count(op) < int(count)
    }
    node_map = {node.id: node for node in child_program.nodes}
    unbounded_weights = []
    unnormalized_compatibility_inputs = []
    unnormalized_tensor_product_inputs = []
    topology_violations = []
    for node in inserted:
        if qualified_op(node) != "core.invariant_weight@1":
            continue
        references = tuple(node.inputs.get("weight", ()))
        root = references[0].split(":", 1)[0] if references else ""
        producer = node_map.get(root)
        function = ""
        if producer is not None and qualified_op(producer) == "core.scalar_activation@1":
            function = str(producer.attrs.get("function", "silu"))
        if function not in {"tanh", "sigmoid"}:
            unbounded_weights.append(
                {
                    "node_id": node.id,
                    "weight_reference": references[0] if references else "",
                    "producer_op": "" if producer is None else qualified_op(producer),
                    "activation_function": function,
                }
            )

    if str(factor_id) == "E6.2":
        for node in inserted:
            if qualified_op(node) != "core.tensor_product@1":
                continue
            normalized_input_nodes = []
            for port in ("left", "right"):
                references = tuple(node.inputs.get(port, ()))
                root = references[0].split(":", 1)[0] if references else ""
                producer = node_map.get(root)
                attrs = {} if producer is None else dict(producer.attrs)
                valid = bool(
                    producer is not None
                    and qualified_op(producer) == "core.norm_activation@1"
                    and bool(attrs.get("normalize", True))
                )
                if not valid:
                    unnormalized_tensor_product_inputs.append(
                        {
                            "node_id": node.id,
                            "port": port,
                            "reference": references[0] if references else "",
                            "producer_op": "" if producer is None else qualified_op(producer),
                            "producer_attrs": attrs,
                        }
                    )
                else:
                    normalized_input_nodes.append(producer)
            if len(normalized_input_nodes) == 2:
                normalized_ids = [producer.id for producer in normalized_input_nodes]
                source_roots = []
                for producer in normalized_input_nodes:
                    sources = tuple(producer.inputs.get("x", ()))
                    source_roots.append(
                        sources[0].split(":", 1)[0] if sources else ""
                    )
                if len(set(normalized_ids)) != 2 or len(set(source_roots)) != 2:
                    topology_violations.append(
                        {
                            "rule": "e62_distinct_normalizers_and_frozen_depths",
                            "tensor_product_node": node.id,
                            "normalizer_ids": normalized_ids,
                            "source_roots": source_roots,
                        }
                    )
                if any(not source.startswith("block") for source in source_roots):
                    topology_violations.append(
                        {
                            "rule": "e62_inputs_are_distinct_frozen_blocks",
                            "tensor_product_node": node.id,
                            "source_roots": source_roots,
                        }
                    )
            post_norms = []
            for candidate in inserted:
                if qualified_op(candidate) != "core.norm_activation@1":
                    continue
                references = tuple(candidate.inputs.get("x", ()))
                root = references[0].split(":", 1)[0] if references else ""
                if root == node.id and bool(dict(candidate.attrs).get("normalize", True)):
                    post_norms.append(candidate.id)
            if not post_norms:
                topology_violations.append(
                    {
                        "rule": "e62_tensor_product_directly_post_normalized",
                        "tensor_product_node": node.id,
                    }
                )

    if str(factor_id) in {"E6.3", "E6.4"}:
        inserted_global_pools = [
            node.id for node in inserted if qualified_op(node) == "core.global_pool@1"
        ]
        if inserted_global_pools:
            topology_violations.append(
                {
                    "rule": "reuse_existing_graph_pool",
                    "inserted_global_pool_nodes": inserted_global_pools,
                }
            )
        weighted_nodes = [
            node for node in inserted if qualified_op(node) == "core.invariant_weight@1"
        ]
        for node in weighted_nodes:
            values = tuple(node.inputs.get("value", ()))
            if values != ("scalar_readout",):
                topology_violations.append(
                    {
                        "rule": "invariant_weight_value_is_scalar_readout",
                        "node_id": node.id,
                        "actual": list(values),
                    }
                )
        residual_nodes = [
            node for node in inserted if qualified_op(node) == "core.residual_add@1"
        ]
        valid_residual_ids = []
        weighted_ids = {node.id for node in weighted_nodes}
        for node in residual_nodes:
            roots = {
                reference.split(":", 1)[0]
                for references in node.inputs.values()
                for reference in references
            }
            if "scalar_readout" in roots and roots.intersection(weighted_ids):
                valid_residual_ids.append(node.id)
        graph_pool = node_map.get("graph_pool")
        pool_roots = set()
        if graph_pool is not None:
            pool_roots = {
                reference.split(":", 1)[0]
                for reference in graph_pool.inputs.get("x", ())
            }
        if not valid_residual_ids or not pool_roots.intersection(valid_residual_ids):
            topology_violations.append(
                {
                    "rule": "scalar_readout_plus_weighted_correction_then_existing_graph_pool",
                    "valid_residual_ids": valid_residual_ids,
                    "graph_pool_input_roots": sorted(pool_roots),
                }
            )
        prediction_sources = [
            output.source for output in child_program.outputs if output.name == "prediction"
        ]
        if prediction_sources != ["graph_pool"]:
            topology_violations.append(
                {
                    "rule": "prediction_remains_existing_graph_pool",
                    "actual": prediction_sources,
                }
            )

    if str(factor_id) == "E6.4":
        for node in inserted:
            if qualified_op(node) != "core.invariant_compatibility@1":
                continue
            for port in ("query", "key"):
                references = tuple(node.inputs.get(port, ()))
                root = references[0].split(":", 1)[0] if references else ""
                producer = node_map.get(root)
                attrs = {} if producer is None else dict(producer.attrs)
                valid = bool(
                    producer is not None
                    and qualified_op(producer) == "core.norm_activation@1"
                    and bool(attrs.get("normalize", True))
                )
                if not valid:
                    unnormalized_compatibility_inputs.append(
                        {
                            "node_id": node.id,
                            "port": port,
                            "reference": references[0] if references else "",
                            "producer_op": "" if producer is None else qualified_op(producer),
                            "producer_attrs": attrs,
                        }
                    )

    passed = bool(
        not outside
        and not forbidden_used
        and not missing_all
        and not missing_any
        and not insufficient_counts
        and not unnormalized_compatibility_inputs
        and not unnormalized_tensor_product_inputs
        and not topology_violations
    )
    if str(factor_id) in {"E6.3", "E6.4"} and unbounded_weights:
        passed = False
    return {
        "factor_id": str(factor_id),
        "purpose": str(contract["purpose"]),
        "inserted_node_ids": [node.id for node in inserted],
        "inserted_operator_signature": list(inserted_ops),
        "allowed_ops": list(contract["allowed"]),
        "outside_allowed_ops": outside,
        "forbidden_ops_used": forbidden_used,
        "missing_required_all": missing_all,
        "missing_required_any": missing_any,
        "insufficient_operator_counts": insufficient_counts,
        "unbounded_invariant_weights": unbounded_weights,
        "unnormalized_compatibility_inputs": unnormalized_compatibility_inputs,
        "unnormalized_tensor_product_inputs": unnormalized_tensor_product_inputs,
        "topology_violations": topology_violations,
        "passed": bool(passed),
    }


def deterministic_stability_repair(program, factor_id: str):
    """Apply only a narrowly authorized bounded-gate normalization.

    This is not an LLM-authored mutation.  It is a compiler-side repair of an
    already selected mechanism, analogous to fixing a declared type after a
    diagnostic.  The raw and repaired programs are both retained.
    """

    if str(factor_id) not in {"E6.3", "E6.4"}:
        return program, ()
    gate_ids = set()
    for node in program.nodes:
        if qualified_op(node) != "core.invariant_weight@1":
            continue
        for reference in node.inputs.get("weight", ()):
            gate_ids.add(reference.split(":", 1)[0])
    repairs = []
    nodes = []
    for node in program.nodes:
        if node.id not in gate_ids or qualified_op(node) != "core.scalar_activation@1":
            nodes.append(node)
            continue
        old_function = str(node.attrs.get("function", "silu"))
        if old_function in {"tanh", "sigmoid"}:
            nodes.append(node)
            continue
        attrs = dict(node.attrs)
        attrs["function"] = "tanh"
        nodes.append(replace(node, attrs=attrs))
        repairs.append(
            {
                "kind": "bounded_gate_normalization",
                "node_id": node.id,
                "old_function": old_function,
                "new_function": "tanh",
                "authorization": "factor_isolated_experience stability contract",
            }
        )
    if not repairs:
        return program, ()
    return replace(program, nodes=tuple(nodes)), tuple(repairs)


_PROMPT_EXPERIENCE_CONTEXT: Dict[str, Any] = {}


def install_experience_prompt_bridge():
    """Inject global policy context and factor-local memory without editing core files."""

    import equivariant_nas.dsl.search as search_module

    original = search_module.synthesizer_prompt
    original_critic = search_module.region_critic_prompt
    if getattr(original, "_experience_bridge", False):
        return

    def wrapped_critic(task, parent, evidence, region, router, vocabulary, primitives=None, motifs=None):
        prompt = original_critic(
            task,
            parent,
            evidence,
            region,
            router,
            vocabulary,
            primitives,
            motifs,
        )
        if not _PROMPT_EXPERIENCE_CONTEXT:
            return prompt
        payload = json.loads(prompt["user"])
        payload["global_factor_policy"] = dict(
            _PROMPT_EXPERIENCE_CONTEXT.get("global_factor_policy", {})
        )
        payload["selected_factor_memory"] = {
            "factor_id": str(_PROMPT_EXPERIENCE_CONTEXT.get("factor_id", "")),
            "factor_operation_contract": dict(
                _PROMPT_EXPERIENCE_CONTEXT.get("factor_operation_contract", {})
            ),
            "audited_experience": list(
                _PROMPT_EXPERIENCE_CONTEXT.get("audited_factor_experience", ())
            ),
        }
        return {
            "system": (
                prompt["system"]
                + " The global factor policy is context for allocation only. "
                + "Your edit plan must remain inside the selected factor contract."
            ),
            "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        }

    def wrapped(task, parent, plan, vocabulary, parent_architecture_id, primitives=None, motifs=None):
        enriched = dict(plan)
        if _PROMPT_EXPERIENCE_CONTEXT:
            enriched["factor_operation_contract"] = dict(
                _PROMPT_EXPERIENCE_CONTEXT.get("factor_operation_contract", {})
            )
            enriched["audited_factor_experience"] = list(
                _PROMPT_EXPERIENCE_CONTEXT.get("audited_factor_experience", ())
            )
            enriched["archive_exclusions"] = list(
                _PROMPT_EXPERIENCE_CONTEXT.get("archive_exclusions", ())
            )
            enriched["stability_contract"] = dict(
                _PROMPT_EXPERIENCE_CONTEXT.get("stability_contract", {})
            )
            enriched["global_factor_policy"] = dict(
                _PROMPT_EXPERIENCE_CONTEXT.get("global_factor_policy", {})
            )
        prompt = original(
            task,
            parent,
            enriched,
            vocabulary,
            parent_architecture_id,
            primitives,
            motifs,
        )
        if not _PROMPT_EXPERIENCE_CONTEXT:
            return prompt
        payload = json.loads(prompt["user"])
        payload["hard_factor_constraints"] = {
            "selected_factor_id": str(
                _PROMPT_EXPERIENCE_CONTEXT.get("factor_id", "")
            ),
            "instruction": (
                "These constraints are mandatory and are checked after compilation. "
                "A patch violating any item is rejected even if it type-checks."
            ),
            "factor_operation_contract": dict(
                _PROMPT_EXPERIENCE_CONTEXT.get("factor_operation_contract", {})
            ),
            "stability_contract": dict(
                _PROMPT_EXPERIENCE_CONTEXT.get("stability_contract", {})
            ),
            "audited_negative_and_positive_experience": list(
                _PROMPT_EXPERIENCE_CONTEXT.get("audited_factor_experience", ())
            ),
            "global_factor_policy": dict(
                _PROMPT_EXPERIENCE_CONTEXT.get("global_factor_policy", {})
            ),
            "mandatory_gate_rule": (
                "Every invariant_weight.weight must directly reference a scalar_activation "
                "whose function attr is exactly tanh or sigmoid; silu and relu are forbidden."
            ),
            "e64_normalization_rule": (
                "For E6.4, both invariant_compatibility inputs must directly reference two "
                "distinct norm_activation nodes with normalize=true; use a bounded scalar gate."
            ),
            "e64_stable_topology_envelope": [
                "norm_activation(block_i) and norm_activation(block_j)",
                "invariant_compatibility(norm_i,norm_j) with node 1x0 invariant output",
                "scalar_activation(function=tanh or sigmoid) preserving node 1x0 invariant",
                "invariant_weight(weight=bounded_gate,value=scalar_readout) producing node 1x0 global",
                "residual_add(left=scalar_readout,right=weighted_correction) producing node 1x0 global",
                "rewire existing graph_pool.x to the residual; insert no new global_pool",
            ],
            "e62_stable_topology_envelope": [
                "two distinct norm_activation(normalize=true) nodes consuming two different frozen block depths",
                "tensor_product.left and tensor_product.right directly reference those normalized nodes",
                "a third norm_activation directly consumes tensor_product output",
                "bounded scalar_activation(tanh) on the normalized l=0 correction, preferably without a post-TP irrep_linear",
                "global_pool the bounded correction and residual-fuse it with the existing graph-level terminal scalar path",
                "never reproduce multiseed-rejected architecture b4568ab650d282e8",
            ],
            "e63_stable_topology_envelope": [
                "derive a node-carried scalar gate from exactly one frozen depth",
                "scalar_activation(function=tanh or sigmoid)",
                "invariant_weight(weight=bounded_gate,value=scalar_readout)",
                "residual_add with scalar_readout then reuse the existing graph_pool",
            ],
        }
        return {
            "system": (
                prompt["system"]
                + " HARD CONSTRAINTS in user.hard_factor_constraints are compiler-audited. "
                + "Do not trade them off. In particular, never use SiLU or ReLU as an "
                + "invariant_weight gate; use tanh or sigmoid exactly."
            ),
            "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        }

    wrapped._experience_bridge = True
    wrapped_critic._experience_bridge = True
    search_module.region_critic_prompt = wrapped_critic
    search_module.synthesizer_prompt = wrapped


def load_experience_file(path: str) -> list:
    if not path:
        return []
    source = Path(path).resolve()
    if not source.is_file():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("experiences", ())
    if not isinstance(payload, list):
        raise ValueError("experience file must contain a list or an experiences object")
    return [dict(item) for item in payload]


def run_numerical_audit(
    args,
    project_root: Path,
    parent_path: Path,
    candidate_path: Path,
    output_path: Path,
    *,
    seed: int = None,
):
    if not args.audit_script:
        return None
    command = [
        sys.executable,
        str(Path(args.audit_script).resolve()),
        "--project-root", str(project_root),
        "--task-contract", str(Path(args.task_contract).resolve()),
        "--program", str(candidate_path),
        "--parent-program", str(parent_path),
        "--output", str(output_path),
        "--device", "cpu",
        "--dtype", str(args.audit_dtype),
        "--rotations", "3",
        "--translations", "2",
        "--permutations", "3",
        "--joint", "2",
        "--seed", str(args.audit_seed if seed is None else int(seed)),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=float(args.audit_timeout),
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            "numerical audit failed with code {}: {}".format(
                completed.returncode,
                completed.stdout[-4000:],
            )
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def maximum_inserted_gradient(audit: Mapping[str, Any]) -> float:
    values = [
        float(item["gradient_norm"])
        for item in audit.get("inserted_parameter_gradients", ())
        if item.get("gradient_norm") is not None
    ]
    return max(values, default=0.0)


def configured_audit_seeds(args) -> Tuple[int, ...]:
    values = [
        int(item.strip())
        for item in str(args.audit_seeds).split(",")
        if item.strip()
    ]
    if not values:
        values = [int(args.audit_seed)]
    return tuple(dict.fromkeys(values))


def experience_row(
    *,
    factor_id: str,
    architecture_id: str,
    operator_signature: Sequence[str],
    outcome: str,
    reason: str = "",
    numerical: Mapping[str, Any] = None,
    output_norm_ratio: Any = None,
) -> Dict[str, Any]:
    numerical = dict(numerical or {})
    activity = numerical.get("inserted_computation_path_activity") or {}
    return {
        "factor_id": str(factor_id),
        "architecture_id": str(architecture_id),
        "operator_signature": list(operator_signature),
        "outcome": str(outcome),
        "reason": str(reason)[:1000],
        "equivariance_passed": numerical.get("equivariance_passed"),
        "path_activity": activity.get("status", "unknown"),
        "output_norm_ratio": output_norm_ratio,
        "maximum_inserted_gradient": (
            maximum_inserted_gradient(numerical) if numerical else None
        ),
    }


def balanced_factor_targets(sequence: Sequence[str], total: int) -> Dict[str, int]:
    ordered = tuple(dict.fromkeys(str(item) for item in sequence if str(item)))
    if not ordered:
        return {}
    base, remainder = divmod(int(total), len(ordered))
    return {
        factor_id: base + (1 if index < remainder else 0)
        for index, factor_id in enumerate(ordered)
    }


def accepted_factor_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in records:
        factor_id = str(item.get("factor_id", ""))
        counts[factor_id] = counts.get(factor_id, 0) + 1
    return counts


def next_balanced_factor(
    sequence: Sequence[str],
    targets: Mapping[str, int],
    records: Sequence[Mapping[str, Any]],
    attempt: int,
) -> str:
    if not sequence:
        return ""
    counts = accepted_factor_counts(records)
    start = (int(attempt) - 1) % len(sequence)
    for offset in range(len(sequence)):
        factor_id = str(sequence[(start + offset) % len(sequence)])
        if counts.get(factor_id, 0) < int(targets.get(factor_id, 0)):
            return factor_id
    return ""


def minimum_factor_targets(
    sequence: Sequence[str], total: int, minimum_per_factor: int
) -> Dict[str, int]:
    ordered = tuple(dict.fromkeys(str(item) for item in sequence if str(item)))
    if not ordered:
        return {}
    minimum = max(0, int(minimum_per_factor))
    required = minimum * len(ordered)
    if int(total) < required:
        raise ValueError(
            "count={} cannot satisfy minimum_per_factor={} across {} factors".format(
                int(total), minimum, len(ordered)
            )
        )
    return {factor_id: minimum for factor_id in ordered}


def factor_policy_statistics(
    sequence: Sequence[str],
    experience: Sequence[Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    ordered = tuple(dict.fromkeys(str(item) for item in sequence if str(item)))
    accepted_counts = accepted_factor_counts(accepted)
    statistics: Dict[str, Dict[str, Any]] = {}
    for factor_id in ordered:
        rows = [item for item in experience if str(item.get("factor_id", "")) == factor_id]
        positive_rows = [
            item for item in rows if str(item.get("outcome", "")).startswith("accepted")
        ]
        robust_positive_rows = [
            item for item in positive_rows if "multiseed" in str(item.get("outcome", ""))
        ]
        negative_rows = [item for item in rows if item not in positive_rows]
        weighted_successes = sum(
            1.0 if item in robust_positive_rows else 0.60 for item in positive_rows
        )
        posterior_success = (weighted_successes + 1.0) / (len(rows) + 2.0)
        signatures = {
            tuple(str(op) for op in item.get("operator_signature", ()))
            for item in positive_rows
            if item.get("operator_signature")
        }
        signature_diversity = min(
            1.0,
            len(signatures) / float(max(1, len(positive_rows))),
        )
        uncertainty = 1.0 / math.sqrt(len(rows) + 1.0)
        coverage_curiosity = 1.0 / math.sqrt(accepted_counts.get(factor_id, 0) + 1.0)
        statistics[factor_id] = {
            "experience_count": len(rows),
            "accepted_experience_count": len(positive_rows),
            "robust_multiseed_accept_count": len(robust_positive_rows),
            "rejected_experience_count": len(negative_rows),
            "accepted_in_current_run": accepted_counts.get(factor_id, 0),
            "unique_positive_operator_signature_count": len(signatures),
            "posterior_success": posterior_success,
            "signature_diversity": signature_diversity,
            "uncertainty": uncertainty,
            "exploration_score": uncertainty + coverage_curiosity,
            "exploitation_score": 0.75 * posterior_success + 0.25 * signature_diversity,
        }
    return statistics


def weighted_factor_choice(
    factors: Sequence[str],
    weights: Sequence[float],
    rng: random.Random,
) -> str:
    positive = [max(0.0, float(weight)) for weight in weights]
    total = sum(positive)
    if total <= 0.0:
        return str(factors[int(rng.random() * len(factors)) % len(factors)])
    threshold = rng.random() * total
    cumulative = 0.0
    for factor_id, weight in zip(factors, positive):
        cumulative += weight
        if cumulative >= threshold:
            return str(factor_id)
    return str(factors[-1])


def select_annealed_epsilon_factor(
    *,
    sequence: Sequence[str],
    targets: Mapping[str, int],
    accepted: Sequence[Mapping[str, Any]],
    experience: Sequence[Mapping[str, Any]],
    attempt: int,
    seed: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_attempts: float,
    stagnation_patience: int,
    stagnation_boost: float,
    total_candidate_count: int,
    maximum_factor_share: float,
    repeat_factor_penalty: float,
) -> Tuple[str, Dict[str, Any]]:
    ordered = tuple(dict.fromkeys(str(item) for item in sequence if str(item)))
    if not ordered:
        return "", {}
    statistics = factor_policy_statistics(ordered, experience, accepted)
    counts = accepted_factor_counts(accepted)
    maximum_count = max(
        max([int(value) for value in targets.values()] or [0]),
        int(math.ceil(max(0.0, float(maximum_factor_share)) * int(total_candidate_count))),
    )
    share_eligible = [
        factor_id for factor_id in ordered if counts.get(factor_id, 0) < maximum_count
    ]
    if not share_eligible:
        share_eligible = list(ordered)
    undercovered = [
        factor_id
        for factor_id in ordered
        if counts.get(factor_id, 0) < int(targets.get(factor_id, 0))
    ]
    decay = max(float(epsilon_decay_attempts), 1.0)
    epsilon = float(epsilon_end) + (
        float(epsilon_start) - float(epsilon_end)
    ) * math.exp(-max(0, int(attempt) - 1) / decay)
    patience = max(0, int(stagnation_patience))
    recent = list(experience[-patience:]) if patience else []
    stagnating = bool(
        patience
        and len(recent) == patience
        and not any(
            str(item.get("outcome", "")).startswith("accepted") for item in recent
        )
    )
    if stagnating:
        epsilon = min(1.0, epsilon + float(stagnation_boost))
    epsilon = min(1.0, max(0.0, epsilon))
    rng = random.Random(int(seed) * 1000003 + int(attempt) * 1009)

    if undercovered:
        pool = undercovered
        mode = "minimum_coverage"
        selected = weighted_factor_choice(
            pool,
            [statistics[item]["exploration_score"] for item in pool],
            rng,
        )
    elif rng.random() < epsilon:
        pool = list(share_eligible)
        mode = "curiosity_explore"
        last_factor = str(experience[-1].get("factor_id", "")) if experience else ""
        selected = weighted_factor_choice(
            pool,
            [
                statistics[item]["exploration_score"]
                * (float(repeat_factor_penalty) if item == last_factor else 1.0)
                for item in pool
            ],
            rng,
        )
    else:
        pool = list(share_eligible)
        mode = "reward_exploit"
        last_factor = str(experience[-1].get("factor_id", "")) if experience else ""
        selected = max(
            pool,
            key=lambda item: (
                statistics[item]["exploitation_score"]
                * (float(repeat_factor_penalty) if item == last_factor else 1.0),
                -ordered.index(item),
            ),
        )

    snapshot = {
        "policy": "annealed_epsilon_greedy",
        "attempt": int(attempt),
        "epsilon": epsilon,
        "epsilon_start": float(epsilon_start),
        "epsilon_end": float(epsilon_end),
        "epsilon_decay_attempts": float(epsilon_decay_attempts),
        "stagnation_detected": stagnating,
        "selection_mode": mode,
        "selected_factor_id": selected,
        "minimum_targets": {str(key): int(value) for key, value in targets.items()},
        "maximum_factor_share": float(maximum_factor_share),
        "maximum_factor_count": int(maximum_count),
        "share_eligible_factors": list(share_eligible),
        "repeat_factor_penalty": float(repeat_factor_penalty),
        "factor_statistics": statistics,
        "curiosity_semantics": (
            "explore uncertain or under-covered factors while keeping each candidate single-factor"
        ),
        "mutation_magnitude_semantics": (
            "epsilon controls direction choice, not the number of simultaneously edited factors"
        ),
    }
    return selected, snapshot


async def run(args):
    project_root = Path(args.project_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from equivariant_nas.dsl import (
        Compiler,
        DSLGenerationEngine,
        EvidenceItem,
        EvidenceStore,
        LanguageVersion,
        core_registry,
        reference_motif_registry,
        select_active_vocabulary,
    )
    from equivariant_nas.dsl.serialization import dumps_program, load_task_contract

    task = load_task_contract(str(Path(args.task_contract).resolve()))
    official_parent, parent = build_audit_parent(task.task_id)
    primitives = core_registry()
    motifs = reference_motif_registry()
    compiler = Compiler(primitives, motifs)
    parent_artifact = compiler.analyze(parent, task)
    parent_lowering = compiler.plan_lowering(parent, task)
    if parent_lowering.mode != "experimental_node_graph":
        raise RuntimeError("audit parent must use experimental_node_graph lowering")
    if parent_lowering.unsupported_nodes:
        raise RuntimeError(
            "audit parent has unsupported lowering nodes: {}".format(parent_lowering.unsupported_nodes)
        )

    language = LanguageVersion.from_registries(
        parent.language_version,
        "",
        primitives,
        motifs,
        now(),
        {"purpose": "audit-only GLM local mutation novelty probe"},
    )
    vocabulary = select_active_vocabulary(
        language,
        task.group,
        primitives,
        motifs,
        allowed_names=visible_names(),
    )
    experience = load_experience_file(args.experience_file)
    resumable_memory = output / "experience_memory.json"
    if resumable_memory.is_file():
        resumed = load_experience_file(str(resumable_memory))
        known = {
            (
                str(item.get("factor_id", "")),
                str(item.get("architecture_id", "")),
                str(item.get("outcome", "")),
                str(item.get("reason", "")),
            )
            for item in experience
        }
        for item in resumed:
            key = (
                str(item.get("factor_id", "")),
                str(item.get("architecture_id", "")),
                str(item.get("outcome", "")),
                str(item.get("reason", "")),
            )
            if key not in known:
                experience.append(item)
                known.add(key)
    install_experience_prompt_bridge()
    regions = build_regions(args.protocol, experience)
    history = load_history(project_root, output, compiler, task)
    store = EvidenceStore(str(output / "evidence.sqlite"))
    store.register_language(language, motifs)
    if args.transport == "local":
        ensemble = LocalGLMEnsemble(
            api_key_env=args.api_key_env,
            api_base=args.api_base,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout=args.request_timeout,
            retries=args.request_retries,
            seed=args.seed,
            log_path=output / "glm_transport.jsonl",
        )
    elif args.transport == "ssh":
        if not args.remote_helper:
            raise ValueError("--remote-helper is required when --transport=ssh")
        ensemble = SSHGLMEnsemble(
            host=args.ssh_host,
            remote_python=args.remote_python,
            remote_helper=args.remote_helper,
            remote_openevolve_root=args.remote_openevolve_root,
            remote_config=args.remote_config,
            seed=args.seed,
            log_path=output / "glm_transport.jsonl",
        )
    else:
        ensemble = BridgeGLMEnsemble(
            seed=args.seed,
            log_path=output / "glm_transport.jsonl",
        )
    engine = DSLGenerationEngine(
        compiler,
        task,
        vocabulary,
        store,
        model_name=args.model,
        repair_attempts=args.repair_attempts,
    )

    (output / "official_parent.dsl.json").write_text(
        dumps_program(official_parent), encoding="utf-8"
    )
    (output / "audit_parent.dsl.json").write_text(
        dumps_program(parent), encoding="utf-8"
    )
    write_json(
        output / "generation_contract.json",
        {
            "created_at": now(),
            "model": args.model,
            "transport": args.transport,
            "api_base": args.api_base if args.transport == "local" else "",
            "api_key_env": args.api_key_env if args.transport == "local" else "",
            "credential_serialized": False,
            "sampling": {
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "max_tokens": int(args.max_tokens),
                "timeout_seconds": float(args.request_timeout),
                "retries": int(args.request_retries),
            },
            "workflow": ["factor_router", "region_critic", "patch_synthesizer", "conditional_compiler_repair"],
            "mutation_protocol": args.protocol,
            "factor_sequence": [
                item.strip() for item in args.factor_sequence.split(",") if item.strip()
            ],
            "factor_policy": {
                "name": str(args.factor_policy),
                "minimum_per_factor": int(args.minimum_per_factor),
                "epsilon_start": float(args.epsilon_start),
                "epsilon_end": float(args.epsilon_end),
                "epsilon_decay_attempts": float(args.epsilon_decay_attempts),
                "stagnation_patience": int(args.stagnation_patience),
                "stagnation_boost": float(args.stagnation_boost),
                "maximum_factor_share": float(args.maximum_factor_share),
                "repeat_factor_penalty": float(args.repeat_factor_penalty),
                "rule": (
                    "epsilon controls factor/topology exploration; every candidate remains single-factor"
                ),
            },
            "experience_file": str(Path(args.experience_file).resolve()) if args.experience_file else "",
            "experience_count_at_start": len(experience),
            "factor_operation_contracts": factor_operation_contracts(),
            "numerical_admission": {
                "audit_script": str(Path(args.audit_script).resolve()) if args.audit_script else "",
                "audit_seed": int(args.audit_seed),
                "audit_seeds": list(configured_audit_seeds(args)),
                "audit_dtype": str(args.audit_dtype),
                "minimum_output_norm_ratio": float(args.minimum_output_norm_ratio),
                "maximum_output_norm_ratio": float(args.maximum_output_norm_ratio),
                "maximum_inserted_gradient": float(args.maximum_inserted_gradient),
                "enforced": args.protocol == "factor_isolated_experience",
            },
            "requested_candidate_count": int(args.count),
            "maximum_attempts": int(args.maximum_attempts),
            "task_contract_hash": task.content_hash(),
            "official_parent_architecture_id": compiler.analyze(official_parent, task).architecture_id,
            "audit_parent_architecture_id": parent_artifact.architecture_id,
            "parent_lowering": parent_lowering.to_dict(),
            "regions": [item.to_dict() for item in regions],
            "visible_vocabulary": vocabulary.to_dict(),
            "training_started": False,
            "test_split_loaded": False,
            "formal_ranking_admitted": False,
        },
    )

    audit_seeds = configured_audit_seeds(args)
    parent_numerical = None
    parent_reference_norm = None
    parent_numerical_by_seed: Dict[int, Mapping[str, Any]] = {}
    parent_reference_norms: Dict[int, float] = {}
    if args.audit_script:
        for seed in audit_seeds:
            parent_audit_path = output / "numerical_audit" / "parent_seed{}.json".format(seed)
            numerical = (
                json.loads(parent_audit_path.read_text(encoding="utf-8"))
                if parent_audit_path.is_file()
                else run_numerical_audit(
                    args,
                    project_root,
                    output / "audit_parent.dsl.json",
                    output / "audit_parent.dsl.json",
                    parent_audit_path,
                    seed=seed,
                )
            )
            parent_numerical_by_seed[int(seed)] = numerical
            parent_reference_norms[int(seed)] = float(
                (numerical.get("reference_output") or {}).get("norm", 0.0)
            )
        parent_numerical = parent_numerical_by_seed[int(audit_seeds[0])]
        parent_reference_norm = parent_reference_norms[int(audit_seeds[0])]

    accepted = load_jsonl(output / "generation.jsonl")
    failures = load_jsonl(output / "failures.jsonl")
    previous_attempt = max(
        [int(item.get("attempt", 0)) for item in accepted + failures],
        default=0,
    )
    factor_sequence = tuple(
        item.strip() for item in args.factor_sequence.split(",") if item.strip()
    )
    factor_targets = (
        balanced_factor_targets(factor_sequence, int(args.count))
        if args.factor_policy == "balanced"
        else minimum_factor_targets(
            factor_sequence,
            int(args.count),
            int(args.minimum_per_factor),
        )
    )
    for attempt in range(previous_attempt + 1, previous_attempt + int(args.maximum_attempts) + 1):
        counts_now = accepted_factor_counts(accepted)
        coverage_reached = all(
            counts_now.get(factor_id, 0) >= target
            for factor_id, target in factor_targets.items()
        )
        if len(accepted) >= int(args.count) and coverage_reached:
            break
        if args.factor_policy == "balanced":
            forced_factor_id = next_balanced_factor(
                factor_sequence,
                factor_targets,
                accepted,
                attempt,
            )
            factor_policy_snapshot = {
                "policy": "balanced",
                "attempt": int(attempt),
                "selected_factor_id": forced_factor_id,
                "minimum_targets": dict(factor_targets),
                "accepted_factor_counts": counts_now,
                "mutation_magnitude_semantics": "one factor per candidate",
            }
        else:
            forced_factor_id, factor_policy_snapshot = select_annealed_epsilon_factor(
                sequence=factor_sequence,
                targets=factor_targets,
                accepted=accepted,
                experience=experience,
                attempt=attempt,
                seed=int(args.seed),
                epsilon_start=float(args.epsilon_start),
                epsilon_end=float(args.epsilon_end),
                epsilon_decay_attempts=float(args.epsilon_decay_attempts),
                stagnation_patience=int(args.stagnation_patience),
                stagnation_boost=float(args.stagnation_boost),
                total_candidate_count=int(args.count),
                maximum_factor_share=float(args.maximum_factor_share),
                repeat_factor_penalty=float(args.repeat_factor_penalty),
            )
        regions = build_regions(args.protocol, experience)
        selected_lessons = compact_factor_experience(experience, forced_factor_id) if forced_factor_id else ()
        _PROMPT_EXPERIENCE_CONTEXT.clear()
        if args.protocol == "factor_isolated_experience":
            _PROMPT_EXPERIENCE_CONTEXT.update(
                {
                    "factor_id": forced_factor_id,
                    "factor_operation_contract": (
                        factor_operation_contracts().get(forced_factor_id, {})
                        if forced_factor_id
                        else factor_operation_contracts()
                    ),
                    "audited_factor_experience": list(selected_lessons),
                    "global_factor_policy": dict(factor_policy_snapshot),
                    "archive_exclusions": [
                        {
                            "architecture_id": item["architecture_id"],
                            "factor_id": item["factor_id"],
                            "operator_signature": item["audit"].get("factor_contract", {}).get(
                                "inserted_operator_signature", []
                            ),
                        }
                        for item in accepted[-12:]
                    ],
                    "stability_contract": {
                        "bounded_gate_functions": ["tanh", "sigmoid"],
                        "robust_audit_seeds": list(audit_seeds),
                        "minimum_random_weight_output_norm_ratio": float(
                            args.minimum_output_norm_ratio
                        ),
                        "maximum_random_weight_output_norm_ratio": float(args.maximum_output_norm_ratio),
                        "maximum_inserted_gradient_norm": float(args.maximum_inserted_gradient),
                        "no_training_evidence_available": True,
                    },
                }
            )
        archive_payload = {
            "instruction": "Prefer a structural mechanism not already present in this audit archive.",
            "forced_factor_id": forced_factor_id,
            "factor_targets": factor_targets,
            "accepted_factor_counts": counts_now,
            "audited_factor_experience": list(selected_lessons),
            "global_factor_policy": dict(factor_policy_snapshot),
            "prior_candidates": [
                {
                    "architecture_id": item["architecture_id"],
                    "factor_id": item["factor_id"],
                    "inserted_source_nodes": item["audit"]["inserted_source_nodes"],
                    "expanded_operator_insertions": item["audit"]["expanded_operator_delta"]["inserted"],
                }
                for item in accepted
            ],
            "performance_evidence_available": False,
        }
        evidence = (
            EvidenceItem(
                "structural_archive_attempt_{:02d}".format(attempt),
                "validation",
                "structural_novelty_archive",
                archive_payload,
                status="hypothesis",
            ),
        )
        try:
            generated = await engine.generate_region_candidate(
                ensemble,
                parent,
                evidence,
                regions,
                forced_factor_id=forced_factor_id,
            )
            factor_id = str(generated.region_audit.get("factor_id", ""))
            raw_child_program = generated.child.source_program
            raw_architecture_id = generated.child.architecture_id
            child_program, deterministic_repairs = (
                deterministic_stability_repair(raw_child_program, factor_id)
                if args.protocol == "factor_isolated_experience"
                else (raw_child_program, ())
            )
            child_artifact = compiler.analyze(child_program, task)
            if deterministic_repairs:
                from equivariant_nas.dsl.regions import validate_region_transition

                selected_region = next(
                    item for item in regions if item.factor_id == factor_id
                )
                region_audit = validate_region_transition(
                    parent_artifact.source_program,
                    child_program,
                    selected_region,
                )
                store.add_compiled_candidate(child_artifact, task)
            else:
                region_audit = dict(generated.region_audit)
            lowering = compiler.plan_lowering(child_program, task)
            audit = candidate_audit(
                parent_artifact,
                child_artifact,
                child_program,
                lowering,
                history,
                primitives,
            )
            factor_audit = factor_contract_audit(
                parent_artifact.source_program,
                child_program,
                factor_id,
            )
            audit["factor_contract"] = factor_audit
            audit["deterministic_stability_repairs"] = list(deterministic_repairs)
            audit["raw_glm_architecture_id"] = raw_architecture_id
            index = len(accepted) + 1
            stem = "attempt_{:02d}_{}".format(attempt, child_artifact.architecture_id)
            candidate_path = output / "all_candidates" / (stem + ".dsl.json")
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(
                dumps_program(child_program), encoding="utf-8"
            )
            if deterministic_repairs:
                raw_path = output / "raw_glm_candidates" / (
                    "attempt_{:02d}_{}.dsl.json".format(attempt, raw_architecture_id)
                )
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(dumps_program(raw_child_program), encoding="utf-8")
            numerical_by_seed: Dict[int, Mapping[str, Any]] = {}
            seed_admission = []
            for audit_seed in audit_seeds:
                numerical_item = run_numerical_audit(
                    args,
                    project_root,
                    output / "audit_parent.dsl.json",
                    candidate_path,
                    output / "numerical_audit" / (stem + "_seed{}.json".format(audit_seed)),
                    seed=audit_seed,
                )
                numerical_by_seed[int(audit_seed)] = numerical_item
                output_norm_ratio_item = None
                if numerical_item is not None and int(audit_seed) in parent_reference_norms:
                    candidate_norm = float(
                        (numerical_item.get("reference_output") or {}).get("norm", 0.0)
                    )
                    output_norm_ratio_item = candidate_norm / max(
                        parent_reference_norms[int(audit_seed)], 1.0e-30
                    )
                gradient_item = maximum_inserted_gradient(numerical_item or {})
                path_status = (
                    "unknown"
                    if numerical_item is None
                    else (numerical_item.get("inserted_computation_path_activity") or {}).get(
                        "status", "unknown"
                    )
                )
                passed_item = bool(
                    numerical_item is not None
                    and numerical_item.get("equivariance_passed") is True
                    and path_status == "active"
                    and output_norm_ratio_item is not None
                    and output_norm_ratio_item >= float(args.minimum_output_norm_ratio)
                    and output_norm_ratio_item <= float(args.maximum_output_norm_ratio)
                    and gradient_item <= float(args.maximum_inserted_gradient)
                )
                seed_admission.append(
                    {
                        "seed": int(audit_seed),
                        "equivariance_passed": None
                        if numerical_item is None
                        else numerical_item.get("equivariance_passed"),
                        "path_activity": path_status,
                        "output_norm_ratio_to_parent": output_norm_ratio_item,
                        "maximum_inserted_gradient": gradient_item,
                        "passed": passed_item,
                    }
                )
            numerical = numerical_by_seed.get(int(audit_seeds[0]))
            output_norm_ratios = [
                item["output_norm_ratio_to_parent"]
                for item in seed_admission
                if item["output_norm_ratio_to_parent"] is not None
            ]
            output_norm_ratio = output_norm_ratios[0] if output_norm_ratios else None
            gradient_maximum = max(
                [float(item["maximum_inserted_gradient"]) for item in seed_admission]
                or [0.0]
            )
            numerical_passed = bool(
                not args.audit_script or (seed_admission and all(item["passed"] for item in seed_admission))
            )
            admission_passed = bool(
                factor_audit["passed"]
                and audit["structurally_unique_within_local_archive"]
                and numerical_passed
            )
            audit["numerical"] = {
                "available": bool(numerical_by_seed),
                "audit_seeds": list(audit_seeds),
                "equivariance_passed": bool(
                    seed_admission
                    and all(item["equivariance_passed"] is True for item in seed_admission)
                ),
                "path_activity": (
                    "active"
                    if seed_admission and all(item["path_activity"] == "active" for item in seed_admission)
                    else "unknown"
                ),
                "output_norm_ratio_to_parent": output_norm_ratio,
                "minimum_output_norm_ratio_to_parent": min(output_norm_ratios)
                if output_norm_ratios
                else None,
                "maximum_output_norm_ratio_to_parent": max(output_norm_ratios)
                if output_norm_ratios
                else None,
                "maximum_inserted_gradient": gradient_maximum,
                "per_seed": seed_admission,
                "passed": numerical_passed,
            }
            audit["admission_passed"] = admission_passed
            memory = experience_row(
                factor_id=factor_id,
                architecture_id=child_artifact.architecture_id,
                operator_signature=factor_audit["inserted_operator_signature"],
                outcome="accepted" if admission_passed else "rejected",
                reason=(
                    "all structural, factor-isolation, SO(3), path-activity, and stability gates passed"
                    if admission_passed
                    else json.dumps(
                        {
                            "factor_contract_passed": factor_audit["passed"],
                            "structurally_unique": audit["structurally_unique_within_local_archive"],
                            "numerical_passed": numerical_passed,
                        },
                        sort_keys=True,
                    )
                ),
                numerical=numerical,
                output_norm_ratio=output_norm_ratio,
            )
            memory["equivariance_passed"] = audit["numerical"]["equivariance_passed"]
            memory["path_activity"] = audit["numerical"]["path_activity"]
            memory["output_norm_ratio"] = audit["numerical"].get(
                "maximum_output_norm_ratio_to_parent"
            )
            memory["minimum_output_norm_ratio"] = audit["numerical"].get(
                "minimum_output_norm_ratio_to_parent"
            )
            memory["maximum_inserted_gradient"] = gradient_maximum
            memory["audit_seeds"] = list(audit_seeds)
            experience.append(memory)
            if args.protocol == "factor_isolated_experience" and not admission_passed:
                rejection = {
                    "attempt": attempt,
                    "architecture_id": child_artifact.architecture_id,
                    "raw_glm_architecture_id": raw_architecture_id,
                    "factor_id": factor_id,
                    "candidate_path": str(candidate_path),
                    "factor_policy_snapshot": dict(factor_policy_snapshot),
                    "error_type": "PostCompilationAdmissionError",
                    "error": memory["reason"],
                    "factor_contract": factor_audit,
                    "numerical": audit["numerical"],
                    "training_started": False,
                    "test_split_loaded": False,
                }
                append_jsonl(output / "failures.jsonl", rejection)
                failures.append(rejection)
                write_json(
                    output / "rejected_evidence" / (stem + ".json"),
                    rejection,
                )
                write_json(
                    output / "experience_memory.json",
                    {
                        "updated_at": now(),
                        "protocol": args.protocol,
                        "experiences": experience,
                    },
                )
                continue
            accepted_path = output / "candidates" / (
                "candidate_{:02d}_{}.dsl.json".format(index, child_artifact.architecture_id)
            )
            accepted_path.parent.mkdir(parents=True, exist_ok=True)
            accepted_path.write_text(
                dumps_program(child_program), encoding="utf-8"
            )
            record = {
                "candidate_index": index,
                "attempt": attempt,
                "architecture_id": child_artifact.architecture_id,
                "raw_glm_architecture_id": raw_architecture_id,
                "factor_id": factor_id,
                "region_id": str(generated.region_audit.get("region_id", "")),
                "candidate_path": str(accepted_path),
                "router_response": dict(generated.router_response),
                "critic_response": dict(generated.critic_response),
                "planner_response": dict(generated.planner_response),
                "patch": generated.patch.to_dict(),
                "region_audit": dict(region_audit),
                "deterministic_stability_repairs": list(deterministic_repairs),
                "repair_count": int(generated.repair_count),
                "critic_repair_count": int(generated.critic_repair_count),
                "factor_policy_snapshot": dict(factor_policy_snapshot),
                "audit": audit,
                "numerical_audit_path": (
                    str(output / "numerical_audit" / (stem + "_seed{}.json".format(audit_seeds[0])))
                    if numerical is not None else ""
                ),
                "numerical_audit_paths": {
                    str(seed): str(
                        output / "numerical_audit" / (stem + "_seed{}.json".format(seed))
                    )
                    for seed in audit_seeds
                },
                "training_started": False,
                "test_split_loaded": False,
            }
            write_json(
                output / "candidate_evidence" / (
                    "candidate_{:02d}_{}.json".format(index, child_artifact.architecture_id)
                ),
                record,
            )
            append_jsonl(output / "generation.jsonl", record)
            accepted.append(record)
        except Exception as exc:
            failure = {
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error": str(exc)[:12000],
                "diagnostics": [
                    item.to_dict() for item in getattr(exc, "diagnostics", ())
                ],
                "traceback": traceback.format_exc()[-12000:],
                "forced_factor_id": forced_factor_id,
                "factor_policy_snapshot": dict(factor_policy_snapshot),
                "training_started": False,
                "test_split_loaded": False,
            }
            append_jsonl(output / "failures.jsonl", failure)
            failures.append(failure)
            experience.append(
                experience_row(
                    factor_id=forced_factor_id,
                    architecture_id="",
                    operator_signature=(),
                    outcome="rejected",
                    reason="{}: {}".format(type(exc).__name__, str(exc)[:2000]),
                )
            )

        write_json(
            output / "experience_memory.json",
            {
                "updated_at": now(),
                "protocol": args.protocol,
                "experiences": experience,
            },
        )

    factor_counts: Dict[str, int] = {}
    for item in accepted:
        factor_counts[item["factor_id"]] = factor_counts.get(item["factor_id"], 0) + 1
    final_coverage = all(
        factor_counts.get(factor_id, 0) >= target
        for factor_id, target in factor_targets.items()
    )
    summary = {
        "status": (
            "complete"
            if len(accepted) >= int(args.count) and final_coverage
            else "attempts_exhausted"
        ),
        "completed_at": now(),
        "mutation_protocol": args.protocol,
        "factor_policy": str(args.factor_policy),
        "factor_policy_parameters": {
            "minimum_per_factor": int(args.minimum_per_factor),
            "epsilon_start": float(args.epsilon_start),
            "epsilon_end": float(args.epsilon_end),
            "epsilon_decay_attempts": float(args.epsilon_decay_attempts),
            "stagnation_patience": int(args.stagnation_patience),
            "stagnation_boost": float(args.stagnation_boost),
            "maximum_factor_share": float(args.maximum_factor_share),
            "repeat_factor_penalty": float(args.repeat_factor_penalty),
        },
        "accepted_candidate_count": len(accepted),
        "requested_candidate_count": int(args.count),
        "attempt_count": len(accepted) + len(failures),
        "failure_count": len(failures),
        "factor_counts": factor_counts,
        "factor_targets": factor_targets,
        "factor_coverage_reached": final_coverage,
        "experience_count_at_end": len(experience),
        "admission_passed_count": sum(
            1 for item in accepted if item["audit"].get("admission_passed")
        ),
        "equivariance_passed_count": sum(
            1 for item in accepted
            if item["audit"].get("numerical", {}).get("equivariance_passed") is True
        ),
        "active_path_count": sum(
            1 for item in accepted
            if item["audit"].get("numerical", {}).get("path_activity") == "active"
        ),
        "structurally_unique_count": sum(
            1 for item in accepted
            if item["audit"]["structurally_unique_within_local_archive"]
        ),
        "candidates": [
            {
                "architecture_id": item["architecture_id"],
                "factor_id": item["factor_id"],
                "region_id": item["region_id"],
                "candidate_path": item["candidate_path"],
                "repair_count": item["repair_count"],
                "factor_selection_mode": item.get("factor_policy_snapshot", {}).get(
                    "selection_mode",
                    item.get("factor_policy_snapshot", {}).get("policy", ""),
                ),
                "factor_selection_epsilon": item.get("factor_policy_snapshot", {}).get(
                    "epsilon"
                ),
                "structurally_unique_within_local_archive": item["audit"]["structurally_unique_within_local_archive"],
                "exact_historical_match_count": item["audit"]["exact_historical_match_count"],
                "inserted_source_nodes": item["audit"]["inserted_source_nodes"],
                "changed_source_nodes": item["audit"]["changed_source_nodes"],
                "lowering_mode": item["audit"]["lowering"]["mode"],
                "factor_contract_passed": item["audit"].get("factor_contract", {}).get("passed"),
                "admission_passed": item["audit"].get("admission_passed"),
                "equivariance_passed": item["audit"].get("numerical", {}).get("equivariance_passed"),
                "path_activity": item["audit"].get("numerical", {}).get("path_activity"),
                "output_norm_ratio_to_parent": item["audit"].get("numerical", {}).get("output_norm_ratio_to_parent"),
                "minimum_output_norm_ratio_to_parent": item["audit"].get("numerical", {}).get("minimum_output_norm_ratio_to_parent"),
                "maximum_output_norm_ratio_to_parent": item["audit"].get("numerical", {}).get("maximum_output_norm_ratio_to_parent"),
                "maximum_inserted_gradient": item["audit"].get("numerical", {}).get("maximum_inserted_gradient"),
            }
            for item in accepted
        ],
        "training_started": False,
        "test_split_loaded": False,
        "formal_ranking_admitted": False,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
