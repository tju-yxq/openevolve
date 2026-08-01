#!/usr/bin/env python
"""Import an official Equiformer V3 model config and export its Typed DSL seed."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from equivariant_nas.dsl import TypeChecker, architecture_id, core_registry, equiformer_v3_direct_model_program
from equivariant_nas.dsl.backends import EquiformerV3Spec, V3_REFERENCE_COMMIT
from equivariant_nas.dsl.serialization import save_program


def _load_payload(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("V3 model config must decode to one mapping")
    return value


def export_seed(model_config: Path, output: Path):
    payload = _load_payload(model_config)
    spec, import_manifest = EquiformerV3Spec.from_official_config(payload)
    program = equiformer_v3_direct_model_program(spec)
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    output.mkdir(parents=True, exist_ok=True)
    program_path = output / "equiformer_v3_seed.dsl.json"
    save_program(program, str(program_path))
    identity = {
        "status": "v3_typed_dsl_seed_exported",
        "architecture_id": architecture_id(program, registry),
        "official_spec_id": spec.architecture_id(),
        "official_source_commit": V3_REFERENCE_COMMIT,
        "node_count": len(program.nodes),
        "node_order_count": len(inference.node_order),
        "outputs": [item.name for item in program.outputs],
        "constructor_bypass": bool(program.annotations.get("constructor_bypass", True)),
        "stress_head_complete": bool(program.annotations.get("stress_head_complete", False)),
        "model_config": str(model_config.resolve()),
        "program": str(program_path.resolve()),
        "import_manifest": import_manifest.to_dict(),
    }
    (output / "model_identity.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return identity


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    args = get_parser().parse_args()
    result = export_seed(Path(args.model_config), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
