"""Strict checkpoint translation for backend-neutral Typed DSL models."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


CHECKPOINT_MAPPING_VERSION = "evoequilang-checkpoint-mapping-v1"


class CheckpointMappingError(RuntimeError):
    """Raised when a checkpoint cannot satisfy a declared mapping contract."""


def _tensor_dtype_name(tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


@dataclass(frozen=True)
class CheckpointTensorGroup:
    """One or more equivalent source keys copied to one or more target keys."""

    source_keys: Tuple[str, ...]
    target_keys: Tuple[str, ...]
    semantic: str

    def __post_init__(self) -> None:
        if not self.source_keys or not self.target_keys:
            raise ValueError("checkpoint tensor groups require source and target keys")
        if len(set(self.source_keys)) != len(self.source_keys):
            raise ValueError("checkpoint source aliases must be unique")
        if len(set(self.target_keys)) != len(self.target_keys):
            raise ValueError("checkpoint target aliases must be unique")
        if not self.semantic:
            raise ValueError("checkpoint tensor groups require a semantic description")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "source_keys": list(self.source_keys),
            "target_keys": list(self.target_keys),
            "semantic": self.semantic,
        }


@dataclass(frozen=True)
class ReconstructedTensorContract:
    """A source checkpoint tensor eliminated by typed structural lowering."""

    source_key: str
    shape: Tuple[int, ...]
    dtype: str
    values: Tuple[int | float | bool, ...]
    semantic: str

    def __post_init__(self) -> None:
        size = 1
        for dimension in self.shape:
            if int(dimension) < 0:
                raise ValueError("reconstructed checkpoint shapes must be nonnegative")
            size *= int(dimension)
        if size != len(self.values):
            raise ValueError("reconstructed checkpoint values do not match the declared shape")
        if not self.source_key or not self.dtype or not self.semantic:
            raise ValueError("reconstructed checkpoint contracts require complete metadata")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "source_key": self.source_key,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "values": list(self.values),
            "semantic": self.semantic,
        }


@dataclass(frozen=True)
class CheckpointMappingManifest:
    """Complete source-state to lowered-state mapping contract."""

    source_format: str
    target_format: str
    architecture_id: str
    tensor_groups: Tuple[CheckpointTensorGroup, ...]
    reconstructed_sources: Tuple[ReconstructedTensorContract, ...] = ()
    version: str = CHECKPOINT_MAPPING_VERSION

    def __post_init__(self) -> None:
        source_keys = [
            key for group in self.tensor_groups for key in group.source_keys
        ] + [item.source_key for item in self.reconstructed_sources]
        target_keys = [key for group in self.tensor_groups for key in group.target_keys]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("checkpoint manifest source keys must be globally unique")
        if len(target_keys) != len(set(target_keys)):
            raise ValueError("checkpoint manifest target keys must be globally unique")
        if not self.source_format or not self.target_format or not self.architecture_id:
            raise ValueError("checkpoint manifest identity fields must be nonempty")

    @property
    def source_keys(self) -> Tuple[str, ...]:
        return tuple(
            [key for group in self.tensor_groups for key in group.source_keys]
            + [item.source_key for item in self.reconstructed_sources]
        )

    @property
    def target_keys(self) -> Tuple[str, ...]:
        return tuple(key for group in self.tensor_groups for key in group.target_keys)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "version": self.version,
            "source_format": self.source_format,
            "target_format": self.target_format,
            "architecture_id": self.architecture_id,
            "source_tensor_count": len(self.source_keys),
            "target_tensor_count": len(self.target_keys),
            "tensor_groups": [item.to_dict() for item in self.tensor_groups],
            "reconstructed_sources": [
                item.to_dict() for item in self.reconstructed_sources
            ],
        }


def extract_checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract a direct, Fair-Chem, or local-training model state dictionary."""

    if not isinstance(checkpoint, Mapping):
        raise CheckpointMappingError("checkpoint payload must be a mapping")
    for key in ("state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    if checkpoint and all(hasattr(value, "shape") and hasattr(value, "dtype") for value in checkpoint.values()):
        return checkpoint
    raise CheckpointMappingError(
        "checkpoint must be a tensor state dictionary or contain 'state_dict'/'model'"
    )


def normalize_checkpoint_prefixes(
    state_dict: Mapping[str, Any],
    prefixes: Sequence[str] = ("module.", "_orig_mod."),
) -> Mapping[str, Any]:
    """Remove distributed/compiled wrapper prefixes with collision detection."""

    normalized = OrderedDict()
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        if key in normalized:
            raise CheckpointMappingError(
                "checkpoint prefix normalization produced duplicate key {!r}".format(key)
            )
        normalized[key] = value
    return normalized


def _assert_same_tensor(left, right, *, context: str) -> None:
    if tuple(left.shape) != tuple(right.shape):
        raise CheckpointMappingError(
            "{} shape mismatch: {} != {}".format(
                context,
                tuple(left.shape),
                tuple(right.shape),
            )
        )
    if left.dtype != right.dtype:
        raise CheckpointMappingError(
            "{} dtype mismatch: {} != {}".format(context, left.dtype, right.dtype)
        )
    if not left.detach().cpu().equal(right.detach().cpu()):
        raise CheckpointMappingError("{} values are not identical".format(context))


def _validate_reconstructed_tensor(tensor, contract: ReconstructedTensorContract) -> None:
    if tuple(tensor.shape) != tuple(contract.shape):
        raise CheckpointMappingError(
            "reconstructed source {} has shape {}, expected {}".format(
                contract.source_key,
                tuple(tensor.shape),
                contract.shape,
            )
        )
    if _tensor_dtype_name(tensor) != contract.dtype:
        raise CheckpointMappingError(
            "reconstructed source {} has dtype {}, expected {}".format(
                contract.source_key,
                _tensor_dtype_name(tensor),
                contract.dtype,
            )
        )
    actual = tuple(tensor.detach().cpu().reshape(-1).tolist())
    if actual != contract.values:
        raise CheckpointMappingError(
            "reconstructed source {} violates its constant contract".format(
                contract.source_key
            )
        )


def translate_checkpoint_state_dict(
    checkpoint: Mapping[str, Any],
    target_state_dict: Mapping[str, Any],
    manifest: CheckpointMappingManifest,
) -> OrderedDict:
    """Translate and strictly validate a source checkpoint for a lowered model."""

    source = normalize_checkpoint_prefixes(extract_checkpoint_state_dict(checkpoint))
    expected_source = set(manifest.source_keys)
    actual_source = set(source)
    if actual_source != expected_source:
        raise CheckpointMappingError(
            "checkpoint source keys differ from the manifest; missing={}, unexpected={}".format(
                sorted(expected_source - actual_source),
                sorted(actual_source - expected_source),
            )
        )
    expected_target = set(manifest.target_keys)
    actual_target = set(target_state_dict)
    if actual_target != expected_target:
        raise CheckpointMappingError(
            "lowered target keys differ from the manifest; missing={}, unexpected={}".format(
                sorted(expected_target - actual_target),
                sorted(actual_target - expected_target),
            )
        )

    translated = {}
    for group in manifest.tensor_groups:
        canonical = source[group.source_keys[0]]
        for alias in group.source_keys[1:]:
            _assert_same_tensor(
                canonical,
                source[alias],
                context="source aliases {} and {}".format(group.source_keys[0], alias),
            )
        for target_key in group.target_keys:
            template = target_state_dict[target_key]
            if tuple(canonical.shape) != tuple(template.shape):
                raise CheckpointMappingError(
                    "checkpoint tensor {} cannot map to {} because shapes differ: {} != {}".format(
                        group.source_keys[0],
                        target_key,
                        tuple(canonical.shape),
                        tuple(template.shape),
                    )
                )
            if canonical.dtype != template.dtype:
                raise CheckpointMappingError(
                    "checkpoint tensor {} cannot map to {} because dtypes differ: {} != {}".format(
                        group.source_keys[0],
                        target_key,
                        canonical.dtype,
                        template.dtype,
                    )
                )
            translated[target_key] = canonical.detach().clone()

    for contract in manifest.reconstructed_sources:
        _validate_reconstructed_tensor(source[contract.source_key], contract)

    return OrderedDict((key, translated[key]) for key in target_state_dict)


def load_mapped_checkpoint_state_dict(
    model,
    checkpoint: Mapping[str, Any],
    manifest: CheckpointMappingManifest,
):
    """Load a translated checkpoint with strict target-state coverage."""

    translated = translate_checkpoint_state_dict(
        checkpoint,
        model.state_dict(),
        manifest,
    )
    incompatible = model.load_state_dict(translated, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CheckpointMappingError(
            "strict lowered checkpoint load returned incompatible keys: {}".format(
                incompatible
            )
        )
    return translated


def export_source_state_dict(
    target_state_dict: Mapping[str, Any],
    manifest: CheckpointMappingManifest,
) -> OrderedDict:
    """Export a lowered state back into the declared source checkpoint namespace."""

    import torch

    expected_target = set(manifest.target_keys)
    actual_target = set(target_state_dict)
    if actual_target != expected_target:
        raise CheckpointMappingError(
            "lowered target keys differ from the manifest during export"
        )
    exported = {}
    for group in manifest.tensor_groups:
        canonical = target_state_dict[group.target_keys[0]]
        for alias in group.target_keys[1:]:
            _assert_same_tensor(
                canonical,
                target_state_dict[alias],
                context="target aliases {} and {}".format(group.target_keys[0], alias),
            )
        for source_key in group.source_keys:
            exported[source_key] = canonical.detach().clone()
    for contract in manifest.reconstructed_sources:
        dtype = getattr(torch, contract.dtype)
        exported[contract.source_key] = torch.tensor(
            contract.values,
            dtype=dtype,
        ).reshape(contract.shape)
    return OrderedDict((key, exported[key]) for key in manifest.source_keys)
