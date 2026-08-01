"""Frozen language snapshots and auditable active-vocabulary selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .groups import GroupSpec
from .diagnostics import DSLValidationError, Diagnostic
from .motifs import MotifRegistry
from .registry import PrimitiveRegistry
from .search_surface import CanonicalSearchSurface


@dataclass(frozen=True)
class LanguageVersion:
    version: str
    parent_version: str
    primitive_names: Tuple[str, ...]
    motif_names: Tuple[str, ...]
    frozen_at: str
    metadata: Mapping[str, Any]
    primitive_hashes: Mapping[str, str] = field(default_factory=dict)
    motif_hashes: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_registries(
        cls,
        version: str,
        parent_version: str,
        primitives: PrimitiveRegistry,
        motifs: MotifRegistry,
        frozen_at: str,
        metadata: Mapping[str, Any],
    ) -> "LanguageVersion":
        return cls(
            version,
            parent_version,
            primitives.names(),
            motifs.names(),
            frozen_at,
            dict(metadata),
            {name: primitives.resolve(name).content_hash() for name in primitives.names()},
            {name: motifs.resolve(name).content_hash() for name in motifs.names()},
        )

    def registry_hash(self) -> str:
        payload = {
            "version": self.version,
            "parent": self.parent_version,
            "primitives": sorted(self.primitive_names),
            "motifs": sorted(self.motif_names),
            "metadata": dict(self.metadata),
            "primitive_hashes": dict(sorted(self.primitive_hashes.items())),
            "motif_hashes": dict(sorted(self.motif_hashes.items())),
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VocabularyDecision:
    visible: Tuple[str, ...]
    excluded: Mapping[str, str]
    language_version: str
    group_family: str
    completion_only: Tuple[str, ...] = ()
    context_only: Tuple[str, ...] = ()
    canonical_families: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    search_surface_version: str = ""
    search_surface_hash: str = ""

    def completion_ops(self) -> Tuple[str, ...]:
        return tuple(sorted({
            name
            for name in tuple(self.visible) + tuple(self.completion_only)
            if name.startswith("core.")
        }))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visible": list(self.visible),
            "excluded": dict(self.excluded),
            "language_version": self.language_version,
            "group_family": self.group_family,
            "completion_only": list(self.completion_only),
            "context_only": list(self.context_only),
            "canonical_families": {
                name: list(values) for name, values in sorted(self.canonical_families.items())
            },
            "search_surface_version": self.search_surface_version,
            "search_surface_hash": self.search_surface_hash,
        }


def describe_active_vocabulary(
    decision: VocabularyDecision,
    primitives: PrimitiveRegistry,
    motifs: MotifRegistry,
) -> Tuple[Mapping[str, Any], ...]:
    """Expose machine-readable signatures and invariants for every visible word."""

    return describe_vocabulary_names(decision.visible, primitives, motifs)


def describe_vocabulary_names(
    names: Sequence[str],
    primitives: PrimitiveRegistry,
    motifs: MotifRegistry,
) -> Tuple[Mapping[str, Any], ...]:
    """Describe an explicit set of primitive or motif names without changing exposure."""

    descriptions = []
    for name in names:
        if name.startswith("motif."):
            definition = motifs.resolve(name)
            descriptions.append({
                "kind": "motif",
                "name": definition.qualified_name,
                "input_ports": list(definition.input_ports),
                "output_ports": sorted(definition.output_bindings),
                "required_attrs": list(definition.required_attrs),
                "group_families": list(definition.group_families),
                "certification": definition.certification,
                "semantic_constraints": list(definition.semantic_constraints),
                "edit_guidance": list(definition.edit_guidance),
                "provenance": dict(definition.provenance),
            })
        else:
            definition = primitives.resolve(name)
            descriptions.append({
                "kind": "primitive",
                "name": definition.qualified_name,
                "input_ports": list(definition.input_ports),
                "output_ports": list(definition.output_ports),
                "group_families": list(definition.group_families),
                "certificate_level": int(definition.certificate_level),
                "backend_keys": list(definition.backend_keys),
                "description": definition.description,
                "required_attrs": list(definition.required_attrs),
                "optional_attrs": dict(definition.optional_attrs),
                "motif_parameter_attrs": list(definition.motif_parameter_attrs),
                "semantic_constraints": list(definition.semantic_constraints),
                "edit_guidance": list(definition.edit_guidance),
            })
    return tuple(descriptions)


def select_active_vocabulary(
    language: LanguageVersion,
    group: GroupSpec,
    primitives: PrimitiveRegistry,
    motifs: MotifRegistry,
    *,
    allowed_names: Sequence[str] = (),
    blocked_names: Sequence[str] = (),
    search_surface: CanonicalSearchSurface = None,
) -> VocabularyDecision:
    """Filter a frozen language without silently changing its contents."""

    allow = set(allowed_names)
    block = set(blocked_names)
    visible = []
    completion_only = []
    context_only = []
    excluded: Dict[str, str] = {}
    for name in tuple(language.primitive_names) + tuple(language.motif_names):
        if allow and name not in allow:
            excluded[name] = "outside_requested_scope"
            continue
        if name in block:
            excluded[name] = "blocked_by_resource_or_safety_policy"
            continue
        if name in language.primitive_names:
            definition = primitives.resolve(name)
            families = definition.group_families
            expected_hash = language.primitive_hashes.get(name)
            actual_hash = definition.content_hash()
        else:
            definition = motifs.resolve(name)
            families = definition.group_families
            expected_hash = language.motif_hashes.get(name)
            actual_hash = definition.content_hash()
        if expected_hash and expected_hash != actual_hash:
            raise DSLValidationError([
                Diagnostic(
                    "E_LANGUAGE_001",
                    "runtime registry content differs from the frozen language snapshot",
                    actual=name,
                    expected=expected_hash,
                    details={"actual_hash": actual_hash},
                )
            ])
        if group.family not in families:
            excluded[name] = "incompatible_group"
            continue
        if search_surface is None:
            visible.append(name)
            continue
        try:
            role = search_surface.role_of(name)
        except KeyError:
            raise DSLValidationError([
                Diagnostic(
                    "E_SEARCH_SURFACE_002",
                    "frozen language contains an entry absent from the selected search surface",
                    actual=name,
                    details={"search_surface_version": search_surface.version},
                )
            ])
        if role == "generatable":
            visible.append(name)
        elif role == "completion_only":
            completion_only.append(name)
            excluded[name] = "trusted_completion_only"
        else:
            context_only.append(name)
            excluded[name] = "compatibility_or_fusion_context_only"

    canonical_families: Dict[str, Tuple[str, ...]] = {}
    if search_surface is not None:
        visible_set = set(visible)
        for canonical, realizations in search_surface.canonical_families.items():
            selected = tuple(name for name in realizations if name in visible_set)
            if selected:
                canonical_families[canonical] = selected
    return VocabularyDecision(
        tuple(sorted(visible)),
        excluded,
        language.version,
        group.family,
        tuple(sorted(completion_only)),
        tuple(sorted(context_only)),
        canonical_families,
        search_surface.version if search_surface is not None else "",
        search_surface.content_hash() if search_surface is not None else "",
    )
