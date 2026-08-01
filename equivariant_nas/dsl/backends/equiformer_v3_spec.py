"""Backend-neutral, lossless architecture specification for Equiformer V3."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional, Tuple


V3_REFERENCE_COMMIT = "a7300c58df683dc99cb48027d5bfd4c887486c48"
V3_MODEL_NAMES = ("equiformer_v3", "equiformer_v3_dens")

_V3_NORM_TYPES = frozenset({
    "equivariant_layer_norm",
    "sep_layer_norm",
    "merge_layer_norm",
    "merge_layer_norm_attn_rms_norm",
    "merge_rms_norm",
})
_V3_ACTIVATIONS = frozenset({
    "gate",
    "s2",
    "sep_s2",
    "s2_swiglu",
    "s2_swiglu_mem",
    "sep_s2_swiglu",
    "sep-merge_s2_swiglu",
    "sep_s2_swiglu_mem",
    "sep-merge_s2_swiglu_mem",
    "sep_s2_square",
    "sep-merge_gates2_swiglu",
    "sep-merge_gates2_swiglu_mem",
})


@dataclass(frozen=True)
class V3SpecImportManifest:
    """Evidence that every official model-config field was consumed or defaulted."""

    model_name: str
    source_commit: str
    consumed_fields: Tuple[str, ...]
    defaulted_fields: Tuple[str, ...]
    dispatch_fields: Tuple[str, ...]
    architecture_id: str

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "source_commit": self.source_commit,
            "consumed_fields": list(self.consumed_fields),
            "defaulted_fields": list(self.defaulted_fields),
            "dispatch_fields": list(self.dispatch_fields),
            "architecture_id": self.architecture_id,
        }


@dataclass(frozen=True)
class EquiformerV3Spec:
    """All constructor fields that change the official V3 model or graph path.

    Optimizer, scheduler, loss and dataset settings remain outside this object.
    Unlike the earlier compositional subset, this schema mirrors the complete
    ``EquiformerV3_OC.__init__`` signature at the pinned official commit.
    """

    use_pbc: bool = True
    use_pbc_single: bool = False
    otf_graph: bool = True
    regress_forces: bool = True
    regress_stress: bool = False
    direct_prediction: bool = True
    max_neighbors: int = 20
    max_radius: float = 12.0
    num_radial_basis: int = 600
    max_num_elements: int = 128
    num_layers: int = 7
    num_channels: int = 128
    attn_hidden_channels: int = 32
    num_heads: int = 8
    attn_alpha_channels: int = 32
    attn_value_channels: int = 16
    ffn_hidden_channels: int = 512
    norm_type: str = "merge_layer_norm"
    lmax: int = 4
    mmax: int = 2
    attn_grid_resolution: Tuple[int, int] = (14, 8)
    ffn_grid_resolution: Tuple[int, int] = (14, 14)
    edge_channels: int = 128
    use_atom_edge_embedding: bool = True
    use_envelope: bool = True
    attn_activation: str = "sep-merge_gates2_swiglu"
    use_attn_renorm: bool = True
    use_add_merge: bool = False
    use_rad_l_parametrization: bool = True
    softcap: Optional[float] = None
    attn_eps: float = 1.0e-16
    ffn_activation: str = "sep-merge_gates2_swiglu"
    use_grid_mlp: bool = True
    use_gate_force_head: bool = True
    alpha_drop: float = 0.0
    attn_mask_rate: float = 0.0
    attn_weights_drop: float = 0.1
    value_drop: float = 0.0
    drop_path_rate: float = 0.05
    proj_drop: float = 0.0
    ffn_drop: float = 0.0
    use_head_reg: bool = False
    gradient_checkpointing_block_list: Tuple[int, ...] = ()
    avg_num_nodes: float = 77.81317
    avg_degree: float = 23.395238876342773
    enforce_max_neighbors_strictly: bool = True

    @classmethod
    def field_names(cls) -> Tuple[str, ...]:
        return tuple(item.name for item in fields(cls))

    def validate(self) -> None:
        positive_ints = {
            "max_neighbors": self.max_neighbors,
            "num_radial_basis": self.num_radial_basis,
            "max_num_elements": self.max_num_elements,
            "num_layers": self.num_layers,
            "num_channels": self.num_channels,
            "attn_hidden_channels": self.attn_hidden_channels,
            "num_heads": self.num_heads,
            "attn_alpha_channels": self.attn_alpha_channels,
            "attn_value_channels": self.attn_value_channels,
            "ffn_hidden_channels": self.ffn_hidden_channels,
            "edge_channels": self.edge_channels,
        }
        invalid = {
            name: value for name, value in positive_ints.items()
            if isinstance(value, bool) or int(value) <= 0 or int(value) != value
        }
        if invalid:
            raise ValueError("Equiformer V3 positive integer fields are invalid: {}".format(invalid))
        positive_floats = {
            "max_radius": self.max_radius,
            "attn_eps": self.attn_eps,
            "avg_num_nodes": self.avg_num_nodes,
            "avg_degree": self.avg_degree,
        }
        invalid_floats = {
            name: value for name, value in positive_floats.items()
            if not math.isfinite(float(value)) or float(value) <= 0.0
        }
        if invalid_floats:
            raise ValueError("Equiformer V3 positive floating fields are invalid: {}".format(invalid_floats))
        if self.lmax < 0 or self.mmax < 0 or self.mmax > self.lmax:
            raise ValueError("Equiformer V3 requires 0 <= mmax <= lmax")
        for name, resolution in (
            ("attn_grid_resolution", self.attn_grid_resolution),
            ("ffn_grid_resolution", self.ffn_grid_resolution),
        ):
            if len(resolution) != 2 or any(int(value) < 2 for value in resolution):
                raise ValueError("{} must contain two resolutions >= 2".format(name))
        if self.norm_type not in _V3_NORM_TYPES:
            raise ValueError("unsupported official Equiformer V3 norm_type: {}".format(self.norm_type))
        if self.attn_activation not in _V3_ACTIVATIONS:
            raise ValueError("unsupported official Equiformer V3 attention activation: {}".format(self.attn_activation))
        if self.ffn_activation not in _V3_ACTIVATIONS:
            raise ValueError("unsupported official Equiformer V3 FFN activation: {}".format(self.ffn_activation))
        if self.softcap is not None and (
            not math.isfinite(float(self.softcap)) or float(self.softcap) <= 0.0
        ):
            raise ValueError("softcap must be None or finite and positive")
        probabilities = {
            "alpha_drop": self.alpha_drop,
            "attn_mask_rate": self.attn_mask_rate,
            "attn_weights_drop": self.attn_weights_drop,
            "value_drop": self.value_drop,
            "drop_path_rate": self.drop_path_rate,
            "proj_drop": self.proj_drop,
            "ffn_drop": self.ffn_drop,
        }
        invalid_probabilities = {
            name: value for name, value in probabilities.items()
            if not math.isfinite(float(value)) or float(value) < 0.0 or float(value) >= 1.0
        }
        if invalid_probabilities:
            raise ValueError("Equiformer V3 probabilities must lie in [0, 1): {}".format(invalid_probabilities))
        if self.use_add_merge and not self.use_rad_l_parametrization:
            raise ValueError("official V3 add-merge requires l-parameterized radial weights")
        if not self.direct_prediction and self.regress_stress and not self.regress_forces:
            raise ValueError("official gradient stress path requires force regression")
        checkpointing = tuple(int(value) for value in self.gradient_checkpointing_block_list)
        if checkpointing and (
            len(checkpointing) != self.num_layers or any(value not in (0, 1) for value in checkpointing)
        ):
            raise ValueError(
                "gradient_checkpointing_block_list must be empty or contain one 0/1 flag per layer"
            )

    def validate_compositional_subset(self) -> None:
        """Guard the legacy 133-node approximation against silent overclaiming."""

        self.validate()
        if self.norm_type != "merge_layer_norm":
            raise ValueError("the legacy compositional V3 importer only represents merge_layer_norm")
        if "swiglu" not in self.attn_activation or "swiglu" not in self.ffn_activation:
            raise ValueError("the legacy compositional V3 importer only represents S2 SwiGLU paths")

    def to_dict(self) -> Mapping[str, Any]:
        data = asdict(self)
        data["attn_grid_resolution"] = list(self.attn_grid_resolution)
        data["ffn_grid_resolution"] = list(self.ffn_grid_resolution)
        data["gradient_checkpointing_block_list"] = list(self.gradient_checkpointing_block_list)
        return data

    def official_constructor_kwargs(self) -> Mapping[str, Any]:
        data = dict(self.to_dict())
        data["attn_grid_resolution_list"] = data.pop("attn_grid_resolution")
        data["ffn_grid_resolution_list"] = data.pop("ffn_grid_resolution")
        checkpointing = data.pop("gradient_checkpointing_block_list")
        data["gradient_checkpointing_block_list"] = checkpointing or None
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, strict: bool = False) -> "EquiformerV3Spec":
        values = dict(data)
        if "attn_grid_resolution_list" in values:
            values["attn_grid_resolution"] = values.pop("attn_grid_resolution_list")
        if "ffn_grid_resolution_list" in values:
            values["ffn_grid_resolution"] = values.pop("ffn_grid_resolution_list")
        allowed = set(cls.field_names())
        unknown = set(values) - allowed
        if strict and unknown:
            raise ValueError("unknown Equiformer V3 constructor fields: {}".format(sorted(unknown)))
        values = {name: value for name, value in values.items() if name in allowed}
        for name in (
            "attn_grid_resolution",
            "ffn_grid_resolution",
            "gradient_checkpointing_block_list",
        ):
            if name in values and values[name] is not None:
                values[name] = tuple(int(value) for value in values[name])
        if values.get("gradient_checkpointing_block_list") is None:
            values["gradient_checkpointing_block_list"] = ()
        spec = cls(**values)
        spec.validate()
        return spec

    @classmethod
    def from_official_config(
        cls,
        config: Mapping[str, Any],
        *,
        allow_dens_dispatch: bool = False,
    ) -> Tuple["EquiformerV3Spec", V3SpecImportManifest]:
        """Strictly consume a full config or its ``model`` section.

        Unknown model fields are rejected instead of being silently discarded.
        DeNS has additional constructor fields and remains an explicit later
        importer stage; accepting its dispatch name here requires an opt-in and
        still does not consume DeNS-only fields.
        """

        if "model" in config and isinstance(config["model"], Mapping):
            raw = dict(config["model"])
        elif "model_attributes" in config and isinstance(config["model_attributes"], Mapping):
            raw = dict(config["model_attributes"])
        else:
            raw = dict(config)
        dispatch_fields = []
        model_name = str(raw.pop("name", "equiformer_v3"))
        if "name" in (config.get("model", {}) if isinstance(config.get("model"), Mapping) else config):
            dispatch_fields.append("name")
        if model_name not in V3_MODEL_NAMES:
            raise ValueError("official config does not select Equiformer V3: {}".format(model_name))
        if model_name == "equiformer_v3_dens" and not allow_dens_dispatch:
            raise ValueError("Equiformer V3 DeNS requires the dedicated DeNS importer")
        provided = set(raw)
        spec = cls.from_mapping(raw, strict=True)
        consumed = tuple(sorted(provided))
        defaulted = tuple(sorted(set(cls.field_names()) - provided))
        manifest = V3SpecImportManifest(
            model_name=model_name,
            source_commit=V3_REFERENCE_COMMIT,
            consumed_fields=consumed,
            defaulted_fields=defaulted,
            dispatch_fields=tuple(sorted(dispatch_fields)),
            architecture_id=spec.architecture_id(),
        )
        return spec, manifest

    def architecture_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def baseline_v3_spec() -> EquiformerV3Spec:
    """Return the historical 7-layer compositional-development baseline."""

    spec = EquiformerV3Spec()
    spec.validate()
    return spec


def official_v3_oc_spec() -> EquiformerV3Spec:
    """Return the exact defaults of ``EquiformerV3_OC`` at the pinned commit."""

    spec = EquiformerV3Spec(
        num_layers=12,
        num_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        lmax=6,
        mmax=2,
        attn_grid_resolution=(20, 8),
        ffn_grid_resolution=(20, 20),
    )
    spec.validate()
    return spec
