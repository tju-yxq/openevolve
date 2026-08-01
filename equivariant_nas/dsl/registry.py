"""Trusted primitive registry and core equivariant type rules."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .diagnostics import DSLValidationError, Diagnostic
from .irreps import Irrep, Irreps
from .obligations import ObligationKind, ProofObligation
from .parameters import PARAMETER_CONTRACT_SCHEMA_VERSION, ParameterAxis, ParameterContract
from .types import AffinePointType, AxisSpec, Carrier, CategoricalTensorType, EquivarianceLevel, EquivariantTensorType, EquivariantType, FeatureRole, Frame, GridSpec, GridTensorType, IndexMapType, InvariantTensorType, LatticeShiftType, LatticeType, ValueType


TypeRule = Callable[[str, Mapping[str, Tuple[ValueType, ...]], Mapping[str, Any]], Tuple[Mapping[str, ValueType], Tuple[ProofObligation, ...]]]
ParameterRule = Callable[[str, Mapping[str, Tuple[ValueType, ...]], Mapping[str, ValueType], Mapping[str, Any]], Tuple[ParameterContract, ...]]


@dataclass(frozen=True)
class PrimitiveDefinition:
    name: str
    version: int
    input_ports: Tuple[str, ...]
    output_ports: Tuple[str, ...]
    type_rule: TypeRule
    group_families: Tuple[str, ...] = ("O3", "SO3", "O2", "SO2")
    certificate_level: EquivarianceLevel = EquivarianceLevel.CORE_CERTIFIED
    backend_keys: Tuple[str, ...] = ()
    description: str = ""
    required_attrs: Tuple[str, ...] = ()
    optional_attrs: Mapping[str, str] = field(default_factory=dict)
    attribute_aliases: Mapping[str, str] = field(default_factory=dict)
    motif_parameter_attrs: Tuple[str, ...] = ()
    semantic_constraints: Tuple[str, ...] = ()
    edit_guidance: Tuple[str, ...] = ()
    parameter_rule: Optional[ParameterRule] = None

    def __post_init__(self) -> None:
        required = set(self.required_attrs)
        optional = set(self.optional_attrs)
        aliases = set(self.attribute_aliases)
        overlap = required.intersection(optional)
        if overlap:
            raise DSLValidationError([
                Diagnostic(
                    "E_REGISTRY_004",
                    "primitive attributes cannot be both required and optional",
                    actual=self.qualified_name,
                    details={"attributes": sorted(overlap)},
                )
            ])
        invalid_targets = set(self.attribute_aliases.values()) - (required | optional)
        if invalid_targets:
            raise DSLValidationError([
                Diagnostic(
                    "E_REGISTRY_005",
                    "primitive attribute alias targets must be declared attributes",
                    actual=self.qualified_name,
                    details={"targets": sorted(invalid_targets)},
                )
            ])
        collisions = aliases.intersection(required | optional)
        if collisions:
            raise DSLValidationError([
                Diagnostic(
                    "E_REGISTRY_006",
                    "primitive attribute aliases cannot shadow canonical attributes",
                    actual=self.qualified_name,
                    details={"attributes": sorted(collisions)},
                )
            ])

    def canonical_attrs(self, node_id: str, attrs: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = dict(attrs)
        for alias, canonical in self.attribute_aliases.items():
            if alias not in normalized:
                continue
            if canonical in normalized:
                raise DSLValidationError([
                    Diagnostic(
                        "E_ATTR_004",
                        "attribute alias and canonical name cannot both be present",
                        node_id=node_id,
                        details={"alias": alias, "canonical": canonical},
                    )
                ])
            normalized[canonical] = normalized.pop(alias)
        allowed = set(self.required_attrs) | set(self.optional_attrs)
        missing = set(self.required_attrs) - set(normalized)
        unknown = set(normalized) - allowed
        if missing or unknown:
            raise DSLValidationError([
                Diagnostic(
                    "E_ATTR_005",
                    "primitive attributes do not match the registered schema",
                    node_id=node_id,
                    details={"missing": sorted(missing), "unknown": sorted(unknown)},
                )
            ])
        return normalized

    @property
    def qualified_name(self) -> str:
        return "{}@{}".format(self.name, self.version)

    def infer_parameter_contracts(
        self,
        node_id: str,
        inputs: Mapping[str, Tuple[ValueType, ...]],
        outputs: Mapping[str, ValueType],
        attrs: Mapping[str, Any],
    ) -> Tuple[ParameterContract, ...]:
        contracts = tuple(self.parameter_rule(node_id, inputs, outputs, attrs)) if self.parameter_rule else ()
        names = [item.name for item in contracts]
        if len(names) != len(set(names)):
            raise DSLValidationError([
                Diagnostic(
                    "E_PARAMETER_018",
                    "primitive parameter contracts must have unique logical names",
                    node_id=node_id,
                    details={"names": names},
                )
            ])
        invalid_external = [
            item.name
            for item in contracts
            if item.storage == "external"
            and (item.external_port not in self.input_ports or not inputs.get(item.external_port))
        ]
        if invalid_external:
            raise DSLValidationError([
                Diagnostic(
                    "E_PARAMETER_019",
                    "external parameter contracts must bind a populated primitive input port",
                    node_id=node_id,
                    details={"parameters": invalid_external},
                )
            ])
        return contracts

    def content_hash(self) -> str:
        payload = {
            "name": self.name,
            "version": self.version,
            "input_ports": self.input_ports,
            "output_ports": self.output_ports,
            "type_rule": getattr(self.type_rule, "__name__", "anonymous"),
            "group_families": self.group_families,
            "certificate_level": int(self.certificate_level),
            "backend_keys": self.backend_keys,
            "description": self.description,
            "required_attrs": self.required_attrs,
            "optional_attrs": dict(self.optional_attrs),
            "attribute_aliases": dict(self.attribute_aliases),
            "motif_parameter_attrs": self.motif_parameter_attrs,
            "semantic_constraints": self.semantic_constraints,
            "edit_guidance": self.edit_guidance,
            "parameter_rule": getattr(self.parameter_rule, "__name__", "") if self.parameter_rule else "",
            "parameter_contract_schema_version": PARAMETER_CONTRACT_SCHEMA_VERSION,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class PrimitiveRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, PrimitiveDefinition] = {}

    def register(self, definition: PrimitiveDefinition) -> None:
        key = definition.qualified_name
        if key in self._items:
            raise DSLValidationError([Diagnostic("E_REGISTRY_001", "duplicate primitive registration", actual=key)])
        self._items[key] = definition

    def resolve(self, name: str) -> PrimitiveDefinition:
        key = name if "@" in name else "{}@1".format(name)
        try:
            return self._items[key]
        except KeyError:
            raise DSLValidationError([Diagnostic("E_REGISTRY_002", "unknown primitive", actual=name)])

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._items))

    def content_hash(self) -> str:
        payload = {
            name: self.resolve(name).content_hash()
            for name in self.names()
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _single(node: str, inputs: Mapping[str, Tuple[ValueType, ...]], port: str) -> ValueType:
    values = inputs.get(port, ())
    if len(values) != 1:
        raise DSLValidationError([Diagnostic("E_PORT_001", "port requires exactly one input", node_id=node, port=port, actual=str(len(values)))])
    return values[0]


def _v2_equivariant(node: str, value: ValueType, port: str) -> EquivariantTensorType:
    if not isinstance(value, EquivariantTensorType):
        raise DSLValidationError([
            Diagnostic("E_TYPE_010", "v2 primitive requires EquivariantTensorType", node_id=node, port=port, actual=type(value).__name__)
        ])
    return value


def _same_context(node: str, left: EquivariantType, right: EquivariantType, *, exact_irreps: bool = True) -> None:
    if not left.compatible(right, exact_irreps=exact_irreps):
        raise DSLValidationError([
            Diagnostic("E_TYPE_004", "input types are incompatible", node_id=node, expected=str(left.to_dict()), actual=str(right.to_dict()))
        ])


def _multiply_measures(left: str, right: str) -> str:
    if left == "dimensionless":
        return right
    if right == "dimensionless":
        return left
    if left == right:
        return "({})^2".format(left)
    return "({})*({})".format(left, right)


def _identity(node, inputs, attrs):
    return {"out": _single(node, inputs, "x")}, ()


def _categorical_remap(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if not isinstance(x, CategoricalTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_CATEGORICAL_REMAP_001",
                "categorical_remap requires CategoricalTensorType input",
                node_id=node,
                actual=type(x).__name__,
            )
        ])
    raw_output_size = attrs["output_vocabulary_size"]
    if isinstance(raw_output_size, bool):
        raise DSLValidationError([
            Diagnostic("E_CATEGORICAL_REMAP_002", "output_vocabulary_size must be a positive integer", node_id=node)
        ])
    try:
        output_size = int(raw_output_size)
    except (TypeError, ValueError):
        output_size = 0
    if output_size <= 0 or str(raw_output_size) != str(output_size):
        raise DSLValidationError([
            Diagnostic(
                "E_CATEGORICAL_REMAP_002",
                "output_vocabulary_size must be a positive integer",
                node_id=node,
                actual=str(raw_output_size),
            )
        ])
    mapping = attrs["mapping"]
    if not isinstance(mapping, (list, tuple)) or len(mapping) != int(x.vocabulary_size):
        raise DSLValidationError([
            Diagnostic(
                "E_CATEGORICAL_REMAP_003",
                "mapping must contain one entry per input category",
                node_id=node,
                expected=str(int(x.vocabulary_size)),
                actual=str(len(mapping) if isinstance(mapping, (list, tuple)) else mapping),
            )
        ])
    normalized = []
    for index, raw_value in enumerate(mapping):
        if isinstance(raw_value, bool):
            value = -2
        else:
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                value = -2
        if value < -1 or value >= output_size or str(raw_value) != str(value):
            raise DSLValidationError([
                Diagnostic(
                    "E_CATEGORICAL_REMAP_004",
                    "mapping entries must be -1 (runtime rejection) or a valid output category",
                    node_id=node,
                    actual=str(raw_value),
                    details={"input_category": index, "output_vocabulary_size": output_size},
                )
            ])
        normalized.append(value)
    if not any(value >= 0 for value in normalized):
        raise DSLValidationError([
            Diagnostic("E_CATEGORICAL_REMAP_005", "mapping must admit at least one category", node_id=node)
        ])
    return {
        "out": replace(
            x,
            vocabulary_size=output_size,
            labels=(),
        )
    }, ()


def _categorical_one_hot(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if not isinstance(x, CategoricalTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_ONE_HOT_001",
                "categorical_one_hot requires CategoricalTensorType input",
                node_id=node,
                actual=type(x).__name__,
            )
        ])
    axis_name = str(attrs.get("axis", "species"))
    if not axis_name:
        raise DSLValidationError([Diagnostic("E_ONE_HOT_002", "one-hot axis name must be nonempty", node_id=node)])
    dtype = str(attrs.get("dtype", "float32"))
    if dtype not in ("float16", "bfloat16", "float32", "float64"):
        raise DSLValidationError([
            Diagnostic("E_ONE_HOT_003", "unsupported one-hot floating dtype", node_id=node, actual=dtype)
        ])
    scalar = Irrep(0, 1, x.group.family)
    size = int(x.vocabulary_size)
    return {
        "out": InvariantTensorType(
            group=x.group,
            carrier=x.carrier,
            irreps=Irreps(((size, scalar),)),
            frame=Frame("invariant"),
            axes=(axis_name,),
            dtype=dtype,
            measure="dimensionless",
            level=EquivarianceLevel.CORE_CERTIFIED,
            axis_specs=(AxisSpec(axis_name, size, FeatureRole.SPECIES, "independent", 0),),
            feature_role=FeatureRole.SPECIES,
        )
    }, ()


def _categorical_embedding(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if not isinstance(x, CategoricalTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_CATEGORICAL_EMBED_001",
                "categorical_embedding requires CategoricalTensorType input",
                node_id=node,
                actual=type(x).__name__,
            )
        ])
    raw_dim = attrs["embedding_dim"]
    if isinstance(raw_dim, bool):
        embedding_dim = 0
    else:
        try:
            embedding_dim = int(raw_dim)
        except (TypeError, ValueError):
            embedding_dim = 0
    if embedding_dim <= 0 or str(raw_dim) != str(embedding_dim):
        raise DSLValidationError([
            Diagnostic(
                "E_CATEGORICAL_EMBED_002",
                "embedding_dim must be a positive integer",
                node_id=node,
                actual=str(raw_dim),
            )
        ])
    axis_name = str(attrs.get("axis", "channel"))
    if not axis_name:
        raise DSLValidationError([
            Diagnostic("E_CATEGORICAL_EMBED_003", "embedding axis name must be nonempty", node_id=node)
        ])
    dtype = str(attrs.get("dtype", "float32"))
    if dtype not in ("float16", "bfloat16", "float32", "float64"):
        raise DSLValidationError([
            Diagnostic("E_CATEGORICAL_EMBED_004", "unsupported embedding dtype", node_id=node, actual=dtype)
        ])
    initializer = str(attrs.get("initializer", "normal_0_1"))
    if initializer not in ("normal_0_1", "normal_then_uniform"):
        raise DSLValidationError([
            Diagnostic(
                "E_CATEGORICAL_EMBED_005",
                "unsupported categorical embedding initializer",
                node_id=node,
                actual=initializer,
            )
        ])
    if initializer == "normal_then_uniform":
        try:
            init_min = float(attrs["init_min"])
            init_max = float(attrs["init_max"])
        except (KeyError, TypeError, ValueError):
            init_min = init_max = float("nan")
        if not math.isfinite(init_min) or not math.isfinite(init_max) or init_max <= init_min:
            raise DSLValidationError([
                Diagnostic(
                    "E_CATEGORICAL_EMBED_006",
                    "normal_then_uniform requires finite init_min < init_max",
                    node_id=node,
                )
            ])
    scalar = Irrep(0, 1, x.group.family)
    return {
        "out": InvariantTensorType(
            group=x.group,
            carrier=x.carrier,
            irreps=Irreps(((embedding_dim, scalar),)),
            frame=Frame("invariant"),
            axes=(axis_name,),
            dtype=dtype,
            measure="dimensionless",
            level=EquivarianceLevel.CORE_CERTIFIED,
            axis_specs=(AxisSpec(axis_name, embedding_dim, FeatureRole.CHANNEL, "independent", 0),),
            feature_role=FeatureRole.CHANNEL,
        )
    }, ()


def _categorical_embedding_parameters(node, inputs, outputs, attrs):
    x = _single(node, inputs, "x")
    output = outputs["out"]
    embedding_dim = output.irreps.dimension
    initializer = str(attrs.get("initializer", "normal_0_1"))
    if initializer == "normal_then_uniform":
        initializer = "normal_0_1_then_uniform_{:.17g}_{:.17g}".format(
            float(attrs["init_min"]), float(attrs["init_max"])
        )
    return (
        ParameterContract(
            "weight",
            (
                ParameterAxis("category", int(x.vocabulary_size), "categorical_vocabulary"),
                ParameterAxis("channel", int(embedding_dim), "embedding_channel"),
            ),
            initializer=initializer,
            checkpoint_names=("{}.weight".format(node),),
            backend_parameter_name="weight",
        ),
    )


def _categorical_embedding_v2(node, inputs, attrs):
    validated = dict(attrs)
    validated["initializer"] = "normal_then_uniform"
    return _categorical_embedding(node, inputs, validated)


def _categorical_embedding_v2_parameters(node, inputs, outputs, attrs):
    contracts = _categorical_embedding_parameters(node, inputs, outputs, attrs)
    return tuple(
        replace(contract, initializer="normal_0_1_then_uniform_at_scheduled_barrier")
        for contract in contracts
    )


def _invariant_concat(node, inputs, attrs):
    values = inputs.get("xs", ())
    if not values or any(not isinstance(value, InvariantTensorType) for value in values):
        raise DSLValidationError([
            Diagnostic(
                "E_INVARIANT_CONCAT_001",
                "invariant_concat requires one or more InvariantTensorType inputs",
                node_id=node,
                port="xs",
            )
        ])
    base = values[0]
    for value in values:
        if (
            value.group != base.group
            or value.carrier != base.carrier
            or value.frame != Frame("invariant")
            or value.dtype != base.dtype
            or value.measure != base.measure
        ):
            raise DSLValidationError([
                Diagnostic(
                    "E_INVARIANT_CONCAT_002",
                    "invariant_concat inputs must share group, carrier, invariant frame, dtype, and measure",
                    node_id=node,
                )
            ])
        axes = _concrete_invariant_axes(node, value, "invariant_concat")
        if len(axes) != 1 or int(axes[0].size) != value.irreps.dimension:
            raise DSLValidationError([
                Diagnostic(
                    "E_INVARIANT_CONCAT_003",
                    "invariant_concat first version requires one complete feature axis per input",
                    node_id=node,
                    actual=str([axis.to_dict() for axis in axes]),
                )
            ])
    axis_name = str(attrs["axis"])
    if not axis_name:
        raise DSLValidationError([
            Diagnostic("E_INVARIANT_CONCAT_004", "invariant_concat output axis must be nonempty", node_id=node)
        ])
    role = FeatureRole.parse(attrs.get("feature_role", FeatureRole.CHANNEL.value))
    total = sum(value.irreps.dimension for value in values)
    scalar = Irrep(0, 1, base.group.family)
    return {
        "out": InvariantTensorType(
            group=base.group,
            carrier=base.carrier,
            irreps=Irreps(((total, scalar),)),
            frame=Frame("invariant"),
            axes=(axis_name,),
            dtype=base.dtype,
            measure=base.measure,
            level=min(value.level for value in values),
            axis_specs=(AxisSpec(axis_name, total, role, "independent", 0),),
            feature_role=role,
        )
    }, ()


def _invariant_slice(node, inputs, attrs):
    x = _single(node, inputs, "x")
    axes = _concrete_invariant_axes(node, x, "invariant_slice")
    if len(axes) != 1:
        raise DSLValidationError([
            Diagnostic("E_INVARIANT_SLICE_001", "invariant_slice requires one concrete feature axis", node_id=node)
        ])
    try:
        start = int(attrs["start"])
        length = int(attrs["length"])
    except (TypeError, ValueError, KeyError):
        start, length = -1, -1
    if start < 0 or length <= 0 or start + length > int(axes[0].size):
        raise DSLValidationError([
            Diagnostic(
                "E_INVARIANT_SLICE_002",
                "invariant_slice range is outside the input feature axis",
                node_id=node,
                expected="0..{}".format(axes[0].size),
                actual="{}:{}".format(start, start + length),
            )
        ])
    axis_name = str(attrs.get("axis", axes[0].name))
    role = FeatureRole.parse(attrs.get("feature_role", axes[0].role.value))
    scalar = Irrep(0, 1, x.group.family)
    return {
        "out": replace(
            x,
            irreps=Irreps(((length, scalar),)),
            axes=(axis_name,),
            axis_specs=(AxisSpec(axis_name, length, role, "independent", 0),),
            feature_role=role,
        )
    }, ()


def _invariant_product(node, inputs, attrs):
    del attrs
    left = _single(node, inputs, "left")
    right = _single(node, inputs, "right")
    if not isinstance(left, InvariantTensorType) or not isinstance(right, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic("E_INVARIANT_PRODUCT_001", "invariant_product requires invariant tensor inputs", node_id=node)
        ])
    left_context = (left.group, left.carrier, left.frame, left.dtype, left.measure)
    right_context = (right.group, right.carrier, right.frame, right.dtype, right.measure)
    if left_context != right_context or left.measure != "dimensionless":
        raise DSLValidationError([
            Diagnostic(
                "E_INVARIANT_PRODUCT_002",
                "invariant_product inputs must share a dimensionless invariant context",
                node_id=node,
            )
        ])
    left_size = left.irreps.dimension
    right_size = right.irreps.dimension
    if left_size != right_size and left_size != 1 and right_size != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_INVARIANT_PRODUCT_003",
                "invariant_product supports equal feature sizes or scalar broadcasting",
                node_id=node,
                actual="{} and {}".format(left_size, right_size),
            )
        ])
    output = left if left_size >= right_size else right
    return {"out": replace(output, level=min(left.level, right.level))}, ()


def _equivariant_channel_concat(node, inputs, attrs):
    del attrs
    values = inputs.get("xs", ())
    if not values:
        raise DSLValidationError([
            Diagnostic("E_EQ_CHANNEL_CONCAT_001", "equivariant_channel_concat requires inputs", node_id=node)
        ])
    typed = tuple(_v2_equivariant(node, value, "xs") for value in values)
    base = typed[0]
    base_lmax, _base_channels = _uniform_so3_multiplicities(node, base.irreps)
    total_channels = 0
    for value in typed:
        _same_context(node, base, value, exact_irreps=False)
        lmax, channels = _uniform_so3_multiplicities(node, value.irreps)
        if lmax != base_lmax or tuple(ir for _, ir in value.irreps) != tuple(ir for _, ir in base.irreps):
            raise DSLValidationError([
                Diagnostic(
                    "E_EQ_CHANNEL_CONCAT_002",
                    "equivariant_channel_concat inputs must contain the same consecutive irrep kinds",
                    node_id=node,
                )
            ])
        if value.axis_specs:
            raise DSLValidationError([
                Diagnostic("E_EQ_CHANNEL_CONCAT_003", "equivariant_channel_concat requires axis-free inputs", node_id=node)
            ])
        total_channels += channels
    output_irreps = Irreps(
        tuple((total_channels, irrep) for _multiplicity, irrep in base.irreps)
    )
    return {"out": base.with_irreps(output_irreps)}, ()


def _degreewise_invariant_scale(node, inputs, attrs):
    del attrs
    weight = _single(node, inputs, "weight")
    value = _v2_equivariant(node, _single(node, inputs, "value"), "value")
    if not isinstance(weight, InvariantTensorType) or weight.carrier != value.carrier:
        raise DSLValidationError([
            Diagnostic("E_DEGREE_SCALE_001", "degreewise scale requires invariant weights on the value carrier", node_id=node)
        ])
    if weight.frame != Frame("invariant") or weight.measure != "dimensionless":
        raise DSLValidationError([
            Diagnostic("E_DEGREE_SCALE_002", "degreewise scale weights must be dimensionless invariants", node_id=node)
        ])
    value_context = (value.group, value.carrier, value.dtype)
    weight_context = (weight.group, weight.carrier, weight.dtype)
    if value_context != weight_context or value.axis_specs:
        raise DSLValidationError([
            Diagnostic("E_DEGREE_SCALE_003", "degreewise scale inputs have incompatible contexts or axes", node_id=node)
        ])
    lmax, channels = _uniform_so3_multiplicities(node, value.irreps)
    expected = (lmax + 1) * channels
    if weight.irreps.dimension != expected:
        raise DSLValidationError([
            Diagnostic(
                "E_DEGREE_SCALE_004",
                "degreewise scale requires one weight per degree and channel",
                node_id=node,
                expected=str(expected),
                actual=str(weight.irreps.dimension),
            )
        ])
    return {"out": replace(value, level=min(value.level, weight.level))}, ()


def _constant_scale(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if not isinstance(x, EquivariantTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_CONSTANT_SCALE_001",
                "constant_scale requires EquivariantTensorType input",
                node_id=node,
                actual=type(x).__name__,
            )
        ])
    try:
        factor = float(attrs["factor"])
    except (TypeError, ValueError):
        factor = float("nan")
    if not math.isfinite(factor):
        raise DSLValidationError([
            Diagnostic("E_CONSTANT_SCALE_002", "constant scale factor must be finite", node_id=node, actual=str(attrs["factor"]))
        ])
    return {"out": x}, ()


def _flatten_invariant_axes(node, inputs, attrs):
    del attrs
    x = _single(node, inputs, "x")
    axes = _concrete_invariant_axes(node, x, "flatten_invariant_axes")
    if math.prod(int(axis.size) for axis in axes) != x.irreps.dimension:
        raise DSLValidationError([
            Diagnostic("E_FLATTEN_INVARIANT_001", "invariant axes do not match scalar multiplicity", node_id=node)
        ])
    return {"out": replace(x, axes=(), axis_specs=())}, ()


def _irrep_linear(node, inputs, attrs):
    x = _single(node, inputs, "x")
    out = Irreps.parse(str(attrs["out_irreps"]), x.group.family)
    input_kinds = {ir for _, ir in x.irreps}
    missing = [str(ir) for _, ir in out if ir not in input_kinds]
    if missing:
        raise DSLValidationError([
            Diagnostic("E_IRREP_011", "equivariant linear map cannot create absent irrep kinds", node_id=node, details={"missing": missing})
        ])
    return {"out": x.with_irreps(out)}, ()


def _irrep_linear_v2(node, inputs, attrs):
    x = _v2_equivariant(node, _single(node, inputs, "x"), "x")
    outputs, obligations = _irrep_linear(node, inputs, attrs)
    if x.axis_specs:
        raise DSLValidationError([
            Diagnostic("E_LINEAR_RS_001", "irrep_linear@2 requires axis-free equivariant input", node_id=node)
        ])
    if (
        x.irreps != x.irreps.simplify()
        or x.layout.storage != "irrep_major"
        or x.layout.coefficient_order != "canonical"
    ):
        raise DSLValidationError([
            Diagnostic("E_LINEAR_RS_002", "irrep_linear@2 requires canonical simplified irrep-major input", node_id=node)
        ])
    output = outputs["out"]
    if output.irreps != output.irreps.simplify():
        raise DSLValidationError([
            Diagnostic("E_LINEAR_RS_003", "irrep_linear@2 output irreps must be simplified", node_id=node)
        ])
    for name in ("bias", "rescale"):
        value = attrs.get(name, True)
        if not isinstance(value, bool):
            raise DSLValidationError([
                Diagnostic("E_LINEAR_RS_004", "irrep_linear@2 {} attr must be boolean".format(name), node_id=node)
            ])
    try:
        initializer_scale = float(attrs.get("initializer_scale", 1.0))
    except (TypeError, ValueError):
        initializer_scale = float("nan")
    if not math.isfinite(initializer_scale) or initializer_scale <= 0.0:
        raise DSLValidationError([
            Diagnostic(
                "E_LINEAR_RS_005",
                "irrep_linear@2 initializer_scale must be finite and positive",
                node_id=node,
                actual=str(attrs.get("initializer_scale")),
            )
        ])
    return {"out": output}, obligations


def _linear_rs_weight_numel(input_irreps, output_irreps):
    return sum(
        input_mul * output_mul
        for input_mul, input_irrep in input_irreps
        for output_mul, output_irrep in output_irreps
        if input_irrep == output_irrep
    )


def _irrep_linear_v2_parameters(node, inputs, outputs, attrs):
    x = _single(node, inputs, "x")
    output = outputs["out"]
    weight_numel = _linear_rs_weight_numel(x.irreps, output.irreps)
    initializer_scale = float(attrs.get("initializer_scale", 1.0))
    base_initializer = (
        "uniform_minus1_1_instruction_fan_in_scaled"
        if attrs.get("rescale", True)
        else "uniform_minus1_1"
    )
    contracts = [
        ParameterContract(
            "weight",
            (ParameterAxis("tp_path", weight_numel, "linear_tensor_product_path"),),
            initializer=(
                base_initializer
                if initializer_scale == 1.0
                else base_initializer + "_times_constant"
            ),
            rescale=initializer_scale,
            checkpoint_names=("{}.tp.weight".format(node),),
            backend_parameter_name="tp.weight",
        )
    ]
    if attrs.get("bias", True):
        bias_index = 0
        for multiplicity, irrep in output.irreps:
            if irrep.degree != 0 or irrep.parity != 1:
                continue
            contracts.append(
                ParameterContract(
                    "bias_{}".format(bias_index),
                    (ParameterAxis("multiplicity", multiplicity, "trivial_scalar_bias"),),
                    is_bias=True,
                    bias_irreps=str(Irreps(((multiplicity, irrep),))),
                    initializer="zeros",
                    checkpoint_names=("{}.bias.{}".format(node, bias_index),),
                    backend_parameter_name="bias.{}".format(bias_index),
                )
            )
            bias_index += 1
    return tuple(contracts)


def _irrep_pad(node, inputs, attrs):
    x = _v2_equivariant(node, _single(node, inputs, "x"), "x")
    if x.axis_specs:
        raise DSLValidationError([
            Diagnostic("E_IRREP_PAD_001", "irrep_pad requires axis-free equivariant input", node_id=node)
        ])
    if x.irreps != x.irreps.simplify() or x.layout.storage != "irrep_major" or x.layout.coefficient_order != "canonical":
        raise DSLValidationError([
            Diagnostic("E_IRREP_PAD_002", "irrep_pad requires canonical simplified irrep-major input", node_id=node)
        ])
    output_irreps = Irreps.parse(str(attrs["out_irreps"]), x.group.family)
    if output_irreps != output_irreps.simplify():
        raise DSLValidationError([
            Diagnostic("E_IRREP_PAD_003", "irrep_pad output irreps must be simplified", node_id=node)
        ])
    available = {irrep: multiplicity for multiplicity, irrep in output_irreps}
    missing = []
    narrowed = []
    for multiplicity, irrep in x.irreps:
        output_multiplicity = available.get(irrep)
        if output_multiplicity is None:
            missing.append(str(irrep))
        elif output_multiplicity < multiplicity:
            narrowed.append({"irrep": str(irrep), "input": multiplicity, "output": output_multiplicity})
    if missing or narrowed:
        raise DSLValidationError([
            Diagnostic(
                "E_IRREP_PAD_004",
                "irrep_pad output must contain every input irrep with at least its input multiplicity",
                node_id=node,
                details={"missing": missing, "narrowed": narrowed},
            )
        ])
    if isinstance(x, InvariantTensorType) and any(
        irrep.degree != 0 or irrep.parity != 1
        for _multiplicity, irrep in output_irreps
    ):
        output = EquivariantTensorType(
            group=x.group,
            carrier=x.carrier,
            irreps=output_irreps,
            frame=Frame("global"),
            axes=x.axes,
            dtype=x.dtype,
            measure=x.measure,
            level=x.level,
            axis_specs=x.axis_specs,
            layout=x.layout,
        )
    else:
        output = replace(x, irreps=output_irreps)
    return {"out": output}, ()


def _scalar_linear(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if not isinstance(x, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_SCALAR_LINEAR_001",
                "scalar_linear requires InvariantTensorType input",
                node_id=node,
                actual=type(x).__name__,
            )
        ])
    if not x.axis_specs:
        raise DSLValidationError([
            Diagnostic("E_SCALAR_LINEAR_002", "scalar_linear requires explicit feature axes", node_id=node)
        ])
    axis_name = str(attrs["axis"])
    matching = [axis for axis in x.axis_specs if axis.name == axis_name]
    if len(matching) != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_SCALAR_LINEAR_003",
                "scalar_linear axis must name exactly one input feature axis",
                node_id=node,
                actual=axis_name,
                details={"available_axes": [axis.name for axis in x.axis_specs]},
            )
        ])
    sizes = [axis.size for axis in x.axis_specs]
    if any(size is None for size in sizes):
        raise DSLValidationError([
            Diagnostic("E_SCALAR_LINEAR_004", "scalar_linear requires statically known feature-axis sizes", node_id=node)
        ])
    if math.prod(int(size) for size in sizes) != x.irreps.dimension:
        raise DSLValidationError([
            Diagnostic(
                "E_SCALAR_LINEAR_005",
                "product of feature-axis sizes must equal the invariant scalar multiplicity",
                node_id=node,
                expected=str(x.irreps.dimension),
                actual=str(math.prod(int(size) for size in sizes)),
            )
        ])
    raw_out = attrs["out_features"]
    if isinstance(raw_out, bool):
        raise DSLValidationError([
            Diagnostic("E_SCALAR_LINEAR_006", "out_features must be a positive integer", node_id=node, actual=str(raw_out))
        ])
    try:
        out_features = int(raw_out)
    except (TypeError, ValueError):
        raise DSLValidationError([
            Diagnostic("E_SCALAR_LINEAR_006", "out_features must be a positive integer", node_id=node, actual=str(raw_out))
        ])
    if out_features <= 0 or str(raw_out) != str(out_features):
        raise DSLValidationError([
            Diagnostic("E_SCALAR_LINEAR_006", "out_features must be a positive integer", node_id=node, actual=str(raw_out))
        ])
    bias = attrs.get("bias", True)
    if not isinstance(bias, bool):
        raise DSLValidationError([
            Diagnostic("E_SCALAR_LINEAR_007", "bias must be boolean", node_id=node, actual=str(bias))
        ])
    output_axes = tuple(
        replace(axis, size=out_features) if axis.name == axis_name else axis
        for axis in x.axis_specs
    )
    output_multiplicity = math.prod(int(axis.size) for axis in output_axes)
    output_irreps = Irreps(((output_multiplicity, Irrep(0, 1, x.group.family)),))
    return {"out": replace(x, irreps=output_irreps, axis_specs=output_axes)}, ()


def _scalar_linear_parameters(node, inputs, outputs, attrs):
    x = _single(node, inputs, "x")
    output = outputs["out"]
    axis_name = str(attrs["axis"])
    input_axis = next(axis for axis in x.axis_specs if axis.name == axis_name)
    output_axis = next(axis for axis in output.axis_specs if axis.name == axis_name)
    sharing_axes = tuple(axis.name for axis in x.axis_specs if axis.name != axis_name)
    contracts = [
        ParameterContract(
            "weight",
            (
                ParameterAxis("out_feature", int(output_axis.size), "output_feature", axis_name),
                ParameterAxis("in_feature", int(input_axis.size), "input_feature", axis_name),
            ),
            sharing_axes=sharing_axes,
            initializer="kaiming_uniform",
            checkpoint_names=("{}.linear.weight".format(node),),
            backend_parameter_name="linear.weight",
        )
    ]
    if attrs.get("bias", True):
        contracts.append(
            ParameterContract(
                "bias",
                (ParameterAxis("out_feature", int(output_axis.size), "output_feature", axis_name),),
                sharing_axes=sharing_axes,
                is_bias=True,
                bias_irreps=str(Irreps(((int(output_axis.size), Irrep(0, 1, x.group.family)),))),
                initializer="uniform_fan_in",
                checkpoint_names=("{}.linear.bias".format(node),),
                backend_parameter_name="linear.bias",
            )
        )
    return tuple(contracts)


def _initializer_scales(node, attrs, size, code):
    raw = attrs.get("initializer_scales")
    if raw is None:
        return (1.0,) * int(size)
    if not isinstance(raw, (list, tuple)) or len(raw) != int(size):
        raise DSLValidationError([
            Diagnostic(
                code,
                "initializer_scales must contain exactly one value per output feature",
                node_id=node,
                expected=str(int(size)),
                actual=str(raw),
            )
        ])
    values = []
    for value in raw:
        try:
            scale = float(value)
        except (TypeError, ValueError):
            scale = float("nan")
        if not math.isfinite(scale) or scale <= 0.0:
            raise DSLValidationError([
                Diagnostic(code, "initializer scales must be finite and positive", node_id=node, actual=str(value))
            ])
        values.append(scale)
    return tuple(values)


def _scalar_linear_v2(node, inputs, attrs):
    outputs, obligations = _scalar_linear(node, inputs, attrs)
    output = outputs["out"]
    source_axis_name = str(attrs["axis"])
    target_axis_name = str(attrs.get("out_axis", source_axis_name))
    if not target_axis_name:
        raise DSLValidationError([
            Diagnostic("E_SCALAR_LINEAR_V2_001", "scalar_linear@2 out_axis must be nonempty", node_id=node)
        ])
    collisions = {
        axis.name for axis in output.axis_specs
        if axis.name != source_axis_name and axis.name == target_axis_name
    }
    if collisions:
        raise DSLValidationError([
            Diagnostic(
                "E_SCALAR_LINEAR_V2_002",
                "scalar_linear@2 out_axis collides with an existing feature axis",
                node_id=node,
                actual=target_axis_name,
            )
        ])
    output_axis_role = FeatureRole.parse(
        attrs.get(
            "output_axis_role",
            next(axis.role for axis in output.axis_specs if axis.name == source_axis_name),
        )
    )
    output_feature_role = FeatureRole.parse(attrs.get("output_feature_role", output.feature_role))
    output_axes = tuple(
        replace(axis, name=target_axis_name, role=output_axis_role)
        if axis.name == source_axis_name else axis
        for axis in output.axis_specs
    )
    out_features = next(int(axis.size) for axis in output_axes if axis.name == target_axis_name)
    _initializer_scales(node, attrs, out_features, "E_SCALAR_LINEAR_V2_003")
    return {
        "out": replace(
            output,
            axes=tuple(axis.name for axis in output_axes),
            axis_specs=output_axes,
            feature_role=output_feature_role,
        )
    }, obligations


def _scalar_linear_v2_parameters(node, inputs, outputs, attrs):
    x = _single(node, inputs, "x")
    output = outputs["out"]
    source_axis_name = str(attrs["axis"])
    target_axis_name = str(attrs.get("out_axis", source_axis_name))
    input_axis = next(axis for axis in x.axis_specs if axis.name == source_axis_name)
    output_axis = next(axis for axis in output.axis_specs if axis.name == target_axis_name)
    sharing_axes = tuple(axis.name for axis in x.axis_specs if axis.name != source_axis_name)
    scales = _initializer_scales(node, attrs, int(output_axis.size), "E_SCALAR_LINEAR_V2_003")
    common_rescale = scales[0] if all(value == scales[0] for value in scales) else 1.0
    initializer = "kaiming_uniform" if all(value == 1.0 for value in scales) else "kaiming_uniform_output_scaled"
    contracts = [
        ParameterContract(
            "weight",
            (
                ParameterAxis("out_feature", int(output_axis.size), "output_feature", target_axis_name),
                ParameterAxis("in_feature", int(input_axis.size), "input_feature", source_axis_name),
            ),
            sharing_axes=sharing_axes,
            initializer=initializer,
            rescale=common_rescale,
            checkpoint_names=("{}.linear.weight".format(node),),
            backend_parameter_name="linear.weight",
        )
    ]
    if attrs.get("bias", True):
        contracts.append(
            ParameterContract(
                "bias",
                (ParameterAxis("out_feature", int(output_axis.size), "output_feature", target_axis_name),),
                sharing_axes=sharing_axes,
                is_bias=True,
                bias_irreps=str(Irreps(((int(output_axis.size), Irrep(0, 1, x.group.family)),))),
                initializer="uniform_fan_in" if all(value == 1.0 for value in scales) else "uniform_fan_in_output_scaled",
                rescale=common_rescale,
                checkpoint_names=("{}.linear.bias".format(node),),
                backend_parameter_name="linear.bias",
            )
        )
    return tuple(contracts)


def _scalar_linear_v3(node, inputs, attrs):
    return _scalar_linear_v2(node, inputs, attrs)


def _scalar_linear_v3_parameters(node, inputs, outputs, attrs):
    contracts = _scalar_linear_v2_parameters(node, inputs, outputs, attrs)
    rewritten = []
    for contract in contracts:
        initializer = (
            "uniform_fan_in_then_zeros_deferred"
            if contract.is_bias
            else "kaiming_uniform_then_uniform_fan_in_deferred"
        )
        rewritten.append(replace(contract, initializer=initializer, rescale=1.0))
    return tuple(rewritten)


def _scalar_linear_v4(node, inputs, attrs):
    """Official V3 non-radial Linear: preserve torch weight, zero bias later."""

    return _scalar_linear_v2(node, inputs, attrs)


def _scalar_linear_v4_parameters(node, inputs, outputs, attrs):
    contracts = _scalar_linear_v2_parameters(node, inputs, outputs, attrs)
    rewritten = []
    for contract in contracts:
        initializer = "uniform_fan_in_then_zeros_deferred" if contract.is_bias else "kaiming_uniform_preserved"
        rewritten.append(replace(contract, initializer=initializer, rescale=1.0))
    return tuple(rewritten)


def _concrete_invariant_axes(node, value, operation):
    if not isinstance(value, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_001",
                "{} requires InvariantTensorType input in its first version".format(operation),
                node_id=node,
                actual=type(value).__name__,
            )
        ])
    if not value.axis_specs or any(axis.size is None for axis in value.axis_specs):
        raise DSLValidationError([
            Diagnostic("E_HEAD_002", "{} requires explicit statically sized axes".format(operation), node_id=node)
        ])
    product = math.prod(int(axis.size) for axis in value.axis_specs)
    if product != value.irreps.dimension:
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_003",
                "{} axis-size product must equal invariant scalar multiplicity".format(operation),
                node_id=node,
                expected=str(value.irreps.dimension),
                actual=str(product),
            )
        ])
    return tuple(sorted(value.axis_specs, key=lambda axis: axis.order))


def _scalar_layer_norm(node, inputs, attrs):
    x = _single(node, inputs, "x")
    axes = _concrete_invariant_axes(node, x, "scalar_layer_norm")
    axis_name = str(attrs["axis"])
    if len([axis for axis in axes if axis.name == axis_name]) != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_SCALAR_NORM_001",
                "scalar_layer_norm axis must name exactly one invariant feature axis",
                node_id=node,
                actual=axis_name,
                details={"available_axes": [axis.name for axis in axes]},
            )
        ])
    try:
        epsilon = float(attrs.get("epsilon", 1.0e-5))
    except (TypeError, ValueError):
        epsilon = float("nan")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise DSLValidationError([
            Diagnostic("E_SCALAR_NORM_002", "scalar_layer_norm epsilon must be finite and positive", node_id=node)
        ])
    affine = attrs.get("affine", True)
    bias = attrs.get("bias", True)
    if not isinstance(affine, bool) or not isinstance(bias, bool):
        raise DSLValidationError([
            Diagnostic("E_SCALAR_NORM_003", "scalar_layer_norm affine and bias attrs must be boolean", node_id=node)
        ])
    if bias and not affine:
        raise DSLValidationError([
            Diagnostic("E_SCALAR_NORM_004", "scalar_layer_norm bias requires affine=True", node_id=node)
        ])
    if x.measure != "dimensionless":
        raise DSLValidationError([
            Diagnostic("E_SCALAR_NORM_005", "scalar_layer_norm requires dimensionless invariant features", node_id=node, actual=x.measure)
        ])
    return {"out": x}, ()


def _scalar_layer_norm_parameters(node, inputs, outputs, attrs):
    del outputs
    x = _single(node, inputs, "x")
    if not attrs.get("affine", True):
        return ()
    axis_name = str(attrs["axis"])
    axis = next(item for item in x.axis_specs if item.name == axis_name)
    sharing_axes = tuple(item.name for item in x.axis_specs if item.name != axis_name)
    contracts = [
        ParameterContract(
            "weight",
            (ParameterAxis("feature", int(axis.size), "normalized_feature", axis_name),),
            sharing_axes=sharing_axes,
            initializer="ones",
            checkpoint_names=("{}.layer_norm.weight".format(node),),
            backend_parameter_name="layer_norm.weight",
        )
    ]
    if attrs.get("bias", True):
        contracts.append(
            ParameterContract(
                "bias",
                (ParameterAxis("feature", int(axis.size), "normalized_feature", axis_name),),
                sharing_axes=sharing_axes,
                is_bias=True,
                bias_irreps=str(Irreps(((int(axis.size), Irrep(0, 1, x.group.family)),))),
                initializer="zeros",
                checkpoint_names=("{}.layer_norm.bias".format(node),),
                backend_parameter_name="layer_norm.bias",
            )
        )
    return tuple(contracts)


def _scalar_offset(node, inputs, attrs):
    x = _single(node, inputs, "x")
    axes = _concrete_invariant_axes(node, x, "scalar_offset")
    axis_name = str(attrs["axis"])
    matches = [axis for axis in axes if axis.name == axis_name]
    if len(matches) != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_SCALAR_OFFSET_001",
                "scalar_offset axis must name exactly one invariant feature axis",
                node_id=node,
                actual=axis_name,
            )
        ])
    fan_in = _positive_int_attr(node, attrs, "fan_in", "E_SCALAR_OFFSET_002")
    del fan_in
    _initializer_scales(node, attrs, int(matches[0].size), "E_SCALAR_OFFSET_003")
    if x.measure != "dimensionless":
        raise DSLValidationError([
            Diagnostic("E_SCALAR_OFFSET_004", "scalar_offset requires dimensionless invariant features", node_id=node, actual=x.measure)
        ])
    return {"out": x}, ()


def _scalar_offset_parameters(node, inputs, outputs, attrs):
    del outputs
    x = _single(node, inputs, "x")
    axis_name = str(attrs["axis"])
    axis = next(item for item in x.axis_specs if item.name == axis_name)
    sharing_axes = tuple(item.name for item in x.axis_specs if item.name != axis_name)
    scales = _initializer_scales(node, attrs, int(axis.size), "E_SCALAR_OFFSET_003")
    common_rescale = scales[0] if all(value == scales[0] for value in scales) else 1.0
    return (
        ParameterContract(
            "offset",
            (ParameterAxis("feature", int(axis.size), "output_feature", axis_name),),
            sharing_axes=sharing_axes,
            is_bias=True,
            bias_irreps=str(Irreps(((int(axis.size), Irrep(0, 1, x.group.family)),))),
            initializer="uniform_fan_in" if all(value == 1.0 for value in scales) else "uniform_fan_in_output_scaled",
            rescale=common_rescale,
            checkpoint_names=("{}.offset".format(node),),
            backend_parameter_name="offset",
        ),
    )


def _positive_int_attr(node, attrs, name, code):
    raw = attrs[name]
    if isinstance(raw, bool):
        raise DSLValidationError([Diagnostic(code, "{} must be a positive integer".format(name), node_id=node, actual=str(raw))])
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise DSLValidationError([Diagnostic(code, "{} must be a positive integer".format(name), node_id=node, actual=str(raw))])
    if value <= 0 or str(raw) != str(value):
        raise DSLValidationError([Diagnostic(code, "{} must be a positive integer".format(name), node_id=node, actual=str(raw))])
    return value


def _head_split(node, inputs, attrs):
    x = _single(node, inputs, "x")
    axes = _concrete_invariant_axes(node, x, "head_split")
    source_name = str(attrs["axis"])
    head_name = str(attrs["head_axis"])
    channel_name = str(attrs["channel_axis"])
    if not head_name or not channel_name or head_name == channel_name:
        raise DSLValidationError([
            Diagnostic("E_HEAD_004", "head_split output axis names must be nonempty and distinct", node_id=node)
        ])
    matches = [axis for axis in axes if axis.name == source_name]
    if len(matches) != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_005",
                "head_split source axis must name exactly one input axis",
                node_id=node,
                actual=source_name,
                details={"available_axes": [axis.name for axis in axes]},
            )
        ])
    source = matches[0]
    if source.role == FeatureRole.HEAD or source.sharing != "independent":
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_006",
                "head_split source must be an independently stored non-head feature axis",
                node_id=node,
                actual=source.name,
            )
        ])
    existing = {axis.name for axis in axes if axis.name != source_name}
    collisions = existing.intersection((head_name, channel_name))
    if collisions:
        raise DSLValidationError([
            Diagnostic("E_HEAD_007", "head_split output axes collide with existing axes", node_id=node, details={"axes": sorted(collisions)})
        ])
    num_heads = _positive_int_attr(node, attrs, "num_heads", "E_HEAD_008")
    if int(source.size) % num_heads != 0:
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_009",
                "head_split source axis size must be divisible by num_heads",
                node_id=node,
                expected="multiple of {}".format(num_heads),
                actual=str(source.size),
            )
        ])
    output_axes = []
    for axis in axes:
        if axis.name != source_name:
            output_axes.append(replace(axis, order=len(output_axes)))
            continue
        output_axes.append(AxisSpec(head_name, num_heads, FeatureRole.HEAD, "independent", len(output_axes), False))
        output_axes.append(
            AxisSpec(
                channel_name,
                int(source.size) // num_heads,
                source.role,
                "per_head",
                len(output_axes),
                source.broadcastable,
            )
        )
    output_axes = tuple(output_axes)
    return {"out": replace(x, axes=tuple(axis.name for axis in output_axes), axis_specs=output_axes)}, ()


def _head_merge(node, inputs, attrs):
    x = _single(node, inputs, "x")
    axes = _concrete_invariant_axes(node, x, "head_merge")
    head_name = str(attrs["head_axis"])
    channel_name = str(attrs["channel_axis"])
    output_name = str(attrs["out_axis"])
    axis_map = {axis.name: axis for axis in axes}
    if head_name not in axis_map or channel_name not in axis_map:
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_010",
                "head_merge requires named head and per-head channel axes",
                node_id=node,
                details={"available_axes": sorted(axis_map)},
            )
        ])
    head = axis_map[head_name]
    channel = axis_map[channel_name]
    if head.role != FeatureRole.HEAD or channel.sharing != "per_head" or channel.order != head.order + 1:
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_011",
                "head_merge requires adjacent head then per-head channel axes",
                node_id=node,
            )
        ])
    if not output_name or (output_name in axis_map and output_name not in (head_name, channel_name)):
        raise DSLValidationError([
            Diagnostic("E_HEAD_012", "head_merge output axis name is empty or collides with another axis", node_id=node, actual=output_name)
        ])
    output_axes = []
    for axis in axes:
        if axis.name == head_name:
            output_axes.append(
                AxisSpec(
                    output_name,
                    int(head.size) * int(channel.size),
                    channel.role,
                    "independent",
                    len(output_axes),
                    channel.broadcastable,
                )
            )
        elif axis.name == channel_name:
            continue
        else:
            output_axes.append(replace(axis, order=len(output_axes)))
    output_axes = tuple(output_axes)
    return {"out": replace(x, axes=tuple(axis.name for axis in output_axes), axis_specs=output_axes)}, ()


def _equivariant_head_input(node, value, operation):
    if not isinstance(value, EquivariantTensorType) or isinstance(value, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_V2_001",
                "{} requires a non-invariant EquivariantTensorType".format(operation),
                node_id=node,
                actual=type(value).__name__,
            )
        ])
    if not value.irreps:
        raise DSLValidationError([
            Diagnostic("E_HEAD_V2_002", "{} requires one or more irrep blocks".format(operation), node_id=node)
        ])
    if value.irreps != value.irreps.simplify():
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_V2_013",
                "{} requires canonical simplified irrep blocks".format(operation),
                node_id=node,
                expected=str(value.irreps.simplify()),
                actual=str(value.irreps),
            )
        ])
    if value.layout.storage != "irrep_major" or value.layout.coefficient_order != "canonical":
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_V2_014",
                "{} requires canonical irrep-major coefficient storage".format(operation),
                node_id=node,
                expected="irrep_major/canonical",
                actual="{}/{}".format(value.layout.storage, value.layout.coefficient_order),
            )
        ])
    return value


def _head_split_v2(node, inputs, attrs):
    x = _equivariant_head_input(node, _single(node, inputs, "x"), "head_split@2")
    if x.axis_specs:
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_V2_003",
                "head_split@2 first version requires axis-free flattened irrep storage",
                node_id=node,
                details={"axes": list(x.axes)},
            )
        ])
    head_name = str(attrs["head_axis"])
    if not head_name:
        raise DSLValidationError([
            Diagnostic("E_HEAD_V2_004", "head_split@2 head_axis must be nonempty", node_id=node)
        ])
    num_heads = _positive_int_attr(node, attrs, "num_heads", "E_HEAD_V2_005")
    invalid = [
        "{}x{}".format(multiplicity, irrep)
        for multiplicity, irrep in x.irreps
        if multiplicity % num_heads != 0
    ]
    if invalid:
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_V2_006",
                "every irrep multiplicity must be divisible by num_heads",
                node_id=node,
                expected="each multiplicity divisible by {}".format(num_heads),
                details={"invalid_blocks": invalid},
            )
        ])
    per_head = Irreps(
        tuple((multiplicity // num_heads, irrep) for multiplicity, irrep in x.irreps)
    ).simplify()
    head_axis = AxisSpec(head_name, num_heads, FeatureRole.HEAD, "independent", 0, False)
    return {
        "out": replace(x, irreps=per_head, axes=(head_name,), axis_specs=(head_axis,))
    }, ()


def _head_merge_v2(node, inputs, attrs):
    x = _equivariant_head_input(node, _single(node, inputs, "x"), "head_merge@2")
    head_name = str(attrs["head_axis"])
    if len(x.axis_specs) != 1 or x.axis_specs[0].name != head_name or x.axis_specs[0].role != FeatureRole.HEAD:
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_V2_007",
                "head_merge@2 requires exactly one named head axis",
                node_id=node,
                expected=head_name,
                actual=str([(axis.name, axis.role.value) for axis in x.axis_specs]),
            )
        ])
    head = x.axis_specs[0]
    if head.size is None or head.sharing != "independent":
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_V2_008",
                "head_merge@2 requires a statically sized independent head axis",
                node_id=node,
                actual=str(head.to_dict()),
            )
        ])
    merged = Irreps(
        tuple((int(head.size) * multiplicity, irrep) for multiplicity, irrep in x.irreps)
    ).simplify()
    return {"out": replace(x, irreps=merged, axes=(), axis_specs=())}, ()


def _headwise_scalar_contraction(node, inputs, attrs):
    x = _single(node, inputs, "x")
    axes = _concrete_invariant_axes(node, x, "headwise_scalar_contraction")
    head_name = str(attrs["head_axis"])
    channel_name = str(attrs["channel_axis"])
    if len(axes) != 2 or tuple(axis.name for axis in axes) != (head_name, channel_name):
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_013",
                "first headwise_scalar_contraction version requires exactly head then channel axes",
                node_id=node,
                expected=str((head_name, channel_name)),
                actual=str(tuple(axis.name for axis in axes)),
            )
        ])
    head, channel = axes
    if head.role != FeatureRole.HEAD or channel.sharing != "per_head":
        raise DSLValidationError([
            Diagnostic("E_HEAD_014", "headwise contraction requires a head axis and per-head channel axis", node_id=node)
        ])
    bias = attrs.get("bias", True)
    if not isinstance(bias, bool):
        raise DSLValidationError([
            Diagnostic("E_HEAD_015", "headwise contraction bias must be boolean", node_id=node, actual=str(bias))
        ])
    output_axis = replace(head, order=0)
    output_irreps = Irreps(((int(head.size), Irrep(0, 1, x.group.family)),))
    return {
        "out": replace(
            x,
            irreps=output_irreps,
            axes=(head.name,),
            axis_specs=(output_axis,),
            feature_role=FeatureRole.ALPHA,
        )
    }, ()


def _headwise_scalar_contraction_parameters(node, inputs, outputs, attrs):
    x = _single(node, inputs, "x")
    axis_map = {axis.name: axis for axis in x.axis_specs}
    head = axis_map[str(attrs["head_axis"])]
    channel = axis_map[str(attrs["channel_axis"])]
    contracts = [
        ParameterContract(
            "weight",
            (
                ParameterAxis("head", int(head.size), "head", head.name),
                ParameterAxis("in_feature", int(channel.size), "input_feature", channel.name),
            ),
            initializer="kaiming_uniform",
            checkpoint_names=("{}.weight".format(node),),
            backend_parameter_name="weight",
        )
    ]
    if attrs.get("bias", True):
        contracts.append(
            ParameterContract(
                "bias",
                (ParameterAxis("head", int(head.size), "head", head.name),),
                is_bias=True,
                bias_irreps=str(Irreps(((int(head.size), Irrep(0, 1, x.group.family)),))),
                initializer="uniform_fan_in",
                checkpoint_names=("{}.bias".format(node),),
                backend_parameter_name="bias",
            )
        )
    return tuple(contracts)


def _headwise_scalar_contraction_v2(node, inputs, attrs):
    x = _equivariant_head_input(
        node,
        _single(node, inputs, "x"),
        "headwise_scalar_contraction@2",
    )
    head_name = str(attrs["head_axis"])
    if len(x.axis_specs) != 1 or x.axis_specs[0].name != head_name or x.axis_specs[0].role != FeatureRole.HEAD:
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_V2_009",
                "headwise_scalar_contraction@2 requires exactly one named head axis",
                node_id=node,
                expected=head_name,
                actual=str([(axis.name, axis.role.value) for axis in x.axis_specs]),
            )
        ])
    nontrivial = [str(irrep) for _, irrep in x.irreps if irrep.degree != 0 or irrep.parity != 1]
    if nontrivial:
        raise DSLValidationError([
            Diagnostic(
                "E_HEAD_V2_010",
                "headwise_scalar_contraction@2 accepts only trivial per-head scalar irreps",
                node_id=node,
                details={"nontrivial_irreps": nontrivial},
            )
        ])
    head = x.axis_specs[0]
    if head.size is None:
        raise DSLValidationError([
            Diagnostic("E_HEAD_V2_011", "headwise_scalar_contraction@2 requires a static head count", node_id=node)
        ])
    bias = attrs.get("bias", False)
    if not isinstance(bias, bool):
        raise DSLValidationError([
            Diagnostic("E_HEAD_V2_012", "headwise scalar contraction bias must be boolean", node_id=node, actual=str(bias))
        ])
    output_irreps = Irreps(((int(head.size), Irrep(0, 1, x.group.family)),))
    output = InvariantTensorType(
        group=x.group,
        carrier=x.carrier,
        irreps=output_irreps,
        frame=x.frame,
        axes=(head.name,),
        dtype=x.dtype,
        measure=x.measure,
        level=x.level,
        axis_specs=(replace(head, order=0),),
        layout=x.layout,
        feature_role=FeatureRole.ALPHA,
    )
    return {"out": output}, ()


def _headwise_scalar_contraction_v2_parameters(node, inputs, outputs, attrs):
    x = _single(node, inputs, "x")
    head = x.axis_specs[0]
    channel_size = x.irreps.dimension
    contracts = [
        ParameterContract(
            "weight",
            (
                ParameterAxis("carrier_broadcast", 1, "carrier_broadcast", x.carrier),
                ParameterAxis("head", int(head.size), "head", head.name),
                ParameterAxis("in_feature", channel_size, "input_feature", "per_head_irrep_scalar"),
            ),
            sharing_axes=(x.carrier,),
            initializer="glorot_uniform",
            checkpoint_names=("{}.alpha_dot".format(node), "{}.weight".format(node)),
            backend_parameter_name="weight",
        )
    ]
    if attrs.get("bias", False):
        contracts.append(
            ParameterContract(
                "bias",
                (ParameterAxis("head", int(head.size), "head", head.name),),
                is_bias=True,
                bias_irreps=str(outputs["out"].irreps),
                initializer="zeros",
                checkpoint_names=("{}.bias".format(node),),
                backend_parameter_name="bias",
            )
        )
    return tuple(contracts)


def _headwise_scalar_contraction_v3(node, inputs, attrs):
    validated = dict(attrs)
    validated["bias"] = False
    return _headwise_scalar_contraction(node, inputs, validated)


def _headwise_scalar_contraction_v3_parameters(node, inputs, outputs, attrs):
    del outputs
    x = _single(node, inputs, "x")
    axis_map = {axis.name: axis for axis in x.axis_specs}
    head = axis_map[str(attrs["head_axis"])]
    channel = axis_map[str(attrs["channel_axis"])]
    return (
        ParameterContract(
            "weight",
            (
                ParameterAxis("head", int(head.size), "head", head.name),
                ParameterAxis("in_feature", int(channel.size), "input_feature", channel.name),
            ),
            initializer="normal_0_1_then_uniform_fan_in",
            checkpoint_names=("{}.alpha_dot".format(node), "{}.weight".format(node)),
            backend_parameter_name="weight",
        ),
    )


def _irrep_concat(node, inputs, attrs):
    values = inputs.get("xs", ())
    if not values:
        raise DSLValidationError([Diagnostic("E_PORT_002", "irrep_concat requires at least one value", node_id=node, port="xs")])
    base = values[0]
    output = base.irreps
    for value in values[1:]:
        _same_context(node, base, value, exact_irreps=False)
        output = output.direct_sum(value.irreps)
    return {"out": base.with_irreps(output)}, ()


def _residual_add(node, inputs, attrs):
    left = _single(node, inputs, "left")
    right = _single(node, inputs, "right")
    _same_context(node, left, right)
    return {"out": left}, ()


def _residual_add_v2(node, inputs, attrs):
    left = _v2_equivariant(node, _single(node, inputs, "left"), "left")
    right = _v2_equivariant(node, _single(node, inputs, "right"), "right")
    left_without_level = replace(left, level=EquivarianceLevel.UNVERIFIED)
    right_without_level = replace(right, level=EquivarianceLevel.UNVERIFIED)
    if left_without_level != right_without_level:
        raise DSLValidationError([
            Diagnostic(
                "E_RESIDUAL_V2_001",
                "residual_add@2 inputs may differ only in certification level",
                node_id=node,
                expected=str(left.to_dict()),
                actual=str(right.to_dict()),
            )
        ])
    return {"out": replace(left, level=min(left.level, right.level))}, ()


def _tensor_product(node, inputs, attrs):
    left = _single(node, inputs, "left")
    right = _single(node, inputs, "right")
    _same_context(node, left, right, exact_irreps=False)
    out = Irreps.parse(str(attrs["out_irreps"]), left.group.family)
    allowed = set(left.irreps.allowed_tensor_product_outputs(right.irreps))
    invalid = [str(ir) for _, ir in out if ir not in allowed]
    if invalid:
        raise DSLValidationError([
            Diagnostic("E_IRREP_012", "tensor-product output has no legal coupling path", node_id=node, details={"invalid_outputs": invalid, "allowed": [str(item) for item in sorted(allowed)]})
        ])
    obligations = (
        ProofObligation("{}:paths".format(node), ObligationKind.IRREP_PATH_EXISTS, node, "discharged", "core.tensor_product@1"),
        ProofObligation("{}:parity".format(node), ObligationKind.PARITY_MATCH, node, "discharged", "core.tensor_product@1"),
    )
    return {"out": left.with_irreps(out).with_measure(_multiply_measures(left.measure, right.measure))}, obligations


def _fully_connected_tp_weight_numel(left: Irreps, right: Irreps, output: Irreps) -> int:
    total = 0
    for left_mul, left_irrep in left:
        for right_mul, right_irrep in right:
            allowed = set(left_irrep.tensor_product(right_irrep))
            for output_mul, output_irrep in output:
                if output_irrep in allowed:
                    total += left_mul * right_mul * output_mul
    return total


def _tensor_product_v2(node, inputs, attrs):
    left = _v2_equivariant(node, _single(node, inputs, "left"), "left")
    right = _v2_equivariant(node, _single(node, inputs, "right"), "right")
    weight = _single(node, inputs, "weight")
    if not isinstance(weight, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic("E_TP_V2_001", "tensor_product@2 weight must use InvariantTensorType", node_id=node, port="weight", actual=type(weight).__name__)
        ])
    if left.axis_specs or right.axis_specs:
        raise DSLValidationError([
            Diagnostic(
                "E_TP_V2_002",
                "first tensor_product@2 version requires axis-free equivariant inputs",
                node_id=node,
                details={"left_axes": list(left.axes), "right_axes": list(right.axes)},
            )
        ])
    context_left = (left.group, left.carrier, left.frame, left.dtype, left.level)
    context_right = (right.group, right.carrier, right.frame, right.dtype, right.level)
    if context_left != context_right:
        raise DSLValidationError([
            Diagnostic("E_TP_V2_003", "tensor_product@2 equivariant inputs must share group, carrier, frame, dtype, and certification level", node_id=node)
        ])
    out = Irreps.parse(str(attrs["out_irreps"]), left.group.family)
    allowed = set(left.irreps.allowed_tensor_product_outputs(right.irreps))
    invalid = [str(irrep) for _, irrep in out if irrep not in allowed]
    if invalid:
        raise DSLValidationError([
            Diagnostic(
                "E_TP_V2_004",
                "tensor_product@2 output has no legal coupling path",
                node_id=node,
                details={"invalid_outputs": invalid, "allowed": [str(item) for item in sorted(allowed)]},
            )
        ])
    weight_numel = _fully_connected_tp_weight_numel(left.irreps, right.irreps, out)
    if weight_numel <= 0:
        raise DSLValidationError([
            Diagnostic("E_TP_V2_005", "tensor_product@2 has no fully connected uvw paths", node_id=node)
        ])
    weight_context = (weight.group, weight.carrier, weight.frame, weight.dtype, weight.level)
    if weight_context != context_left:
        raise DSLValidationError([
            Diagnostic("E_TP_V2_006", "external TP weight must share group, carrier, frame, dtype, and certification level", node_id=node, port="weight")
        ])
    if weight.measure != "dimensionless":
        raise DSLValidationError([
            Diagnostic("E_TP_V2_007", "external TP weight must be dimensionless", node_id=node, port="weight", actual=weight.measure)
        ])
    if weight.feature_role != FeatureRole.RADIAL_WEIGHT:
        raise DSLValidationError([
            Diagnostic("E_TP_V2_008", "external TP weight must use radial_weight feature role", node_id=node, port="weight", actual=weight.feature_role.value)
        ])
    if len(weight.axis_specs) != 1 or weight.axis_specs[0].role != FeatureRole.TP_PATH:
        raise DSLValidationError([
            Diagnostic("E_TP_V2_009", "external TP weight must contain exactly one tp_path axis", node_id=node, port="weight")
        ])
    path_axis = weight.axis_specs[0]
    if path_axis.size != weight_numel or weight.irreps.dimension != weight_numel:
        raise DSLValidationError([
            Diagnostic(
                "E_TP_V2_010",
                "external TP weight path axis and scalar multiplicity must match fully connected uvw weight_numel",
                node_id=node,
                port="weight",
                expected=str(weight_numel),
                actual=str({"axis_size": path_axis.size, "scalar_dimension": weight.irreps.dimension}),
            )
        ])
    obligations = (
        ProofObligation("{}:paths".format(node), ObligationKind.IRREP_PATH_EXISTS, node, "discharged", "core.tensor_product@2", {"weight_numel": weight_numel, "connection_mode": "uvw"}),
        ProofObligation("{}:parity".format(node), ObligationKind.PARITY_MATCH, node, "discharged", "core.tensor_product@2"),
    )
    output = replace(
        left,
        irreps=out,
        measure=_multiply_measures(left.measure, right.measure),
    )
    return {"out": output}, obligations


def _tensor_product_v2_parameters(node, inputs, outputs, attrs):
    left = _single(node, inputs, "left")
    right = _single(node, inputs, "right")
    weight = _single(node, inputs, "weight")
    output = outputs["out"]
    weight_numel = _fully_connected_tp_weight_numel(left.irreps, right.irreps, output.irreps)
    return (
        ParameterContract(
            "weight",
            (ParameterAxis("tp_path", weight_numel, "tensor_product_path", weight.axis_specs[0].name),),
            storage="external",
            trainable=False,
            initializer="supplied_by_input",
            external_port="weight",
            architecture_identity="contract_only",
        ),
    )


def _tensor_product_v4(node, inputs, attrs):
    left = _v2_equivariant(node, _single(node, inputs, "left"), "left")
    right = _v2_equivariant(node, _single(node, inputs, "right"), "right")
    if left.axis_specs or right.axis_specs:
        raise DSLValidationError([
            Diagnostic("E_TP_V4_001", "tensor_product@4 requires axis-free equivariant inputs", node_id=node)
        ])
    for port, value in (("left", left), ("right", right)):
        if (
            value.irreps != value.irreps.simplify()
            or value.layout.storage != "irrep_major"
            or value.layout.coefficient_order != "canonical"
        ):
            raise DSLValidationError([
                Diagnostic(
                    "E_TP_V4_002",
                    "tensor_product@4 requires canonical simplified irrep-major inputs",
                    node_id=node,
                    port=port,
                )
            ])
    context_left = (left.group, left.carrier, left.frame, left.dtype, left.level)
    context_right = (right.group, right.carrier, right.frame, right.dtype, right.level)
    if context_left != context_right:
        raise DSLValidationError([
            Diagnostic("E_TP_V4_003", "tensor_product@4 inputs must share group, carrier, frame, dtype, and level", node_id=node)
        ])
    out = Irreps.parse(str(attrs["out_irreps"]), left.group.family)
    if out != out.simplify():
        raise DSLValidationError([
            Diagnostic("E_TP_V4_004", "tensor_product@4 output irreps must be simplified", node_id=node)
        ])
    allowed = set(left.irreps.allowed_tensor_product_outputs(right.irreps))
    invalid = [str(irrep) for _multiplicity, irrep in out if irrep not in allowed]
    if invalid:
        raise DSLValidationError([
            Diagnostic(
                "E_TP_V4_005",
                "tensor_product@4 output has no legal coupling path",
                node_id=node,
                details={"invalid_outputs": invalid},
            )
        ])
    for name in ("bias", "rescale"):
        value = attrs.get(name, True)
        if not isinstance(value, bool):
            raise DSLValidationError([
                Diagnostic("E_TP_V4_006", "tensor_product@4 {} must be boolean".format(name), node_id=node)
            ])
    output = replace(
        left,
        irreps=out,
        measure=_multiply_measures(left.measure, right.measure),
    )
    obligations = (
        ProofObligation(
            "{}:paths".format(node),
            ObligationKind.IRREP_PATH_EXISTS,
            node,
            "discharged",
            "core.tensor_product@4",
            {
                "weight_numel": _fully_connected_tp_weight_numel(left.irreps, right.irreps, out),
                "connection_mode": "uvw",
            },
        ),
        ProofObligation("{}:parity".format(node), ObligationKind.PARITY_MATCH, node, "discharged", "core.tensor_product@4"),
    )
    return {"out": output}, obligations


def _tensor_product_v4_parameters(node, inputs, outputs, attrs):
    left = _single(node, inputs, "left")
    right = _single(node, inputs, "right")
    output = outputs["out"]
    weight_numel = _fully_connected_tp_weight_numel(left.irreps, right.irreps, output.irreps)
    contracts = [
        ParameterContract(
            "weight",
            (ParameterAxis("tp_path", weight_numel, "fully_connected_uvw_path"),),
            initializer=(
                "uniform_minus1_1_instruction_fan_in_scaled"
                if attrs.get("rescale", True)
                else "uniform_minus1_1"
            ),
            checkpoint_names=("{}.tp.weight".format(node),),
            backend_parameter_name="tp.weight",
        )
    ]
    if attrs.get("bias", True):
        bias_index = 0
        for multiplicity, irrep in output.irreps:
            if irrep.degree != 0 or irrep.parity != 1:
                continue
            contracts.append(
                ParameterContract(
                    "bias_{}".format(bias_index),
                    (ParameterAxis("multiplicity", multiplicity, "trivial_scalar_bias"),),
                    is_bias=True,
                    bias_irreps=str(Irreps(((multiplicity, irrep),))),
                    initializer="zeros",
                    checkpoint_names=("{}.bias.{}".format(node, bias_index),),
                    backend_parameter_name="bias.{}".format(bias_index),
                )
            )
            bias_index += 1
    return tuple(contracts)


def _tensor_product_v3_path_contract(node, left, right, attrs):
    raw_blocks = attrs["path_blocks"]
    if not isinstance(raw_blocks, (list, tuple)) or not raw_blocks:
        raise DSLValidationError([
            Diagnostic("E_TP_V3_001", "tensor_product@3 requires a nonempty path_blocks list", node_id=node)
        ])
    path_terms = []
    for block_index, raw in enumerate(raw_blocks):
        if not isinstance(raw, Mapping):
            raise DSLValidationError([
                Diagnostic("E_TP_V3_002", "each tensor_product@3 path block must be an object", node_id=node, actual=str(raw))
            ])
        unknown = set(raw) - {"multiplicity", "irrep"}
        missing = {"multiplicity", "irrep"} - set(raw)
        if unknown or missing:
            raise DSLValidationError([
                Diagnostic(
                    "E_TP_V3_003",
                    "tensor_product@3 path block fields must be exactly multiplicity and irrep",
                    node_id=node,
                    details={"block": block_index, "unknown": sorted(unknown), "missing": sorted(missing)},
                )
            ])
        raw_multiplicity = raw["multiplicity"]
        if isinstance(raw_multiplicity, bool):
            raise DSLValidationError([
                Diagnostic("E_TP_V3_004", "path block multiplicity must be a positive integer", node_id=node, actual=str(raw_multiplicity))
            ])
        try:
            multiplicity = int(raw_multiplicity)
        except (TypeError, ValueError):
            raise DSLValidationError([
                Diagnostic("E_TP_V3_004", "path block multiplicity must be a positive integer", node_id=node, actual=str(raw_multiplicity))
            ])
        if multiplicity <= 0 or str(raw_multiplicity) != str(multiplicity):
            raise DSLValidationError([
                Diagnostic("E_TP_V3_004", "path block multiplicity must be a positive integer", node_id=node, actual=str(raw_multiplicity))
            ])
        parsed = Irreps.parse("1x{}".format(str(raw["irrep"])), left.group.family)
        if len(parsed.terms) != 1 or parsed.terms[0][0] != 1:
            raise DSLValidationError([
                Diagnostic("E_TP_V3_005", "path block irrep must name exactly one representation kind", node_id=node, actual=str(raw["irrep"]))
            ])
        path_terms.append((multiplicity, parsed.terms[0][1]))
    path_irreps = Irreps(tuple(path_terms))
    path_kinds = [irrep for _, irrep in path_irreps]
    if path_kinds != sorted(path_kinds):
        raise DSLValidationError([
            Diagnostic(
                "E_TP_V3_006",
                "tensor_product@3 path blocks must follow canonical irrep order",
                node_id=node,
                expected=str([str(irrep) for irrep in sorted(path_kinds)]),
                actual=str([str(irrep) for irrep in path_kinds]),
            )
        ])

    raw_instructions = attrs["instructions"]
    if not isinstance(raw_instructions, (list, tuple)) or not raw_instructions:
        raise DSLValidationError([
            Diagnostic("E_TP_V3_007", "tensor_product@3 requires a nonempty instructions list", node_id=node)
        ])
    instructions = []
    weight_numel = 0
    referenced_outputs = set()
    allowed_fields = {"left", "right", "out", "mode", "has_weight", "path_weight"}
    required_fields = {"left", "right", "out", "mode"}
    for instruction_index, raw in enumerate(raw_instructions):
        if not isinstance(raw, Mapping):
            raise DSLValidationError([
                Diagnostic("E_TP_V3_008", "each tensor_product@3 instruction must be an object", node_id=node, actual=str(raw))
            ])
        unknown = set(raw) - allowed_fields
        missing = required_fields - set(raw)
        if unknown or missing:
            raise DSLValidationError([
                Diagnostic(
                    "E_TP_V3_009",
                    "tensor_product@3 instruction has unknown or missing fields",
                    node_id=node,
                    details={"instruction": instruction_index, "unknown": sorted(unknown), "missing": sorted(missing)},
                )
            ])
        indices = []
        for field_name, upper_bound in (
            ("left", len(left.irreps.terms)),
            ("right", len(right.irreps.terms)),
            ("out", len(path_irreps.terms)),
        ):
            raw_index = raw[field_name]
            if isinstance(raw_index, bool):
                value = -1
            else:
                try:
                    value = int(raw_index)
                except (TypeError, ValueError):
                    value = -1
            if value < 0 or value >= upper_bound or str(raw_index) != str(value):
                raise DSLValidationError([
                    Diagnostic(
                        "E_TP_V3_010",
                        "tensor_product@3 instruction block index is invalid",
                        node_id=node,
                        actual=str({"field": field_name, "value": raw_index, "upper_bound": upper_bound}),
                    )
                ])
            indices.append(value)
        left_index, right_index, output_index = indices
        mode = str(raw["mode"])
        if mode != "uvu":
            raise DSLValidationError([
                Diagnostic("E_TP_V3_011", "tensor_product@3 first version supports only uvu instructions", node_id=node, actual=mode)
            ])
        has_weight = raw.get("has_weight", True)
        if has_weight is not True:
            raise DSLValidationError([
                Diagnostic("E_TP_V3_012", "tensor_product@3 first version requires every uvu instruction to have an external weight", node_id=node, actual=str(has_weight))
            ])
        try:
            path_weight = float(raw.get("path_weight", 1.0))
        except (TypeError, ValueError):
            path_weight = float("nan")
        if not math.isfinite(path_weight) or path_weight <= 0.0:
            raise DSLValidationError([
                Diagnostic("E_TP_V3_013", "tensor_product@3 path_weight must be finite and positive", node_id=node, actual=str(raw.get("path_weight")))
            ])
        left_mul, left_irrep = left.irreps.terms[left_index]
        right_mul, right_irrep = right.irreps.terms[right_index]
        output_mul, output_irrep = path_irreps.terms[output_index]
        if output_irrep not in left_irrep.tensor_product(right_irrep):
            raise DSLValidationError([
                Diagnostic(
                    "E_TP_V3_014",
                    "tensor_product@3 instruction has no legal Clebsch-Gordan path",
                    node_id=node,
                    details={
                        "instruction": instruction_index,
                        "left": str(left_irrep),
                        "right": str(right_irrep),
                        "out": str(output_irrep),
                    },
                )
            ])
        if output_mul != left_mul:
            raise DSLValidationError([
                Diagnostic(
                    "E_TP_V3_015",
                    "uvu output block multiplicity must equal the selected left input multiplicity",
                    node_id=node,
                    expected=str(left_mul),
                    actual=str(output_mul),
                    details={"instruction": instruction_index, "output_block": output_index},
                )
            ])
        weight_numel += left_mul * right_mul
        referenced_outputs.add(output_index)
        instructions.append((left_index, right_index, output_index, mode, True, path_weight))
    missing_outputs = sorted(set(range(len(path_irreps.terms))) - referenced_outputs)
    if missing_outputs:
        raise DSLValidationError([
            Diagnostic(
                "E_TP_V3_016",
                "every tensor_product@3 path block must be referenced by at least one instruction",
                node_id=node,
                details={"unreferenced_output_blocks": missing_outputs},
            )
        ])
    return path_irreps, tuple(instructions), weight_numel


def _tensor_product_v3(node, inputs, attrs):
    left = _v2_equivariant(node, _single(node, inputs, "left"), "left")
    right = _v2_equivariant(node, _single(node, inputs, "right"), "right")
    weight = _single(node, inputs, "weight")
    if not isinstance(weight, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic("E_TP_V3_017", "tensor_product@3 weight must use InvariantTensorType", node_id=node, port="weight", actual=type(weight).__name__)
        ])
    if left.axis_specs or right.axis_specs:
        raise DSLValidationError([
            Diagnostic("E_TP_V3_018", "tensor_product@3 first version requires axis-free equivariant inputs", node_id=node)
        ])
    if (
        left.irreps != left.irreps.simplify()
        or right.irreps != right.irreps.simplify()
        or left.layout.storage != "irrep_major"
        or right.layout.storage != "irrep_major"
        or left.layout.coefficient_order != "canonical"
        or right.layout.coefficient_order != "canonical"
    ):
        raise DSLValidationError([
            Diagnostic("E_TP_V3_019", "tensor_product@3 requires canonical simplified irrep-major inputs", node_id=node)
        ])
    context_left = (left.group, left.carrier, left.frame, left.dtype, left.level)
    context_right = (right.group, right.carrier, right.frame, right.dtype, right.level)
    weight_context = (weight.group, weight.carrier, weight.frame, weight.dtype, weight.level)
    if context_left != context_right or context_left != weight_context:
        raise DSLValidationError([
            Diagnostic("E_TP_V3_020", "tensor_product@3 inputs must share group, carrier, frame, dtype, and certification level", node_id=node)
        ])
    if weight.measure != "dimensionless" or weight.feature_role != FeatureRole.RADIAL_WEIGHT:
        raise DSLValidationError([
            Diagnostic("E_TP_V3_021", "tensor_product@3 external weight must be dimensionless radial_weight", node_id=node, port="weight")
        ])
    if len(weight.axis_specs) != 1 or weight.axis_specs[0].role != FeatureRole.TP_PATH:
        raise DSLValidationError([
            Diagnostic("E_TP_V3_022", "tensor_product@3 weight must contain exactly one tp_path axis", node_id=node, port="weight")
        ])
    path_irreps, _instructions, weight_numel = _tensor_product_v3_path_contract(node, left, right, attrs)
    if weight.axis_specs[0].size != weight_numel or weight.irreps.dimension != weight_numel:
        raise DSLValidationError([
            Diagnostic(
                "E_TP_V3_023",
                "tensor_product@3 external weight size must equal the total uvu instruction path size",
                node_id=node,
                expected=str(weight_numel),
                actual=str({"axis_size": weight.axis_specs[0].size, "scalar_dimension": weight.irreps.dimension}),
            )
        ])
    output = replace(
        left,
        irreps=path_irreps.simplify(),
        measure=_multiply_measures(left.measure, right.measure),
    )
    obligations = (
        ProofObligation(
            "{}:paths".format(node),
            ObligationKind.IRREP_PATH_EXISTS,
            node,
            "discharged",
            "core.tensor_product@3",
            {
                "instruction_count": len(_instructions),
                "weight_numel": weight_numel,
                "connection_mode": "uvu",
                "path_block_irreps": str(path_irreps),
            },
        ),
        ProofObligation("{}:parity".format(node), ObligationKind.PARITY_MATCH, node, "discharged", "core.tensor_product@3"),
    )
    return {"out": output}, obligations


def _tensor_product_v3_parameters(node, inputs, outputs, attrs):
    left = _single(node, inputs, "left")
    right = _single(node, inputs, "right")
    weight = _single(node, inputs, "weight")
    _path_irreps, _instructions, weight_numel = _tensor_product_v3_path_contract(node, left, right, attrs)
    return (
        ParameterContract(
            "weight",
            (ParameterAxis("tp_path", weight_numel, "tensor_product_path", weight.axis_specs[0].name),),
            storage="external",
            trainable=False,
            initializer="supplied_by_input",
            external_port="weight",
            architecture_identity="contract_only",
        ),
    )


def _tensor_product_v5(node, inputs, attrs):
    left = _v2_equivariant(node, _single(node, inputs, "left"), "left")
    right = _v2_equivariant(node, _single(node, inputs, "right"), "right")
    if left.axis_specs or right.axis_specs:
        raise DSLValidationError([
            Diagnostic("E_TP_V5_001", "tensor_product@5 requires axis-free equivariant inputs", node_id=node)
        ])
    if (
        left.irreps != left.irreps.simplify()
        or right.irreps != right.irreps.simplify()
        or left.layout.storage != "irrep_major"
        or right.layout.storage != "irrep_major"
        or left.layout.coefficient_order != "canonical"
        or right.layout.coefficient_order != "canonical"
    ):
        raise DSLValidationError([
            Diagnostic("E_TP_V5_002", "tensor_product@5 requires canonical simplified irrep-major inputs", node_id=node)
        ])
    context_left = (left.group, left.carrier, left.frame, left.dtype, left.level)
    context_right = (right.group, right.carrier, right.frame, right.dtype, right.level)
    if context_left != context_right:
        raise DSLValidationError([
            Diagnostic("E_TP_V5_003", "tensor_product@5 inputs must share group, carrier, frame, dtype, and level", node_id=node)
        ])
    if attrs.get("bias", False) is not False:
        raise DSLValidationError([
            Diagnostic("E_TP_V5_004", "tensor_product@5 first version requires bias=False", node_id=node)
        ])
    if not isinstance(attrs.get("rescale", True), bool):
        raise DSLValidationError([
            Diagnostic("E_TP_V5_005", "tensor_product@5 rescale must be boolean", node_id=node)
        ])
    path_irreps, instructions, weight_numel = _tensor_product_v3_path_contract(node, left, right, attrs)
    output = replace(
        left,
        irreps=path_irreps.simplify(),
        measure=_multiply_measures(left.measure, right.measure),
    )
    obligations = (
        ProofObligation(
            "{}:paths".format(node),
            ObligationKind.IRREP_PATH_EXISTS,
            node,
            "discharged",
            "core.tensor_product@5",
            {
                "instruction_count": len(instructions),
                "weight_numel": weight_numel,
                "connection_mode": "uvu",
                "weight_storage": "internal_shared",
                "path_block_irreps": str(path_irreps),
            },
        ),
        ProofObligation("{}:parity".format(node), ObligationKind.PARITY_MATCH, node, "discharged", "core.tensor_product@5"),
    )
    return {"out": output}, obligations


def _tensor_product_v5_parameters(node, inputs, outputs, attrs):
    left = _single(node, inputs, "left")
    right = _single(node, inputs, "right")
    _path_irreps, _instructions, weight_numel = _tensor_product_v3_path_contract(node, left, right, attrs)
    return (
        ParameterContract(
            "weight",
            (ParameterAxis("tp_path", weight_numel, "instruction_uvu_path"),),
            initializer=(
                "uniform_minus1_1_instruction_fan_in_scaled"
                if attrs.get("rescale", True)
                else "uniform_minus1_1"
            ),
            checkpoint_names=("{}.tp.weight".format(node),),
            backend_parameter_name="tp.weight",
        ),
    )


def _scalar_activation(node, inputs, attrs):
    x = _single(node, inputs, "x")
    non_scalars = [str(ir) for _, ir in x.irreps if ir.degree != 0]
    if non_scalars:
        raise DSLValidationError([
            Diagnostic("E_NONLINEAR_001", "ordinary elementwise activation is only valid on scalar irreps", node_id=node, details={"non_scalars": non_scalars})
        ])
    if x.measure != "dimensionless":
        raise DSLValidationError([
            Diagnostic("E_UNIT_001", "ordinary scalar activation requires dimensionless inputs", node_id=node, actual=x.measure)
        ])
    return {"out": x}, ()


def _invariant_weight(node, inputs, attrs):
    weight = _single(node, inputs, "weight")
    value = _single(node, inputs, "value")
    if any(ir.degree != 0 or (ir.family in ("O3", "O2") and ir.parity != 1) for _, ir in weight.irreps):
        raise DSLValidationError([Diagnostic("E_ATTN_001", "attention weights must be invariant scalars", node_id=node, port="weight")])
    if weight.irreps.dimension != 1:
        raise DSLValidationError([Diagnostic("E_ATTN_004", "invariant_weight requires one scalar until an explicit head axis is declared", node_id=node, port="weight")])
    if weight.measure != "dimensionless":
        raise DSLValidationError([Diagnostic("E_UNIT_002", "attention weights must be dimensionless", node_id=node, port="weight", actual=weight.measure)])
    if weight.carrier != value.carrier or weight.frame not in (value.frame, Frame("invariant")):
        raise DSLValidationError([Diagnostic("E_ATTN_002", "attention weight and value must share carrier and compatible frame", node_id=node)])
    obligation = ProofObligation("{}:weight".format(node), ObligationKind.INVARIANT_ATTENTION_WEIGHT, node, "discharged", "core.invariant_weight@1")
    return {"out": value}, (obligation,)


def _invariant_scale(node, inputs, attrs):
    weight = _single(node, inputs, "weight")
    value = _v2_equivariant(node, _single(node, inputs, "value"), "value")
    if not isinstance(weight, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_SCALE_001",
                "invariant_scale weight must use InvariantTensorType",
                node_id=node,
                port="weight",
                actual=type(weight).__name__,
            )
        ])
    if weight.measure != "dimensionless":
        raise DSLValidationError([
            Diagnostic("E_SCALE_002", "invariant_scale weight must be dimensionless", node_id=node, actual=weight.measure)
        ])
    value_context = (value.group, value.carrier, value.dtype, value.level)
    weight_context = (weight.group, weight.carrier, weight.dtype, weight.level)
    if value_context != weight_context or weight.frame not in (value.frame, Frame("invariant")):
        raise DSLValidationError([
            Diagnostic(
                "E_SCALE_003",
                "invariant_scale inputs must share group, carrier, dtype, certification level, and a compatible frame",
                node_id=node,
            )
        ])
    if not weight.axis_specs:
        if weight.irreps.dimension != 1:
            raise DSLValidationError([
                Diagnostic(
                    "E_SCALE_004",
                    "axis-free invariant_scale weight must contain exactly one scalar",
                    node_id=node,
                    actual=str(weight.irreps.dimension),
                )
            ])
    else:
        if len(weight.axis_specs) != 1 or weight.axis_specs[0].role != FeatureRole.HEAD:
            raise DSLValidationError([
                Diagnostic(
                    "E_SCALE_005",
                    "first invariant_scale version supports either one scalar or one explicit head axis",
                    node_id=node,
                    details={"weight_axes": [axis.to_dict() for axis in weight.axis_specs]},
                )
            ])
        if len(value.axis_specs) != 1 or value.axis_specs[0].role != FeatureRole.HEAD:
            raise DSLValidationError([
                Diagnostic(
                    "E_SCALE_006",
                    "headwise invariant_scale requires an equivariant value with one head axis",
                    node_id=node,
                    details={"value_axes": [axis.to_dict() for axis in value.axis_specs]},
                )
            ])
        weight_head = weight.axis_specs[0]
        value_head = value.axis_specs[0]
        if weight_head.name != value_head.name or weight_head.size != value_head.size:
            raise DSLValidationError([
                Diagnostic(
                    "E_SCALE_007",
                    "invariant_scale head axes must have the same name and size",
                    node_id=node,
                    expected=str((value_head.name, value_head.size)),
                    actual=str((weight_head.name, weight_head.size)),
                )
            ])
        if weight.irreps.dimension != int(weight_head.size):
            raise DSLValidationError([
                Diagnostic(
                    "E_SCALE_008",
                    "headwise invariant scale requires one trivial scalar per head",
                    node_id=node,
                    expected=str(weight_head.size),
                    actual=str(weight.irreps.dimension),
                )
            ])
    obligation = ProofObligation(
        "{}:scale".format(node),
        ObligationKind.INVARIANT_ATTENTION_WEIGHT,
        node,
        "discharged",
        "core.invariant_scale@1",
        {"broadcast_axes": [axis.name for axis in weight.axis_specs]},
    )
    return {"out": value}, (obligation,)


def _invariant_scale_v2(node, inputs, attrs):
    weight = _single(node, inputs, "weight")
    value = _v2_equivariant(node, _single(node, inputs, "value"), "value")
    if not isinstance(weight, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic("E_SCALE_V2_001", "invariant_scale@2 requires invariant head weights", node_id=node)
        ])
    if weight.measure != "dimensionless" or weight.frame != Frame("invariant"):
        raise DSLValidationError([
            Diagnostic("E_SCALE_V2_002", "invariant_scale@2 weights must be dimensionless invariants", node_id=node)
        ])
    if value.frame.kind != "edge" or value.axis_specs:
        raise DSLValidationError([
            Diagnostic("E_SCALE_V2_003", "invariant_scale@2 requires an axis-free edge-frame value", node_id=node)
        ])
    if len(weight.axis_specs) != 1 or weight.axis_specs[0].role != FeatureRole.HEAD:
        raise DSLValidationError([
            Diagnostic("E_SCALE_V2_004", "invariant_scale@2 requires exactly one head axis", node_id=node)
        ])
    head_count = int(weight.axis_specs[0].size)
    _lmax, channels = _uniform_so3_multiplicities(node, value.irreps)
    if channels % head_count != 0 or weight.irreps.dimension != head_count:
        raise DSLValidationError([
            Diagnostic(
                "E_SCALE_V2_005",
                "edge-frame channel multiplicity must be divisible by the attention head count",
                node_id=node,
            )
        ])
    value_context = (value.group, value.carrier, value.dtype)
    weight_context = (weight.group, weight.carrier, weight.dtype)
    if value_context != weight_context:
        raise DSLValidationError([
            Diagnostic("E_SCALE_V2_006", "invariant_scale@2 inputs have incompatible contexts", node_id=node)
        ])
    obligation = ProofObligation(
        "{}:head-scale".format(node),
        ObligationKind.INVARIANT_ATTENTION_WEIGHT,
        node,
        "discharged",
        "core.invariant_scale@2",
        {"head_count": head_count},
    )
    return {"out": replace(value, level=min(value.level, weight.level))}, (obligation,)


def _edge_lift(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if x.carrier != Carrier.NODE:
        raise DSLValidationError([Diagnostic("E_CARRIER_001", "edge_lift expects node features", node_id=node)])
    return {"out": x.with_carrier(Carrier.EDGE)}, ()


def _endpoint_gather(node, inputs, attrs):
    x = _v2_equivariant(node, _single(node, inputs, "x"), "x")
    index = _single(node, inputs, "index")
    if not isinstance(index, IndexMapType):
        raise DSLValidationError([Diagnostic("E_INDEX_005", "endpoint_gather requires an IndexMapType", node_id=node, port="index")])
    if index.group != x.group or index.source_carrier != x.carrier:
        raise DSLValidationError([
            Diagnostic(
                "E_INDEX_006",
                "endpoint_gather index domain does not match its value input",
                node_id=node,
                expected="{} on {}".format(x.group.family, x.carrier),
                actual="{} on {}".format(index.group.family, index.source_carrier),
            )
        ])
    if index.endpoint not in ("source", "target") or index.target_carrier != Carrier.EDGE:
        raise DSLValidationError([Diagnostic("E_INDEX_007", "endpoint_gather requires a source or target node-to-edge map", node_id=node)])
    return {"out": x.with_carrier(index.target_carrier)}, ()


def _endpoint_gather_v2(node, inputs, attrs):
    x = _single(node, inputs, "x")
    index = _single(node, inputs, "index")
    if not isinstance(index, IndexMapType):
        raise DSLValidationError([
            Diagnostic("E_INDEX_V2_001", "endpoint_gather@2 requires an IndexMapType", node_id=node, port="index")
        ])
    if index.endpoint not in ("source", "target") or index.target_carrier != Carrier.EDGE:
        raise DSLValidationError([
            Diagnostic("E_INDEX_V2_002", "endpoint_gather@2 requires a source or target node-to-edge map", node_id=node)
        ])
    if isinstance(x, EquivariantTensorType):
        if index.group != x.group or index.source_carrier != x.carrier:
            raise DSLValidationError([
                Diagnostic("E_INDEX_V2_003", "endpoint_gather@2 index domain does not match equivariant input", node_id=node)
            ])
        return {"out": x.with_carrier(Carrier.EDGE)}, ()
    if isinstance(x, CategoricalTensorType):
        if index.group != x.group or index.source_carrier != x.carrier:
            raise DSLValidationError([
                Diagnostic("E_INDEX_V2_004", "endpoint_gather@2 index domain does not match categorical input", node_id=node)
            ])
        return {"out": replace(x, carrier=Carrier.EDGE)}, ()
    raise DSLValidationError([
        Diagnostic(
            "E_INDEX_V2_005",
            "endpoint_gather@2 supports equivariant or categorical node values",
            node_id=node,
            actual=type(x).__name__,
        )
    ])


def _segment_sum(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if x.carrier != Carrier.EDGE:
        raise DSLValidationError([Diagnostic("E_CARRIER_002", "segment_sum expects edge values", node_id=node)])
    if x.frame.kind == "edge":
        raise DSLValidationError([
            Diagnostic("E_FRAME_004", "edge-frame values must return to global frame before node aggregation", node_id=node, actual=str(x.frame), repairs=("insert core.from_edge_frame",))
        ])
    obligation = ProofObligation("{}:permutation".format(node), ObligationKind.PERMUTATION_SAFE_AGGREGATION, node, "discharged", "core.segment_sum@1")
    return {"out": x.with_carrier(Carrier.NODE)}, (obligation,)


def _segment_reduce_v2(node, inputs, attrs):
    x = _v2_equivariant(node, _single(node, inputs, "x"), "x")
    index = _single(node, inputs, "index")
    if not isinstance(index, IndexMapType):
        raise DSLValidationError([Diagnostic("E_INDEX_008", "segment_reduce requires an IndexMapType", node_id=node, port="index")])
    if index.group != x.group or index.source_carrier != x.carrier:
        raise DSLValidationError([Diagnostic("E_INDEX_009", "segment_reduce index domain does not match its value input", node_id=node)])
    if index.endpoint not in ("segment", "batch"):
        raise DSLValidationError([Diagnostic("E_INDEX_010", "segment_reduce requires a segment or batch index map", node_id=node)])
    if x.frame.kind == "edge" and index.target_carrier != Carrier.EDGE:
        raise DSLValidationError([
            Diagnostic("E_FRAME_004", "edge-frame values must return to global frame before segment reduction", node_id=node, actual=str(x.frame))
        ])
    reduce = str(attrs.get("reduce", "sum"))
    normalization = str(attrs.get("normalization", "none"))
    if reduce not in ("sum", "mean"):
        raise DSLValidationError([Diagnostic("E_ATTR_006", "segment_reduce supports sum or mean", node_id=node, actual=reduce)])
    if normalization not in ("none", "target_cardinality"):
        raise DSLValidationError([
            Diagnostic(
                "E_ATTR_007",
                "segment_reduce normalization must be none or target_cardinality",
                node_id=node,
                actual=normalization,
            )
        ])
    if normalization == "target_cardinality" and reduce != "sum":
        raise DSLValidationError([
            Diagnostic(
                "E_ATTR_008",
                "target_cardinality rescale is defined only for sum reduction",
                node_id=node,
                actual=reduce,
            )
        ])
    obligation = ProofObligation("{}:permutation".format(node), ObligationKind.PERMUTATION_SAFE_AGGREGATION, node, "discharged", "core.segment_reduce@1")
    return {"out": x.with_carrier(index.target_carrier)}, (obligation,)


def _global_pool(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if x.carrier != Carrier.NODE:
        raise DSLValidationError([Diagnostic("E_CARRIER_003", "global_pool expects node values", node_id=node)])
    return {"out": x.with_carrier(Carrier.GRAPH)}, ()


def _select_scalars(node, inputs, attrs):
    x = _single(node, inputs, "x")
    terms = tuple((mul, ir) for mul, ir in x.irreps if ir.degree == 0 and (ir.family not in ("O3", "O2") or ir.parity == 1))
    if not terms:
        raise DSLValidationError([Diagnostic("E_READOUT_001", "no invariant scalar irreps are available", node_id=node)])
    requested = attrs.get("multiplicity")
    output = Irreps(terms).simplify()
    if requested is not None:
        scalar = output.terms[0][1]
        if int(requested) > output.multiplicity(scalar):
            raise DSLValidationError([Diagnostic("E_READOUT_002", "requested scalar multiplicity exceeds available channels", node_id=node)])
        output = Irreps(((int(requested), scalar),))
    return {"out": x.with_irreps(output)}, ()


def _select_scalars_v2(node, inputs, attrs):
    """Extract trivial coefficients into an explicit invariant feature axis."""

    x = _v2_equivariant(node, _single(node, inputs, "x"), "x")
    outputs, obligations = _select_scalars(node, inputs, attrs)
    selected = outputs["out"]
    axis_name = str(attrs.get("axis", "scalar_channel"))
    if not axis_name:
        raise DSLValidationError([
            Diagnostic("E_READOUT_003", "select_scalars@2 axis must be nonempty", node_id=node)
        ])
    role = FeatureRole.parse(attrs.get("feature_role", FeatureRole.CHANNEL.value))
    output = InvariantTensorType(
        group=x.group,
        carrier=x.carrier,
        irreps=selected.irreps,
        frame=Frame("invariant"),
        axes=(axis_name,),
        dtype=x.dtype,
        measure=x.measure,
        level=x.level,
        axis_specs=(AxisSpec(axis_name, selected.irreps.dimension, role, "independent", 0),),
        feature_role=role,
    )
    return {"out": output}, obligations


def _to_edge_frame(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if x.group.dimension != 3 or x.carrier != Carrier.EDGE:
        raise DSLValidationError([
            Diagnostic(
                "E_FRAME_007",
                "to_edge_frame expects 3D edge-carried features",
                node_id=node,
                actual="{} on {}".format(x.group, x.carrier),
            )
        ])
    if x.frame.kind != "global":
        raise DSLValidationError([Diagnostic("E_FRAME_005", "to_edge_frame expects global features", node_id=node, actual=str(x.frame))])
    lmax = max(ir.degree for _, ir in x.irreps)
    mmax = int(attrs.get("mmax", lmax))
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic(
                "E_FRAME_008",
                "edge-frame mmax must satisfy 0 <= mmax <= lmax",
                node_id=node,
                expected="0..{}".format(lmax),
                actual=str(mmax),
            )
        ])
    reference = str(attrs.get("frame_id", node))
    obligation = ProofObligation("{}:frame-return".format(node), ObligationKind.FRAME_BALANCE, node, details={"frame_id": reference})
    return {"out": x.with_frame(Frame("edge", reference))}, (obligation,)


def _from_edge_frame(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if x.group.dimension != 3 or x.carrier != Carrier.EDGE:
        raise DSLValidationError([
            Diagnostic(
                "E_FRAME_008",
                "from_edge_frame expects 3D edge-carried features",
                node_id=node,
                actual="{} on {}".format(x.group, x.carrier),
            )
        ])
    expected = str(attrs.get("frame_id", ""))
    if x.frame.kind != "edge" or (expected and x.frame.reference != expected):
        raise DSLValidationError([Diagnostic("E_FRAME_006", "from_edge_frame received a mismatched frame", node_id=node, expected=expected, actual=str(x.frame))])
    obligation = ProofObligation("{}:frame-restored".format(node), ObligationKind.FRAME_BALANCE, node, "discharged", "core.from_edge_frame@1", {"frame_id": x.frame.reference})
    return {"out": x.with_frame(Frame("global"))}, (obligation,)


def _to_edge_frame_v2(node, inputs, attrs):
    x = _single(node, inputs, "x")
    direction = _v2_equivariant(node, _single(node, inputs, "direction"), "direction")
    outputs, obligations = _to_edge_frame(node, {"x": (x,)}, attrs)
    if (
        direction.group != x.group
        or direction.carrier != Carrier.EDGE
        or direction.frame.kind != "global"
        or direction.irreps.dimension != 3
        or any(ir.degree != 1 for _, ir in direction.irreps)
    ):
        raise DSLValidationError([
            Diagnostic(
                "E_FRAME_V2_001",
                "to_edge_frame@2 requires an explicit global edge direction vector",
                node_id=node,
            )
        ])
    if not isinstance(attrs.get("use_rotation_mask", False), bool):
        raise DSLValidationError([
            Diagnostic("E_FRAME_V2_002", "use_rotation_mask must be boolean", node_id=node)
        ])
    return outputs, obligations


def _from_edge_frame_v2(node, inputs, attrs):
    if not isinstance(attrs.get("use_rotation_mask", False), bool):
        raise DSLValidationError([
            Diagnostic("E_FRAME_V2_003", "use_rotation_mask must be boolean", node_id=node)
        ])
    return _from_edge_frame(node, inputs, attrs)


def _irrep_slice(node, inputs, attrs):
    x = _single(node, inputs, "x")
    out = Irreps.parse(str(attrs["irreps"]), x.group.family)
    unavailable = [
        "{}x{}".format(mul, ir)
        for mul, ir in out
        if x.irreps.multiplicity(ir) < mul
    ]
    if unavailable:
        raise DSLValidationError([
            Diagnostic("E_IRREP_013", "irrep_slice requests unavailable channels", node_id=node, details={"unavailable": unavailable})
        ])
    return {"out": x.with_irreps(out)}, ()


def _selection_irrep(node, raw, family):
    if not isinstance(raw, Mapping):
        raise DSLValidationError([
            Diagnostic("E_SELECT_V2_001", "each irrep_select@2 selection must be an object", node_id=node, actual=str(raw))
        ])
    unknown = set(raw) - {"irrep", "start", "multiplicity"}
    if unknown:
        raise DSLValidationError([
            Diagnostic("E_SELECT_V2_002", "unknown irrep_select@2 selection fields", node_id=node, details={"fields": sorted(unknown)})
        ])
    missing = {"irrep", "start", "multiplicity"} - set(raw)
    if missing:
        raise DSLValidationError([
            Diagnostic("E_SELECT_V2_003", "irrep_select@2 selection is missing required fields", node_id=node, details={"fields": sorted(missing)})
        ])
    parsed = Irreps.parse("1x{}".format(str(raw["irrep"])), family)
    if len(parsed.terms) != 1 or parsed.terms[0][0] != 1:
        raise DSLValidationError([
            Diagnostic("E_SELECT_V2_004", "irrep_select@2 irrep must name exactly one representation kind", node_id=node, actual=str(raw["irrep"]))
        ])
    try:
        start = int(raw["start"])
        multiplicity = int(raw["multiplicity"])
    except (TypeError, ValueError):
        raise DSLValidationError([
            Diagnostic("E_SELECT_V2_005", "irrep_select@2 start and multiplicity must be integers", node_id=node, actual=str(raw))
        ])
    if isinstance(raw["start"], bool) or isinstance(raw["multiplicity"], bool) or start < 0 or multiplicity <= 0:
        raise DSLValidationError([
            Diagnostic("E_SELECT_V2_006", "irrep_select@2 requires start >= 0 and multiplicity > 0", node_id=node, actual=str(raw))
        ])
    return parsed.terms[0][1], start, multiplicity


def _irrep_select_v2_spec(node, source, attrs):
    if source.irreps != source.irreps.simplify():
        raise DSLValidationError([
            Diagnostic("E_SELECT_V2_010", "irrep_select@2 requires canonical simplified input irreps", node_id=node)
        ])
    if source.layout.storage != "irrep_major" or source.layout.coefficient_order != "canonical":
        raise DSLValidationError([
            Diagnostic(
                "E_SELECT_V2_012",
                "irrep_select@2 requires canonical irrep-major coefficient storage",
                node_id=node,
                expected="irrep_major/canonical",
                actual="{}/{}".format(source.layout.storage, source.layout.coefficient_order),
            )
        ])
    raw_selections = attrs["selections"]
    if not isinstance(raw_selections, (list, tuple)) or not raw_selections:
        raise DSLValidationError([
            Diagnostic("E_SELECT_V2_007", "irrep_select@2 requires a nonempty selections list", node_id=node)
        ])
    selections = tuple(_selection_irrep(node, raw, source.group.family) for raw in raw_selections)
    irreps = [irrep for irrep, _, _ in selections]
    if len(irreps) != len(set(irreps)):
        raise DSLValidationError([
            Diagnostic("E_SELECT_V2_008", "irrep_select@2 first version permits at most one range per irrep kind", node_id=node)
        ])
    if irreps != sorted(irreps):
        raise DSLValidationError([
            Diagnostic(
                "E_SELECT_V2_011",
                "irrep_select@2 selections must follow canonical irrep order",
                node_id=node,
                expected=str([str(irrep) for irrep in sorted(irreps)]),
                actual=str([str(irrep) for irrep in irreps]),
            )
        ])
    unavailable = []
    for irrep, start, multiplicity in selections:
        available = source.irreps.multiplicity(irrep)
        if start + multiplicity > available:
            unavailable.append({
                "irrep": str(irrep),
                "start": start,
                "multiplicity": multiplicity,
                "available": available,
            })
    if unavailable:
        raise DSLValidationError([
            Diagnostic("E_SELECT_V2_009", "irrep_select@2 range exceeds available multiplicity", node_id=node, details={"unavailable": unavailable})
        ])
    return selections


def _irrep_select_v2(node, inputs, attrs):
    x = _v2_equivariant(node, _single(node, inputs, "x"), "x")
    selections = _irrep_select_v2_spec(node, x, attrs)
    output = Irreps(tuple((multiplicity, irrep) for irrep, _start, multiplicity in selections)).simplify()
    return {"out": replace(x, irreps=output)}, ()


def _relative_position(node, inputs, attrs):
    source = _single(node, inputs, "source")
    target = _single(node, inputs, "target")
    _same_context(node, source, target)
    if source.carrier != Carrier.NODE or source.irreps.dimension != source.group.dimension:
        raise DSLValidationError([Diagnostic("E_GEOMETRY_001", "relative_position expects one Cartesian node position", node_id=node)])
    if source.group.translation != "relative_coordinates":
        raise DSLValidationError([Diagnostic("E_GEOMETRY_002", "task group does not authorize relative-coordinate translation handling", node_id=node)])
    return {"out": EquivariantType(source.group, Carrier.EDGE, source.irreps, Frame("global"), source.axes, source.dtype, source.measure, source.level)}, ()


def _displacement_inputs(node, inputs):
    positions = _single(node, inputs, "positions")
    source_index = _single(node, inputs, "source_index")
    target_index = _single(node, inputs, "target_index")
    if not isinstance(positions, AffinePointType):
        raise DSLValidationError([Diagnostic("E_AFFINE_005", "displacement requires AffinePointType positions", node_id=node, port="positions")])
    if positions.group.dimension != 3 or positions.group.family not in ("O3", "SO3"):
        raise DSLValidationError([Diagnostic("E_GEOMETRY_006", "displacement currently supports 3D O(3)/SO(3) groups", node_id=node)])
    for role, index in (("source", source_index), ("target", target_index)):
        if not isinstance(index, IndexMapType):
            raise DSLValidationError([Diagnostic("E_INDEX_011", "{} endpoint requires IndexMapType".format(role), node_id=node, port="{}_index".format(role))])
        if index.group != positions.group or index.source_carrier != Carrier.NODE or index.target_carrier != Carrier.EDGE or index.endpoint != role:
            raise DSLValidationError([Diagnostic("E_INDEX_012", "{} endpoint index is incompatible with affine positions".format(role), node_id=node)])
    if source_index.target_size is not None and target_index.target_size is not None and source_index.target_size != target_index.target_size:
        raise DSLValidationError([Diagnostic("E_INDEX_013", "source and target endpoint maps must describe the same edge count", node_id=node)])
    return positions, source_index, target_index


def _displacement_output(positions: AffinePointType) -> EquivariantTensorType:
    vector_irreps = Irreps.parse("1x1o" if positions.group.family == "O3" else "1x1", positions.group.family)
    return EquivariantTensorType(
        group=positions.group,
        carrier=Carrier.EDGE,
        irreps=vector_irreps,
        frame=Frame("global"),
        axes=positions.axes,
        dtype=positions.dtype,
        measure=positions.measure,
        level=EquivarianceLevel.CONSTRUCTIVE,
        axis_specs=positions.axis_specs,
    )


def _relative_displacement_v2(node, inputs, attrs):
    positions, _source_index, _target_index = _displacement_inputs(node, inputs)
    if positions.group.translation not in ("relative_coordinates", "explicit"):
        raise DSLValidationError([Diagnostic("E_GEOMETRY_007", "group does not authorize translation-aware displacement", node_id=node)])
    obligation = ProofObligation(
        "{}:translation".format(node),
        ObligationKind.TRANSLATION_COVARIANCE,
        node,
        "discharged",
        "core.relative_displacement@2",
        {"direction": "target-source"},
    )
    return {"out": _displacement_output(positions)}, (obligation,)


def _relative_displacement_v3(node, inputs, attrs):
    positions, _source_index, _target_index = _displacement_inputs(node, inputs)
    if positions.group.translation not in ("relative_coordinates", "explicit"):
        raise DSLValidationError([Diagnostic("E_GEOMETRY_007", "group does not authorize translation-aware displacement", node_id=node)])
    obligation = ProofObligation(
        "{}:translation".format(node),
        ObligationKind.TRANSLATION_COVARIANCE,
        node,
        "discharged",
        "core.relative_displacement@3",
        {"direction": "source-target"},
    )
    return {"out": _displacement_output(positions)}, (obligation,)


def _periodic_displacement(node, inputs, attrs):
    positions, _source_index, _target_index = _displacement_inputs(node, inputs)
    lattice = _single(node, inputs, "lattice")
    shift = _single(node, inputs, "lattice_shift")
    if not isinstance(lattice, LatticeType) or not isinstance(shift, LatticeShiftType):
        raise DSLValidationError([Diagnostic("E_LATTICE_013", "periodic_displacement requires LatticeType and LatticeShiftType", node_id=node)])
    if positions.group.periodicity != "lattice" or lattice.group != positions.group or shift.group != positions.group:
        raise DSLValidationError([Diagnostic("E_LATTICE_014", "periodic displacement inputs must share one periodic group", node_id=node)])
    if lattice.lattice_id != shift.lattice_id:
        raise DSLValidationError([Diagnostic("E_LATTICE_015", "lattice and shift references do not match", node_id=node, expected=lattice.lattice_id, actual=shift.lattice_id)])
    if lattice.dtype != positions.dtype or lattice.measure != positions.measure:
        raise DSLValidationError([Diagnostic("E_LATTICE_016", "lattice dtype and length measure must match affine positions", node_id=node)])
    if shift.convention != "target_image":
        raise DSLValidationError([Diagnostic("E_LATTICE_017", "periodic_displacement@1 requires target_image shifts", node_id=node)])
    obligations = (
        ProofObligation(
            "{}:translation".format(node), ObligationKind.TRANSLATION_COVARIANCE, node,
            "discharged", "core.periodic_displacement@1", {"direction": "target+shift@lattice-source"},
        ),
        ProofObligation(
            "{}:periodic-image".format(node), ObligationKind.PERIODIC_IMAGE_CONSISTENCY, node,
            "discharged", "core.periodic_displacement@1", {"lattice_id": lattice.lattice_id},
        ),
    )
    return {"out": _displacement_output(positions)}, obligations


def _periodic_displacement_v2(node, inputs, attrs):
    positions, _source_index, _target_index = _displacement_inputs(node, inputs)
    lattice = _single(node, inputs, "lattice")
    shift = _single(node, inputs, "lattice_shift")
    if not isinstance(lattice, LatticeType) or not isinstance(shift, LatticeShiftType):
        raise DSLValidationError([Diagnostic("E_LATTICE_V2_001", "periodic_displacement@2 requires lattice and shift inputs", node_id=node)])
    if positions.group.periodicity != "lattice" or lattice.group != positions.group or shift.group != positions.group:
        raise DSLValidationError([Diagnostic("E_LATTICE_V2_002", "periodic displacement inputs must share one periodic group", node_id=node)])
    if lattice.lattice_id != shift.lattice_id or shift.convention != "source_image":
        raise DSLValidationError([Diagnostic("E_LATTICE_V2_003", "periodic_displacement@2 requires a matching source_image shift", node_id=node)])
    if lattice.dtype != positions.dtype or lattice.measure != positions.measure:
        raise DSLValidationError([Diagnostic("E_LATTICE_V2_004", "lattice dtype and measure must match positions", node_id=node)])
    obligations = (
        ProofObligation(
            "{}:translation".format(node), ObligationKind.TRANSLATION_COVARIANCE, node,
            "discharged", "core.periodic_displacement@2", {"direction": "source+shift@lattice-target"},
        ),
        ProofObligation(
            "{}:periodic-image".format(node), ObligationKind.PERIODIC_IMAGE_CONSISTENCY, node,
            "discharged", "core.periodic_displacement@2", {"lattice_id": lattice.lattice_id},
        ),
    )
    return {"out": _displacement_output(positions)}, obligations


def _distance(node, inputs, attrs):
    vector = _single(node, inputs, "vector")
    if vector.carrier != Carrier.EDGE or vector.irreps.dimension != vector.group.dimension:
        raise DSLValidationError([Diagnostic("E_GEOMETRY_003", "distance expects one Cartesian edge vector", node_id=node)])
    scalar_family = vector.group.family
    suffix = "e" if scalar_family in ("O3", "O2") else ""
    scalar = Irreps.parse("1x0{}".format(suffix) if scalar_family in ("O3", "SO3") else "1xm0{}".format(suffix), scalar_family)
    return {"out": EquivariantType(vector.group, Carrier.EDGE, scalar, Frame("invariant"), vector.axes, vector.dtype, vector.measure, vector.level)}, ()


def _distance_v2(node, inputs, attrs):
    vector = _single(node, inputs, "vector")
    if not isinstance(vector, EquivariantTensorType):
        raise DSLValidationError([
            Diagnostic("E_DISTANCE_V2_001", "distance@2 requires EquivariantTensorType input", node_id=node)
        ])
    if vector.carrier != Carrier.EDGE or vector.irreps.dimension != vector.group.dimension:
        raise DSLValidationError([
            Diagnostic("E_DISTANCE_V2_002", "distance@2 expects one Cartesian edge vector", node_id=node)
        ])
    if vector.axis_specs:
        raise DSLValidationError([
            Diagnostic("E_DISTANCE_V2_003", "distance@2 first version requires an axis-free Cartesian vector", node_id=node)
        ])
    scalar = Irrep(0, 1, vector.group.family)
    return {
        "out": InvariantTensorType(
            group=vector.group,
            carrier=Carrier.EDGE,
            irreps=Irreps(((1, scalar),)),
            frame=Frame("invariant"),
            dtype=vector.dtype,
            measure=vector.measure,
            level=vector.level,
            feature_role=FeatureRole.CHANNEL,
        )
    }, ()


def _radial_basis(node, inputs, attrs):
    distance = _single(node, inputs, "distance")
    if distance.carrier != Carrier.EDGE or any(ir.degree != 0 for _, ir in distance.irreps):
        raise DSLValidationError([Diagnostic("E_GEOMETRY_004", "radial_basis expects invariant edge scalars", node_id=node)])
    count = int(attrs.get("num_basis", 0))
    if count <= 0:
        raise DSLValidationError([Diagnostic("E_ATTR_001", "num_basis must be positive", node_id=node)])
    family = distance.group.family
    scalar = "{}x0e".format(count) if family == "O3" else "{}x0".format(count) if family == "SO3" else "{}xm0e".format(count) if family == "O2" else "{}xm0".format(count)
    return {"out": distance.with_irreps(Irreps.parse(scalar, family)).with_measure("dimensionless")}, ()


def _fixed_gaussian_radial_basis(node, inputs, attrs):
    distance = _single(node, inputs, "distance")
    if not isinstance(distance, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_FIXED_GAUSSIAN_001",
                "fixed_gaussian_radial_basis requires InvariantTensorType distance input",
                node_id=node,
                actual=type(distance).__name__,
            )
        ])
    if distance.carrier != Carrier.EDGE or distance.irreps.dimension != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_FIXED_GAUSSIAN_002",
                "fixed Gaussian radial basis expects one invariant scalar per edge",
                node_id=node,
            )
        ])
    raw_count = attrs["num_basis"]
    if isinstance(raw_count, bool):
        count = 0
    else:
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            count = 0
    if count < 2 or str(raw_count) != str(count):
        raise DSLValidationError([
            Diagnostic(
                "E_FIXED_GAUSSIAN_003",
                "fixed Gaussian radial basis requires num_basis >= 2",
                node_id=node,
                actual=str(raw_count),
            )
        ])
    try:
        start = float(attrs.get("start", 0.0))
        stop = float(attrs["stop"])
        width_scalar = float(attrs.get("basis_width_scalar", 1.0))
    except (TypeError, ValueError):
        start = stop = width_scalar = float("nan")
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise DSLValidationError([
            Diagnostic(
                "E_FIXED_GAUSSIAN_004",
                "fixed Gaussian radial basis requires finite start < stop",
                node_id=node,
            )
        ])
    if not math.isfinite(width_scalar) or width_scalar <= 0.0:
        raise DSLValidationError([
            Diagnostic(
                "E_FIXED_GAUSSIAN_005",
                "basis_width_scalar must be finite and positive",
                node_id=node,
                actual=str(width_scalar),
            )
        ])
    axis_name = str(attrs.get("axis", "radial_basis"))
    if not axis_name:
        raise DSLValidationError([
            Diagnostic("E_FIXED_GAUSSIAN_006", "radial basis axis name must be nonempty", node_id=node)
        ])
    construction_dtype = str(attrs.get("construction_dtype", "float32"))
    if construction_dtype not in ("float32", "float64"):
        raise DSLValidationError([
            Diagnostic(
                "E_FIXED_GAUSSIAN_007",
                "fixed Gaussian construction_dtype must be float32 or float64",
                node_id=node,
                actual=construction_dtype,
            )
        ])
    scalar = Irrep(0, 1, distance.group.family)
    return {
        "out": InvariantTensorType(
            group=distance.group,
            carrier=Carrier.EDGE,
            irreps=Irreps(((count, scalar),)),
            frame=Frame("invariant"),
            axes=(axis_name,),
            dtype=distance.dtype,
            measure="dimensionless",
            level=distance.level,
            axis_specs=(AxisSpec(axis_name, count, FeatureRole.BASIS, "independent", 0),),
            feature_role=FeatureRole.BASIS,
        )
    }, ()


def _cutoff_envelope_v2(node, inputs, attrs):
    distance = _single(node, inputs, "x")
    if not isinstance(distance, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic("E_CUTOFF_V2_001", "cutoff_envelope@2 requires invariant distance input", node_id=node)
        ])
    if distance.carrier != Carrier.EDGE or distance.irreps.dimension != 1:
        raise DSLValidationError([
            Diagnostic("E_CUTOFF_V2_002", "cutoff_envelope@2 expects one scalar per edge", node_id=node)
        ])
    try:
        cutoff = float(attrs["cutoff"])
        order = int(attrs.get("order", 5))
    except (KeyError, TypeError, ValueError):
        cutoff = float("nan")
        order = 0
    if not math.isfinite(cutoff) or cutoff <= 0.0 or order <= 0:
        raise DSLValidationError([
            Diagnostic("E_CUTOFF_V2_003", "cutoff must be positive and order must be a positive integer", node_id=node)
        ])
    return {"out": replace(distance, measure="dimensionless")}, ()


def _axisymmetric_spherical_lift(node, inputs, attrs):
    amplitudes = _single(node, inputs, "amplitudes")
    direction = _single(node, inputs, "direction")
    if not isinstance(amplitudes, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic("E_AXISYMMETRIC_LIFT_001", "axisymmetric lift amplitudes must be invariant", node_id=node)
        ])
    if not isinstance(direction, EquivariantTensorType):
        raise DSLValidationError([
            Diagnostic("E_AXISYMMETRIC_LIFT_002", "axisymmetric lift direction must be an equivariant tensor", node_id=node)
        ])
    if (
        amplitudes.group != direction.group
        or amplitudes.carrier != Carrier.EDGE
        or direction.carrier != Carrier.EDGE
        or amplitudes.frame != Frame("invariant")
        or direction.frame != Frame("global")
        or amplitudes.dtype != direction.dtype
        or direction.irreps.dimension != direction.group.dimension
    ):
        raise DSLValidationError([
            Diagnostic(
                "E_AXISYMMETRIC_LIFT_003",
                "axisymmetric lift requires compatible invariant amplitudes and one global Cartesian edge vector",
                node_id=node,
            )
        ])
    if amplitudes.measure != "dimensionless":
        raise DSLValidationError([
            Diagnostic("E_AXISYMMETRIC_LIFT_004", "axisymmetric lift amplitudes must be dimensionless", node_id=node)
        ])
    output_irreps = Irreps.parse(str(attrs["out_irreps"]), amplitudes.group.family)
    degrees = [irrep.degree for _multiplicity, irrep in output_irreps]
    lmax = max(degrees) if degrees else -1
    multiplicities = {multiplicity for multiplicity, _irrep in output_irreps}
    if degrees != list(range(lmax + 1)) or len(multiplicities) != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_AXISYMMETRIC_LIFT_005",
                "axisymmetric lift output must contain every degree 0..lmax with uniform multiplicity",
                node_id=node,
                actual=str(output_irreps),
            )
        ])
    channels = next(iter(multiplicities))
    if amplitudes.irreps.dimension != (lmax + 1) * channels:
        raise DSLValidationError([
            Diagnostic(
                "E_AXISYMMETRIC_LIFT_006",
                "axisymmetric lift requires one m=0 amplitude per degree and channel",
                node_id=node,
                expected=str((lmax + 1) * channels),
                actual=str(amplitudes.irreps.dimension),
            )
        ])
    mmax = int(attrs.get("mmax", lmax))
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic("E_AXISYMMETRIC_LIFT_007", "mmax must satisfy 0 <= mmax <= lmax", node_id=node)
        ])
    if not isinstance(attrs.get("use_rotation_mask", False), bool):
        raise DSLValidationError([
            Diagnostic("E_AXISYMMETRIC_LIFT_008", "use_rotation_mask must be boolean", node_id=node)
        ])
    return {
        "out": EquivariantTensorType(
            group=amplitudes.group,
            carrier=Carrier.EDGE,
            irreps=output_irreps,
            frame=Frame("global"),
            dtype=amplitudes.dtype,
            measure="dimensionless",
            level=min(amplitudes.level, direction.level),
        )
    }, ()


def _gaussian_radial_basis(node, inputs, attrs):
    distance = _single(node, inputs, "distance")
    if not isinstance(distance, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_GAUSSIAN_RBF_001",
                "gaussian_radial_basis requires InvariantTensorType distance input",
                node_id=node,
                actual=type(distance).__name__,
            )
        ])
    if distance.carrier != Carrier.EDGE or distance.irreps.dimension != 1:
        raise DSLValidationError([
            Diagnostic("E_GAUSSIAN_RBF_002", "gaussian radial basis expects one invariant scalar per edge", node_id=node)
        ])
    raw_count = attrs["num_basis"]
    if isinstance(raw_count, bool):
        count = 0
    else:
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            count = 0
    if count <= 0 or str(raw_count) != str(count):
        raise DSLValidationError([
            Diagnostic("E_GAUSSIAN_RBF_003", "num_basis must be a positive integer", node_id=node, actual=str(raw_count))
        ])
    try:
        cutoff = float(attrs["cutoff"])
    except (TypeError, ValueError):
        cutoff = float("nan")
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise DSLValidationError([
            Diagnostic("E_GAUSSIAN_RBF_004", "cutoff must be finite and positive", node_id=node, actual=str(attrs["cutoff"]))
        ])
    axis_name = str(attrs.get("axis", "radial_channel"))
    if not axis_name:
        raise DSLValidationError([Diagnostic("E_GAUSSIAN_RBF_005", "radial basis axis must be nonempty", node_id=node)])
    scalar = Irrep(0, 1, distance.group.family)
    return {
        "out": InvariantTensorType(
            group=distance.group,
            carrier=Carrier.EDGE,
            irreps=Irreps(((count, scalar),)),
            frame=Frame("invariant"),
            axes=(axis_name,),
            dtype=distance.dtype,
            measure="dimensionless",
            level=distance.level,
            axis_specs=(AxisSpec(axis_name, count, FeatureRole.BASIS, "independent", 0),),
            feature_role=FeatureRole.BASIS,
        )
    }, ()


def _gaussian_radial_basis_parameters(node, inputs, outputs, attrs):
    del outputs
    distance = _single(node, inputs, "distance")
    count = int(attrs["num_basis"])
    family = distance.group.family
    scalar_bias = (
        "1x0e" if family == "O3" else
        "1x0" if family == "SO3" else
        "1xm0e" if family == "O2" else
        "1xm0"
    )
    return (
        ParameterContract(
            "mean",
            (
                ParameterAxis("broadcast", 1, "broadcast"),
                ParameterAxis("basis", count, "radial_basis"),
            ),
            initializer="uniform_0_1",
            checkpoint_names=("{}.mean".format(node),),
            backend_parameter_name="mean",
        ),
        ParameterContract(
            "std",
            (
                ParameterAxis("broadcast", 1, "broadcast"),
                ParameterAxis("basis", count, "radial_basis"),
            ),
            initializer="uniform_1_over_basis_1",
            checkpoint_names=("{}.std".format(node),),
            backend_parameter_name="std",
        ),
        ParameterContract(
            "weight",
            (
                ParameterAxis("output_broadcast", 1, "scalar_affine_output"),
                ParameterAxis("input_broadcast", 1, "scalar_affine_input"),
            ),
            initializer="ones",
            checkpoint_names=("{}.weight".format(node),),
            backend_parameter_name="weight",
        ),
        ParameterContract(
            "bias",
            (
                ParameterAxis("output_broadcast", 1, "scalar_affine_output"),
                ParameterAxis("input_broadcast", 1, "scalar_affine_input"),
            ),
            is_bias=True,
            bias_irreps=scalar_bias,
            initializer="zeros",
            checkpoint_names=("{}.bias".format(node),),
            backend_parameter_name="bias",
        ),
    )


def _spherical_harmonics(node, inputs, attrs):
    direction = _single(node, inputs, "direction")
    if direction.carrier != Carrier.EDGE or direction.group.dimension != 3:
        raise DSLValidationError([Diagnostic("E_GEOMETRY_005", "spherical_harmonics currently supports 3D edge directions", node_id=node)])
    lmax = int(attrs.get("lmax", -1))
    if lmax < 0:
        raise DSLValidationError([Diagnostic("E_ATTR_002", "lmax must be nonnegative", node_id=node)])
    terms = []
    for degree in range(lmax + 1):
        parity = 1 if direction.group.family == "SO3" or degree % 2 == 0 else -1
        terms.append((1, Irrep(degree, parity, direction.group.family)))
    return {"out": EquivariantType(direction.group, Carrier.EDGE, Irreps(tuple(terms)), Frame("global"), direction.axes, direction.dtype, "dimensionless", direction.level)}, ()


def _norm_activation(node, inputs, attrs):
    return {"out": _single(node, inputs, "x")}, ()


def _gate(node, inputs, attrs):
    gates = _single(node, inputs, "gates")
    value = _single(node, inputs, "value")
    if gates.group != value.group or gates.carrier != value.carrier or gates.frame not in (value.frame, Frame("invariant")):
        raise DSLValidationError([Diagnostic("E_GATE_001", "gates and values have incompatible contexts", node_id=node)])
    if any(ir.degree != 0 or (ir.family in ("O3", "O2") and ir.parity != 1) for _, ir in gates.irreps):
        raise DSLValidationError([Diagnostic("E_GATE_002", "gates must be invariant scalars", node_id=node)])
    if any(ir.degree == 0 for _, ir in value.irreps):
        raise DSLValidationError([Diagnostic("E_GATE_004", "gate values must contain only non-scalar irreps; activate scalar paths separately", node_id=node)])
    required = sum(mul for mul, ir in value.irreps if ir.degree != 0)
    available = sum(mul for mul, _ in gates.irreps)
    if available not in (1, required) and required:
        raise DSLValidationError([Diagnostic("E_GATE_003", "gate multiplicity must be one shared gate or one gate per non-scalar multiplicity", node_id=node, expected="1 or {}".format(required), actual=str(available))])
    return {"out": value}, ()


def _segment_mean(node, inputs, attrs):
    return _segment_sum(node, inputs, attrs)


def _segment_softmax(node, inputs, attrs):
    logits = _single(node, inputs, "logits")
    if logits.carrier != Carrier.EDGE or any(ir.degree != 0 or (ir.family in ("O3", "O2") and ir.parity != 1) for _, ir in logits.irreps):
        raise DSLValidationError([Diagnostic("E_ATTN_003", "segment_softmax expects invariant scalar edge logits", node_id=node)])
    if logits.measure != "dimensionless":
        raise DSLValidationError([Diagnostic("E_UNIT_003", "segment_softmax logits must be dimensionless", node_id=node, actual=logits.measure)])
    return {"out": logits}, ()


def _segment_softmax_v2(node, inputs, attrs):
    logits = _single(node, inputs, "logits")
    index = _single(node, inputs, "index")
    if not isinstance(logits, InvariantTensorType) or logits.carrier != Carrier.EDGE:
        raise DSLValidationError([
            Diagnostic(
                "E_SOFTMAX_V2_001",
                "segment_softmax@2 requires edge-carried InvariantTensorType logits",
                node_id=node,
                port="logits",
                actual=type(logits).__name__,
            )
        ])
    if logits.measure != "dimensionless":
        raise DSLValidationError([
            Diagnostic("E_SOFTMAX_V2_002", "segment_softmax@2 logits must be dimensionless", node_id=node, actual=logits.measure)
        ])
    if not isinstance(index, IndexMapType):
        raise DSLValidationError([
            Diagnostic("E_SOFTMAX_V2_003", "segment_softmax@2 requires an explicit IndexMapType", node_id=node, port="index")
        ])
    if (
        index.group != logits.group
        or index.source_carrier != Carrier.EDGE
        or index.endpoint != "segment"
    ):
        raise DSLValidationError([
            Diagnostic(
                "E_SOFTMAX_V2_004",
                "segment_softmax@2 index must map edge logits to destination segments under the same group",
                node_id=node,
            )
        ])
    return {"out": logits}, ()


def _segment_softmax_v3(node, inputs, attrs):
    logits = _single(node, inputs, "logits")
    index = _single(node, inputs, "index")
    exp_rescale = _single(node, inputs, "exp_rescale")
    _segment_softmax_v2(node, {"logits": (logits,), "index": (index,)}, attrs)
    if not isinstance(exp_rescale, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic("E_SOFTMAX_V3_001", "segment_softmax@3 exp_rescale must be invariant", node_id=node)
        ])
    if (
        exp_rescale.group != logits.group
        or exp_rescale.carrier != logits.carrier
        or exp_rescale.frame != Frame("invariant")
        or exp_rescale.dtype != logits.dtype
        or exp_rescale.measure != "dimensionless"
    ):
        raise DSLValidationError([
            Diagnostic("E_SOFTMAX_V3_002", "segment_softmax@3 exp_rescale has an incompatible context", node_id=node)
        ])
    if exp_rescale.irreps.dimension not in (1, logits.irreps.dimension):
        raise DSLValidationError([
            Diagnostic(
                "E_SOFTMAX_V3_003",
                "segment_softmax@3 exp_rescale must be scalar or match the logit width",
                node_id=node,
            )
        ])
    try:
        epsilon = float(attrs.get("epsilon", 1.0e-16))
        dropout = float(attrs.get("exp_dropout", 0.0))
    except (TypeError, ValueError):
        epsilon, dropout = float("nan"), float("nan")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise DSLValidationError([
            Diagnostic("E_SOFTMAX_V3_004", "segment_softmax@3 epsilon must be finite and positive", node_id=node)
        ])
    if not math.isfinite(dropout) or dropout < 0.0 or dropout >= 1.0:
        raise DSLValidationError([
            Diagnostic("E_SOFTMAX_V3_005", "segment_softmax@3 exp_dropout must satisfy 0 <= p < 1", node_id=node)
        ])
    softcap = attrs.get("softcap")
    if softcap is not None:
        try:
            softcap = float(softcap)
        except (TypeError, ValueError):
            softcap = float("nan")
        if not math.isfinite(softcap) or softcap <= 0.0:
            raise DSLValidationError([
                Diagnostic("E_SOFTMAX_V3_006", "segment_softmax@3 softcap must be finite and positive", node_id=node)
            ])
    return {"out": replace(logits, level=min(logits.level, exp_rescale.level))}, ()


def _equivariant_preserving(node, inputs, attrs):
    return {"out": _single(node, inputs, "x")}, ()


def _irrep_layer_norm(node, inputs, attrs):
    x = _v2_equivariant(node, _single(node, inputs, "x"), "x")
    if x.axis_specs:
        raise DSLValidationError([
            Diagnostic("E_IRREP_LN_001", "irrep_layer_norm requires axis-free equivariant input", node_id=node)
        ])
    if (
        x.irreps != x.irreps.simplify()
        or x.layout.storage != "irrep_major"
        or x.layout.coefficient_order != "canonical"
    ):
        raise DSLValidationError([
            Diagnostic("E_IRREP_LN_002", "irrep_layer_norm requires canonical simplified irrep-major input", node_id=node)
        ])
    epsilon = float(attrs.get("epsilon", 1.0e-5))
    if epsilon <= 0.0:
        raise DSLValidationError([
            Diagnostic("E_IRREP_LN_003", "irrep_layer_norm epsilon must be positive", node_id=node, actual=str(epsilon))
        ])
    affine = attrs.get("affine", True)
    if not isinstance(affine, bool):
        raise DSLValidationError([
            Diagnostic("E_IRREP_LN_004", "irrep_layer_norm affine must be boolean", node_id=node, actual=str(affine))
        ])
    if not affine:
        raise DSLValidationError([
            Diagnostic(
                "E_IRREP_LN_006",
                "irrep_layer_norm@1 requires affine=True to match the supported Equiformer V1 LayerNormV2 path",
                node_id=node,
            )
        ])
    normalization = str(attrs.get("normalization", "component"))
    if normalization not in ("component", "norm"):
        raise DSLValidationError([
            Diagnostic(
                "E_IRREP_LN_005",
                "irrep_layer_norm normalization must be component or norm",
                node_id=node,
                actual=normalization,
            )
        ])
    return {"out": x}, ()


def _irrep_layer_norm_parameters(node, inputs, outputs, attrs):
    if not attrs.get("affine", True):
        return ()
    x = _single(node, inputs, "x")
    feature_count = sum(multiplicity for multiplicity, _irrep in x.irreps)
    scalar_count = sum(
        multiplicity
        for multiplicity, irrep in x.irreps
        if irrep.degree == 0 and irrep.parity == 1
    )
    contracts = [
        ParameterContract(
            "affine_weight",
            (ParameterAxis("irrep_instance", feature_count, "complete_irrep_affine_scale"),),
            initializer="ones",
            checkpoint_names=("{}.affine_weight".format(node),),
            backend_parameter_name="affine_weight",
        )
    ]
    if scalar_count:
        contracts.append(
            ParameterContract(
                "affine_bias",
                (ParameterAxis("trivial_scalar", scalar_count, "centered_scalar_affine_bias"),),
                is_bias=True,
                bias_irreps="{}x0e".format(scalar_count),
                initializer="zeros",
                checkpoint_names=("{}.affine_bias".format(node),),
                backend_parameter_name="affine_bias",
            )
        )
    return tuple(contracts)


def _dropout_probability(node, attrs):
    probability = float(attrs.get("p", 0.0))
    if probability < 0.0 or probability >= 1.0:
        raise DSLValidationError([Diagnostic("E_ATTR_003", "dropout probability must satisfy 0 <= p < 1", node_id=node, actual=str(probability))])
    return probability


def _dropout_preserving(node, inputs, attrs):
    _dropout_probability(node, attrs)
    return {"out": _single(node, inputs, "x")}, ()


def _graph_stochastic_depth(node, inputs, attrs):
    _dropout_probability(node, attrs)
    x = _single(node, inputs, "x")
    batch = _single(node, inputs, "batch")
    if not isinstance(x, EquivariantTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_GRAPH_DROP_001",
                "graph_stochastic_depth requires an EquivariantTensorType value",
                node_id=node,
                port="x",
                actual=type(x).__name__,
            )
        ])
    if not isinstance(batch, IndexMapType):
        raise DSLValidationError([
            Diagnostic(
                "E_GRAPH_DROP_002",
                "graph_stochastic_depth requires an explicit batch IndexMapType",
                node_id=node,
                port="batch",
                actual=type(batch).__name__,
            )
        ])
    if (
        batch.endpoint != "batch"
        or batch.source_carrier != x.carrier
        or batch.target_carrier != Carrier.GRAPH
        or batch.group != x.group
        or batch.allows_empty_targets
    ):
        raise DSLValidationError([
            Diagnostic(
                "E_GRAPH_DROP_003",
                "batch map must connect the value carrier to graph carrier in the same group",
                node_id=node,
                port="batch",
                expected="{}->graph endpoint=batch group={} allows_empty_targets=False".format(x.carrier, x.group),
                actual="{}->{} endpoint={} group={} allows_empty_targets={}".format(
                    batch.source_carrier,
                    batch.target_carrier,
                    batch.endpoint,
                    batch.group,
                    batch.allows_empty_targets,
                ),
            )
        ])
    return {"out": x}, ()


def _scalar_dropout(node, inputs, attrs):
    _dropout_probability(node, attrs)
    x = _single(node, inputs, "x")
    if not isinstance(x, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_SCALAR_DROP_001",
                "scalar_dropout requires InvariantTensorType input",
                node_id=node,
                port="x",
                actual=type(x).__name__,
            )
        ])
    return {"out": x}, ()


def _equivariant_dropout_v1(node, inputs, attrs):
    _dropout_probability(node, attrs)
    x = _single(node, inputs, "x")
    if not isinstance(x, EquivariantTensorType) or x.axis_specs:
        raise DSLValidationError([
            Diagnostic(
                "E_EQ_DROPOUT_001",
                "equivariant_dropout requires axis-free EquivariantTensorType input",
                node_id=node,
                port="x",
                actual=type(x).__name__,
            )
        ])
    if (
        x.irreps != x.irreps.simplify()
        or x.layout.storage != "irrep_major"
        or x.layout.coefficient_order != "canonical"
    ):
        raise DSLValidationError([
            Diagnostic(
                "E_EQ_DROPOUT_002",
                "equivariant_dropout requires canonical simplified irrep-major storage",
                node_id=node,
                expected="irrep_major/canonical simplified",
                actual="{}/{} {}".format(x.layout.storage, x.layout.coefficient_order, x.irreps),
            )
        ])
    return {"out": x}, ()


def _so2_convolution(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if x.group.dimension != 3 or x.frame.kind != "edge":
        raise DSLValidationError([Diagnostic("E_SO2_001", "SO(2) convolution requires 3D features expressed in an edge frame", node_id=node, actual=str(x.frame))])
    out = Irreps.parse(str(attrs.get("out_irreps", str(x.irreps))), x.group.family)
    input_kinds = {ir for _, ir in x.irreps}
    unavailable = [str(ir) for _, ir in out if ir not in input_kinds]
    if unavailable:
        raise DSLValidationError([
            Diagnostic("E_SO2_002", "SO(2) convolution cannot create representation degrees absent from its edge-frame input", node_id=node, details={"unavailable": unavailable})
        ])
    lmax = max(ir.degree for _, ir in x.irreps)
    mmax = int(attrs.get("mmax", lmax))
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic(
                "E_SO2_003",
                "mmax must satisfy 0 <= mmax <= lmax",
                node_id=node,
                expected="0..{}".format(lmax),
                actual=str(mmax),
            )
        ])
    return {"out": x.with_irreps(out)}, ()


def _so2_linear_layout(node, x, out_irreps, mmax):
    if not isinstance(x, EquivariantType):
        raise DSLValidationError([
            Diagnostic("E_SO2_LINEAR_001", "SO(2) linear requires an equivariant tensor", node_id=node)
        ])
    if x.group.family != "SO3" or x.carrier != Carrier.EDGE or x.frame.kind != "edge":
        raise DSLValidationError([
            Diagnostic(
                "E_SO2_LINEAR_002",
                "SO(2) linear requires an SO(3) edge-carried tensor in an edge frame",
                node_id=node,
                actual="{}/{}/{}".format(x.group.family, x.carrier, x.frame),
            )
        ])
    input_degrees = [irrep.degree for _, irrep in x.irreps]
    if not input_degrees:
        raise DSLValidationError([
            Diagnostic("E_SO2_LINEAR_003", "SO(2) linear input irreps must be nonempty", node_id=node)
        ])
    lmax = max(input_degrees)
    if input_degrees != list(range(lmax + 1)) or len({multiplicity for multiplicity, _ in x.irreps}) != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_SO2_LINEAR_004",
                "SO(2) linear requires every degree from zero through lmax with uniform multiplicity",
                node_id=node,
                actual=str(x.irreps),
            )
        ])
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic(
                "E_SO2_LINEAR_005",
                "mmax must satisfy 0 <= mmax <= lmax",
                node_id=node,
                expected="0..{}".format(lmax),
                actual=str(mmax),
            )
        ])
    output_degrees = [irrep.degree for _, irrep in out_irreps]
    if (
        out_irreps.family != "SO3"
        or output_degrees != input_degrees
        or len({multiplicity for multiplicity, _ in out_irreps}) != 1
    ):
        raise DSLValidationError([
            Diagnostic(
                "E_SO2_LINEAR_006",
                "SO(2) linear output must preserve all degrees with one uniform output multiplicity",
                node_id=node,
                expected="degrees 0..{}".format(lmax),
                actual=str(out_irreps),
            )
        ])
    return lmax, next(iter(multiplicity for multiplicity, _ in x.irreps)), next(
        iter(multiplicity for multiplicity, _ in out_irreps)
    )


def _so2_linear_common(node, inputs, attrs, *, with_extra_m0):
    x = _single(node, inputs, "x")
    out_irreps = Irreps.parse(str(attrs["out_irreps"]), x.group.family)
    lmax = max(irrep.degree for _, irrep in x.irreps) if x.irreps.terms else -1
    mmax = int(attrs.get("mmax", lmax))
    _lmax, _input_channels, output_channels = _so2_linear_layout(
        node, x, out_irreps, mmax
    )
    raw_extra = attrs.get("extra_m0_channels", 0)
    if isinstance(raw_extra, bool):
        extra_m0_channels = -1
    else:
        try:
            extra_m0_channels = int(raw_extra)
        except (TypeError, ValueError):
            extra_m0_channels = -1
    if with_extra_m0 and extra_m0_channels <= 0:
        raise DSLValidationError([
            Diagnostic(
                "E_SO2_LINEAR_007",
                "SO(2) linear@2 requires a positive extra_m0_channels value",
                node_id=node,
                actual=str(raw_extra),
            )
        ])
    if not with_extra_m0 and extra_m0_channels != 0:
        raise DSLValidationError([
            Diagnostic(
                "E_SO2_LINEAR_008",
                "SO(2) linear@1 does not produce an extra m=0 output",
                node_id=node,
                actual=str(raw_extra),
            )
        ])
    total_m0_rows = (lmax + 1) * output_channels + extra_m0_channels
    prefix_rows = int(attrs.get("m0_prefix_rows", 0))
    prefix_scale = float(attrs.get("m0_prefix_scale", 1.0))
    if prefix_rows < 0 or prefix_rows > total_m0_rows:
        raise DSLValidationError([
            Diagnostic(
                "E_SO2_LINEAR_009",
                "m0_prefix_rows must lie within the m=0 output row range",
                node_id=node,
                expected="0..{}".format(total_m0_rows),
                actual=str(prefix_rows),
            )
        ])
    if not math.isfinite(prefix_scale) or prefix_scale <= 0.0:
        raise DSLValidationError([
            Diagnostic(
                "E_SO2_LINEAR_010",
                "m0_prefix_scale must be finite and positive",
                node_id=node,
                actual=str(prefix_scale),
            )
        ])
    if not isinstance(attrs.get("zero_bias", False), bool):
        raise DSLValidationError([
            Diagnostic("E_SO2_LINEAR_011", "zero_bias must be boolean", node_id=node)
        ])
    outputs = {"out": x.with_irreps(out_irreps)}
    if with_extra_m0:
        scalar = Irrep(0, 1, x.group.family)
        outputs["extra_m0"] = InvariantTensorType(
            group=x.group,
            carrier=x.carrier,
            irreps=Irreps(((extra_m0_channels, scalar),)),
            frame=Frame("invariant"),
            axes=("m0_channel",),
            dtype=x.dtype,
            measure=x.measure,
            level=x.level,
            axis_specs=(
                AxisSpec("m0_channel", extra_m0_channels, FeatureRole.CHANNEL, "independent", 0),
            ),
            feature_role=FeatureRole.CHANNEL,
        )
    return outputs, ()


def _so2_linear_v1(node, inputs, attrs):
    return _so2_linear_common(node, inputs, attrs, with_extra_m0=False)


def _so2_linear_v2(node, inputs, attrs):
    return _so2_linear_common(node, inputs, attrs, with_extra_m0=True)


def _so2_linear_parameters(node, inputs, outputs, attrs):
    x = _single(node, inputs, "x")
    output = outputs["out"]
    lmax = max(irrep.degree for _, irrep in x.irreps)
    input_channels = next(iter(multiplicity for multiplicity, _ in x.irreps))
    output_channels = next(iter(multiplicity for multiplicity, _ in output.irreps))
    extra_m0_channels = int(attrs.get("extra_m0_channels", 0))
    input_m0 = (lmax + 1) * input_channels
    output_m0 = (lmax + 1) * output_channels + extra_m0_channels
    prefix_rows = int(attrs.get("m0_prefix_rows", 0))
    prefix_scale = float(attrs.get("m0_prefix_scale", 1.0))
    initializer = "kaiming_uniform"
    if prefix_rows:
        initializer = "kaiming_uniform_prefix_{}_rows_times_{:.17g}".format(
            prefix_rows, prefix_scale
        )
    contracts = [
        ParameterContract(
            "fc_m0_weight",
            (
                ParameterAxis("out_m0", output_m0, "m0_output"),
                ParameterAxis("in_m0", input_m0, "m0_input"),
            ),
            initializer=initializer,
            checkpoint_names=("{}.fc_m0.weight".format(node),),
            backend_parameter_name="fc_m0.weight",
        ),
        ParameterContract(
            "fc_m0_bias",
            (ParameterAxis("out_m0", output_m0, "m0_output"),),
            is_bias=True,
            bias_irreps="{}xm0".format(output_m0),
            initializer="zeros" if attrs.get("zero_bias", False) else "uniform_fan_in",
            checkpoint_names=("{}.fc_m0.bias".format(node),),
            backend_parameter_name="fc_m0.bias",
        ),
    ]
    mmax = int(attrs.get("mmax", lmax))
    for m_value in range(1, mmax + 1):
        component_count = lmax - m_value + 1
        input_width = component_count * input_channels
        output_width = component_count * output_channels
        backend_name = "so2_m_linear.{}.fc.weight".format(m_value - 1)
        contracts.append(
            ParameterContract(
                "m{}_weight".format(m_value),
                (
                    ParameterAxis("complex_out", 2 * output_width, "so2_complex_output"),
                    ParameterAxis("in", input_width, "so2_order_input"),
                ),
                initializer="kaiming_uniform_times_inv_sqrt2",
                rescale=1.0 / math.sqrt(2.0),
                checkpoint_names=("{}.{}".format(node, backend_name),),
                backend_parameter_name=backend_name,
            )
        )
    return tuple(contracts)


def _validate_s2_parameters(node, x, attrs):
    if x.group.dimension != 3:
        raise DSLValidationError([
            Diagnostic("E_S2_003", "S2 activation requires a 3D rotation group", node_id=node)
        ])
    lmax = max(ir.degree for _, ir in x.irreps)
    mmax = int(attrs.get("mmax", lmax))
    resolution_value = attrs.get("grid_resolution", 18)
    if isinstance(resolution_value, (list, tuple)):
        resolutions = tuple(int(value) for value in resolution_value)
    else:
        resolutions = (int(resolution_value),)
    normalization = str(attrs.get("normalization", "component"))
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic(
                "E_S2_004",
                "mmax must satisfy 0 <= mmax <= lmax",
                node_id=node,
                expected="0..{}".format(lmax),
                actual=str(mmax),
            )
        ])
    if not resolutions or len(resolutions) > 2 or any(value < 2 for value in resolutions):
        raise DSLValidationError([
            Diagnostic(
                "E_S2_005",
                "grid_resolution must be one or two integers, each at least two",
                node_id=node,
            )
        ])
    if normalization not in ("component", "integral", "norm"):
        raise DSLValidationError([
            Diagnostic(
                "E_S2_006",
                "unsupported S2 grid normalization",
                node_id=node,
                actual=normalization,
            )
        ])


def _s2_activation(node, inputs, attrs):
    x = _single(node, inputs, "x")
    _validate_s2_parameters(node, x, attrs)
    return {"out": x}, ()


def _separable_s2_activation(node, inputs, attrs):
    scalars = _single(node, inputs, "scalars")
    x = _single(node, inputs, "x")
    _validate_s2_parameters(node, x, attrs)
    _same_context(node, scalars, x, exact_irreps=False)
    if any(ir.degree != 0 for _, ir in scalars.irreps):
        raise DSLValidationError([Diagnostic("E_S2_001", "separable S2 activation requires a scalar side path", node_id=node, port="scalars")])
    scalar_count = sum(mul for mul, _ in scalars.irreps)
    x_scalar_count = sum(mul for mul, ir in x.irreps if ir.degree == 0)
    if scalar_count != x_scalar_count:
        raise DSLValidationError([Diagnostic("E_S2_002", "scalar side path must match the l=0 multiplicity", node_id=node, expected=str(x_scalar_count), actual=str(scalar_count))])
    return {"out": x}, ()


def _uniform_so3_multiplicities(node, irreps):
    if not irreps.terms:
        raise DSLValidationError([
            Diagnostic("E_V3_TYPE_001", "V3 dense operators require nonempty irreps", node_id=node)
        ])
    degrees = [ir.degree for _, ir in irreps]
    lmax = max(degrees)
    if degrees != list(range(lmax + 1)):
        raise DSLValidationError([
            Diagnostic(
                "E_V3_TYPE_002",
                "V3 dense operators require every degree from zero through lmax",
                node_id=node,
                actual=str(irreps),
            )
        ])
    multiplicities = {mul for mul, _ in irreps}
    if len(multiplicities) != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_TYPE_003",
                "V3 dense operators require one uniform multiplicity across degrees",
                node_id=node,
                actual=str(irreps),
            )
        ])
    return lmax, next(iter(multiplicities))


def _grid_irreps_with_channels(source_irreps, channels):
    return Irreps(tuple((int(channels), irrep) for _multiplicity, irrep in source_irreps))


def _grid_project(node, inputs, attrs):
    x = _v2_equivariant(node, _single(node, inputs, "x"), "x")
    if x.axis_specs or x.frame.kind == "invariant":
        raise DSLValidationError([
            Diagnostic("E_GRID_PROJECT_001", "grid_project requires axis-free non-invariant coefficients", node_id=node)
        ])
    if x.layout.storage != "irrep_major" or x.layout.coefficient_order != "canonical":
        raise DSLValidationError([
            Diagnostic(
                "E_GRID_PROJECT_002",
                "grid_project requires canonical irrep-major coefficient storage",
                node_id=node,
                actual="{}/{}".format(x.layout.storage, x.layout.coefficient_order),
            )
        ])
    lmax, channels = _uniform_so3_multiplicities(node, x.irreps)
    raw_resolution = attrs["grid_resolution"]
    if not isinstance(raw_resolution, (list, tuple)) or len(raw_resolution) != 2:
        raise DSLValidationError([
            Diagnostic("E_GRID_PROJECT_003", "grid_resolution must contain latitude and longitude", node_id=node)
        ])
    try:
        latitude, longitude = (int(value) for value in raw_resolution)
        mmax = int(attrs.get("mmax", lmax))
    except (TypeError, ValueError):
        raise DSLValidationError([
            Diagnostic("E_GRID_PROJECT_004", "grid resolution and mmax must be integers", node_id=node)
        ])
    use_m_primary = attrs.get("use_m_primary", False)
    if not isinstance(use_m_primary, bool):
        raise DSLValidationError([
            Diagnostic("E_GRID_PROJECT_005", "use_m_primary must be boolean", node_id=node)
        ])
    grid = GridSpec(
        latitude=latitude,
        longitude=longitude,
        lmax=lmax,
        mmax=mmax,
        normalization=str(attrs.get("normalization", "component")),
        quadrature=str(attrs.get("quadrature", "e3nn_s2grid")),
        sampling=str(attrs.get("sampling", "equiangular")),
        use_m_primary=use_m_primary,
    )
    return {
        "out": GridTensorType(
            group=x.group,
            carrier=x.carrier,
            source_irreps=x.irreps,
            grid=grid,
            channels=channels,
            frame=x.frame,
            dtype=x.dtype,
            measure=x.measure,
            level=x.level,
            channel_role=FeatureRole.CHANNEL,
            aliasing_model="linear_projection_only",
        )
    }, ()


def _grid_unproject(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if not isinstance(x, GridTensorType):
        raise DSLValidationError([
            Diagnostic("E_GRID_UNPROJECT_001", "grid_unproject requires GridTensorType input", node_id=node)
        ])
    output_irreps = Irreps.parse(str(attrs.get("out_irreps", x.source_irreps)), x.group.family)
    lmax, channels = _uniform_so3_multiplicities(node, output_irreps)
    if lmax != x.grid.lmax or channels != x.channels or tuple(ir for _, ir in output_irreps) != tuple(ir for _, ir in x.source_irreps):
        raise DSLValidationError([
            Diagnostic(
                "E_GRID_UNPROJECT_002",
                "grid_unproject output must preserve the grid bandlimit, irrep kinds, and channel count",
                node_id=node,
                expected=str(x.source_irreps),
                actual=str(output_irreps),
            )
        ])
    return {
        "out": EquivariantTensorType(
            group=x.group,
            carrier=x.carrier,
            irreps=output_irreps,
            frame=x.frame,
            dtype=x.dtype,
            measure=x.measure,
            level=x.level,
        )
    }, ()


def _grid_split(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if not isinstance(x, GridTensorType):
        raise DSLValidationError([
            Diagnostic("E_GRID_SPLIT_001", "grid_split requires GridTensorType input", node_id=node)
        ])
    try:
        left_channels = int(attrs["left_channels"])
    except (TypeError, ValueError):
        left_channels = 0
    if left_channels <= 0 or left_channels >= x.channels:
        raise DSLValidationError([
            Diagnostic(
                "E_GRID_SPLIT_002",
                "grid_split left_channels must lie strictly inside the channel range",
                node_id=node,
                actual=str(attrs.get("left_channels")),
            )
        ])
    right_channels = x.channels - left_channels
    return {
        "left": x.with_channels(left_channels, _grid_irreps_with_channels(x.source_irreps, left_channels)),
        "right": x.with_channels(right_channels, _grid_irreps_with_channels(x.source_irreps, right_channels)),
    }, ()


def _same_grid_contract(node, left, right):
    fields = (
        "group", "carrier", "grid", "frame", "dtype", "measure", "level",
        "aliasing_model",
    )
    mismatches = [name for name in fields if getattr(left, name) != getattr(right, name)]
    if mismatches or tuple(ir for _, ir in left.source_irreps) != tuple(ir for _, ir in right.source_irreps):
        raise DSLValidationError([
            Diagnostic(
                "E_GRID_CONTEXT_001",
                "grid values have incompatible sampling or representation contexts",
                node_id=node,
                details={"mismatches": mismatches},
            )
        ])


def _grid_concat(node, inputs, attrs):
    del attrs
    values = tuple(inputs.get("xs", ()))
    if not values or any(not isinstance(value, GridTensorType) for value in values):
        raise DSLValidationError([
            Diagnostic("E_GRID_CONCAT_001", "grid_concat requires one or more GridTensorType values", node_id=node)
        ])
    first = values[0]
    for value in values[1:]:
        _same_grid_contract(node, first, value)
    channels = sum(value.channels for value in values)
    return {
        "out": first.with_channels(channels, _grid_irreps_with_channels(first.source_irreps, channels))
    }, ()


def _grid_pointwise_activation(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if not isinstance(x, GridTensorType):
        raise DSLValidationError([
            Diagnostic("E_GRID_ACT_001", "grid pointwise activation requires GridTensorType input", node_id=node)
        ])
    activation = str(attrs.get("activation", "silu"))
    if activation not in ("identity", "silu", "sigmoid", "square"):
        raise DSLValidationError([
            Diagnostic("E_GRID_ACT_002", "unsupported grid activation", node_id=node, actual=activation)
        ])
    if activation == "identity":
        return {"out": x}, ()
    return {
        "out": x.with_level(
            min(x.level, EquivarianceLevel.EMPIRICAL),
            aliasing_model="finite_grid_truncation",
        )
    }, ()


def _grid_pointwise_product(node, inputs, attrs):
    del attrs
    left = _single(node, inputs, "left")
    right = _single(node, inputs, "right")
    grid = left if isinstance(left, GridTensorType) else right if isinstance(right, GridTensorType) else None
    other = right if grid is left else left
    if grid is None:
        raise DSLValidationError([
            Diagnostic("E_GRID_PRODUCT_001", "grid product requires at least one GridTensorType input", node_id=node)
        ])
    aliasing = grid.aliasing_model
    level = grid.level
    measure = grid.measure
    if isinstance(other, GridTensorType):
        _same_grid_contract(node, grid, other)
        if grid.channels != other.channels and grid.channels != 1 and other.channels != 1:
            raise DSLValidationError([
                Diagnostic("E_GRID_PRODUCT_002", "grid channel counts are not broadcast-compatible", node_id=node)
            ])
        output_channels = max(grid.channels, other.channels)
        grid = grid.with_channels(output_channels, _grid_irreps_with_channels(grid.source_irreps, output_channels))
        aliasing = "finite_grid_truncation"
        level = min(grid.level, other.level, EquivarianceLevel.EMPIRICAL)
        measure = _multiply_measures(grid.measure, other.measure)
    elif isinstance(other, InvariantTensorType):
        if (
            other.group != grid.group
            or other.carrier != grid.carrier
            or other.dtype != grid.dtype
            or other.frame != Frame("invariant")
        ):
            raise DSLValidationError([
                Diagnostic("E_GRID_PRODUCT_003", "grid broadcast scalar has an incompatible context", node_id=node)
            ])
        axes = _concrete_invariant_axes(node, other, "grid_pointwise_product")
        scalar_channels = math.prod(int(axis.size) for axis in axes)
        if scalar_channels not in (1, grid.channels):
            raise DSLValidationError([
                Diagnostic(
                    "E_GRID_PRODUCT_004",
                    "grid broadcast scalar must contain one or exactly grid.channels values",
                    node_id=node,
                    expected="1 or {}".format(grid.channels),
                    actual=str(scalar_channels),
                )
            ])
        level = min(grid.level, other.level)
        measure = _multiply_measures(grid.measure, other.measure)
    else:
        raise DSLValidationError([
            Diagnostic("E_GRID_PRODUCT_005", "grid product side input must be a grid or invariant tensor", node_id=node)
        ])
    return {"out": replace(grid, level=level, measure=measure, aliasing_model=aliasing)}, ()


def _grid_channel_linear(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if not isinstance(x, GridTensorType):
        raise DSLValidationError([
            Diagnostic("E_GRID_LINEAR_001", "grid_channel_linear requires GridTensorType input", node_id=node)
        ])
    try:
        out_channels = int(attrs["out_channels"])
    except (TypeError, ValueError):
        out_channels = 0
    if out_channels <= 0:
        raise DSLValidationError([
            Diagnostic("E_GRID_LINEAR_002", "grid out_channels must be positive", node_id=node)
        ])
    if not isinstance(attrs.get("bias", True), bool):
        raise DSLValidationError([
            Diagnostic("E_GRID_LINEAR_003", "grid channel-linear bias must be boolean", node_id=node)
        ])
    return {
        "out": x.with_channels(out_channels, _grid_irreps_with_channels(x.source_irreps, out_channels))
    }, ()


def _grid_channel_linear_parameters(node, inputs, outputs, attrs):
    x = _single(node, inputs, "x")
    output = outputs["out"]
    contracts = [
        ParameterContract(
            "weight",
            (
                ParameterAxis("out_channel", output.channels, "output_channel"),
                ParameterAxis("in_channel", x.channels, "input_channel"),
            ),
            sharing_axes=("grid_latitude", "grid_longitude"),
            initializer="kaiming_uniform",
            checkpoint_names=("{}.weight".format(node),),
            backend_parameter_name="weight",
        )
    ]
    if attrs.get("bias", True):
        contracts.append(
            ParameterContract(
                "bias",
                (ParameterAxis("out_channel", output.channels, "output_channel"),),
                sharing_axes=("grid_latitude", "grid_longitude"),
                is_bias=True,
                bias_irreps="{}x0".format(output.channels),
                initializer="uniform_fan_in",
                checkpoint_names=("{}.bias".format(node),),
                backend_parameter_name="bias",
            )
        )
    return tuple(contracts)


def _grid_dropout(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if not isinstance(x, GridTensorType):
        raise DSLValidationError([
            Diagnostic("E_GRID_DROPOUT_001", "grid_dropout requires GridTensorType input", node_id=node)
        ])
    try:
        probability = float(attrs.get("probability", 0.0))
    except (TypeError, ValueError):
        probability = float("nan")
    if not math.isfinite(probability) or probability < 0.0 or probability >= 1.0:
        raise DSLValidationError([
            Diagnostic("E_GRID_DROPOUT_002", "grid dropout probability must satisfy 0 <= p < 1", node_id=node)
        ])
    if probability == 0.0:
        return {"out": x}, ()
    return {"out": x.with_level(min(x.level, EquivarianceLevel.EMPIRICAL))}, ()


def _s2_swiglu(node, inputs, attrs):
    x = _single(node, inputs, "x")
    _validate_s2_parameters(node, x, attrs)
    _input_lmax, input_channels = _uniform_so3_multiplicities(node, x.irreps)
    if "out_irreps" not in attrs:
        raise DSLValidationError([
            Diagnostic("E_V3_TYPE_004", "S2 SwiGLU requires out_irreps", node_id=node)
        ])
    out = Irreps.parse(str(attrs["out_irreps"]), x.group.family)
    _output_lmax, output_channels = _uniform_so3_multiplicities(node, out)
    input_kinds = tuple(ir for _, ir in x.irreps)
    output_kinds = tuple(ir for _, ir in out)
    if input_kinds != output_kinds or input_channels != 2 * output_channels:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_TYPE_005",
                "S2 SwiGLU halves one paired channel axis while preserving every irrep kind",
                node_id=node,
                expected="same irrep kinds and input multiplicity = 2 * output multiplicity",
                actual="{} -> {}".format(x.irreps, out),
            )
        ])
    obligations = (
        ProofObligation(
            "{}:s2-product".format(node),
            ObligationKind.IRREP_PATH_EXISTS,
            node,
            "discharged",
            "core.s2_swiglu@1",
            {"interpretation": "band-limited S2 pointwise product / truncated self tensor product"},
        ),
    )
    return {"out": x.with_irreps(out)}, obligations


def _s2_gated_swiglu_merge(node, inputs, attrs):
    x = _v2_equivariant(node, _single(node, inputs, "x"), "x")
    scalars = _single(node, inputs, "scalars")
    if x.frame.kind != "edge" or not isinstance(scalars, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_V3_GATE_001",
                "V3 gated S2 SwiGLU requires an edge-frame value and invariant scalar side path",
                node_id=node,
            )
        ])
    if (
        scalars.group != x.group
        or scalars.carrier != x.carrier
        or scalars.frame != Frame("invariant")
        or scalars.dtype != x.dtype
        or scalars.measure != x.measure
    ):
        raise DSLValidationError([
            Diagnostic("E_V3_GATE_002", "V3 gated S2 SwiGLU inputs have incompatible contexts", node_id=node)
        ])
    _validate_s2_parameters(node, x, attrs)
    input_lmax, input_channels = _uniform_so3_multiplicities(node, x.irreps)
    out_irreps = Irreps.parse(str(attrs["out_irreps"]), x.group.family)
    output_lmax, output_channels = _uniform_so3_multiplicities(node, out_irreps)
    if (
        input_lmax != output_lmax
        or input_channels != 2 * output_channels
        or tuple(ir for _, ir in x.irreps) != tuple(ir for _, ir in out_irreps)
    ):
        raise DSLValidationError([
            Diagnostic(
                "E_V3_GATE_003",
                "V3 gated S2 SwiGLU halves paired channels while preserving degrees",
                node_id=node,
            )
        ])
    if scalars.irreps.dimension != input_channels + output_channels:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_GATE_004",
                "V3 gated S2 SwiGLU scalar side path must contain paired scalars plus gates",
                node_id=node,
                expected=str(input_channels + output_channels),
                actual=str(scalars.irreps.dimension),
            )
        ])
    try:
        dropout = float(attrs.get("dropout", 0.0))
    except (TypeError, ValueError):
        dropout = float("nan")
    if not math.isfinite(dropout) or dropout < 0.0 or dropout >= 1.0:
        raise DSLValidationError([
            Diagnostic("E_V3_GATE_005", "V3 gated S2 SwiGLU dropout must satisfy 0 <= p < 1", node_id=node)
        ])
    obligation = ProofObligation(
        "{}:gated-s2-product".format(node),
        ObligationKind.IRREP_PATH_EXISTS,
        node,
        "discharged",
        "core.s2_gated_swiglu_merge@1",
        {"interpretation": "official sep-merge_gates2_swiglu"},
    )
    return {"out": x.with_irreps(out_irreps)}, (obligation,)


def _edge_frame_gate_activation(node, inputs, attrs):
    x = _v2_equivariant(node, _single(node, inputs, "x"), "x")
    scalars = _single(node, inputs, "scalars")
    if x.frame.kind != "edge" or not isinstance(scalars, InvariantTensorType):
        raise DSLValidationError([
            Diagnostic(
                "E_V3_EDGE_GATE_001",
                "V3 edge-frame gate requires an edge-frame value and invariant scalar gates",
                node_id=node,
            )
        ])
    if (
        scalars.group != x.group
        or scalars.carrier != x.carrier
        or scalars.frame != Frame("invariant")
        or scalars.dtype != x.dtype
        or scalars.measure != x.measure
    ):
        raise DSLValidationError([
            Diagnostic(
                "E_V3_EDGE_GATE_002",
                "V3 edge-frame gate inputs have incompatible contexts",
                node_id=node,
            )
        ])
    lmax, channels = _uniform_so3_multiplicities(node, x.irreps)
    try:
        mmax = int(attrs.get("mmax", lmax))
    except (TypeError, ValueError):
        mmax = -1
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_EDGE_GATE_003",
                "V3 edge-frame gate requires 0 <= mmax <= lmax",
                node_id=node,
                actual=str(attrs.get("mmax")),
            )
        ])
    expected_scalars = lmax * channels
    if scalars.irreps.dimension != expected_scalars:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_EDGE_GATE_004",
                "V3 edge-frame gate requires one gate per non-scalar degree and channel",
                node_id=node,
                expected=str(expected_scalars),
                actual=str(scalars.irreps.dimension),
            )
        ])
    obligation = ProofObligation(
        "{}:degree-gate".format(node),
        ObligationKind.IRREP_PATH_EXISTS,
        node,
        "discharged",
        "core.edge_frame_gate_activation@1",
        {"interpretation": "official V3 GateActivation in m-primary edge-frame storage"},
    )
    return {"out": x}, (obligation,)


def _so3_linear(node, inputs, attrs):
    x = _v2_equivariant(node, _single(node, inputs, "x"), "x")
    if x.frame.kind != "global" or x.axis_specs:
        raise DSLValidationError([
            Diagnostic("E_SO3_LINEAR_001", "V3 SO3Linear requires axis-free global coefficients", node_id=node)
        ])
    input_lmax, _input_channels = _uniform_so3_multiplicities(node, x.irreps)
    out_irreps = Irreps.parse(str(attrs["out_irreps"]), x.group.family)
    output_lmax, _output_channels = _uniform_so3_multiplicities(node, out_irreps)
    if input_lmax != output_lmax or tuple(ir for _, ir in x.irreps) != tuple(ir for _, ir in out_irreps):
        raise DSLValidationError([
            Diagnostic("E_SO3_LINEAR_002", "V3 SO3Linear preserves every degree from zero through lmax", node_id=node)
        ])
    if not isinstance(attrs.get("bias", True), bool):
        raise DSLValidationError([
            Diagnostic("E_SO3_LINEAR_003", "V3 SO3Linear bias attr must be boolean", node_id=node)
        ])
    return {"out": x.with_irreps(out_irreps)}, ()


def _so3_linear_parameters(node, inputs, outputs, attrs):
    x = _single(node, inputs, "x")
    output = outputs["out"]
    lmax, input_channels = _uniform_so3_multiplicities(node, x.irreps)
    _output_lmax, output_channels = _uniform_so3_multiplicities(node, output.irreps)
    contracts = [
        ParameterContract(
            "weight",
            (
                ParameterAxis("degree", lmax + 1, "irrep_degree"),
                ParameterAxis("out_channel", output_channels, "output_channel"),
                ParameterAxis("in_channel", input_channels, "input_channel"),
            ),
            initializer="normal_0_1_then_uniform_fan_in",
            checkpoint_names=("{}.weight".format(node),),
            backend_parameter_name="weight",
        )
    ]
    if attrs.get("bias", True):
        contracts.append(
            ParameterContract(
                "bias",
                (
                    ParameterAxis("singleton_batch", 1, "broadcast"),
                    ParameterAxis("singleton_coefficient", 1, "l0_only"),
                    ParameterAxis("out_channel", output_channels, "output_channel"),
                ),
                is_bias=True,
                bias_irreps="{}x0".format(output_channels),
                initializer="zeros",
                checkpoint_names=("{}.bias".format(node),),
                backend_parameter_name="bias",
            )
        )
    return tuple(contracts)


def _so3_linear_v2(node, inputs, attrs):
    outputs, obligations = _so3_linear(node, inputs, attrs)
    try:
        scale = float(attrs.get("l0_weight_scale", 1.0))
    except (TypeError, ValueError):
        scale = float("nan")
    if not math.isfinite(scale) or scale <= 0.0:
        raise DSLValidationError([
            Diagnostic(
                "E_SO3_LINEAR_004",
                "SO3Linear l0_weight_scale must be finite and positive",
                node_id=node,
                actual=str(attrs.get("l0_weight_scale")),
            )
        ])
    return outputs, obligations


def _so3_linear_v2_parameters(node, inputs, outputs, attrs):
    contracts = _so3_linear_parameters(node, inputs, outputs, attrs)
    scale = float(attrs.get("l0_weight_scale", 1.0))
    return tuple(
        replace(
            contract,
            initializer=(
                "normal_0_1_then_uniform_fan_in_l0_scaled:{}".format(scale)
                if contract.name == "weight" and scale != 1.0
                else contract.initializer
            ),
        )
        for contract in contracts
    )


def _equivariant_merge_norm(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if x.group.dimension != 3:
        raise DSLValidationError([
            Diagnostic("E_V3_TYPE_006", "merged irrep normalization requires a 3D rotation group", node_id=node)
        ])
    _uniform_so3_multiplicities(node, x.irreps)
    epsilon = float(attrs.get("epsilon", 1.0e-5))
    normalization = str(attrs.get("normalization", "component"))
    if epsilon <= 0.0:
        raise DSLValidationError([
            Diagnostic("E_V3_TYPE_007", "merged normalization epsilon must be positive", node_id=node)
        ])
    if normalization not in ("component", "norm"):
        raise DSLValidationError([
            Diagnostic(
                "E_V3_TYPE_008",
                "merged normalization supports component or norm scaling",
                node_id=node,
                actual=normalization,
            )
        ])
    return {"out": x}, ()


def _compatibility(node, inputs, attrs):
    query = _single(node, inputs, "query")
    key = _single(node, inputs, "key")
    _same_context(node, query, key, exact_irreps=True)
    family = query.group.family
    scalar = "1x0e" if family == "O3" else "1x0" if family == "SO3" else "1xm0e" if family == "O2" else "1xm0"
    return {
        "out": query.with_irreps(Irreps.parse(scalar, family))
        .with_frame(Frame("invariant"))
        .with_measure(_multiply_measures(query.measure, key.measure))
    }, ()


_PRIMITIVE_CONTRACTS = {
    "core.grid_project": {
        "description": "Project complete SO(3) coefficient blocks to an explicitly typed finite S2 sampling grid.",
        "required_attrs": ("grid_resolution",),
        "optional_attrs": {
            "mmax": "maximum represented spherical order",
            "normalization": "S2 transform normalization",
            "quadrature": "quadrature implementation contract",
            "sampling": "finite sampling-domain contract",
            "use_m_primary": "whether the input coefficient layout is m-primary",
        },
        "motif_parameter_attrs": ("grid_resolution", "mmax", "normalization", "quadrature", "sampling", "use_m_primary"),
        "semantic_constraints": (
            "The input contains every degree through lmax with one uniform channel multiplicity.",
            "Grid resolution, bandlimit, normalization, quadrature and layout enter the output GridTensorType.",
        ),
    },
    "core.grid_unproject": {
        "description": "Unproject a typed finite S2 grid back to complete canonical SO(3) coefficients.",
        "optional_attrs": {"out_irreps": "explicit output irreps; defaults to the grid source irreps"},
        "motif_parameter_attrs": ("out_irreps",),
    },
    "core.grid_split": {
        "description": "Split the pointwise grid channel axis into two ordered typed grid values.",
        "required_attrs": ("left_channels",),
        "motif_parameter_attrs": ("left_channels",),
    },
    "core.grid_concat": {
        "description": "Concatenate compatible finite-grid values along the shared pointwise channel axis.",
    },
    "core.grid_pointwise_activation": {
        "description": "Apply an explicit pointwise nonlinearity on finite S2 grid samples.",
        "required_attrs": ("activation",),
        "motif_parameter_attrs": ("activation",),
        "semantic_constraints": ("Non-identity finite-grid nonlinearities have empirical rather than analytic equivariance certification.",),
    },
    "core.grid_pointwise_product": {
        "description": "Multiply two grid functions or broadcast typed invariant channels over one grid.",
        "semantic_constraints": ("Grid-grid products record finite-bandwidth truncation and empirical equivariance.",),
    },
    "core.grid_channel_linear": {
        "description": "Apply one shared torch-compatible channel linear map at every finite S2 sample.",
        "required_attrs": ("out_channels",),
        "optional_attrs": {"bias": "whether to add one spatially shared channel bias"},
        "motif_parameter_attrs": ("out_channels", "bias"),
    },
    "core.grid_dropout": {
        "description": "Apply explicit elementwise dropout to finite-grid samples with typed probability and training semantics.",
        "required_attrs": ("probability",),
        "motif_parameter_attrs": ("probability",),
    },
    "core.identity": {
        "description": "Type-preserving identity map.",
        "semantic_constraints": ("Output type equals input type exactly.",),
    },
    "core.categorical_remap": {
        "description": "Map one typed discrete vocabulary into another with explicit rejection entries.",
        "required_attrs": ("mapping", "output_vocabulary_size"),
        "motif_parameter_attrs": ("mapping", "output_vocabulary_size"),
        "semantic_constraints": (
            "Input is a CategoricalTensorType with exactly one integer category per carrier item.",
            "mapping has one entry per input category; -1 marks a runtime-rejected category.",
            "All nonnegative entries are bounded by output_vocabulary_size and output remains categorical.",
        ),
        "edit_guidance": ("Use before one-hot encoding when dataset labels differ from the model species vocabulary.",),
    },
    "core.categorical_one_hot": {
        "description": "Convert a typed categorical index into a floating invariant one-hot species axis.",
        "optional_attrs": {
            "axis": "name of the explicit species feature axis",
            "dtype": "floating output dtype",
        },
        "motif_parameter_attrs": ("axis", "dtype"),
        "semantic_constraints": (
            "Input is categorical and output is a dimensionless InvariantTensorType.",
            "The output species-axis size equals the input vocabulary size.",
            "No learned embedding or network-specific constructor is hidden in this primitive.",
        ),
    },
    "core.categorical_embedding": {
        "description": "Lookup a trainable dense invariant embedding vector for each typed category index.",
        "required_attrs": ("embedding_dim",),
        "optional_attrs": {
            "axis": "name of the output embedding-channel axis",
            "dtype": "floating output dtype",
            "initializer": "normal_0_1 or normal_then_uniform",
            "init_min": "lower bound used by normal_then_uniform",
            "init_max": "upper bound used by normal_then_uniform",
        },
        "motif_parameter_attrs": (
            "embedding_dim", "axis", "dtype", "initializer", "init_min", "init_max",
        ),
        "semantic_constraints": (
            "input is a categorical tensor and never enters arithmetic before the lookup",
            "the trainable table has shape [vocabulary, embedding_channel] and PyTorch normal(0,1) initialization",
            "output is an invariant tensor on the same carrier",
        ),
    },
    "core.categorical_embedding@2": {
        "description": "Categorical embedding whose uniform overwrite occurs at an explicit graph-level initialization barrier.",
        "required_attrs": ("embedding_dim", "init_min", "init_max"),
        "optional_attrs": {
            "axis": "name of the output embedding-channel axis",
            "dtype": "floating output dtype",
        },
        "motif_parameter_attrs": ("embedding_dim", "axis", "dtype", "init_min", "init_max"),
        "semantic_constraints": (
            "torch Embedding first consumes its normal initialization during module construction",
            "uniform overwrite is delayed until the lowering_contract initializer barrier",
            "the barrier order is architecture-identifying and can group multiple embeddings before later modules are constructed",
        ),
    },
    "core.invariant_concat": {
        "description": "Concatenate complete invariant feature vectors into one explicit output feature axis.",
        "required_attrs": ("axis",),
        "optional_attrs": {"feature_role": "semantic role assigned to the concatenated feature axis"},
        "motif_parameter_attrs": ("axis", "feature_role"),
        "semantic_constraints": (
            "all inputs are invariant values on the same group, carrier, frame, dtype, and measure",
            "each input has one complete explicit feature axis and concatenation occurs only along storage feature dimension",
            "the output certification level is the minimum of the input levels",
        ),
    },
    "core.constant_scale": {
        "description": "Multiply a complete typed equivariant value by one finite dimensionless constant.",
        "required_attrs": ("factor",),
        "motif_parameter_attrs": ("factor",),
        "semantic_constraints": (
            "The factor is a compile-time finite scalar and the output type equals the input type exactly.",
            "Every representation coordinate is scaled uniformly, so equivariance is preserved.",
        ),
        "edit_guidance": ("Use for sign conventions and fixed normalization constants, not learned gates.",),
    },
    "core.flatten_invariant_axes": {
        "description": "Reinterpret explicit invariant feature axes as one canonical trivial-irrep multiplicity axis without changing storage.",
        "semantic_constraints": (
            "Input is an InvariantTensorType with concrete axes whose size product equals scalar multiplicity.",
            "Runtime is identity; output removes the explicit axes and retains the same invariant scalar coefficients.",
            "This boundary is explicit before e3nn-style irrep-multiplicity linear maps.",
        ),
    },
    "core.irrep_linear": {
        "description": "Equivariant linear map within representation kinds already present in the input.",
        "required_attrs": ("out_irreps",),
        "semantic_constraints": ("May change multiplicities but cannot create a new irrep degree or parity absent from the input.",),
        "edit_guidance": ("Use at representation-width boundaries; use tensor_product when a new irrep degree is required.",),
    },
    "core.irrep_linear@2": {
        "description": "Equiformer V1 LinearRS parameterization as a shared internal tensor product with 1x0e plus optional even-scalar bias.",
        "required_attrs": ("out_irreps",),
        "optional_attrs": {
            "bias": "whether to learn a zero-initialized bias for every even scalar output block",
            "rescale": "whether to apply official per-instruction fan-in scaling at initialization",
            "initializer_scale": "additional positive constant applied to the initialized TP weight",
        },
        "motif_parameter_attrs": ("out_irreps", "bias", "rescale", "initializer_scale"),
        "semantic_constraints": (
            "Input and output are canonical simplified axis-free irrep-major EquivariantTensorType values.",
            "Only representation kinds present in the input may appear in the output.",
            "The numerical map is an internal/shared uvw TensorProduct with 1x0e, normalization=None and path_normalization=none.",
            "Rescale affects initialization only; forward applies the tensor product and optional trivial-scalar bias.",
            "initializer_scale is an explicit initialization-only multiplier used by official atom embeddings.",
        ),
    },
    "core.irrep_pad": {
        "description": "Embed an equivariant value into a wider direct-sum representation by appending exact zero channels.",
        "required_attrs": ("out_irreps",),
        "motif_parameter_attrs": ("out_irreps",),
        "semantic_constraints": (
            "Input and output use canonical simplified axis-free irrep-major storage.",
            "Every input irrep appears in the output with at least its original multiplicity.",
            "Existing channels are copied unchanged and every additional irrep coordinate is exactly zero.",
        ),
        "edit_guidance": ("Use when a later block expects representation kinds that are initially absent, such as V1 atomic embeddings.",),
    },
    "core.scalar_linear": {
        "description": "Linear map along one explicit invariant feature axis with typed weight and optional bias contracts.",
        "required_attrs": ("axis", "out_features"),
        "optional_attrs": {"bias": "whether to learn one trivial-scalar bias per output feature"},
        "motif_parameter_attrs": ("axis", "out_features", "bias"),
        "semantic_constraints": (
            "Input must be InvariantTensorType with statically sized axes whose product equals scalar multiplicity.",
            "Only the named feature axis changes size; all other axes are parameter-sharing axes.",
            "Bias is restricted to trivial scalar representations.",
        ),
        "edit_guidance": ("Use for radial MLP, attention logits, invariant readout, and per-head scalar projections.",),
    },
    "core.scalar_linear@2": {
        "description": "Axis-aware invariant linear map with explicit output semantic role and per-output initialization rescale.",
        "required_attrs": ("axis", "out_features"),
        "optional_attrs": {
            "bias": "whether to learn one trivial-scalar bias per output feature",
            "out_axis": "optional output axis rename",
            "output_axis_role": "semantic role of the output feature axis",
            "output_feature_role": "semantic role of the output invariant tensor",
            "initializer_scales": "one positive initialization multiplier per output feature",
        },
        "motif_parameter_attrs": (
            "axis", "out_features", "bias", "out_axis", "output_axis_role",
            "output_feature_role", "initializer_scales",
        ),
        "semantic_constraints": (
            "The input contract is identical to scalar_linear@1.",
            "Output axis renaming and semantic role changes are explicit and architecture-identifying.",
            "initializer_scales affect parameter initialization only and never alter the forward equation after parameters are loaded.",
        ),
    },
    "core.scalar_linear@3": {
        "description": "Axis-aware invariant linear map with deferred official V3 uniform fan-in reinitialization.",
        "required_attrs": ("axis", "out_features"),
        "optional_attrs": {
            "bias": "whether to learn a zeroed scalar bias",
            "out_axis": "optional output axis rename",
            "output_axis_role": "semantic role of the output feature axis",
            "output_feature_role": "semantic role of the output invariant tensor",
        },
        "motif_parameter_attrs": (
            "axis", "out_features", "bias", "out_axis", "output_axis_role", "output_feature_role",
        ),
        "semantic_constraints": (
            "module construction first consumes ordinary torch Linear initialization",
            "after every program module has been constructed, weight is redrawn uniformly in plus/minus one over sqrt(fan_in) and bias is zeroed",
            "module construction order is stored in the architecture-identifying lowering_contract",
        ),
    },
    "core.scalar_linear@4": {
        "description": "Axis-aware invariant Linear that preserves torch default weight initialization and only zeroes bias after construction.",
        "required_attrs": ("axis", "out_features"),
        "optional_attrs": {
            "bias": "whether to learn a scalar bias that is zeroed by the official V3 global initializer",
            "out_axis": "optional output axis rename",
            "output_axis_role": "semantic role of the output feature axis",
            "output_feature_role": "semantic role of the output invariant tensor",
        },
        "motif_parameter_attrs": (
            "axis", "out_features", "bias", "out_axis", "output_axis_role", "output_feature_role",
        ),
        "semantic_constraints": (
            "module construction consumes exactly one ordinary torch Linear initialization",
            "the weight is not redrawn by the post-construction initializer",
            "only an existing bias is zeroed without consuming RNG",
        ),
    },
    "core.scalar_layer_norm": {
        "description": "Torch-compatible LayerNorm over one explicit invariant feature axis.",
        "required_attrs": ("axis",),
        "optional_attrs": {
            "epsilon": "positive numerical stabilizer",
            "affine": "whether to learn a normalized-feature scale",
            "bias": "whether to learn a normalized-feature offset when affine is enabled",
        },
        "motif_parameter_attrs": ("axis", "epsilon", "affine", "bias"),
        "semantic_constraints": (
            "Input must be a dimensionless InvariantTensorType with concrete feature axes.",
            "Mean and variance are computed only over the named axis; all remaining axes share parameters.",
        ),
    },
    "core.scalar_offset": {
        "description": "Add one learned invariant offset along an explicit feature axis with fan-in initialization.",
        "required_attrs": ("axis", "fan_in"),
        "optional_attrs": {
            "initializer_scales": "one positive initialization multiplier per output feature",
        },
        "motif_parameter_attrs": ("axis", "fan_in", "initializer_scales"),
        "semantic_constraints": (
            "Input must be a dimensionless InvariantTensorType with concrete feature axes.",
            "The offset is a trivial representation and broadcasts only over axes other than the named feature axis.",
            "initializer_scales affect initialization only, matching Equiformer V1 radial offset rescaling.",
        ),
    },
    "core.head_split": {
        "description": "Split one invariant feature axis into explicit head and per-head channel axes without changing scalar storage order.",
        "required_attrs": ("axis", "head_axis", "channel_axis", "num_heads"),
        "motif_parameter_attrs": ("axis", "head_axis", "channel_axis", "num_heads"),
        "semantic_constraints": (
            "First version accepts only InvariantTensorType.",
            "Source axis is independently stored and divisible by num_heads.",
            "Output head and per-head channel axes are explicit and collision-free.",
        ),
    },
    "core.head_merge": {
        "description": "Merge adjacent invariant head and per-head channel axes back into one independent feature axis.",
        "required_attrs": ("head_axis", "channel_axis", "out_axis"),
        "motif_parameter_attrs": ("head_axis", "channel_axis", "out_axis"),
        "semantic_constraints": (
            "First version accepts only InvariantTensorType.",
            "Input axes must be adjacent head then per-head channel axes.",
            "Runtime scalar storage order is unchanged.",
        ),
    },
    "core.head_split@2": {
        "description": "Split a flattened equivariant value into an explicit head axis by dividing every irrep multiplicity, never an irrep coordinate dimension.",
        "required_attrs": ("head_axis", "num_heads"),
        "motif_parameter_attrs": ("head_axis", "num_heads"),
        "semantic_constraints": (
            "Input is an axis-free non-invariant EquivariantTensorType in canonical flattened irrep storage.",
            "Every irrep multiplicity is divisible by num_heads; degree/parity coordinates are never split.",
            "Runtime output is [carrier..., head, per_head_irrep_dimension] and uses an explicit equivariant-head value kind.",
        ),
    },
    "core.head_merge@2": {
        "description": "Merge an explicit equivariant head axis by multiplying every per-head irrep multiplicity and restoring flattened irrep-major storage.",
        "required_attrs": ("head_axis",),
        "motif_parameter_attrs": ("head_axis",),
        "semantic_constraints": (
            "Input has exactly one statically sized independent head axis in canonical simplified irrep-major storage.",
            "Each per-head irrep block is flattened independently before blocks are concatenated.",
            "This is the exact inverse of core.head_split@2 for the same head axis.",
        ),
    },
    "core.headwise_scalar_contraction": {
        "description": "Independently contract one invariant channel vector to one scalar logit per head.",
        "required_attrs": ("head_axis", "channel_axis"),
        "optional_attrs": {"bias": "whether to learn one trivial-scalar bias per head"},
        "motif_parameter_attrs": ("head_axis", "channel_axis", "bias"),
        "semantic_constraints": (
            "First version requires exactly one head axis followed by one per-head channel axis.",
            "Each head owns an independent channel weight vector and optional scalar bias.",
            "Output keeps only the head axis and has alpha feature role.",
        ),
    },
    "core.headwise_scalar_contraction@2": {
        "description": "Contract trivial scalar channels inside an equivariant-head runtime value to one invariant logit per head.",
        "required_attrs": ("head_axis",),
        "optional_attrs": {"bias": "whether to add one invariant scalar bias per head"},
        "motif_parameter_attrs": ("head_axis", "bias"),
        "semantic_constraints": (
            "Input has exactly one head axis and contains only per-head trivial scalar irreps.",
            "Every head has an independent weight vector; optional bias is restricted to trivial scalars.",
            "Output is a dense InvariantTensorType with one scalar per head.",
        ),
    },
    "core.change_multiplicity": {
        "description": "Named equivariant adapter for changing irrep multiplicities.",
        "required_attrs": ("out_irreps",),
        "semantic_constraints": ("Cannot create a new irrep degree or parity absent from the input.",),
        "edit_guidance": ("Insert before a residual region whose hidden_irreps differ from its predecessor.",),
    },
    "core.irrep_concat": {
        "description": "Direct-sum concatenation of compatible equivariant values.",
        "semantic_constraints": ("All inputs must share group, carrier, frame, axes, dtype, and measure; irreps are direct-summed.",),
    },
    "core.residual_add": {
        "description": "Residual addition of two exactly type-compatible equivariant values.",
        "semantic_constraints": ("Both inputs must have exactly equal irreps and all other type fields; output equals that type.",),
    },
    "core.residual_add@2": {
        "description": "Residual addition with certification-level meet for otherwise identical equivariant values.",
        "semantic_constraints": (
            "group, carrier, irreps, frame, axes, dtype, measure, and layout must be identical",
            "the output certification level is the minimum of the two input levels",
        ),
    },
    "core.tensor_product": {
        "description": "Clebsch-Gordan-compatible equivariant tensor-product coupling.",
        "required_attrs": ("out_irreps",),
        "semantic_constraints": ("Every requested output irrep must occur in a legal input tensor-product path, including parity.",),
        "edit_guidance": ("Use this operation, not gate or irrep_linear, to couple hidden l>0 paths into l=0 or other new degrees.",),
    },
    "core.tensor_product@2": {
        "description": "Fully connected uvw tensor product with explicit per-carrier external path weights.",
        "required_attrs": ("out_irreps",),
        "semantic_constraints": (
            "First version uses axis-free EquivariantTensorType inputs with shared group/carrier/frame/dtype.",
            "External weight is a dimensionless InvariantTensorType with radial_weight role and one tp_path axis.",
            "tp_path size equals the fully connected uvw path weight count and weights are unshared across carrier items.",
            "No custom instructions, bias, rescale, or shared external weights are supported in this version.",
        ),
        "edit_guidance": ("Use as the explicit radial-MLP-to-tensor-product boundary; do not treat it as official Depthwise TP exactness.",),
    },
    "core.tensor_product@3": {
        "description": "Instruction-explicit external-weight tensor product with repeated output path blocks; first version freezes Equiformer V1-style uvu paths.",
        "required_attrs": ("path_blocks", "instructions"),
        "motif_parameter_attrs": ("path_blocks", "instructions"),
        "semantic_constraints": (
            "Axis-free inputs use canonical simplified irrep-major storage and share group, carrier, frame, dtype, and certification level.",
            "path_blocks may repeat one irrep kind but must be canonically ordered; every block is referenced by at least one instruction.",
            "First version supports only weighted uvu instructions, whose output multiplicity equals the selected left input multiplicity.",
            "External radial weights are dimensionless, use one tp_path axis, and have length sum(left_mul * right_mul) over instructions.",
            "Lowering uses generic e3nn TensorProduct with path_normalization=none and does not call an official DepthwiseTensorProduct constructor.",
        ),
        "edit_guidance": ("Use for explicit Equiformer V1 Depthwise TP paths; keep radial initializer rescaling in the radial-profile contract.",),
    },
    "core.tensor_product@4": {
        "description": "Internal/shared fully connected uvw TensorProduct with explicit V1 fan-in initialization and scalar bias contracts.",
        "required_attrs": ("out_irreps",),
        "optional_attrs": {
            "bias": "whether to add one parameter per even scalar output multiplicity",
            "rescale": "whether to scale each instruction weight view by inverse sqrt accumulated output fan-in at initialization",
        },
        "semantic_constraints": (
            "Both inputs are axis-free canonical simplified irrep-major values in the same context.",
            "All legal input/output irrep triples use internal shared uvw weights, normalization=None, and path_normalization=none.",
            "Runtime output is not rescaled; rescale changes only parameter initialization.",
        ),
        "edit_guidance": ("Use for Equiformer V1 FullyConnectedTensorProductRescale and FFN layers.",),
    },
    "core.tensor_product@5": {
        "description": "Instruction-explicit internal/shared uvu TensorProduct with V1 per-output fan-in initialization.",
        "required_attrs": ("path_blocks", "instructions"),
        "optional_attrs": {
            "bias": "first version requires false",
            "rescale": "whether to scale internal instruction weights by inverse sqrt accumulated output fan-in at initialization",
        },
        "motif_parameter_attrs": ("path_blocks", "instructions", "bias", "rescale"),
        "semantic_constraints": (
            "Inputs and explicit path blocks obey the same canonical uvu instruction graph contract as tensor_product@3.",
            "Weights are trainable internal parameters shared across carrier items; no external radial-weight port is accepted.",
            "First version requires bias=false and uses normalization=None with path_normalization=none.",
            "Runtime output is not rescaled; rescale changes only the initialization of each internal instruction weight view.",
        ),
        "edit_guidance": ("Use for Equiformer V1 nonlinear-message sep_value DepthwiseTensorProduct; use @3 for radial external weights.",),
    },
    "core.scalar_activation": {
        "description": "Ordinary pointwise nonlinearity restricted to invariant scalar irreps.",
        "optional_attrs": {
            "activation": "backend activation name",
            "negative_slope": "slope in [0,1] for smooth_leaky_relu",
            "normalization": "none or deterministic second_moment normalization",
        },
        "attribute_aliases": {"function": "activation"},
        "motif_parameter_attrs": ("activation", "negative_slope", "normalization"),
        "semantic_constraints": (
            "All input irreps must be l=0 invariant scalars; output type is unchanged.",
            "second_moment uses e3nn-compatible deterministic N(0,1) normalization with seed 0 and 1,000,000 float64 samples.",
        ),
    },
    "core.invariant_weight": {
        "description": "Multiply an equivariant value by one invariant scalar attention weight.",
        "semantic_constraints": ("weight must be one invariant scalar with compatible carrier and frame; output type equals value type.",),
    },
    "core.invariant_scale": {
        "description": "Broadcast one invariant scalar or one scalar per explicit head over an equivariant value.",
        "semantic_constraints": (
            "Scale is dimensionless and invariant, with compatible group, carrier, dtype, frame, and certification level.",
            "The first version supports an axis-free scalar or one named head axis matching the value head axis.",
            "The output transformation type and runtime representation equal the value input.",
        ),
    },
    "core.edge_lift": {
        "description": "Lift node-carried values to directed edges without changing irreps.",
        "optional_attrs": {"endpoint": "source or target node endpoint"},
        "semantic_constraints": ("Input carrier must be node and output carrier is edge.",),
    },
    "core.endpoint_gather": {
        "description": "Gather values through an explicit typed source or target endpoint index map.",
        "semantic_constraints": (
            "x must use EquivariantTensorType and its carrier/group must equal the index-map domain.",
            "index must map node-carried values to source or target edge endpoints.",
            "No hidden graph_context endpoint indices are read.",
        ),
    },
    "core.endpoint_gather@2": {
        "description": "Gather equivariant or categorical node values through an explicit typed endpoint map.",
        "semantic_constraints": (
            "the input is an EquivariantTensorType or CategoricalTensorType whose group/carrier matches the index domain",
            "the index maps nodes to source or target edge endpoints",
            "categorical identity is preserved and no hidden graph context is read",
        ),
    },
    "core.segment_sum": {
        "description": "Permutation-safe edge-to-node sum aggregation.",
        "semantic_constraints": ("Input carrier must be edge and must be in global, not edge-local, frame.",),
    },
    "core.segment_mean": {
        "description": "Permutation-safe edge-to-node mean aggregation.",
        "semantic_constraints": ("Input carrier must be edge and must be in global, not edge-local, frame.",),
    },
    "core.segment_reduce": {
        "description": "Permutation-safe reduction through an explicit typed segment or batch index map.",
        "optional_attrs": {
            "reduce": "sum or mean",
            "normalization": "none or target_cardinality",
        },
        "motif_parameter_attrs": ("reduce", "normalization"),
        "semantic_constraints": (
            "x must use EquivariantTensorType and match the index-map domain.",
            "The output carrier is the index-map codomain.",
            "The runtime index payload carries indices and target_size, including empty targets.",
            "target_cardinality multiplies each summed target by its number of source items and is invalid with mean reduction.",
            "No hidden edge_dst, batch, num_nodes, or num_graphs context is read.",
        ),
    },
    "core.global_pool": {
        "description": "Permutation-invariant node-to-graph pooling.",
        "optional_attrs": {"reduce": "sum or mean"},
        "semantic_constraints": ("Input carrier must be node; irreps are preserved while carrier becomes graph.",),
    },
    "core.select_scalars": {
        "description": "Select invariant l=0 channels from a mixed-irrep value.",
        "optional_attrs": {"multiplicity": "positive number of scalar channels to retain"},
        "semantic_constraints": ("At least one invariant scalar channel must exist; non-scalars are discarded, not coupled.",),
    },
    "core.select_scalars@2": {
        "description": "Extract trivial coefficients into one explicit invariant feature axis for editable scalar side paths.",
        "optional_attrs": {
            "multiplicity": "optional leading scalar multiplicity",
            "axis": "explicit output scalar feature axis",
            "feature_role": "semantic role of the extracted invariant channels",
        },
        "motif_parameter_attrs": ("multiplicity", "axis", "feature_role"),
    },
    "core.so3_linear@2": {
        "description": "Official V3 degree-wise SO3Linear with an explicit construction-time l=0 weight rescale.",
        "required_attrs": ("out_irreps",),
        "optional_attrs": {
            "bias": "l=0-only bias",
            "l0_weight_scale": "positive construction-time multiplier for the degree-zero weight slice",
        },
        "motif_parameter_attrs": ("out_irreps", "bias", "l0_weight_scale"),
    },
    "core.to_edge_frame": {
        "description": "Express global-frame 3D irreps in an edge-aligned local frame.",
        "optional_attrs": {
            "frame_id": "stable identifier paired with from_edge_frame",
            "mmax": "optional retained SO(2) order, satisfying 0 <= mmax <= lmax",
            "use_rotation_mask": "whether to use the stabilized official V3 rotation-gradient mask",
        },
        "semantic_constraints": ("Input must be in global frame and creates an open frame-balance proof obligation.",),
        "edit_guidance": ("Pair with from_edge_frame using the same frame_id before node aggregation.",),
    },
    "core.from_edge_frame": {
        "description": "Return edge-local 3D irreps to the global frame.",
        "optional_attrs": {
            "frame_id": "identifier matching to_edge_frame",
            "mmax": "retained SO(2) order used by the paired forward rotation",
            "use_rotation_mask": "whether the paired V3 rotation used its stabilized gradient mask",
        },
        "semantic_constraints": ("Input must be in the matching edge frame; output is global-frame.",),
    },
    "core.irrep_slice": {
        "description": "Select available irrep blocks without changing their transformation law.",
        "required_attrs": ("irreps",),
        "semantic_constraints": ("Requested multiplicity for every irrep cannot exceed the input multiplicity.",),
    },
    "core.irrep_select@2": {
        "description": "Select an explicit contiguous multiplicity range independently inside each requested irrep kind.",
        "required_attrs": ("selections",),
        "motif_parameter_attrs": ("selections",),
        "semantic_constraints": (
            "Each selection declares irrep, zero-based multiplicity start, and positive multiplicity.",
            "Selections never split the coordinate dimension of an irrep and must remain within the available multiplicity.",
            "Input irreps and coefficient storage must be canonical, simplified, and irrep-major.",
            "Runtime kind, carrier axes, frame, dtype, measure, and layout are preserved.",
        ),
    },
    "core.relative_position": {
        "description": "Construct translation-safe directed edge displacement vectors from node positions.",
        "semantic_constraints": ("Inputs are Cartesian node vectors and the task must authorize relative-coordinate translation handling.",),
    },
    "core.relative_displacement": {
        "description": "Construct non-periodic target-source edge displacements from affine points and explicit endpoint maps.",
        "semantic_constraints": (
            "positions must be AffinePointType rather than an ordinary l=1 tensor.",
            "source_index and target_index must be matching node-to-edge endpoint maps.",
            "the output is one global-frame polar vector with the same dtype and length measure as positions.",
            "adding one common translation to every point leaves the output unchanged.",
        ),
    },
    "core.relative_displacement@3": {
        "description": "Construct non-periodic source-target edge displacements from affine points and explicit endpoint maps.",
        "semantic_constraints": (
            "the output is pos[source] - pos[target] in the global frame",
            "the endpoint maps are explicit and a common translation cancels exactly",
            "this sign convention matches official fairchem Equiformer V3 edge_distance_vec",
        ),
    },
    "core.periodic_displacement": {
        "description": "Construct target + lattice_shift @ lattice - source under an explicit periodic-image convention.",
        "semantic_constraints": (
            "all inputs share one group with periodicity=lattice.",
            "lattice and lattice_shift carry the same lattice_id.",
            "the target_image convention and row-vector lattice convention are fixed by the type contract.",
            "no hidden cell, pbc, or cell_offsets context is read.",
        ),
    },
    "core.periodic_displacement@2": {
        "description": "Construct source + lattice_shift @ lattice - target using the official fairchem source-image convention.",
        "semantic_constraints": (
            "lattice_shift uses source_image convention and shares lattice_id with the row-vector lattice",
            "the output equals official fairchem pos[row] - pos[col] + offsets",
            "no hidden cell or cell_offsets context is read",
        ),
    },
    "core.distance": {
        "description": "Compute invariant scalar distances from Cartesian edge vectors.",
        "semantic_constraints": ("Input must be one edge-carried Cartesian vector; output is an invariant edge scalar.",),
    },
    "core.radial_basis": {
        "description": "Expand invariant edge distances in a scalar radial basis.",
        "required_attrs": ("num_basis",),
        "optional_attrs": {
            "cutoff": "positive maximum radial extent",
            "width": "positive Gaussian width",
        },
        "semantic_constraints": ("num_basis must be positive and input must be invariant edge scalars.",),
    },
    "core.fixed_gaussian_radial_basis": {
        "description": "Fixed evenly-spaced Gaussian smearing of invariant edge distances.",
        "required_attrs": ("num_basis", "stop"),
        "optional_attrs": {
            "start": "first Gaussian center",
            "basis_width_scalar": "positive multiplier of adjacent-center spacing",
            "axis": "name of the output basis axis",
            "construction_dtype": "dtype used to construct centers and freeze the scalar coefficient before module dtype conversion",
        },
        "motif_parameter_attrs": (
            "num_basis", "start", "stop", "basis_width_scalar", "axis", "construction_dtype",
        ),
        "semantic_constraints": (
            "centers are linspace(start, stop, num_basis) in construction_dtype and then follow module dtype/device conversion",
            "coefficient is frozen as a host scalar from construction-dtype adjacent spacing",
            "the basis is fixed and contains no trainable parameters",
        ),
    },
    "core.distance@2": {
        "description": "Compute a typed invariant distance scalar from one axis-free Cartesian edge vector.",
        "semantic_constraints": (
            "Input is an axis-free EquivariantTensorType carrying one Cartesian vector on edges.",
            "Output is an InvariantTensorType in the invariant frame and preserves the input length measure.",
        ),
    },
    "core.gaussian_radial_basis": {
        "description": "Official Graphormer/Equiformer V1 learnable Gaussian radial basis expansion.",
        "required_attrs": ("num_basis", "cutoff"),
        "optional_attrs": {"axis": "name of the output radial-basis feature axis"},
        "motif_parameter_attrs": ("num_basis", "cutoff", "axis"),
        "semantic_constraints": (
            "Input is one invariant edge distance and output has one explicit radial basis axis.",
            "mean and std are trainable per basis; weight and bias are trainable shared scalar affine parameters.",
            "Forward uses pi=3.14159, abs(std)+1e-5, distance/cutoff, and the official normalized Gaussian equation.",
            "This primitive does not call the official GaussianRadialBasisLayer constructor.",
        ),
    },
    "core.cutoff_envelope": {
        "description": "Type-preserving invariant cutoff envelope.",
        "optional_attrs": {"cutoff": "positive radial cutoff", "order": "envelope polynomial order"},
        "motif_parameter_attrs": ("cutoff", "order"),
        "semantic_constraints": ("Output type equals input type.",),
    },
    "core.cutoff_envelope@2": {
        "description": "Dimensionless polynomial cutoff envelope of one invariant edge distance.",
        "required_attrs": ("cutoff",),
        "optional_attrs": {"order": "positive polynomial exponent"},
        "motif_parameter_attrs": ("cutoff", "order"),
        "semantic_constraints": (
            "input is one invariant edge distance and output is one dimensionless invariant edge scalar",
            "forward matches the official Equiformer V3 PolynomialEnvelope equation and strict d/cutoff < 1 boundary",
        ),
    },
    "core.axisymmetric_spherical_lift": {
        "description": "Lift per-degree invariant m=0 amplitudes along an edge direction into a global SO(3) feature.",
        "required_attrs": ("out_irreps",),
        "optional_attrs": {
            "mmax": "maximum edge-frame order used by the official Wigner adapter",
            "use_rotation_mask": "whether to use the official differentiable rotation masking convention",
        },
        "motif_parameter_attrs": ("out_irreps", "mmax", "use_rotation_mask"),
        "semantic_constraints": (
            "output contains every degree 0..lmax with one uniform channel multiplicity",
            "amplitudes contain exactly one invariant m=0 value per degree and channel",
            "the inverse Wigner rotation produces a global equivariant edge feature without calling a V3 Block",
        ),
    },
    "core.spherical_harmonics": {
        "description": "Construct 3D spherical-harmonic edge features through degree lmax.",
        "required_attrs": ("lmax",),
        "optional_attrs": {"normalization": "component, integral, or norm"},
        "semantic_constraints": ("Only 3D SO(3)/O(3) groups are supported and lmax must be nonnegative.",),
    },
    "core.norm_activation": {
        "description": "Equivariant nonlinearity that acts through irrep norms.",
        "optional_attrs": {
            "activation": "scalar activation applied to norms",
            "normalize": "whether to divide by the irrep norm before rescaling",
            "epsilon": "positive numerical stabilizer",
            "bias": "whether to add a scalar norm bias",
        },
        "attribute_aliases": {"function": "activation"},
        "motif_parameter_attrs": ("activation",),
        "semantic_constraints": ("Preserves every input irrep block and its output type.",),
    },
    "core.gate": {
        "description": "Scale non-scalar irrep blocks by invariant scalar gates.",
        "semantic_constraints": (
            "gates must contain only invariant l=0 scalars.",
            "value must contain only non-scalar irreps.",
            "output irreps equal value irreps exactly; gate does not convert tensors into scalars.",
            "gate multiplicity must be one shared gate or one gate per non-scalar multiplicity.",
        ),
        "edit_guidance": ("Use select_scalars and irrep_slice to form gates and values; use tensor_product to couple l>0 into l=0.",),
    },
    "core.equivariant_norm": {
        "description": "Type-preserving normalization over complete irrep blocks.",
        "optional_attrs": {
            "epsilon": "positive numerical stabilizer",
            "affine": "whether to learn invariant affine parameters",
            "instance": "whether to use instance rather than batch statistics",
            "normalization": "component or norm",
        },
        "semantic_constraints": ("Output type equals input type; normalization must not mix representation coordinates illegally.",),
    },
    "core.irrep_layer_norm": {
        "description": "Equiformer V1 LayerNormV2 over multiplicities and complete irrep coordinates.",
        "optional_attrs": {
            "epsilon": "positive numerical stabilizer",
            "affine": "whether to learn one scale per irrep instance and one bias per trivial scalar",
            "normalization": "component or norm",
        },
        "semantic_constraints": (
            "Input is axis-free canonical simplified irrep-major storage and output type is unchanged.",
            "Even scalar multiplicities are centered before normalization; non-scalars are never coordinate-wise centered.",
            "component uses the mean square over representation coordinates, while norm uses the sum of squares.",
            "The first exact V1 version requires affine=True; the official affine=False path is not admitted.",
        ),
        "edit_guidance": ("Use for Equiformer V1 pre-norm blocks; do not substitute e3nn BatchNorm.",),
    },
    "core.stochastic_depth": {
        "description": "Drop complete equivariant residual paths stochastically.",
        "optional_attrs": {"p": "drop probability in [0,1)"},
        "motif_parameter_attrs": ("p",),
        "semantic_constraints": ("p must satisfy 0 <= p < 1 and output type equals input type.",),
    },
    "core.graph_stochastic_depth": {
        "description": "Drop a complete residual path with one shared mask per graph.",
        "optional_attrs": {"p": "drop probability in [0,1)"},
        "motif_parameter_attrs": ("p",),
        "semantic_constraints": (
            "x is an EquivariantTensorType and batch is an explicit same-group IndexMapType from its carrier to graph.",
            "Training uses one Bernoulli survival mask per graph and broadcasts it to every item and irrep coordinate in that graph.",
            "Evaluation and p=0 are exact identity maps.",
        ),
        "edit_guidance": ("Use for graph-level residual DropPath; do not substitute node-wise stochastic_depth@1.",),
    },
    "core.scalar_dropout": {
        "description": "Elementwise dropout over explicitly invariant scalar values such as attention alpha weights.",
        "optional_attrs": {"p": "drop probability in [0,1)"},
        "motif_parameter_attrs": ("p",),
        "semantic_constraints": (
            "Input must be InvariantTensorType and output type is unchanged.",
            "Each runtime scalar entry receives an independent inverted-dropout mask in training mode.",
        ),
    },
    "core.equivariant_dropout": {
        "description": "Drop complete irrep instances while sharing a mask over every m coordinate.",
        "optional_attrs": {"p": "drop probability in [0,1)"},
        "motif_parameter_attrs": ("p",),
        "semantic_constraints": (
            "Input is axis-free canonical simplified irrep-major EquivariantTensorType and output type is unchanged.",
            "One inverted-dropout mask is sampled per carrier item and irrep instance, never per representation coordinate.",
        ),
    },
    "core.invariant_dropout": {
        "description": "Drop complete multiplicity channels without splitting irrep coordinates.",
        "optional_attrs": {"p": "drop probability in [0,1)"},
        "motif_parameter_attrs": ("p",),
        "semantic_constraints": ("p must satisfy 0 <= p < 1 and output type equals input type.",),
    },
    "core.segment_softmax": {
        "description": "Permutation-consistent neighbor softmax over invariant edge logits.",
        "semantic_constraints": ("Input must be edge-carried invariant scalar logits; output type is unchanged.",),
    },
    "core.segment_softmax@2": {
        "description": "Permutation-consistent neighbor softmax over invariant edge logits using an explicit typed segment index.",
        "semantic_constraints": (
            "logits are dimensionless edge-carried InvariantTensorType values and may contain an explicit head axis.",
            "index maps edges to destination segments under the same group and carries target_size explicitly.",
            "No hidden edge_dst graph context is read.",
        ),
    },
    "core.invariant_compatibility": {
        "description": "Produce one invariant scalar compatibility score from equal-type query and key values.",
        "semantic_constraints": ("query and key types must match exactly; output is one invariant scalar in an invariant frame.",),
    },
    "core.so2_convolution": {
        "description": "Equiformer V2-style SO(2) frequency convolution in an edge-aligned frame.",
        "optional_attrs": {
            "out_irreps": "output multiplicities over irrep kinds present in input",
            "mmax": "maximum SO(2) order satisfying 0 <= mmax <= lmax",
        },
        "semantic_constraints": ("Requires 3D features in an edge frame and cannot create irrep degrees absent from the input.",),
    },
    "core.so2_linear@1": {
        "description": "SO(2)-equivariant linear map over an m-primary edge-frame coefficient layout, without an auxiliary m=0 output.",
        "required_attrs": ("out_irreps",),
        "optional_attrs": {
            "mmax": "maximum represented SO(2) order satisfying 0 <= mmax <= lmax",
            "extra_m0_channels": "must remain zero for the single-output version",
            "m0_prefix_rows": "number of leading m=0 output rows rescaled after initialization",
            "m0_prefix_scale": "positive initializer scale for the selected leading rows",
            "zero_bias": "whether the final m=0 bias is explicitly zeroed after Linear construction",
        },
        "motif_parameter_attrs": (
            "out_irreps", "mmax", "m0_prefix_rows", "m0_prefix_scale", "zero_bias",
        ),
        "semantic_constraints": (
            "input and output contain every degree 0..lmax with uniform channel multiplicity",
            "m=0 is a real dense linear map; each nonzero order is a complex SO(2) intertwiner shared over +/-m",
            "the runtime coefficient order is m-primary and remains in the same edge frame",
        ),
    },
    "core.so2_linear@2": {
        "description": "SO(2)-equivariant m-primary linear map with a separate invariant extra-m0 output for attention or gate scalars.",
        "required_attrs": ("out_irreps", "extra_m0_channels"),
        "optional_attrs": {
            "mmax": "maximum represented SO(2) order satisfying 0 <= mmax <= lmax",
            "m0_prefix_rows": "number of leading m=0 output rows rescaled after initialization",
            "m0_prefix_scale": "positive initializer scale for the selected leading rows",
            "zero_bias": "whether the final m=0 bias is explicitly zeroed after Linear construction",
        },
        "motif_parameter_attrs": (
            "out_irreps", "extra_m0_channels", "mmax", "m0_prefix_rows", "m0_prefix_scale", "zero_bias",
        ),
        "semantic_constraints": (
            "the equivariant output follows the same SO(2) intertwiner contract as so2_linear@1",
            "extra_m0 is split before reshaping the remaining m=0 coefficients and is an edge-carried invariant tensor",
            "the parameter tree is fc_m0 plus one bias-free complex linear weight per nonzero order",
        ),
    },
    "core.s2_activation": {
        "description": "S2-grid equivariant activation for 3D spherical representations.",
        "optional_attrs": {
            "mmax": "maximum represented SO(2) order",
            "grid_resolution": "backend quadrature resolution, at least two",
            "normalization": "component, integral, or norm",
        },
        "motif_parameter_attrs": ("mmax", "grid_resolution", "normalization"),
        "semantic_constraints": ("Preserves the declared equivariant type and is restricted to SO(3)/O(3).",),
    },
    "core.separable_s2_activation": {
        "description": "Equiformer V2 separable S2 activation with an explicit scalar side path.",
        "optional_attrs": {
            "mmax": "maximum represented SO(2) order",
            "grid_resolution": "backend quadrature resolution, at least two",
            "normalization": "component, integral, or norm",
        },
        "motif_parameter_attrs": ("mmax", "grid_resolution", "normalization"),
        "semantic_constraints": (
            "scalars and x share group, carrier, and frame.",
            "the scalar side path contains only l=0 and matches x's l=0 multiplicity.",
            "output type equals x type.",
        ),
    },
    "core.s2_swiglu": {
        "description": "Equiformer V3 S2-grid SwiGLU: project to a spherical grid, split paired channels, apply SiLU-gated pointwise multiplication, and project back.",
        "required_attrs": ("out_irreps",),
        "optional_attrs": {
            "mmax": "maximum represented SO(2) order",
            "grid_resolution": "one or two backend quadrature resolutions",
            "normalization": "component, integral, or norm",
        },
        "motif_parameter_attrs": ("out_irreps", "mmax", "grid_resolution", "normalization"),
        "semantic_constraints": (
            "input and output contain the same consecutive SO(3)/O(3) irrep kinds",
            "input multiplicity is exactly twice output multiplicity for every degree",
            "the pointwise grid product is interpreted as a band-limited, output-truncated self tensor product",
        ),
        "edit_guidance": (
            "precede this operation with an equivariant linear expansion to twice the desired output width",
            "do not use it with an odd or nonuniform paired channel layout",
        ),
    },
    "core.edge_frame_gate_activation": {
        "description": "Apply the official V3 degree-wise scalar gate to an m-primary edge-frame SO(3) value.",
        "optional_attrs": {"mmax": "maximum represented SO(2) order"},
        "motif_parameter_attrs": ("mmax",),
        "semantic_constraints": (
            "x is an edge-frame value with uniform channels across degrees",
            "the invariant scalar side path contains exactly lmax times channels gates",
            "SiLU acts on l=0 while sigmoid gates scale complete l>0 representation blocks",
        ),
    },
    "core.equivariant_merge_norm": {
        "description": "Equiformer V3 merged normalization over all complete angular-momentum blocks.",
        "optional_attrs": {
            "epsilon": "positive numerical stabilizer",
            "affine": "whether to learn degree-wise scale and scalar bias",
            "normalization": "component or norm",
            "centering": "whether to center the l=0 channel path",
        },
        "motif_parameter_attrs": ("epsilon", "affine", "normalization", "centering"),
        "semantic_constraints": (
            "all degrees from zero through lmax must be present with one uniform multiplicity",
            "the output type equals the input type exactly",
            "centering is restricted to invariant l=0 channels; non-scalar coordinates are only rescaled by invariant norms",
        ),
    },
}


def core_registry() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()
    definitions = (
        PrimitiveDefinition("core.identity", 1, ("x",), ("out",), _identity),
        PrimitiveDefinition("core.categorical_remap", 1, ("x",), ("out",), _categorical_remap),
        PrimitiveDefinition("core.categorical_one_hot", 1, ("x",), ("out",), _categorical_one_hot),
        PrimitiveDefinition("core.categorical_embedding", 1, ("x",), ("out",), _categorical_embedding, parameter_rule=_categorical_embedding_parameters),
        PrimitiveDefinition("core.categorical_embedding", 2, ("x",), ("out",), _categorical_embedding_v2, parameter_rule=_categorical_embedding_v2_parameters),
        PrimitiveDefinition("core.invariant_concat", 1, ("xs",), ("out",), _invariant_concat),
        PrimitiveDefinition(
            "core.invariant_slice", 1, ("x",), ("out",), _invariant_slice,
            required_attrs=("start", "length"),
            optional_attrs={"axis": "output feature axis", "feature_role": "output feature role"},
        ),
        PrimitiveDefinition("core.invariant_product", 1, ("left", "right"), ("out",), _invariant_product),
        PrimitiveDefinition(
            "core.equivariant_channel_concat", 1, ("xs",), ("out",), _equivariant_channel_concat,
            group_families=("O3", "SO3"),
        ),
        PrimitiveDefinition(
            "core.degreewise_invariant_scale", 1, ("weight", "value"), ("out",), _degreewise_invariant_scale,
            group_families=("O3", "SO3"),
        ),
        PrimitiveDefinition("core.constant_scale", 1, ("x",), ("out",), _constant_scale),
        PrimitiveDefinition("core.flatten_invariant_axes", 1, ("x",), ("out",), _flatten_invariant_axes),
        PrimitiveDefinition("core.irrep_linear", 1, ("x",), ("out",), _irrep_linear, backend_keys=("e3nn.linear",)),
        PrimitiveDefinition(
            "core.irrep_linear", 2, ("x",), ("out",), _irrep_linear_v2,
            group_families=("O3", "SO3"), backend_keys=("e3nn.tensor_product.linear_rs",),
            parameter_rule=_irrep_linear_v2_parameters,
        ),
        PrimitiveDefinition("core.irrep_pad", 1, ("x",), ("out",), _irrep_pad),
        PrimitiveDefinition(
            "core.scalar_linear", 1, ("x",), ("out",), _scalar_linear,
            backend_keys=("torch.linear",), parameter_rule=_scalar_linear_parameters,
        ),
        PrimitiveDefinition(
            "core.scalar_linear", 2, ("x",), ("out",), _scalar_linear_v2,
            backend_keys=("torch.linear.output_scaled",), parameter_rule=_scalar_linear_v2_parameters,
        ),
        PrimitiveDefinition(
            "core.scalar_linear", 3, ("x",), ("out",), _scalar_linear_v3,
            backend_keys=("torch.linear.deferred_uniform_fan_in",), parameter_rule=_scalar_linear_v3_parameters,
        ),
        PrimitiveDefinition(
            "core.scalar_linear", 4, ("x",), ("out",), _scalar_linear_v4,
            backend_keys=("torch.linear.default_weight_zero_bias",), parameter_rule=_scalar_linear_v4_parameters,
        ),
        PrimitiveDefinition(
            "core.scalar_layer_norm", 1, ("x",), ("out",), _scalar_layer_norm,
            backend_keys=("torch.layer_norm",), parameter_rule=_scalar_layer_norm_parameters,
        ),
        PrimitiveDefinition(
            "core.scalar_offset", 1, ("x",), ("out",), _scalar_offset,
            backend_keys=("torch.parameter.offset",), parameter_rule=_scalar_offset_parameters,
        ),
        PrimitiveDefinition("core.head_split", 1, ("x",), ("out",), _head_split),
        PrimitiveDefinition("core.head_merge", 1, ("x",), ("out",), _head_merge),
        PrimitiveDefinition("core.head_split", 2, ("x",), ("out",), _head_split_v2),
        PrimitiveDefinition("core.head_merge", 2, ("x",), ("out",), _head_merge_v2),
        PrimitiveDefinition(
            "core.headwise_scalar_contraction", 1, ("x",), ("out",), _headwise_scalar_contraction,
            backend_keys=("torch.parameter",), parameter_rule=_headwise_scalar_contraction_parameters,
        ),
        PrimitiveDefinition(
            "core.headwise_scalar_contraction", 2, ("x",), ("out",), _headwise_scalar_contraction_v2,
            backend_keys=("torch.parameter",), parameter_rule=_headwise_scalar_contraction_v2_parameters,
        ),
        PrimitiveDefinition(
            "core.headwise_scalar_contraction", 3, ("x",), ("out",), _headwise_scalar_contraction_v3,
            required_attrs=("head_axis", "channel_axis"),
            optional_attrs={"bias": "must remain false for the official V3 alpha-dot path"},
            backend_keys=("torch.parameter.v3_alpha_dot",),
            parameter_rule=_headwise_scalar_contraction_v3_parameters,
        ),
        PrimitiveDefinition("core.irrep_concat", 1, ("xs",), ("out",), _irrep_concat),
        PrimitiveDefinition("core.residual_add", 1, ("left", "right"), ("out",), _residual_add),
        PrimitiveDefinition("core.residual_add", 2, ("left", "right"), ("out",), _residual_add_v2),
        PrimitiveDefinition("core.tensor_product", 1, ("left", "right"), ("out",), _tensor_product, backend_keys=("e3nn.tensor_product",)),
        PrimitiveDefinition(
            "core.tensor_product", 2, ("left", "right", "weight"), ("out",), _tensor_product_v2,
            group_families=("O3", "SO3"), backend_keys=("e3nn.tensor_product.external",),
            parameter_rule=_tensor_product_v2_parameters,
        ),
        PrimitiveDefinition(
            "core.tensor_product", 3, ("left", "right", "weight"), ("out",), _tensor_product_v3,
            group_families=("O3", "SO3"), backend_keys=("e3nn.tensor_product.instructions.external",),
            parameter_rule=_tensor_product_v3_parameters,
        ),
        PrimitiveDefinition(
            "core.tensor_product", 4, ("left", "right"), ("out",), _tensor_product_v4,
            group_families=("O3", "SO3"), backend_keys=("e3nn.tensor_product.instructions.internal",),
            parameter_rule=_tensor_product_v4_parameters,
        ),
        PrimitiveDefinition(
            "core.tensor_product", 5, ("left", "right"), ("out",), _tensor_product_v5,
            group_families=("O3", "SO3"), backend_keys=("e3nn.tensor_product.instructions.internal_shared_uvu",),
            parameter_rule=_tensor_product_v5_parameters,
        ),
        PrimitiveDefinition("core.scalar_activation", 1, ("x",), ("out",), _scalar_activation),
        PrimitiveDefinition("core.invariant_weight", 1, ("weight", "value"), ("out",), _invariant_weight),
        PrimitiveDefinition("core.invariant_scale", 1, ("weight", "value"), ("out",), _invariant_scale),
        PrimitiveDefinition("core.invariant_scale", 2, ("weight", "value"), ("out",), _invariant_scale_v2),
        PrimitiveDefinition("core.edge_lift", 1, ("x",), ("out",), _edge_lift),
        PrimitiveDefinition("core.endpoint_gather", 1, ("x", "index"), ("out",), _endpoint_gather),
        PrimitiveDefinition("core.endpoint_gather", 2, ("x", "index"), ("out",), _endpoint_gather_v2),
        PrimitiveDefinition("core.segment_sum", 1, ("x",), ("out",), _segment_sum),
        PrimitiveDefinition("core.segment_reduce", 1, ("x", "index"), ("out",), _segment_reduce_v2),
        PrimitiveDefinition("core.global_pool", 1, ("x",), ("out",), _global_pool),
        PrimitiveDefinition("core.select_scalars", 1, ("x",), ("out",), _select_scalars),
        PrimitiveDefinition("core.select_scalars", 2, ("x",), ("out",), _select_scalars_v2),
        PrimitiveDefinition("core.to_edge_frame", 1, ("x",), ("out",), _to_edge_frame),
        PrimitiveDefinition("core.from_edge_frame", 1, ("x",), ("out",), _from_edge_frame),
        PrimitiveDefinition(
            "core.to_edge_frame", 2, ("x", "direction"), ("out",), _to_edge_frame_v2,
            group_families=("SO3",),
            optional_attrs={
                "mmax": "maximum represented order",
                "frame_id": "edge-frame identity",
                "use_rotation_mask": "official V3 stabilized rotation mask",
            },
        ),
        PrimitiveDefinition(
            "core.from_edge_frame", 2, ("x",), ("out",), _from_edge_frame_v2,
            group_families=("SO3",),
            optional_attrs={
                "mmax": "maximum represented order",
                "frame_id": "edge-frame identity",
                "use_rotation_mask": "official V3 stabilized rotation mask",
            },
        ),
        PrimitiveDefinition("core.irrep_slice", 1, ("x",), ("out",), _irrep_slice),
        PrimitiveDefinition("core.irrep_select", 2, ("x",), ("out",), _irrep_select_v2),
        PrimitiveDefinition("core.change_multiplicity", 1, ("x",), ("out",), _irrep_linear, backend_keys=("e3nn.linear",)),
        PrimitiveDefinition("core.relative_position", 1, ("source", "target"), ("out",), _relative_position),
        PrimitiveDefinition(
            "core.relative_displacement", 2,
            ("positions", "source_index", "target_index"), ("out",),
            _relative_displacement_v2,
            group_families=("O3", "SO3"),
        ),
        PrimitiveDefinition(
            "core.relative_displacement", 3,
            ("positions", "source_index", "target_index"), ("out",),
            _relative_displacement_v3,
            group_families=("O3", "SO3"),
        ),
        PrimitiveDefinition(
            "core.periodic_displacement", 1,
            ("positions", "source_index", "target_index", "lattice", "lattice_shift"), ("out",),
            _periodic_displacement,
            group_families=("O3", "SO3"),
        ),
        PrimitiveDefinition(
            "core.periodic_displacement", 2,
            ("positions", "source_index", "target_index", "lattice", "lattice_shift"), ("out",),
            _periodic_displacement_v2,
            group_families=("O3", "SO3"),
        ),
        PrimitiveDefinition("core.distance", 1, ("vector",), ("out",), _distance),
        PrimitiveDefinition("core.distance", 2, ("vector",), ("out",), _distance_v2),
        PrimitiveDefinition("core.radial_basis", 1, ("distance",), ("out",), _radial_basis),
        PrimitiveDefinition("core.fixed_gaussian_radial_basis", 1, ("distance",), ("out",), _fixed_gaussian_radial_basis),
        PrimitiveDefinition(
            "core.gaussian_radial_basis", 1, ("distance",), ("out",), _gaussian_radial_basis,
            backend_keys=("torch.learnable_gaussian_rbf",), parameter_rule=_gaussian_radial_basis_parameters,
        ),
        PrimitiveDefinition("core.cutoff_envelope", 1, ("x",), ("out",), _equivariant_preserving),
        PrimitiveDefinition("core.cutoff_envelope", 2, ("x",), ("out",), _cutoff_envelope_v2),
        PrimitiveDefinition(
            "core.axisymmetric_spherical_lift", 1,
            ("amplitudes", "direction"), ("out",), _axisymmetric_spherical_lift,
            group_families=("SO3",), backend_keys=("equiformer_v3.wigner_inv_m0",),
        ),
        PrimitiveDefinition("core.spherical_harmonics", 1, ("direction",), ("out",), _spherical_harmonics, group_families=("O3", "SO3"), backend_keys=("e3nn.spherical_harmonics",)),
        PrimitiveDefinition("core.norm_activation", 1, ("x",), ("out",), _norm_activation),
        PrimitiveDefinition("core.gate", 1, ("gates", "value"), ("out",), _gate),
        PrimitiveDefinition("core.equivariant_norm", 1, ("x",), ("out",), _equivariant_preserving),
        PrimitiveDefinition(
            "core.irrep_layer_norm",
            1,
            ("x",),
            ("out",),
            _irrep_layer_norm,
            parameter_rule=_irrep_layer_norm_parameters,
        ),
        PrimitiveDefinition("core.stochastic_depth", 1, ("x",), ("out",), _dropout_preserving),
        PrimitiveDefinition("core.graph_stochastic_depth", 1, ("x", "batch"), ("out",), _graph_stochastic_depth),
        PrimitiveDefinition("core.scalar_dropout", 1, ("x",), ("out",), _scalar_dropout),
        PrimitiveDefinition("core.equivariant_dropout", 1, ("x",), ("out",), _equivariant_dropout_v1),
        PrimitiveDefinition("core.invariant_dropout", 1, ("x",), ("out",), _dropout_preserving),
        PrimitiveDefinition("core.segment_mean", 1, ("x",), ("out",), _segment_mean),
        PrimitiveDefinition("core.segment_softmax", 1, ("logits",), ("out",), _segment_softmax),
        PrimitiveDefinition("core.segment_softmax", 2, ("logits", "index"), ("out",), _segment_softmax_v2),
        PrimitiveDefinition(
            "core.segment_softmax", 3,
            ("logits", "index", "exp_rescale"), ("out",), _segment_softmax_v3,
            optional_attrs={
                "epsilon": "positive denominator stabilizer",
                "exp_dropout": "dropout applied to exponentiated neighbor contributions",
                "softcap": "optional positive tanh logit cap",
            },
        ),
        PrimitiveDefinition("core.invariant_compatibility", 1, ("query", "key"), ("out",), _compatibility),
        PrimitiveDefinition("core.so2_convolution", 1, ("x",), ("out",), _so2_convolution, group_families=("O3", "SO3"), backend_keys=("equiformer_v2.so2_convolution",)),
        PrimitiveDefinition("core.so2_linear", 1, ("x",), ("out",), _so2_linear_v1, group_families=("SO3",), backend_keys=("torch.so2_linear",), parameter_rule=_so2_linear_parameters),
        PrimitiveDefinition("core.so2_linear", 2, ("x",), ("out", "extra_m0"), _so2_linear_v2, group_families=("SO3",), backend_keys=("torch.so2_linear_with_extra_m0",), parameter_rule=_so2_linear_parameters),
        PrimitiveDefinition("core.s2_activation", 1, ("x",), ("out",), _s2_activation, group_families=("O3", "SO3"), backend_keys=("equiformer_v2.s2_activation",)),
        PrimitiveDefinition("core.separable_s2_activation", 1, ("scalars", "x"), ("out",), _separable_s2_activation, group_families=("O3", "SO3"), backend_keys=("equiformer_v2.separable_s2_activation",)),
        PrimitiveDefinition(
            "core.grid_project", 1, ("x",), ("out",), _grid_project,
            group_families=("O3", "SO3"), certificate_level=EquivarianceLevel.CONSTRUCTIVE,
            backend_keys=("s2.grid_project",),
        ),
        PrimitiveDefinition(
            "core.grid_unproject", 1, ("x",), ("out",), _grid_unproject,
            group_families=("O3", "SO3"), certificate_level=EquivarianceLevel.CONSTRUCTIVE,
            backend_keys=("s2.grid_unproject",),
        ),
        PrimitiveDefinition(
            "core.grid_split", 1, ("x",), ("left", "right"), _grid_split,
            group_families=("O3", "SO3"), certificate_level=EquivarianceLevel.CONSTRUCTIVE,
            backend_keys=("torch.grid_split",),
        ),
        PrimitiveDefinition(
            "core.grid_concat", 1, ("xs",), ("out",), _grid_concat,
            group_families=("O3", "SO3"), certificate_level=EquivarianceLevel.CONSTRUCTIVE,
            backend_keys=("torch.grid_concat",),
        ),
        PrimitiveDefinition(
            "core.grid_pointwise_activation", 1, ("x",), ("out",), _grid_pointwise_activation,
            group_families=("O3", "SO3"), certificate_level=EquivarianceLevel.EMPIRICAL,
            backend_keys=("torch.grid_pointwise_activation",),
        ),
        PrimitiveDefinition(
            "core.grid_pointwise_product", 1, ("left", "right"), ("out",), _grid_pointwise_product,
            group_families=("O3", "SO3"), certificate_level=EquivarianceLevel.EMPIRICAL,
            backend_keys=("torch.grid_pointwise_product",),
        ),
        PrimitiveDefinition(
            "core.grid_channel_linear", 1, ("x",), ("out",), _grid_channel_linear,
            group_families=("O3", "SO3"), certificate_level=EquivarianceLevel.CONSTRUCTIVE,
            backend_keys=("torch.grid_channel_linear",), parameter_rule=_grid_channel_linear_parameters,
        ),
        PrimitiveDefinition(
            "core.grid_dropout", 1, ("x",), ("out",), _grid_dropout,
            group_families=("O3", "SO3"), certificate_level=EquivarianceLevel.EMPIRICAL,
            backend_keys=("torch.grid_dropout",),
        ),
        PrimitiveDefinition("core.s2_swiglu", 1, ("x",), ("out",), _s2_swiglu, group_families=("O3", "SO3"), certificate_level=EquivarianceLevel.EMPIRICAL, backend_keys=("equiformer_v3.s2_swiglu",)),
        PrimitiveDefinition(
            "core.s2_gated_swiglu_merge", 1,
            ("x", "scalars"), ("out",), _s2_gated_swiglu_merge,
            group_families=("SO3",), certificate_level=EquivarianceLevel.EMPIRICAL,
            required_attrs=("out_irreps",),
            optional_attrs={
                "mmax": "maximum represented order",
                "grid_resolution": "official V3 latitude/longitude grid resolution",
                "normalization": "S2 grid normalization",
                "dropout": "grid-product dropout",
            },
            backend_keys=("equiformer_v3.sep_merge_gates2_swiglu",),
        ),
        PrimitiveDefinition(
            "core.edge_frame_gate_activation", 1,
            ("x", "scalars"), ("out",), _edge_frame_gate_activation,
            group_families=("SO3",), certificate_level=EquivarianceLevel.CONSTRUCTIVE,
            optional_attrs={"mmax": "maximum represented order"},
            backend_keys=("equiformer_v3.edge_frame_gate",),
        ),
        PrimitiveDefinition(
            "core.so3_linear", 1, ("x",), ("out",), _so3_linear,
            group_families=("SO3",), required_attrs=("out_irreps",),
            optional_attrs={"bias": "l=0-only bias"},
            backend_keys=("torch.v3_so3_linear",), parameter_rule=_so3_linear_parameters,
        ),
        PrimitiveDefinition(
            "core.so3_linear", 2, ("x",), ("out",), _so3_linear_v2,
            group_families=("SO3",),
            backend_keys=("torch.v3_so3_linear.l0_scaled",), parameter_rule=_so3_linear_v2_parameters,
        ),
        PrimitiveDefinition("core.equivariant_merge_norm", 1, ("x",), ("out",), _equivariant_merge_norm, group_families=("O3", "SO3"), backend_keys=("equiformer_v3.merge_layer_norm",)),
    )
    for definition in definitions:
        contract = _PRIMITIVE_CONTRACTS.get(
            definition.qualified_name,
            _PRIMITIVE_CONTRACTS.get(definition.name, {}),
        )
        definition = replace(definition, **contract)
        registry.register(definition)
    return registry
