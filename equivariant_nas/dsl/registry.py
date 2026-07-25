"""Trusted primitive registry and core equivariant type rules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence, Tuple

from .diagnostics import DSLValidationError, Diagnostic
from .irreps import Irrep, Irreps
from .obligations import ObligationKind, ProofObligation
from .types import Carrier, EquivarianceLevel, EquivariantType, Frame


TypeRule = Callable[[str, Mapping[str, Tuple[EquivariantType, ...]], Mapping[str, Any]], Tuple[Mapping[str, EquivariantType], Tuple[ProofObligation, ...]]]


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
    motif_parameter_attrs: Tuple[str, ...] = ()
    semantic_constraints: Tuple[str, ...] = ()
    edit_guidance: Tuple[str, ...] = ()

    @property
    def qualified_name(self) -> str:
        return "{}@{}".format(self.name, self.version)

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
            "motif_parameter_attrs": self.motif_parameter_attrs,
            "semantic_constraints": self.semantic_constraints,
            "edit_guidance": self.edit_guidance,
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


def _single(node: str, inputs: Mapping[str, Tuple[EquivariantType, ...]], port: str) -> EquivariantType:
    values = inputs.get(port, ())
    if len(values) != 1:
        raise DSLValidationError([Diagnostic("E_PORT_001", "port requires exactly one input", node_id=node, port=port, actual=str(len(values)))])
    return values[0]


def _same_context(node: str, left: EquivariantType, right: EquivariantType, *, exact_irreps: bool = True) -> None:
    if not left.compatible(right, exact_irreps=exact_irreps):
        raise DSLValidationError([
            Diagnostic("E_TYPE_004", "input types are incompatible", node_id=node, expected=str(left.to_dict()), actual=str(right.to_dict()))
        ])


def _identity(node, inputs, attrs):
    return {"out": _single(node, inputs, "x")}, ()


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
    return {"out": left.with_irreps(out)}, obligations


def _scalar_activation(node, inputs, attrs):
    x = _single(node, inputs, "x")
    non_scalars = [str(ir) for _, ir in x.irreps if ir.degree != 0]
    if non_scalars:
        raise DSLValidationError([
            Diagnostic("E_NONLINEAR_001", "ordinary elementwise activation is only valid on scalar irreps", node_id=node, details={"non_scalars": non_scalars})
        ])
    return {"out": x}, ()


def _invariant_weight(node, inputs, attrs):
    weight = _single(node, inputs, "weight")
    value = _single(node, inputs, "value")
    if any(ir.degree != 0 or (ir.family in ("O3", "O2") and ir.parity != 1) for _, ir in weight.irreps):
        raise DSLValidationError([Diagnostic("E_ATTN_001", "attention weights must be invariant scalars", node_id=node, port="weight")])
    if weight.irreps.dimension != 1:
        raise DSLValidationError([Diagnostic("E_ATTN_004", "invariant_weight requires one scalar until an explicit head axis is declared", node_id=node, port="weight")])
    if weight.carrier != value.carrier or weight.frame not in (value.frame, Frame("invariant")):
        raise DSLValidationError([Diagnostic("E_ATTN_002", "attention weight and value must share carrier and compatible frame", node_id=node)])
    obligation = ProofObligation("{}:weight".format(node), ObligationKind.INVARIANT_ATTENTION_WEIGHT, node, "discharged", "core.invariant_weight@1")
    return {"out": value}, (obligation,)


def _edge_lift(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if x.carrier != Carrier.NODE:
        raise DSLValidationError([Diagnostic("E_CARRIER_001", "edge_lift expects node features", node_id=node)])
    return {"out": x.with_carrier(Carrier.EDGE)}, ()


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


def _to_edge_frame(node, inputs, attrs):
    x = _single(node, inputs, "x")
    if x.frame.kind != "global":
        raise DSLValidationError([Diagnostic("E_FRAME_005", "to_edge_frame expects global features", node_id=node, actual=str(x.frame))])
    reference = str(attrs.get("frame_id", node))
    obligation = ProofObligation("{}:frame-return".format(node), ObligationKind.FRAME_BALANCE, node, details={"frame_id": reference})
    return {"out": x.with_frame(Frame("edge", reference))}, (obligation,)


def _from_edge_frame(node, inputs, attrs):
    x = _single(node, inputs, "x")
    expected = str(attrs.get("frame_id", ""))
    if x.frame.kind != "edge" or (expected and x.frame.reference != expected):
        raise DSLValidationError([Diagnostic("E_FRAME_006", "from_edge_frame received a mismatched frame", node_id=node, expected=expected, actual=str(x.frame))])
    obligation = ProofObligation("{}:frame-restored".format(node), ObligationKind.FRAME_BALANCE, node, "discharged", "core.from_edge_frame@1", {"frame_id": x.frame.reference})
    return {"out": x.with_frame(Frame("global"))}, (obligation,)


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


def _relative_position(node, inputs, attrs):
    source = _single(node, inputs, "source")
    target = _single(node, inputs, "target")
    _same_context(node, source, target)
    if source.carrier != Carrier.NODE or source.irreps.dimension != source.group.dimension:
        raise DSLValidationError([Diagnostic("E_GEOMETRY_001", "relative_position expects one Cartesian node position", node_id=node)])
    if source.group.translation != "relative_coordinates":
        raise DSLValidationError([Diagnostic("E_GEOMETRY_002", "task group does not authorize relative-coordinate translation handling", node_id=node)])
    return {"out": EquivariantType(source.group, Carrier.EDGE, source.irreps, Frame("global"), source.axes, source.dtype, source.measure, source.level)}, ()


def _distance(node, inputs, attrs):
    vector = _single(node, inputs, "vector")
    if vector.carrier != Carrier.EDGE or vector.irreps.dimension != vector.group.dimension:
        raise DSLValidationError([Diagnostic("E_GEOMETRY_003", "distance expects one Cartesian edge vector", node_id=node)])
    scalar_family = vector.group.family
    suffix = "e" if scalar_family in ("O3", "O2") else ""
    scalar = Irreps.parse("1x0{}".format(suffix) if scalar_family in ("O3", "SO3") else "1xm0{}".format(suffix), scalar_family)
    return {"out": EquivariantType(vector.group, Carrier.EDGE, scalar, Frame("invariant"), vector.axes, vector.dtype, vector.measure, vector.level)}, ()


def _radial_basis(node, inputs, attrs):
    distance = _single(node, inputs, "distance")
    if distance.carrier != Carrier.EDGE or any(ir.degree != 0 for _, ir in distance.irreps):
        raise DSLValidationError([Diagnostic("E_GEOMETRY_004", "radial_basis expects invariant edge scalars", node_id=node)])
    count = int(attrs.get("num_basis", 0))
    if count <= 0:
        raise DSLValidationError([Diagnostic("E_ATTR_001", "num_basis must be positive", node_id=node)])
    family = distance.group.family
    scalar = "{}x0e".format(count) if family == "O3" else "{}x0".format(count) if family == "SO3" else "{}xm0e".format(count) if family == "O2" else "{}xm0".format(count)
    return {"out": distance.with_irreps(Irreps.parse(scalar, family))}, ()


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
    return {"out": logits}, ()


def _equivariant_preserving(node, inputs, attrs):
    return {"out": _single(node, inputs, "x")}, ()


def _dropout_preserving(node, inputs, attrs):
    probability = float(attrs.get("p", 0.0))
    if probability < 0.0 or probability >= 1.0:
        raise DSLValidationError([Diagnostic("E_ATTR_003", "dropout probability must satisfy 0 <= p < 1", node_id=node, actual=str(probability))])
    return {"out": _single(node, inputs, "x")}, ()


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
    return {"out": x.with_irreps(out)}, ()


def _separable_s2_activation(node, inputs, attrs):
    scalars = _single(node, inputs, "scalars")
    x = _single(node, inputs, "x")
    _same_context(node, scalars, x, exact_irreps=False)
    if any(ir.degree != 0 for _, ir in scalars.irreps):
        raise DSLValidationError([Diagnostic("E_S2_001", "separable S2 activation requires a scalar side path", node_id=node, port="scalars")])
    scalar_count = sum(mul for mul, _ in scalars.irreps)
    x_scalar_count = sum(mul for mul, ir in x.irreps if ir.degree == 0)
    if scalar_count != x_scalar_count:
        raise DSLValidationError([Diagnostic("E_S2_002", "scalar side path must match the l=0 multiplicity", node_id=node, expected=str(x_scalar_count), actual=str(scalar_count))])
    return {"out": x}, ()


def _compatibility(node, inputs, attrs):
    query = _single(node, inputs, "query")
    key = _single(node, inputs, "key")
    _same_context(node, query, key, exact_irreps=True)
    family = query.group.family
    scalar = "1x0e" if family == "O3" else "1x0" if family == "SO3" else "1xm0e" if family == "O2" else "1xm0"
    return {"out": query.with_irreps(Irreps.parse(scalar, family)).with_frame(Frame("invariant"))}, ()


_PRIMITIVE_CONTRACTS = {
    "core.identity": {
        "description": "Type-preserving identity map.",
        "semantic_constraints": ("Output type equals input type exactly.",),
    },
    "core.irrep_linear": {
        "description": "Equivariant linear map within representation kinds already present in the input.",
        "required_attrs": ("out_irreps",),
        "semantic_constraints": ("May change multiplicities but cannot create a new irrep degree or parity absent from the input.",),
        "edit_guidance": ("Use at representation-width boundaries; use tensor_product when a new irrep degree is required.",),
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
    "core.tensor_product": {
        "description": "Clebsch-Gordan-compatible equivariant tensor-product coupling.",
        "required_attrs": ("out_irreps",),
        "semantic_constraints": ("Every requested output irrep must occur in a legal input tensor-product path, including parity.",),
        "edit_guidance": ("Use this operation, not gate or irrep_linear, to couple hidden l>0 paths into l=0 or other new degrees.",),
    },
    "core.scalar_activation": {
        "description": "Ordinary pointwise nonlinearity restricted to invariant scalar irreps.",
        "optional_attrs": {"activation": "backend activation name"},
        "motif_parameter_attrs": ("activation",),
        "semantic_constraints": ("All input irreps must be l=0 invariant scalars; output type is unchanged.",),
    },
    "core.invariant_weight": {
        "description": "Multiply an equivariant value by one invariant scalar attention weight.",
        "semantic_constraints": ("weight must be one invariant scalar with compatible carrier and frame; output type equals value type.",),
    },
    "core.edge_lift": {
        "description": "Lift node-carried values to directed edges without changing irreps.",
        "semantic_constraints": ("Input carrier must be node and output carrier is edge.",),
    },
    "core.segment_sum": {
        "description": "Permutation-safe edge-to-node sum aggregation.",
        "semantic_constraints": ("Input carrier must be edge and must be in global, not edge-local, frame.",),
    },
    "core.segment_mean": {
        "description": "Permutation-safe edge-to-node mean aggregation.",
        "semantic_constraints": ("Input carrier must be edge and must be in global, not edge-local, frame.",),
    },
    "core.global_pool": {
        "description": "Permutation-invariant node-to-graph pooling.",
        "semantic_constraints": ("Input carrier must be node; irreps are preserved while carrier becomes graph.",),
    },
    "core.select_scalars": {
        "description": "Select invariant l=0 channels from a mixed-irrep value.",
        "optional_attrs": {"multiplicity": "positive number of scalar channels to retain"},
        "semantic_constraints": ("At least one invariant scalar channel must exist; non-scalars are discarded, not coupled.",),
    },
    "core.to_edge_frame": {
        "description": "Express global-frame 3D irreps in an edge-aligned local frame.",
        "optional_attrs": {"frame_id": "stable identifier paired with from_edge_frame"},
        "semantic_constraints": ("Input must be in global frame and creates an open frame-balance proof obligation.",),
        "edit_guidance": ("Pair with from_edge_frame using the same frame_id before node aggregation.",),
    },
    "core.from_edge_frame": {
        "description": "Return edge-local 3D irreps to the global frame.",
        "optional_attrs": {"frame_id": "identifier matching to_edge_frame"},
        "semantic_constraints": ("Input must be in the matching edge frame; output is global-frame.",),
    },
    "core.irrep_slice": {
        "description": "Select available irrep blocks without changing their transformation law.",
        "required_attrs": ("irreps",),
        "semantic_constraints": ("Requested multiplicity for every irrep cannot exceed the input multiplicity.",),
    },
    "core.relative_position": {
        "description": "Construct translation-safe directed edge displacement vectors from node positions.",
        "semantic_constraints": ("Inputs are Cartesian node vectors and the task must authorize relative-coordinate translation handling.",),
    },
    "core.distance": {
        "description": "Compute invariant scalar distances from Cartesian edge vectors.",
        "semantic_constraints": ("Input must be one edge-carried Cartesian vector; output is an invariant edge scalar.",),
    },
    "core.radial_basis": {
        "description": "Expand invariant edge distances in a scalar radial basis.",
        "required_attrs": ("num_basis",),
        "semantic_constraints": ("num_basis must be positive and input must be invariant edge scalars.",),
    },
    "core.cutoff_envelope": {
        "description": "Type-preserving invariant cutoff envelope.",
        "optional_attrs": {"cutoff": "positive radial cutoff", "order": "envelope polynomial order"},
        "motif_parameter_attrs": ("cutoff", "order"),
        "semantic_constraints": ("Output type equals input type.",),
    },
    "core.spherical_harmonics": {
        "description": "Construct 3D spherical-harmonic edge features through degree lmax.",
        "required_attrs": ("lmax",),
        "semantic_constraints": ("Only 3D SO(3)/O(3) groups are supported and lmax must be nonnegative.",),
    },
    "core.norm_activation": {
        "description": "Equivariant nonlinearity that acts through irrep norms.",
        "optional_attrs": {"activation": "scalar activation applied to norms"},
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
        "semantic_constraints": ("Output type equals input type; normalization must not mix representation coordinates illegally.",),
    },
    "core.stochastic_depth": {
        "description": "Drop complete equivariant residual paths stochastically.",
        "optional_attrs": {"p": "drop probability in [0,1)"},
        "motif_parameter_attrs": ("p",),
        "semantic_constraints": ("p must satisfy 0 <= p < 1 and output type equals input type.",),
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
    "core.invariant_compatibility": {
        "description": "Produce one invariant scalar compatibility score from equal-type query and key values.",
        "semantic_constraints": ("query and key types must match exactly; output is one invariant scalar in an invariant frame.",),
    },
    "core.so2_convolution": {
        "description": "Equiformer V2-style SO(2) frequency convolution in an edge-aligned frame.",
        "optional_attrs": {"out_irreps": "output multiplicities over irrep kinds present in input"},
        "semantic_constraints": ("Requires 3D features in an edge frame and cannot create irrep degrees absent from the input.",),
    },
    "core.s2_activation": {
        "description": "S2-grid equivariant activation for 3D spherical representations.",
        "optional_attrs": {"grid_resolution": "backend quadrature resolution"},
        "motif_parameter_attrs": ("grid_resolution",),
        "semantic_constraints": ("Preserves the declared equivariant type and is restricted to SO(3)/O(3).",),
    },
    "core.separable_s2_activation": {
        "description": "Equiformer V2 separable S2 activation with an explicit scalar side path.",
        "optional_attrs": {"grid_resolution": "backend quadrature resolution"},
        "motif_parameter_attrs": ("grid_resolution",),
        "semantic_constraints": (
            "scalars and x share group, carrier, and frame.",
            "the scalar side path contains only l=0 and matches x's l=0 multiplicity.",
            "output type equals x type.",
        ),
    },
}


def core_registry() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()
    definitions = (
        PrimitiveDefinition("core.identity", 1, ("x",), ("out",), _identity),
        PrimitiveDefinition("core.irrep_linear", 1, ("x",), ("out",), _irrep_linear, backend_keys=("e3nn.linear",)),
        PrimitiveDefinition("core.irrep_concat", 1, ("xs",), ("out",), _irrep_concat),
        PrimitiveDefinition("core.residual_add", 1, ("left", "right"), ("out",), _residual_add),
        PrimitiveDefinition("core.tensor_product", 1, ("left", "right"), ("out",), _tensor_product, backend_keys=("e3nn.tensor_product",)),
        PrimitiveDefinition("core.scalar_activation", 1, ("x",), ("out",), _scalar_activation),
        PrimitiveDefinition("core.invariant_weight", 1, ("weight", "value"), ("out",), _invariant_weight),
        PrimitiveDefinition("core.edge_lift", 1, ("x",), ("out",), _edge_lift),
        PrimitiveDefinition("core.segment_sum", 1, ("x",), ("out",), _segment_sum),
        PrimitiveDefinition("core.global_pool", 1, ("x",), ("out",), _global_pool),
        PrimitiveDefinition("core.select_scalars", 1, ("x",), ("out",), _select_scalars),
        PrimitiveDefinition("core.to_edge_frame", 1, ("x",), ("out",), _to_edge_frame),
        PrimitiveDefinition("core.from_edge_frame", 1, ("x",), ("out",), _from_edge_frame),
        PrimitiveDefinition("core.irrep_slice", 1, ("x",), ("out",), _irrep_slice),
        PrimitiveDefinition("core.change_multiplicity", 1, ("x",), ("out",), _irrep_linear, backend_keys=("e3nn.linear",)),
        PrimitiveDefinition("core.relative_position", 1, ("source", "target"), ("out",), _relative_position),
        PrimitiveDefinition("core.distance", 1, ("vector",), ("out",), _distance),
        PrimitiveDefinition("core.radial_basis", 1, ("distance",), ("out",), _radial_basis),
        PrimitiveDefinition("core.cutoff_envelope", 1, ("x",), ("out",), _equivariant_preserving),
        PrimitiveDefinition("core.spherical_harmonics", 1, ("direction",), ("out",), _spherical_harmonics, group_families=("O3", "SO3"), backend_keys=("e3nn.spherical_harmonics",)),
        PrimitiveDefinition("core.norm_activation", 1, ("x",), ("out",), _norm_activation),
        PrimitiveDefinition("core.gate", 1, ("gates", "value"), ("out",), _gate),
        PrimitiveDefinition("core.equivariant_norm", 1, ("x",), ("out",), _equivariant_preserving),
        PrimitiveDefinition("core.stochastic_depth", 1, ("x",), ("out",), _dropout_preserving),
        PrimitiveDefinition("core.invariant_dropout", 1, ("x",), ("out",), _dropout_preserving),
        PrimitiveDefinition("core.segment_mean", 1, ("x",), ("out",), _segment_mean),
        PrimitiveDefinition("core.segment_softmax", 1, ("logits",), ("out",), _segment_softmax),
        PrimitiveDefinition("core.invariant_compatibility", 1, ("query", "key"), ("out",), _compatibility),
        PrimitiveDefinition("core.so2_convolution", 1, ("x",), ("out",), _so2_convolution, group_families=("O3", "SO3"), backend_keys=("equiformer_v2.so2_convolution",)),
        PrimitiveDefinition("core.s2_activation", 1, ("x",), ("out",), _equivariant_preserving, group_families=("O3", "SO3"), backend_keys=("equiformer_v2.s2_activation",)),
        PrimitiveDefinition("core.separable_s2_activation", 1, ("scalars", "x"), ("out",), _separable_s2_activation, group_families=("O3", "SO3"), backend_keys=("equiformer_v2.separable_s2_activation",)),
    )
    for definition in definitions:
        contract = _PRIMITIVE_CONTRACTS.get(definition.name, {})
        definition = replace(definition, **contract)
        registry.register(definition)
    return registry
