"""Backend-neutral contracts for primitive-by-primitive numerical lowering."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..ast import ArchitectureProgram, Node
from ..diagnostics import DSLValidationError, Diagnostic
from ..parameters import ParameterContract
from ..types import AffinePointType, CategoricalTensorType, GraphTopologyType, GridTensorType, IndexMapType, LatticeShiftType, LatticeType


ModuleBuilder = Callable[["ModuleBuildContext"], Any]
ModuleInitializer = Callable[[Any, "ModuleBuildContext"], None]
NodeValidator = Callable[["ModuleBuildContext"], None]
RuntimeExecutor = Callable[["RuntimeExecutionContext"], Any]
RuntimeKindRule = Callable[[Node, Mapping[str, Tuple[str, ...]]], Mapping[str, str]]


CERTIFIED_EXACTNESS = frozenset({
    "constructive_exact",
    "library_exact",
    "library_exact_fusion",
})


class RuntimeValueKind(str):
    """Backend value representations that affect legal graph composition."""

    DENSE_TENSOR = "dense_tensor"
    CATEGORICAL_TENSOR = "categorical_tensor"
    EQUIVARIANT_HEADS = "equivariant_heads"
    SO3_EDGE_FRAME = "so3_edge_frame"
    INDEX_MAP = "index_map"
    GRAPH_TOPOLOGY = "graph_topology"
    AFFINE_POINT = "affine_point"
    LATTICE = "lattice"
    LATTICE_SHIFT = "lattice_shift"
    GRID_TENSOR = "grid_tensor"


def runtime_kind_for_value_type(value_type: Any) -> str:
    if isinstance(value_type, CategoricalTensorType):
        return RuntimeValueKind.CATEGORICAL_TENSOR
    if isinstance(value_type, IndexMapType):
        return RuntimeValueKind.INDEX_MAP
    if isinstance(value_type, GraphTopologyType):
        return RuntimeValueKind.GRAPH_TOPOLOGY
    if isinstance(value_type, AffinePointType):
        return RuntimeValueKind.AFFINE_POINT
    if isinstance(value_type, LatticeType):
        return RuntimeValueKind.LATTICE
    if isinstance(value_type, LatticeShiftType):
        return RuntimeValueKind.LATTICE_SHIFT
    if isinstance(value_type, GridTensorType):
        return RuntimeValueKind.GRID_TENSOR
    return RuntimeValueKind.DENSE_TENSOR


@dataclass(frozen=True)
class DependencyRequirement:
    """One importable Python module required by a numerical lowering rule."""

    module: str
    label: str = ""
    checker: Optional[Callable[[], bool]] = None

    @property
    def display_name(self) -> str:
        return self.label or self.module


@dataclass(frozen=True)
class ModuleBuildContext:
    node: Node
    output_types: Mapping[str, Any]
    value_type: Callable[[str], Any]
    libraries: Mapping[str, Any]
    parameter_contracts: Tuple[ParameterContract, ...] = ()

    @property
    def output_type(self) -> Any:
        if "out" in self.output_types:
            return self.output_types["out"]
        if len(self.output_types) == 1:
            return next(iter(self.output_types.values()))
        raise RuntimeError(
            "multi-output node {} must select an explicit output type".format(self.node.id)
        )


@dataclass(frozen=True)
class RuntimeExecutionContext:
    node: Node
    resolved: Mapping[str, Tuple[Any, ...]]
    output_types: Mapping[str, Any]
    value_type: Callable[[str], Any]
    module: Any
    graph_context: Mapping[str, Any]
    program_inputs: Mapping[str, Any]
    training: bool
    libraries: Mapping[str, Any]

    @property
    def output_type(self) -> Any:
        if "out" in self.output_types:
            return self.output_types["out"]
        if len(self.output_types) == 1:
            return next(iter(self.output_types.values()))
        raise RuntimeError(
            "multi-output node {} must select an explicit output type".format(self.node.id)
        )

    def require_context(self, *keys: str) -> None:
        missing = tuple(key for key in keys if key not in self.graph_context)
        if missing:
            raise RuntimeError(
                "lowering of {} requires graph context keys {}".format(
                    self.node.op,
                    ", ".join(missing),
                )
            )


@dataclass(frozen=True)
class LoweringRule:
    """Complete executable contract for one qualified DSL primitive."""

    primitive: str
    executor: RuntimeExecutor
    module_builder: Optional[ModuleBuilder] = None
    validator: Optional[NodeValidator] = None
    dependencies: Tuple[DependencyRequirement, ...] = ()
    required_context_keys: Tuple[str, ...] = ()
    exactness: str = "experimental"
    description: str = ""
    runtime_kind_rule: Optional[RuntimeKindRule] = None
    post_build_initializer: Optional[ModuleInitializer] = None

    def __post_init__(self) -> None:
        if "@" not in self.primitive:
            raise ValueError("lowering rule primitive must be version-qualified")
        if not callable(self.executor):
            raise TypeError("lowering rule executor must be callable")

    def validate(self, context: ModuleBuildContext) -> None:
        if self.validator is not None:
            self.validator(context)

    def build_module(self, context: ModuleBuildContext) -> Any:
        return self.module_builder(context) if self.module_builder is not None else None

    def execute(self, context: RuntimeExecutionContext) -> Any:
        if self.required_context_keys:
            context.require_context(*self.required_context_keys)
        return self.executor(context)

    def initialize(self, module: Any, context: ModuleBuildContext) -> None:
        if self.post_build_initializer is not None:
            if module is None:
                raise RuntimeError(
                    "post-build initializer for {} requires a constructed module".format(
                        self.primitive
                    )
                )
            self.post_build_initializer(module, context)

    def infer_runtime_kinds(
        self,
        node: Node,
        resolved: Mapping[str, Tuple[str, ...]],
    ) -> Mapping[str, str]:
        if self.runtime_kind_rule is not None:
            outputs = dict(self.runtime_kind_rule(node, resolved))
        else:
            invalid = {
                port: tuple(kind for kind in kinds if kind != RuntimeValueKind.DENSE_TENSOR)
                for port, kinds in resolved.items()
            }
            invalid = {port: kinds for port, kinds in invalid.items() if kinds}
            if invalid:
                raise ValueError(
                    "{} only supports dense tensor inputs; received {}".format(
                        self.primitive,
                        invalid,
                    )
                )
            outputs = {port: RuntimeValueKind.DENSE_TENSOR for port in node.outputs}
        if set(outputs) != set(node.outputs):
            raise ValueError(
                "runtime kind rule for {} returned {} but node declares {}".format(
                    self.primitive,
                    sorted(outputs),
                    list(node.outputs),
                )
            )
        return outputs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primitive": self.primitive,
            "dependencies": [item.display_name for item in self.dependencies],
            "required_context_keys": list(self.required_context_keys),
            "exactness": self.exactness,
            "has_executor": callable(self.executor),
            "has_module_builder": self.module_builder is not None,
            "has_validator": self.validator is not None,
            "description": self.description,
            "has_runtime_kind_rule": self.runtime_kind_rule is not None,
            "has_post_build_initializer": self.post_build_initializer is not None,
        }


@dataclass(frozen=True)
class BackendSupportReport:
    backend: str
    supported: bool
    unsupported_nodes: Tuple[Tuple[str, str], ...]
    missing_dependencies: Tuple[str, ...] = ()
    node_exactness: Tuple[Tuple[str, str], ...] = ()
    runtime_kinds: Tuple[Tuple[str, str], ...] = ()
    composition_errors: Tuple[Tuple[str, str], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "supported": self.supported,
            "unsupported_nodes": [list(item) for item in self.unsupported_nodes],
            "missing_dependencies": list(self.missing_dependencies),
            "node_exactness": {
                node_id: exactness for node_id, exactness in self.node_exactness
            },
            "runtime_kinds": {
                reference: kind for reference, kind in self.runtime_kinds
            },
            "composition_errors": [list(item) for item in self.composition_errors],
        }


class LoweringRuleRegistry:
    """Single source of truth for backend support and primitive execution."""

    def __init__(self, backend: str, base_dependencies: Sequence[DependencyRequirement] = ()):
        self.backend = str(backend)
        self.base_dependencies = tuple(base_dependencies)
        self._rules: Dict[str, LoweringRule] = {}

    @staticmethod
    def qualify(name: str) -> str:
        return name if "@" in name else "{}@1".format(name)

    def register(self, rule: LoweringRule) -> None:
        key = self.qualify(rule.primitive)
        if key in self._rules:
            raise DSLValidationError([
                Diagnostic("E_LOWERING_001", "duplicate numerical lowering rule", actual=key)
            ])
        self._rules[key] = rule

    def resolve(self, name: str) -> LoweringRule:
        key = self.qualify(name)
        try:
            return self._rules[key]
        except KeyError:
            raise DSLValidationError([
                Diagnostic("E_LOWERING_002", "no numerical lowering rule is registered", actual=key)
            ])

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._rules))

    def rules(self) -> Tuple[LoweringRule, ...]:
        return tuple(self._rules[name] for name in self.names())

    @staticmethod
    def _dependency_available(requirement: DependencyRequirement) -> bool:
        if requirement.checker is not None:
            try:
                return bool(requirement.checker())
            except Exception:
                return False
        try:
            return importlib.util.find_spec(requirement.module) is not None
        except (ImportError, AttributeError, ValueError):
            return False

    def dependency_requirements(
        self,
        program: ArchitectureProgram,
        ignored_nodes: Sequence[str] = (),
    ) -> Tuple[DependencyRequirement, ...]:
        ignored = set(ignored_nodes)
        requirements = {
            item.module: item
            for item in self.base_dependencies
        }
        for node in program.nodes:
            if node.id in ignored:
                continue
            key = self.qualify(node.op)
            rule = self._rules.get(key)
            if rule is None:
                continue
            for item in rule.dependencies:
                requirements[item.module] = item
        return tuple(requirements[name] for name in sorted(requirements))

    def support_report(
        self,
        program: ArchitectureProgram,
        ignored_nodes: Sequence[str] = (),
    ) -> BackendSupportReport:
        ignored = set(ignored_nodes)
        unsupported = []
        exactness = []
        for node in program.nodes:
            if node.id in ignored:
                continue
            qualified = self.qualify(node.op)
            rule = self._rules.get(qualified)
            if rule is None:
                unsupported.append((node.id, qualified))
            else:
                exactness.append((node.id, rule.exactness))
        missing = tuple(
            requirement.display_name
            for requirement in self.dependency_requirements(program, ignored_nodes)
            if not self._dependency_available(requirement)
        )
        runtime_values = {
            "input:{}".format(item.name): runtime_kind_for_value_type(item.value_type)
            for item in program.inputs
        }
        composition_errors = []
        pending = list(program.nodes)
        while pending:
            progressed = False
            for node in tuple(pending):
                references = [
                    reference
                    for values in node.inputs.values()
                    for reference in values
                ]
                if any(reference not in runtime_values for reference in references):
                    continue
                resolved_kinds = {
                    port: tuple(runtime_values[reference] for reference in references)
                    for port, references in node.inputs.items()
                }
                qualified = self.qualify(node.op)
                rule = self._rules.get(qualified)
                if rule is None:
                    if node.id in ignored:
                        output_kinds = {
                            port: RuntimeValueKind.DENSE_TENSOR for port in node.outputs
                        }
                    else:
                        output_kinds = {}
                else:
                    try:
                        output_kinds = dict(rule.infer_runtime_kinds(node, resolved_kinds))
                    except (TypeError, ValueError) as exc:
                        composition_errors.append((node.id, str(exc)))
                        output_kinds = {
                            port: RuntimeValueKind.DENSE_TENSOR for port in node.outputs
                        }
                for port, kind in output_kinds.items():
                    runtime_values["{}:{}".format(node.id, port)] = kind
                if len(node.outputs) == 1 and output_kinds:
                    runtime_values[node.id] = output_kinds[node.outputs[0]]
                pending.remove(node)
                progressed = True
            if not progressed:
                for node in pending:
                    composition_errors.append(
                        (node.id, "runtime value kinds cannot resolve one or more input references")
                    )
                break
        return BackendSupportReport(
            self.backend,
            not unsupported and not missing and not composition_errors,
            tuple(unsupported),
            tuple(sorted(set(missing))),
            tuple(sorted(exactness)),
            tuple(sorted(runtime_values.items())),
            tuple(sorted(set(composition_errors))),
        )

    def audit(self) -> Mapping[str, Any]:
        exactness = {}
        for rule in self.rules():
            exactness[rule.exactness] = exactness.get(rule.exactness, 0) + 1
        return {
            "backend": self.backend,
            "rule_count": len(self._rules),
            "base_dependencies": [item.display_name for item in self.base_dependencies],
            "exactness_counts": exactness,
            "rules": [rule.to_dict() for rule in self.rules()],
        }
