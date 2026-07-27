#!/usr/bin/env python3
"""Extract source-grounded structural facts from a Python model implementation.

The extractor intentionally stays at the syntax/evidence layer. It records
classes, assignments, control flow, repeated-layer factories, parallelism hints,
and checkpoint mappings without inventing model semantics.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "0.3"

PARALLELISM_SYMBOLS = {
    "get_pp_group",
    "get_ep_group",
    "get_tensor_model_parallel_world_size",
    "get_tensor_model_parallel_rank",
    "QKVParallelLinear",
    "MergedColumnParallelLinear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "ParallelLMHead",
    "FusedMoE",
    "PPMissingLayer",
    "make_layers",
}

IMPORTANT_METHODS = {
    "__init__",
    "forward",
    "compute_logits",
    "load_weights",
    "get_expert_mapping",
    "update_physical_experts_metadata",
    "embed_input_ids",
    "_filter_weights",
}

KNOWN_SUBMODULE_CONSTRUCTORS = {
    "Attention",
    "FusedMoE",
    "GateLinear",
    "HpcRopeNorm",
    "LogitsProcessor",
    "MergedColumnParallelLinear",
    "ParallelLMHead",
    "QKVParallelLinear",
    "RMSNorm",
    "RowParallelLinear",
    "SiluAndMul",
    "VocabParallelEmbedding",
    "ColumnParallelLinear",
    "Embedding",
    "LayerNorm",
    "Linear",
    "ModuleList",
    "nn.ModuleList",
    "nn.Sequential",
}

STATE_CONSTRUCTORS = {
    "bool",
    "dict",
    "enumerate",
    "float",
    "getattr",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "set",
    "torch.get_default_dtype",
    "tuple",
}

PARAMETER_CONSTRUCTORS = {
    "Parameter",
    "nn.Parameter",
    "torch.nn.Parameter",
}

FACTORY_CONSTRUCTORS = {
    "make_layers",
    "make_empty_intermediate_tensors_factory",
}


class Context:
    def __init__(
        self,
        class_name: str | None = None,
        method_name: str | None = None,
    ) -> None:
        self.class_name = class_name
        self.method_name = method_name

    def label(self) -> str:
        if self.class_name and self.method_name:
            return f"{self.class_name}.{self.method_name}"
        return self.class_name or self.method_name or "<module>"


def dotted_name(node: ast.AST | None) -> str | None:
    """Return a dotted name for Name/Attribute nodes, otherwise None."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def safe_unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def literal_value(node: ast.AST) -> Any:
    """Convert simple literal containers to JSON-compatible values."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return safe_unparse(node)


def target_names(node: ast.AST | None) -> list[str]:
    """Return assignment target names, preserving self prefixes."""
    if node is None:
        return []
    if isinstance(node, (ast.Tuple, ast.List)):
        result: list[str] = []
        for element in node.elts:
            result.extend(target_names(element))
        return result
    name = dotted_name(node)
    return [name] if name else [safe_unparse(node) or "<unknown>"]


def self_attribute(node: ast.AST | None) -> str | None:
    name = dotted_name(node)
    if name and name.startswith("self."):
        return name
    return None


def slug(value: str | None, fallback: str = "unknown") -> str:
    text = value or fallback
    rendered = "".join(ch if ch.isalnum() or ch in "._" else "_" for ch in text)
    rendered = "_".join(part for part in rendered.split("_") if part)
    return rendered[:96] or fallback


def make_fact_id(
    owner_class: str | None,
    method: str | None,
    line: int | None,
    kind: str,
    label: str | None,
) -> str:
    qualified = ".".join(part for part in (owner_class, method) if part) or "<module>"
    return f"fact:{qualified}:{line or 0}:{kind}:{slug(label)}"


def collect_references(node: ast.AST | None) -> list[str]:
    """Collect stable name/attribute references from an expression."""
    if node is None:
        return []
    refs: set[str] = set()
    for candidate in ast.walk(node):
        name = dotted_name(candidate)
        if name:
            refs.add(name)
    return sorted(refs)


def collect_local_module_classes(tree: ast.Module) -> set[str]:
    """Find locally declared Module subclasses, including indirect subclasses."""
    class_bases: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_bases[node.name] = [safe_unparse(base) or "" for base in node.bases]

    module_classes = {
        name
        for name, bases in class_bases.items()
        if any(base == "Module" or base.endswith(".Module") for base in bases)
    }
    changed = True
    while changed:
        changed = False
        for name, bases in class_bases.items():
            if name in module_classes:
                continue
            if any(base in module_classes for base in bases):
                module_classes.add(name)
                changed = True
    return module_classes


def classify_assignment(
    constructor: str | None,
    local_module_classes: set[str],
) -> str:
    """Classify a call-valued assignment without pretending every call is a module."""
    if not constructor:
        return "unknown"
    short_name = constructor.split(".")[-1]
    if constructor in PARAMETER_CONSTRUCTORS or short_name == "Parameter":
        return "parameter"
    if constructor in FACTORY_CONSTRUCTORS or constructor.endswith("_factory"):
        return "factory_result"
    if constructor in STATE_CONSTRUCTORS or short_name in STATE_CONSTRUCTORS:
        return "state"
    if constructor.startswith("self."):
        return "state"
    if (
        constructor in KNOWN_SUBMODULE_CONSTRUCTORS
        or short_name in KNOWN_SUBMODULE_CONSTRUCTORS
        or short_name in local_module_classes
    ):
        return "submodule"
    return "unknown"


def assignment_record(
    target: ast.AST,
    value: ast.AST,
    statement: ast.stmt,
    local_module_classes: set[str],
) -> dict[str, Any] | None:
    if not isinstance(value, ast.Call):
        return None
    constructor = dotted_name(value.func) or safe_unparse(value.func)
    targets = target_names(target)
    self_targets = [name for name in targets if name.startswith("self.")]
    if not self_targets:
        return None
    attribute = (
        self_targets[0].removeprefix("self.") if len(self_targets) == 1 else None
    )
    return {
        "targets": [name.removeprefix("self.") for name in self_targets],
        "attribute": attribute,
        "constructor": constructor,
        "assignment_kind": classify_assignment(constructor, local_module_classes),
        "line": getattr(statement, "lineno", None),
        "source": safe_unparse(statement),
    }


def iter_self_call_assignments(
    statements: Iterable[ast.stmt],
    local_module_classes: set[str],
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for statement in statements:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            record = assignment_record(
                statement.targets[0], statement.value, statement, local_module_classes
            )
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            record = assignment_record(
                statement.target, statement.value, statement, local_module_classes
            )
        else:
            record = None
        if record:
            assignments.append(record)
    return assignments


def find_self_collections(node: ast.AST) -> list[str]:
    collections: list[str] = []
    for candidate in ast.walk(node):
        name = self_attribute(candidate)
        if name and name not in collections:
            collections.append(name)
    return collections


def serialize_call(
    node: ast.Call,
    local_bindings: dict[str, str] | None = None,
    *,
    owner_class: str | None = None,
    method: str | None = None,
    assignment_targets: list[str] | None = None,
) -> dict[str, Any]:
    local_bindings = local_bindings or {}
    target = dotted_name(node.func) or safe_unparse(node.func) or "<call>"
    receiver = None
    if isinstance(node.func, ast.Attribute):
        receiver = dotted_name(node.func.value)
    result: dict[str, Any] = {
        "fact_id": make_fact_id(owner_class, method, getattr(node, "lineno", None), "call", target),
        "type": "call",
        "target": target,
        "receiver": receiver,
        "line": getattr(node, "lineno", None),
        "args": [
            {
                "position": index,
                "expression": safe_unparse(argument),
                "references": collect_references(argument),
            }
            for index, argument in enumerate(node.args)
        ],
        "kwargs": {
            keyword.arg or "**": {
                "expression": safe_unparse(keyword.value),
                "references": collect_references(keyword.value),
            }
            for keyword in node.keywords
        },
        "source": safe_unparse(node),
    }
    if assignment_targets:
        result["assignment_targets"] = assignment_targets
    if target in local_bindings:
        result["resolved_collection"] = local_bindings[target]
    return result


def serialize_expression(
    node: ast.AST | None,
    local_bindings: dict[str, str] | None = None,
    *,
    owner_class: str | None = None,
    method: str | None = None,
    assignment_targets: list[str] | None = None,
) -> dict[str, Any] | None:
    if node is None:
        return None
    if isinstance(node, ast.Call):
        return serialize_call(
            node,
            local_bindings,
            owner_class=owner_class,
            method=method,
            assignment_targets=assignment_targets,
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return {
            "type": "add",
            "line": getattr(node, "lineno", None),
            "left": safe_unparse(node.left),
            "right": safe_unparse(node.right),
            "source": safe_unparse(node),
        }
    return {
        "type": "expression",
        "line": getattr(node, "lineno", None),
        "source": safe_unparse(node),
    }


def serialize_statements(
    statements: Iterable[ast.stmt],
    local_bindings: dict[str, str] | None = None,
    *,
    owner_class: str | None = None,
    method: str | None = None,
) -> list[dict[str, Any]]:
    local_bindings = dict(local_bindings or {})
    result: list[dict[str, Any]] = []

    for statement in statements:
        if isinstance(statement, ast.If):
            result.append(
                {
                    "fact_id": make_fact_id(owner_class, method, statement.lineno, "branch", safe_unparse(statement.test)),
                    "type": "if",
                    "condition": safe_unparse(statement.test),
                    "condition_references": collect_references(statement.test),
                    "line": statement.lineno,
                    "then": serialize_statements(statement.body, local_bindings, owner_class=owner_class, method=method),
                    "else": serialize_statements(statement.orelse, local_bindings, owner_class=owner_class, method=method),
                }
            )
            continue

        if isinstance(statement, (ast.For, ast.AsyncFor)):
            loop_targets = target_names(statement.target)
            collections = find_self_collections(statement.iter)
            loop_bindings = dict(local_bindings)
            if collections:
                collection = collections[0]
                for name in loop_targets:
                    if name and not name.startswith("self."):
                        loop_bindings[name] = collection
            result.append(
                {
                    "fact_id": make_fact_id(owner_class, method, statement.lineno, "for", safe_unparse(statement.iter)),
                    "type": "for",
                    "target": safe_unparse(statement.target),
                    "iterable": safe_unparse(statement.iter),
                    "line": statement.lineno,
                    "body": serialize_statements(statement.body, loop_bindings, owner_class=owner_class, method=method),
                    "else": serialize_statements(statement.orelse, loop_bindings, owner_class=owner_class, method=method),
                }
            )
            continue

        if isinstance(statement, ast.Return):
            result.append(
                {
                    "fact_id": make_fact_id(owner_class, method, statement.lineno, "return", safe_unparse(statement.value)),
                    "type": "return",
                    "line": statement.lineno,
                    "value": serialize_expression(statement.value, local_bindings, owner_class=owner_class, method=method),
                }
            )
            continue

        if isinstance(statement, ast.Assign):
            targets: list[str] = []
            for target in statement.targets:
                targets.extend(target_names(target))
            result.append(
                {
                    "fact_id": make_fact_id(owner_class, method, statement.lineno, "assignment", ",".join(targets)),
                    "type": "assignment",
                    "targets": targets,
                    "line": statement.lineno,
                    "value": serialize_expression(
                        statement.value,
                        local_bindings,
                        owner_class=owner_class,
                        method=method,
                        assignment_targets=targets,
                    ),
                    "source": safe_unparse(statement),
                }
            )
            continue

        if isinstance(statement, ast.AnnAssign):
            result.append(
                {
                    "fact_id": make_fact_id(owner_class, method, statement.lineno, "assignment", ",".join(target_names(statement.target))),
                    "type": "assignment",
                    "targets": target_names(statement.target),
                    "line": statement.lineno,
                    "value": serialize_expression(
                        statement.value,
                        local_bindings,
                        owner_class=owner_class,
                        method=method,
                        assignment_targets=target_names(statement.target),
                    ),
                    "source": safe_unparse(statement),
                }
            )
            continue

        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            result.append(serialize_call(statement.value, local_bindings, owner_class=owner_class, method=method))
            continue

        if isinstance(statement, (ast.Assert, ast.Raise, ast.Continue, ast.Break)):
            result.append(
                {
                    "fact_id": make_fact_id(owner_class, method, getattr(statement, "lineno", None), statement.__class__.__name__.lower(), safe_unparse(statement)),
                    "type": statement.__class__.__name__.lower(),
                    "line": getattr(statement, "lineno", None),
                    "source": safe_unparse(statement),
                }
            )
            continue

        # Keep unsupported control statements visible without expanding every
        # Python syntax form into the stable schema.
        calls = [item for item in ast.walk(statement) if isinstance(item, ast.Call)]
        if calls:
            result.append(
                {
                    "type": "statement",
                    "line": getattr(statement, "lineno", None),
                    "source": safe_unparse(statement),
                    "calls": [serialize_call(call, local_bindings, owner_class=owner_class, method=method) for call in calls],
                }
            )

    return result


def extract_make_layers(
    target: ast.AST,
    value: ast.AST,
    statement: ast.stmt,
) -> dict[str, Any] | None:
    if not isinstance(value, ast.Call):
        return None
    factory = dotted_name(value.func) or safe_unparse(value.func)
    if factory != "make_layers":
        return None

    repeat_expression = safe_unparse(value.args[0]) if value.args else None
    layer_constructor: str | None = None
    layer_factory_source: str | None = None
    if len(value.args) >= 2 and isinstance(value.args[1], ast.Lambda):
        layer_factory_source = safe_unparse(value.args[1])
        if isinstance(value.args[1].body, ast.Call):
            layer_constructor = (
                dotted_name(value.args[1].body.func)
                or safe_unparse(value.args[1].body.func)
            )

    return {
        "fact_id": make_fact_id(None, None, getattr(statement, "lineno", None), "call", factory),
        "targets": target_names(target),
        "factory": factory,
        "repeat_expression": repeat_expression,
        "layer_constructor": layer_constructor,
        "layer_factory_source": layer_factory_source,
        "line": getattr(statement, "lineno", None),
        "source": safe_unparse(statement),
    }


def assignment_value_kind(value: ast.AST) -> str:
    if isinstance(value, ast.Call):
        return "call"
    if isinstance(value, ast.Attribute):
        return "attribute"
    if isinstance(value, ast.Constant):
        return "constant"
    if isinstance(value, ast.BinOp):
        return "binary_operation"
    if isinstance(value, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        return "collection"
    if isinstance(value, ast.Subscript):
        return "subscript"
    if isinstance(value, ast.Name):
        return "name"
    return "unknown"


def config_accesses_from_expression(
    value: ast.AST,
    targets: list[str],
    *,
    owner_class: str | None,
    method: str | None,
    line: int | None,
) -> list[dict[str, Any]]:
    accesses: list[dict[str, Any]] = []
    target = targets[0] if targets else None
    for node in ast.walk(value):
        name = dotted_name(node)
        if not name or "." not in name:
            continue
        parts = name.split(".")
        root = parts[0]
        if root not in {"config", "vllm_config", "parallel_config", "eplb_config", "self.config"}:
            continue
        accesses.append(
            {
                "fact_id": make_fact_id(owner_class, method, line, "config_access", name),
                "root": root,
                "path": ".".join(parts[1:]),
                "owner_class": owner_class,
                "method": method,
                "line": line,
                "target": target,
                "source": name,
            }
        )
    unique: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for access in accesses:
        unique[(access["root"], access["path"], access["target"])] = access
    return list(unique.values())


class ArchitectureVisitor(ast.NodeVisitor):
    def __init__(self, source_text: str, local_module_classes: set[str]) -> None:
        self.source_text = source_text
        self.local_module_classes = local_module_classes
        self.context = Context()
        self.imports: list[dict[str, Any]] = []
        self.classes: list[dict[str, Any]] = []
        self.class_attributes: list[dict[str, Any]] = []
        self.module_assignments: list[dict[str, Any]] = []
        self.layer_factories: list[dict[str, Any]] = []
        self.forward_flows: list[dict[str, Any]] = []
        self.forward_control_flows: list[dict[str, Any]] = []
        self.methods: list[dict[str, Any]] = []
        self.method_control_flows: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.assignments: list[dict[str, Any]] = []
        self.branches: list[dict[str, Any]] = []
        self.returns: list[dict[str, Any]] = []
        self.config_accesses: list[dict[str, Any]] = []
        self.parallelism_facts: list[dict[str, Any]] = []
        self.weight_loading_flows: list[dict[str, Any]] = []
        self.semantic_facts: list[dict[str, Any]] = []
        self.conditions: list[dict[str, Any]] = []
        self.parallelism_hints: list[dict[str, Any]] = []
        self.weight_loading_hints: list[dict[str, Any]] = []
        self.weight_mappings: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self._seen_parallelism: set[tuple[str, int | None, str]] = set()
        self._seen_parallelism_facts: set[tuple[str, str, int | None]] = set()
        self._seen_weight_mappings: set[tuple[str, str, str]] = set()

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.imports.append(
                {
                    "module": None,
                    "name": alias.name,
                    "asname": alias.asname,
                    "fact_id": make_fact_id(None, None, node.lineno, "import", alias.asname or alias.name),
                    "line": node.lineno,
                    "source": safe_unparse(node),
                }
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            self.imports.append(
                {
                    "module": module,
                    "name": alias.name,
                    "asname": alias.asname,
                    "fact_id": make_fact_id(None, None, node.lineno, "import", alias.asname or alias.name),
                    "line": node.lineno,
                    "source": safe_unparse(node),
                }
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        previous = self.context
        self.context = Context(class_name=node.name)

        methods = [
            {
                "fact_id": make_fact_id(node.name, item.name, item.lineno, "method", item.name),
                "name": item.name,
                "line": item.lineno,
                "end_line": getattr(item, "end_lineno", item.lineno),
                "important": item.name in IMPORTANT_METHODS,
            }
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.methods.extend(
            {
                **method,
                "owner_class": node.name,
                "qualified_name": f"{node.name}.{method['name']}",
                "parameters": [
                    arg.arg for arg in item.args.args
                ],
            }
            for method, item in zip(
                methods,
                [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))],
            )
        )

        self.classes.append(
            {
                "name": node.name,
                "fact_id": make_fact_id(node.name, None, node.lineno, "class", node.name),
                "bases": [safe_unparse(base) for base in node.bases],
                "decorators": [
                    {
                        "name": dotted_name(decorator) or safe_unparse(decorator),
                        "line": getattr(decorator, "lineno", node.lineno),
                        "source": safe_unparse(decorator),
                    }
                    for decorator in node.decorator_list
                ],
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "methods": methods,
            }
        )

        for item in node.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                target = dotted_name(item.targets[0])
                if target and not target.startswith("self.") and isinstance(
                    item.value, (ast.Dict, ast.List, ast.Tuple, ast.Constant)
                ):
                    value = literal_value(item.value)
                    self.class_attributes.append(
                        {
                            "fact_id": make_fact_id(node.name, None, item.lineno, "class_attribute", target),
                            "owner_class": node.name,
                            "name": target,
                            "value": value,
                            "line": item.lineno,
                            "source": safe_unparse(item),
                        }
                    )
                    if target == "packed_modules_mapping" and isinstance(value, dict):
                        self._add_packed_module_mappings(value, item.lineno)

        self.generic_visit(node)
        self.context = previous

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        previous = self.context
        self.context = Context(previous.class_name, node.name)
        qualified_name = f"{previous.class_name}.{node.name}" if previous.class_name else node.name
        body = serialize_statements(
            node.body,
            owner_class=previous.class_name,
            method=node.name,
        )
        if previous.class_name and (node.name in IMPORTANT_METHODS or node.name.startswith("_")):
            self.method_control_flows.append(
                {
                    "fact_id": make_fact_id(previous.class_name, node.name, node.lineno, "method_control_flow", node.name),
                    "owner_class": previous.class_name,
                    "method": node.name,
                    "qualified_name": qualified_name,
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "parameters": [arg.arg for arg in node.args.args],
                    "body": body,
                }
            )
        self._collect_flat_facts(body, previous.class_name, node.name)

        if node.name == "forward" and previous.class_name:
            calls: list[dict[str, Any]] = []
            for candidate in ast.walk(node):
                if not isinstance(candidate, ast.Call):
                    continue
                target = dotted_name(candidate.func)
                if not target or not target.startswith("self."):
                    continue
                calls.append(
                    {
                        "fact_id": make_fact_id(previous.class_name, node.name, getattr(candidate, "lineno", None), "call", target),
                        "target": target,
                        "line": getattr(candidate, "lineno", None),
                        "source": safe_unparse(candidate),
                    }
                )
            calls.sort(key=lambda item: (item["line"] or 0, item["target"]))
            for index, call in enumerate(calls, start=1):
                call["order"] = index
            self.forward_flows.append(
                {
                    "class": previous.class_name,
                    "method": node.name,
                    "line": node.lineno,
                    "calls": calls,
                }
            )
            self.forward_control_flows.append(
                {
                    "class": previous.class_name,
                    "method": node.name,
                    "line": node.lineno,
                    "body": body,
                }
            )

        if node.name == "load_weights":
            self._collect_weight_loading_hints(node)
            self._collect_normalized_weight_mappings(node)
            self._collect_weight_loading_flow(node, body, previous.class_name)

        self.generic_visit(node)
        self.context = previous

    def visit_Assign(self, node: ast.Assign) -> Any:
        if len(node.targets) == 1:
            self._record_generic_assignment(node.targets, node.value, node)
            self._record_assignment(node.targets[0], node.value, node)
            factory = extract_make_layers(node.targets[0], node.value, node)
            if factory:
                factory["owner_class"] = self.context.class_name
                factory["method"] = self.context.method_name
                self.layer_factories.append(factory)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if node.value is not None:
            self._record_generic_assignment([node.target], node.value, node)
            self._record_assignment(node.target, node.value, node)
        self.generic_visit(node)

    def _record_assignment(
        self,
        target: ast.AST,
        value: ast.AST,
        statement: ast.stmt,
    ) -> None:
        record = assignment_record(
            target, value, statement, self.local_module_classes
        )
        if not record:
            return
        record["fact_id"] = make_fact_id(self.context.class_name, self.context.method_name, getattr(statement, "lineno", None), "assignment", ",".join(record.get("targets", [])))
        record["owner_class"] = self.context.class_name
        record["method"] = self.context.method_name
        self.module_assignments.append(record)

    def _record_generic_assignment(
        self,
        targets: list[ast.AST],
        value: ast.AST,
        statement: ast.stmt,
    ) -> None:
        names: list[str] = []
        for target in targets:
            names.extend(target_names(target))
        line = getattr(statement, "lineno", None)
        call_fact_id = None
        if isinstance(value, ast.Call):
            call_fact_id = make_fact_id(
                self.context.class_name,
                self.context.method_name,
                line,
                "call",
                dotted_name(value.func) or safe_unparse(value.func),
            )
        record = {
            "fact_id": make_fact_id(self.context.class_name, self.context.method_name, line, "assignment", ",".join(names)),
            "owner_class": self.context.class_name,
            "method": self.context.method_name,
            "targets": names,
            "value_kind": assignment_value_kind(value),
            "expression": safe_unparse(value),
            "references": collect_references(value),
            "call_fact_id": call_fact_id,
            "line": line,
            "source": safe_unparse(statement),
        }
        self.assignments.append(record)
        self.config_accesses.extend(
            config_accesses_from_expression(
                value,
                names,
                owner_class=self.context.class_name,
                method=self.context.method_name,
                line=line,
            )
        )

    def _collect_flat_facts(
        self,
        body: list[dict[str, Any]],
        owner_class: str | None,
        method: str | None,
    ) -> None:
        for item in body:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "return":
                self.returns.append(
                    {
                        "fact_id": item.get("fact_id"),
                        "owner_class": owner_class,
                        "method": method,
                        "line": item.get("line"),
                        "value": item.get("value"),
                    }
                )
            if item_type == "if":
                then_ids = [fact.get("fact_id") for fact in item.get("then", []) if isinstance(fact, dict) and fact.get("fact_id")]
                else_ids = [fact.get("fact_id") for fact in item.get("else", []) if isinstance(fact, dict) and fact.get("fact_id")]
                self.branches.append(
                    {
                        "fact_id": item.get("fact_id"),
                        "owner_class": owner_class,
                        "method": method,
                        "condition": item.get("condition"),
                        "condition_references": item.get("condition_references", []),
                        "line": item.get("line"),
                        "then_fact_ids": then_ids,
                        "else_fact_ids": else_ids,
                    }
                )
                self._collect_flat_facts(item.get("then", []), owner_class, method)
                self._collect_flat_facts(item.get("else", []), owner_class, method)
            elif item_type == "for":
                self._collect_flat_facts(item.get("body", []), owner_class, method)
                self._collect_flat_facts(item.get("else", []), owner_class, method)
            elif item_type == "assignment":
                value = item.get("value")
                if isinstance(value, dict) and value.get("type") == "call":
                    self.calls.append(value)
            elif item_type == "call":
                self.calls.append(item)
            elif item_type == "statement":
                for call in item.get("calls", []):
                    if isinstance(call, dict):
                        self.calls.append(call)

    def _add_parallelism_fact(
        self,
        category: str,
        symbol: str,
        line: int | None,
        fact_ids: list[str],
        summary: str,
    ) -> None:
        key = (category, symbol, line)
        if key in self._seen_parallelism_facts:
            return
        self._seen_parallelism_facts.add(key)
        self.parallelism_facts.append(
            {
                "fact_id": make_fact_id(self.context.class_name, self.context.method_name, line, "parallelism", f"{category}:{symbol}"),
                "category": category,
                "symbol": symbol,
                "owner_class": self.context.class_name,
                "method": self.context.method_name,
                "line": line,
                "evidence_fact_ids": fact_ids,
                "summary": summary,
            }
        )

    def _collect_weight_loading_flow(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        body: list[dict[str, Any]],
        owner_class: str | None,
    ) -> None:
        stages: list[dict[str, Any]] = []
        flat = list(self._iter_control(body))
        for item in flat:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or item.get("condition") or item.get("target") or item.get("type"))
            line = item.get("line")
            fact_id = item.get("fact_id")
            if not isinstance(fact_id, str):
                continue
            stage_kind = self._weight_stage_kind(source)
            if stage_kind is None:
                continue
            stages.append(
                {
                    "fact_id": f"stage:{fact_id}",
                    "stage_kind": stage_kind,
                    "line": line,
                    "condition": item.get("condition") if item.get("type") == "if" else None,
                    "input_references": item.get("condition_references", []),
                    "output_references": item.get("targets", []),
                    "call_fact_id": item.get("value", {}).get("fact_id") if isinstance(item.get("value"), dict) else None,
                    "evidence_fact_ids": [fact_id],
                    "summary": source,
                }
            )
        self.weight_loading_flows.append(
            {
                "entrypoint": f"{owner_class}.{node.name}" if owner_class else node.name,
                "owner_class": owner_class,
                "method": node.name,
                "line": node.lineno,
                "stages": stages,
            }
        )

    def _iter_control(self, body: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for item in body:
            yield item
            if item.get("type") == "if":
                yield from self._iter_control(item.get("then", []))
                yield from self._iter_control(item.get("else", []))
            elif item.get("type") == "for":
                yield from self._iter_control(item.get("body", []))
                yield from self._iter_control(item.get("else", []))

    def _weight_stage_kind(self, source: str) -> str | None:
        if "weight_loader" in source or "AutoWeightsLoader" in source or "default_weight_loader" in source:
            return "loader"
        if "_filter_weights" in source or "spec_layer" in source or "tie_word_embeddings" in source:
            return "filter"
        if "maybe_remap_kv_scale_name" in source or "replace(" in source:
            return "rename"
        if "stacked_params_mapping" in source or "expert_params_mapping" in source or "mapping" in source:
            return "mapping"
        if "params_dict" in source or "named_parameters" in source:
            return "lookup"
        if "loaded_params.add" in source:
            return "collect"
        if "return" in source or "loader.load_weights" in source:
            return "return"
        if "continue" in source or "skip" in source.lower() or "is_pp_missing_parameter" in source:
            return "skip"
        return None

    def visit_If(self, node: ast.If) -> Any:
        branch_record = {
            "fact_id": make_fact_id(self.context.class_name, self.context.method_name, node.lineno, "branch", safe_unparse(node.test)),
            "owner_class": self.context.class_name,
            "method": self.context.method_name,
            "condition": safe_unparse(node.test),
            "condition_references": collect_references(node.test),
            "line": node.lineno,
            "then_fact_ids": [],
            "else_fact_ids": [],
        }
        self.branches.append(branch_record)
        true_assignments = iter_self_call_assignments(
            node.body, self.local_module_classes
        )
        false_assignments = iter_self_call_assignments(
            node.orelse, self.local_module_classes
        )
        if true_assignments or false_assignments:
            self.conditions.append(
                {
                    "fact_id": branch_record["fact_id"],
                    "owner_class": self.context.class_name,
                    "method": self.context.method_name,
                    "condition": safe_unparse(node.test),
                    "line": node.lineno,
                    "true_assignments": true_assignments,
                    "false_assignments": false_assignments,
                }
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        self.calls.append(
            serialize_call(
                node,
                owner_class=self.context.class_name,
                method=self.context.method_name,
            )
        )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        self._record_parallelism_hint(node.id, node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        self._record_parallelism_hint(node.attr, node)
        self.generic_visit(node)

    def _record_parallelism_hint(self, symbol: str, node: ast.AST) -> None:
        if symbol not in PARALLELISM_SYMBOLS:
            return
        key = (symbol, getattr(node, "lineno", None), self.context.label())
        if key in self._seen_parallelism:
            return
        self._seen_parallelism.add(key)
        self.parallelism_hints.append(
            {
                "fact_id": make_fact_id(self.context.class_name, self.context.method_name, getattr(node, "lineno", None), "parallelism_hint", symbol),
                "symbol": symbol,
                "line": getattr(node, "lineno", None),
                "context": self.context.label(),
                "source": ast.get_source_segment(self.source_text, node)
                or safe_unparse(node),
            }
        )
        category = "tensor_parallel"
        if symbol in {"get_pp_group", "make_layers", "PPMissingLayer"}:
            category = "pipeline_parallel"
        elif symbol in {"get_ep_group", "FusedMoE"}:
            category = "expert_parallel"
        fact_id = make_fact_id(self.context.class_name, self.context.method_name, getattr(node, "lineno", None), "parallelism_hint", symbol)
        self._add_parallelism_fact(category, symbol, getattr(node, "lineno", None), [fact_id], f"{category}: {symbol}")

    def _collect_weight_loading_hints(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for candidate in ast.walk(node):
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                text = candidate.value
                if any(
                    token in text
                    for token in (
                        "proj",
                        "qkv",
                        "gate",
                        "expert",
                        "weight",
                        "scale",
                    )
                ):
                    self.weight_loading_hints.append(
                        {
                            "fact_id": make_fact_id(self.context.class_name, self.context.method_name, getattr(candidate, "lineno", None), "weight_hint", text),
                            "kind": "string",
                            "value": text,
                            "line": getattr(candidate, "lineno", None),
                            "context": self.context.label(),
                        }
                    )
            elif isinstance(candidate, (ast.Tuple, ast.List, ast.Dict)):
                value = literal_value(candidate)
                rendered = json.dumps(value, ensure_ascii=False, default=str)
                if any(token in rendered for token in ("proj", "expert", "weight")):
                    self.weight_loading_hints.append(
                        {
                            "fact_id": make_fact_id(self.context.class_name, self.context.method_name, getattr(candidate, "lineno", None), "weight_hint", rendered),
                            "kind": candidate.__class__.__name__.lower(),
                            "value": value,
                            "line": getattr(candidate, "lineno", None),
                            "context": self.context.label(),
                        }
                    )

    def _collect_normalized_weight_mappings(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for candidate in ast.walk(node):
            if not isinstance(candidate, ast.Assign) or len(candidate.targets) != 1:
                continue
            target = dotted_name(candidate.targets[0])
            if target not in {"stacked_params_mapping", "packed_modules_mapping"}:
                continue
            value = literal_value(candidate.value)
            if target == "stacked_params_mapping" and isinstance(value, list):
                for item in value:
                    if not isinstance(item, (list, tuple)) or len(item) < 3:
                        continue
                    self._add_weight_mapping(
                        source=str(item[1]),
                        target=str(item[0]),
                        shard=item[2],
                        line=getattr(candidate, "lineno", None),
                        context=self.context.label(),
                        mapping_kind="stacked_parameter",
                    )

    def _add_packed_module_mappings(self, value: dict[Any, Any], line: int) -> None:
        for target, sources in value.items():
            if not isinstance(sources, (list, tuple)):
                continue
            for source in sources:
                self._add_weight_mapping(
                    source=str(source),
                    target=str(target),
                    shard=None,
                    line=line,
                    context=self.context.label(),
                    mapping_kind="packed_module",
                )

    def _add_weight_mapping(
        self,
        *,
        source: str,
        target: str,
        shard: Any,
        line: int | None,
        context: str,
        mapping_kind: str,
    ) -> None:
        key = (source, target, json.dumps(shard, ensure_ascii=False, default=str))
        if key in self._seen_weight_mappings:
            return
        self._seen_weight_mappings.add(key)
        self.weight_mappings.append(
            {
                "fact_id": make_fact_id(
                    self.context.class_name,
                    self.context.method_name,
                    line,
                    "weight_mapping",
                    f"{mapping_kind}:{source}->{target}",
                ),
                "source": source,
                "target": target,
                "shard": shard,
                "mapping_kind": mapping_kind,
                "line": line,
                "context": context,
            }
        )


def extract_architecture(source_path: Path) -> dict[str, Any]:
    warnings: list[str] = []
    try:
        source_text = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source_text = source_path.read_text(encoding="utf-8-sig")
        warnings.append("Input file required UTF-8 BOM handling.")

    try:
        tree = ast.parse(source_text, filename=str(source_path))
    except SyntaxError as exc:
        raise ValueError(
            f"Unable to parse {source_path}: line {exc.lineno}: {exc.msg}"
        ) from exc

    visitor = ArchitectureVisitor(source_text, collect_local_module_classes(tree))
    visitor.visit(tree)
    visitor.warnings.extend(warnings)
    semantic_facts = build_semantic_facts(visitor)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_file": str(source_path.resolve()),
        "imports": sorted(
            visitor.imports,
            key=lambda item: (item["line"] or 0, item["module"] or "", item["name"]),
        ),
        "classes": visitor.classes,
        "class_attributes": visitor.class_attributes,
        "module_assignments": visitor.module_assignments,
        "layer_factories": visitor.layer_factories,
        "forward_flows": visitor.forward_flows,
        "forward_control_flows": visitor.forward_control_flows,
        "methods": sorted(visitor.methods, key=lambda item: (item.get("line") or 0, item.get("qualified_name") or "")),
        "method_control_flows": sorted(visitor.method_control_flows, key=lambda item: (item.get("line") or 0, item.get("qualified_name") or "")),
        "calls": sorted(visitor.calls, key=lambda item: (item.get("line") or 0, item.get("target") or "")),
        "assignments": sorted(visitor.assignments, key=lambda item: (item.get("line") or 0, ",".join(item.get("targets", [])))),
        "branches": sorted(visitor.branches, key=lambda item: (item.get("line") or 0, item.get("condition") or "")),
        "returns": sorted(visitor.returns, key=lambda item: (item.get("line") or 0, item.get("method") or "")),
        "config_accesses": sorted(visitor.config_accesses, key=lambda item: (item.get("line") or 0, item.get("source") or "")),
        "parallelism_facts": sorted(visitor.parallelism_facts, key=lambda item: (item.get("line") or 0, item.get("category") or "", item.get("symbol") or "")),
        "weight_loading_flows": visitor.weight_loading_flows,
        "semantic_facts": semantic_facts,
        "conditions": visitor.conditions,
        "parallelism_hints": sorted(
            visitor.parallelism_hints,
            key=lambda item: (item["line"] or 0, item["symbol"]),
        ),
        "weight_loading_hints": sorted(
            visitor.weight_loading_hints,
            key=lambda item: (item["line"] or 0, item["kind"]),
        ),
        "weight_mappings": sorted(
            visitor.weight_mappings,
            key=lambda item: (
                item["line"] or 0,
                item["target"],
                item["source"],
            ),
        ),
        "warnings": visitor.warnings,
    }


def _source_location(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "owner_class": record.get("owner_class"),
        "method": record.get("method"),
        "line": record.get("line"),
    }


def build_semantic_facts(visitor: ArchitectureVisitor) -> list[dict[str, Any]]:
    """Build a conservative source fact inventory for semantic coverage."""
    facts: dict[str, dict[str, Any]] = {}

    def add(record: dict[str, Any], category: str, summary: str, relevance: str = "required") -> None:
        fact_id = record.get("fact_id")
        if not isinstance(fact_id, str):
            return
        facts[fact_id] = {
            "fact_id": fact_id,
            "category": category,
            "relevance": relevance,
            "source_location": _source_location(record),
            "summary": summary,
        }

    for item in visitor.module_assignments:
        kind = "construction" if item.get("method") == "__init__" else "model_structure"
        add(item, kind, str(item.get("source") or item.get("constructor")))
    for item in visitor.assignments:
        text = str(item.get("source") or item.get("expression"))
        category = "configuration" if any(ref.startswith(("config", "vllm_config", "parallel_config")) for ref in item.get("references", [])) else "runtime_flow"
        if item.get("method") == "load_weights":
            category = "checkpoint_loading"
        add(item, category, text)
    for item in visitor.calls:
        target = str(item.get("target") or "")
        category = "runtime_flow"
        if item.get("method") == "__init__":
            category = "construction"
        if item.get("method") == "load_weights":
            category = "checkpoint_loading"
        if target in {"AutoWeightsLoader", "Attention", "FusedMoE"} or target.endswith(("Attention", "FusedMoE")):
            category = "external_boundary" if item.get("method") != "__init__" else "construction"
        add(item, category, str(item.get("source") or target))
    for item in visitor.branches:
        category = "checkpoint_loading" if item.get("method") == "load_weights" else "runtime_flow"
        add(item, category, str(item.get("condition")))
    for item in visitor.config_accesses:
        add(item, "configuration", f"{item.get('root')}.{item.get('path')} -> {item.get('target')}")
    for item in visitor.parallelism_facts:
        add(item, str(item.get("category")), str(item.get("summary")))
    for flow in visitor.weight_loading_flows:
        for stage in flow.get("stages", []):
            add(
                {
                    "fact_id": stage.get("fact_id"),
                    "owner_class": flow.get("owner_class"),
                    "method": flow.get("method"),
                    "line": stage.get("line"),
                },
                "checkpoint_loading",
                str(stage.get("summary")),
            )
    return sorted(facts.values(), key=lambda item: (item["source_location"].get("line") or 0, item["fact_id"]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract source-grounded architecture facts from Python code."
    )
    parser.add_argument("input", type=Path, help="Input Python source file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSON path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"error: input file does not exist: {args.input}", file=sys.stderr)
        return 2
    if not args.input.is_file() or args.input.suffix.lower() != ".py":
        print(f"error: input must be a Python file: {args.input}", file=sys.stderr)
        return 2

    try:
        result = extract_architecture(args.input)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote architecture analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
