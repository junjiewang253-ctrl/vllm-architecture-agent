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

SCHEMA_VERSION = "0.2"

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
) -> dict[str, Any]:
    local_bindings = local_bindings or {}
    target = dotted_name(node.func) or safe_unparse(node.func) or "<call>"
    result: dict[str, Any] = {
        "type": "call",
        "target": target,
        "line": getattr(node, "lineno", None),
        "source": safe_unparse(node),
    }
    if target in local_bindings:
        result["resolved_collection"] = local_bindings[target]
    return result


def serialize_expression(
    node: ast.AST | None,
    local_bindings: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    if node is None:
        return None
    if isinstance(node, ast.Call):
        return serialize_call(node, local_bindings)
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
) -> list[dict[str, Any]]:
    local_bindings = dict(local_bindings or {})
    result: list[dict[str, Any]] = []

    for statement in statements:
        if isinstance(statement, ast.If):
            result.append(
                {
                    "type": "if",
                    "condition": safe_unparse(statement.test),
                    "line": statement.lineno,
                    "then": serialize_statements(statement.body, local_bindings),
                    "else": serialize_statements(statement.orelse, local_bindings),
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
                    "type": "for",
                    "target": safe_unparse(statement.target),
                    "iterable": safe_unparse(statement.iter),
                    "line": statement.lineno,
                    "body": serialize_statements(statement.body, loop_bindings),
                    "else": serialize_statements(statement.orelse, loop_bindings),
                }
            )
            continue

        if isinstance(statement, ast.Return):
            result.append(
                {
                    "type": "return",
                    "line": statement.lineno,
                    "value": serialize_expression(statement.value, local_bindings),
                }
            )
            continue

        if isinstance(statement, ast.Assign):
            targets: list[str] = []
            for target in statement.targets:
                targets.extend(target_names(target))
            result.append(
                {
                    "type": "assignment",
                    "targets": targets,
                    "line": statement.lineno,
                    "value": serialize_expression(statement.value, local_bindings),
                    "source": safe_unparse(statement),
                }
            )
            continue

        if isinstance(statement, ast.AnnAssign):
            result.append(
                {
                    "type": "assignment",
                    "targets": target_names(statement.target),
                    "line": statement.lineno,
                    "value": serialize_expression(statement.value, local_bindings),
                    "source": safe_unparse(statement),
                }
            )
            continue

        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            result.append(serialize_call(statement.value, local_bindings))
            continue

        if isinstance(statement, (ast.Assert, ast.Raise, ast.Continue, ast.Break)):
            result.append(
                {
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
                    "calls": [serialize_call(call, local_bindings) for call in calls],
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
        "targets": target_names(target),
        "factory": factory,
        "repeat_expression": repeat_expression,
        "layer_constructor": layer_constructor,
        "layer_factory_source": layer_factory_source,
        "line": getattr(statement, "lineno", None),
        "source": safe_unparse(statement),
    }


class ArchitectureVisitor(ast.NodeVisitor):
    def __init__(self, source_text: str, local_module_classes: set[str]) -> None:
        self.source_text = source_text
        self.local_module_classes = local_module_classes
        self.context = Context()
        self.classes: list[dict[str, Any]] = []
        self.class_attributes: list[dict[str, Any]] = []
        self.module_assignments: list[dict[str, Any]] = []
        self.layer_factories: list[dict[str, Any]] = []
        self.forward_flows: list[dict[str, Any]] = []
        self.forward_control_flows: list[dict[str, Any]] = []
        self.conditions: list[dict[str, Any]] = []
        self.parallelism_hints: list[dict[str, Any]] = []
        self.weight_loading_hints: list[dict[str, Any]] = []
        self.weight_mappings: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self._seen_parallelism: set[tuple[str, int | None, str]] = set()
        self._seen_weight_mappings: set[tuple[str, str, str]] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        previous = self.context
        self.context = Context(class_name=node.name)

        methods = [
            {
                "name": item.name,
                "line": item.lineno,
                "end_line": getattr(item, "end_lineno", item.lineno),
                "important": item.name in IMPORTANT_METHODS,
            }
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        self.classes.append(
            {
                "name": node.name,
                "bases": [safe_unparse(base) for base in node.bases],
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
                    "body": serialize_statements(node.body),
                }
            )

        if node.name == "load_weights":
            self._collect_weight_loading_hints(node)
            self._collect_normalized_weight_mappings(node)

        self.generic_visit(node)
        self.context = previous

    def visit_Assign(self, node: ast.Assign) -> Any:
        if len(node.targets) == 1:
            self._record_assignment(node.targets[0], node.value, node)
            factory = extract_make_layers(node.targets[0], node.value, node)
            if factory:
                factory["owner_class"] = self.context.class_name
                factory["method"] = self.context.method_name
                self.layer_factories.append(factory)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if node.value is not None:
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
        record["owner_class"] = self.context.class_name
        record["method"] = self.context.method_name
        self.module_assignments.append(record)

    def visit_If(self, node: ast.If) -> Any:
        true_assignments = iter_self_call_assignments(
            node.body, self.local_module_classes
        )
        false_assignments = iter_self_call_assignments(
            node.orelse, self.local_module_classes
        )
        if true_assignments or false_assignments:
            self.conditions.append(
                {
                    "owner_class": self.context.class_name,
                    "method": self.context.method_name,
                    "condition": safe_unparse(node.test),
                    "line": node.lineno,
                    "true_assignments": true_assignments,
                    "false_assignments": false_assignments,
                }
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
                "symbol": symbol,
                "line": getattr(node, "lineno", None),
                "context": self.context.label(),
                "source": ast.get_source_segment(self.source_text, node)
                or safe_unparse(node),
            }
        )

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

    return {
        "schema_version": SCHEMA_VERSION,
        "source_file": str(source_path.resolve()),
        "classes": visitor.classes,
        "class_attributes": visitor.class_attributes,
        "module_assignments": visitor.module_assignments,
        "layer_factories": visitor.layer_factories,
        "forward_flows": visitor.forward_flows,
        "forward_control_flows": visitor.forward_control_flows,
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
