#!/usr/bin/env python3
"""Extract source-grounded structural facts from a Python model implementation.

This script intentionally performs syntax-level extraction only. It does not
try to infer a complete Transformer architecture or generate Draw.io XML.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "0.1"

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


def self_attribute(node: ast.AST | None) -> str | None:
    """Return the full self attribute path, such as self.self_attn."""
    name = dotted_name(node)
    if name and name.startswith("self."):
        return name
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


def iter_self_call_assignments(statements: Iterable[ast.stmt]) -> list[dict[str, Any]]:
    """Collect direct self.attr = Constructor(...) assignments in statements."""
    assignments: list[dict[str, Any]] = []
    for statement in statements:
        target: ast.AST | None = None
        value: ast.AST | None = None

        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value

        attribute = self_attribute(target)
        if attribute and isinstance(value, ast.Call):
            assignments.append(
                {
                    "attribute": attribute.removeprefix("self."),
                    "constructor": dotted_name(value.func) or safe_unparse(value.func),
                    "line": getattr(statement, "lineno", None),
                    "source": safe_unparse(statement),
                }
            )
    return assignments


class ArchitectureVisitor(ast.NodeVisitor):
    def __init__(self, source_text: str) -> None:
        self.source_text = source_text
        self.context = Context()
        self.classes: list[dict[str, Any]] = []
        self.module_assignments: list[dict[str, Any]] = []
        self.forward_flows: list[dict[str, Any]] = []
        self.conditions: list[dict[str, Any]] = []
        self.parallelism_hints: list[dict[str, Any]] = []
        self.weight_loading_hints: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self._seen_parallelism: set[tuple[str, int | None, str]] = set()

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
                target = self_attribute(candidate.func)
                if not target:
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

        if node.name == "load_weights":
            self._collect_weight_loading_hints(node)

        self.generic_visit(node)
        self.context = previous

    def visit_Assign(self, node: ast.Assign) -> Any:
        if len(node.targets) == 1:
            self._record_module_assignment(node.targets[0], node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if node.value is not None:
            self._record_module_assignment(node.target, node.value, node)
        self.generic_visit(node)

    def _record_module_assignment(
        self,
        target: ast.AST,
        value: ast.AST,
        statement: ast.stmt,
    ) -> None:
        attribute = self_attribute(target)
        if not attribute or not isinstance(value, ast.Call):
            return

        self.module_assignments.append(
            {
                "owner_class": self.context.class_name,
                "method": self.context.method_name,
                "attribute": attribute.removeprefix("self."),
                "constructor": dotted_name(value.func) or safe_unparse(value.func),
                "line": getattr(statement, "lineno", None),
                "source": safe_unparse(statement),
            }
        )

    def visit_If(self, node: ast.If) -> Any:
        true_assignments = iter_self_call_assignments(node.body)
        false_assignments = iter_self_call_assignments(node.orelse)
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

    visitor = ArchitectureVisitor(source_text)
    visitor.visit(tree)
    visitor.warnings.extend(warnings)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_file": str(source_path.resolve()),
        "classes": visitor.classes,
        "module_assignments": visitor.module_assignments,
        "forward_flows": visitor.forward_flows,
        "conditions": visitor.conditions,
        "parallelism_hints": sorted(
            visitor.parallelism_hints,
            key=lambda item: (item["line"] or 0, item["symbol"]),
        ),
        "weight_loading_hints": sorted(
            visitor.weight_loading_hints,
            key=lambda item: (item["line"] or 0, item["kind"]),
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
