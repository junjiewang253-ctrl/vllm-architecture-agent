"""Collect v2.1 complete Source Context for Agent-native diagram design."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from resolve_model_target import resolve_model_target  # noqa: E402
from vllm_architecture_agent.patterns import (  # noqa: E402
    categorize_constructor,
    classify_model_source,
    detect_capabilities,
)

CORE_METHOD_NAMES = {
    "__init__",
    "forward",
    "compute_logits",
    "load_weights",
    "get_expert_mapping",
    "update_physical_experts_metadata",
    "pooling",
    "embed_input_ids",
    "encode",
    "encoder",
}

SUPPORTING_NAME_PARTS = (
    "prepare",
    "reshape",
    "metadata",
    "filter",
    "remap",
    "mapping",
    "loader",
    "input",
)

HIGH_BRANCH_TERMS = (
    "rank",
    "pp",
    "moe",
    "expert",
    "hpc",
    "attention",
    "load",
    "weight",
    "mapping",
    "multimodal",
    "shared",
    "first_k",
)

MEDIUM_BRANCH_TERMS = (
    "none",
    "norm",
    "config",
    "metadata",
    "embed",
    "quant",
    "scale",
)

MAPPING_KIND_TERMS: tuple[tuple[str, str], ...] = (
    ("packed_modules_mapping", "packed_mapping"),
    ("stacked_params_mapping", "stacked_mapping"),
    ("expert", "expert_mapping"),
    ("replace", "rename"),
    ("remap", "rename"),
    ("filter", "filter"),
    ("skip", "skip"),
    ("continue", "skip"),
    ("AutoWeightsLoader", "automatic_loader"),
    ("default_weight_loader", "default_loader"),
    ("weight_loader", "loader_dispatch"),
)


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _range_sha(lines: list[str], start: int, end: int) -> str:
    snippet = "\n".join(lines[start - 1 : end])
    return hashlib.sha256(snippet.encode("utf-8")).hexdigest()


def _end_line(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 0)))


def _decorators(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [_unparse(item) for item in getattr(node, "decorator_list", [])]


def _call_target(node: ast.Call) -> str:
    return _unparse(node.func)


def _self_attr_name(target: ast.AST) -> str | None:
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return target.attr
    return None


def _class_categories(name: str, bases: list[str], source: str) -> list[str]:
    blob = f"{name} {' '.join(bases)} {source}"
    categories: list[str] = []
    for term, category in (
        ("Attention", "attention"),
        ("MoE", "moe"),
        ("Expert", "moe"),
        ("FeedForward", "ffn"),
        ("MLP", "ffn"),
        ("Decoder", "decoder"),
        ("Model", "model"),
        ("ForCausalLM", "wrapper"),
        ("Embedding", "embedding"),
        ("Pool", "pooling"),
        ("Vision", "multimodal"),
        ("Mamba", "state_space"),
        ("SSM", "state_space"),
    ):
        if term in blob and category not in categories:
            categories.append(category)
    return categories or ["unknown"]


def _branch_relevance(condition: str) -> str:
    lowered = condition.lower()
    if any(term in lowered for term in HIGH_BRANCH_TERMS):
        return "high"
    if any(term in lowered for term in MEDIUM_BRANCH_TERMS):
        return "medium"
    return "low"


def _mapping_kind(text: str) -> str:
    lowered = text.lower()
    for term, kind in MAPPING_KIND_TERMS:
        if term.lower() in lowered:
            return kind
    return "unknown"


def _mapping_relevance(kind: str) -> str:
    return "high" if kind != "unknown" else "medium"


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    params = [arg.arg for arg in node.args.posonlyargs + node.args.args]
    if node.args.vararg:
        params.append("*" + node.args.vararg.arg)
    params.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        params.append("**" + node.args.kwarg.arg)
    return params


def _importance(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    calls: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    name = node.name
    lowered = name.lower()
    if name in CORE_METHOD_NAMES:
        reasons.append(f"{name} entrypoint")
    if name == "__init__" and assignments:
        reasons.append("constructs model components")
    if len([item for item in branches if item["architecture_relevance_candidate"] in {"high", "medium"}]) >= 2:
        reasons.append("contains multiple architecture-relevant branches")
    if any("self." in call["target"] for call in calls):
        reasons.append("calls model components")
    if "weight" in lowered or "expert" in lowered:
        reasons.append("handles weights or experts")
    if reasons:
        return "core", reasons
    if any(part in lowered for part in SUPPORTING_NAME_PARTS):
        return "supporting", ["name suggests supporting architecture helper"]
    if _end_line(node) - node.lineno <= 3:
        return "trivial", ["short helper or property-like method"]
    return "supporting", ["non-trivial method"]


class ScopeAwareBodyVisitor(ast.NodeVisitor):
    """Inspect one function/method body without descending into nested scopes."""

    def __init__(self, owner_id: str, owner_qualified_name: str, lines: list[str]) -> None:
        self.owner_id = owner_id
        self.owner_qualified_name = owner_qualified_name
        self.lines = lines
        self.calls: list[dict[str, Any]] = []
        self.branches: list[dict[str, Any]] = []
        self.loops: list[dict[str, Any]] = []
        self.returns: list[dict[str, Any]] = []
        self.module_assignments: list[dict[str, Any]] = []
        self.weight_mappings: list[dict[str, Any]] = []
        self._call_ordinal = 0
        self._branch_ordinal = 0
        self._loop_ordinal = 0
        self._mapping_ordinal = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        self._call_ordinal += 1
        target = _call_target(node)
        self.calls.append(
            {
                "call_id": f"call:{self.owner_qualified_name}:{node.lineno}:{self._call_ordinal}",
                "owner_id": self.owner_id,
                "target": target,
                "line": node.lineno,
                "args_source": [_unparse(arg) for arg in node.args],
                "kwargs_source": {
                    keyword.arg or "**": _unparse(keyword.value)
                    for keyword in node.keywords
                },
            }
        )
        lowered = " ".join([target] + [_unparse(arg) for arg in node.args]).lower()
        if any(term in lowered for term in ("mapping", "weight", "loader", "shard", "expert", "remap", "filter")):
            self._mapping_ordinal += 1
            kind = _mapping_kind(lowered)
            self.weight_mappings.append(
                {
                    "mapping_id": f"mapping:{self.owner_qualified_name}:{node.lineno}:{self._mapping_ordinal}",
                    "owner_id": self.owner_id,
                    "kind": kind,
                    "source_summary": target,
                    "target_summary": ", ".join(_unparse(arg) for arg in node.args),
                    "start_line": node.lineno,
                    "end_line": _end_line(node),
                    "architecture_relevance_candidate": _mapping_relevance(kind),
                }
            )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        self._branch_ordinal += 1
        condition = _unparse(node.test)
        self.branches.append(
            {
                "branch_id": f"branch:{self.owner_qualified_name}:{node.lineno}:{self._branch_ordinal}",
                "owner_id": self.owner_id,
                "kind": "if",
                "condition": condition,
                "start_line": node.lineno,
                "end_line": _end_line(node),
                "has_else": bool(node.orelse),
                "architecture_relevance_candidate": _branch_relevance(condition),
            }
        )
        for child in node.body + node.orelse:
            self.visit(child)

    def visit_Match(self, node: ast.Match) -> None:  # noqa: N802
        self._branch_ordinal += 1
        condition = f"match {_unparse(node.subject)}"
        self.branches.append(
            {
                "branch_id": f"branch:{self.owner_qualified_name}:{node.lineno}:{self._branch_ordinal}",
                "owner_id": self.owner_id,
                "kind": "match",
                "condition": condition,
                "start_line": node.lineno,
                "end_line": _end_line(node),
                "has_else": any(isinstance(case.pattern, ast.MatchAs) and case.pattern.name is None for case in node.cases),
                "architecture_relevance_candidate": _branch_relevance(condition),
            }
        )
        for case in node.cases:
            for child in case.body:
                self.visit(child)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._visit_loop(node, "for")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
        self._visit_loop(node, "async_for")

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self._visit_loop(node, "while")

    def _visit_loop(self, node: ast.For | ast.AsyncFor | ast.While, kind: str) -> None:
        self._loop_ordinal += 1
        iter_or_condition = _unparse(getattr(node, "iter", getattr(node, "test", None)))
        self.loops.append(
            {
                "loop_id": f"loop:{self.owner_qualified_name}:{node.lineno}:{self._loop_ordinal}",
                "owner_id": self.owner_id,
                "kind": kind,
                "target": _unparse(getattr(node, "target", None)),
                "iter_or_condition": iter_or_condition,
                "start_line": node.lineno,
                "end_line": _end_line(node),
                "architecture_relevance_candidate": "high" if "layer" in iter_or_condition.lower() else "medium",
            }
        )
        for child in node.body + node.orelse:
            self.visit(child)

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        self.returns.append({"expression": _unparse(node.value), "line": node.lineno})
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self._capture_assignment(node.targets, node.value, node.lineno, _end_line(node))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is not None:
            self._capture_assignment([node.target], node.value, node.lineno, _end_line(node))
        self.generic_visit(node)

    def _capture_assignment(self, targets: list[ast.AST], value: ast.AST, start: int, end: int) -> None:
        expression = _unparse(value)
        target_names = [_unparse(target) for target in targets]
        if isinstance(value, ast.Call):
            for target in targets:
                attr = _self_attr_name(target)
                if attr is None:
                    continue
                constructor = _call_target(value)
                match = categorize_constructor(constructor)
                self.module_assignments.append(
                    {
                        "assignment_id": f"assignment:{self.owner_qualified_name}:{start}:{len(self.module_assignments) + 1}",
                        "owner_id": self.owner_id,
                        "owner_qualified_name": self.owner_qualified_name,
                        "attribute": attr,
                        "constructor": constructor,
                        "args_source": [_unparse(arg) for arg in value.args],
                        "kwargs_source": {
                            keyword.arg or "**": _unparse(keyword.value)
                            for keyword in value.keywords
                        },
                        "line": start,
                        "category_candidate": match.category,
                        "confidence": match.confidence,
                        "reason": match.reason,
                    }
                )
        lowered = " ".join(target_names + [expression]).lower()
        if any(term in lowered for term in ("mapping", "weight", "loader", "shard", "expert", "remap", "filter")):
            self._mapping_ordinal += 1
            kind = _mapping_kind(lowered)
            self.weight_mappings.append(
                {
                    "mapping_id": f"mapping:{self.owner_qualified_name}:{start}:{self._mapping_ordinal}",
                    "owner_id": self.owner_id,
                    "kind": kind,
                    "source_summary": ", ".join(target_names),
                    "target_summary": expression,
                    "start_line": start,
                    "end_line": end,
                    "architecture_relevance_candidate": _mapping_relevance(kind),
                }
            )


class CatalogVisitor(ast.NodeVisitor):
    """Recursively catalog classes, methods and module-level functions."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.classes: list[dict[str, Any]] = []
        self.methods: list[dict[str, Any]] = []
        self.module_functions: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.branches: list[dict[str, Any]] = []
        self.loops: list[dict[str, Any]] = []
        self.module_assignments: list[dict[str, Any]] = []
        self.weight_mappings: list[dict[str, Any]] = []
        self.class_stack: list[tuple[str, str]] = []
        self.function_depth = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        parent_class_id = self.class_stack[-1][1] if self.class_stack else None
        qualified_name = ".".join([item[0].split(".")[-1] for item in self.class_stack] + [node.name])
        class_id = f"class:{qualified_name}:{node.lineno}"
        direct_methods = [
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        direct_nested = [child for child in node.body if isinstance(child, ast.ClassDef)]
        source = "\n".join(self.lines[node.lineno - 1 : _end_line(node)])
        bases = [_unparse(base) for base in node.bases]
        self.classes.append(
            {
                "class_id": class_id,
                "name": node.name,
                "qualified_name": qualified_name,
                "parent_class_id": parent_class_id,
                "bases": bases,
                "decorators": _decorators(node),
                "start_line": node.lineno,
                "end_line": _end_line(node),
                "source_range_sha256": _range_sha(self.lines, node.lineno, _end_line(node)),
                "method_ids": [f"method:{qualified_name}.{method.name}:{method.lineno}" for method in direct_methods],
                "nested_class_ids": [f"class:{qualified_name}.{child.name}:{child.lineno}" for child in direct_nested],
                "category_candidates": _class_categories(node.name, bases, source),
            }
        )
        self.class_stack.append((qualified_name, class_id))
        for child in node.body:
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(child)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.class_stack:
            class_qualified, class_id = self.class_stack[-1]
            qualified_name = f"{class_qualified}.{node.name}"
            owner_id = f"method:{qualified_name}:{node.lineno}"
            inspector = ScopeAwareBodyVisitor(owner_id, qualified_name, self.lines)
            for child in node.body:
                inspector.visit(child)
            importance, reasons = _importance(node, inspector.calls, inspector.branches, inspector.module_assignments)
            method = {
                "method_id": owner_id,
                "qualified_name": qualified_name,
                "owner_class_id": class_id,
                "name": node.name,
                "parameters": _parameters(node),
                "start_line": node.lineno,
                "end_line": _end_line(node),
                "call_ids": [item["call_id"] for item in inspector.calls],
                "branch_ids": [item["branch_id"] for item in inspector.branches],
                "loop_ids": [item["loop_id"] for item in inspector.loops],
                "return_summaries": inspector.returns,
                "importance_candidate": importance,
                "importance_reasons": reasons,
            }
            self.methods.append(method)
            self._extend_from_inspector(inspector)
            for child in node.body:
                if isinstance(child, ast.ClassDef):
                    self.visit(child)
            return

        if self.function_depth > 0:
            return
        self.function_depth += 1
        qualified_name = node.name
        function_id = f"function:{qualified_name}:{node.lineno}"
        inspector = ScopeAwareBodyVisitor(function_id, qualified_name, self.lines)
        for child in node.body:
            inspector.visit(child)
        importance, reasons = _importance(node, inspector.calls, inspector.branches, inspector.module_assignments)
        self.module_functions.append(
            {
                "function_id": function_id,
                "name": node.name,
                "qualified_name": qualified_name,
                "start_line": node.lineno,
                "end_line": _end_line(node),
                "parameters": _parameters(node),
                "call_ids": [item["call_id"] for item in inspector.calls],
                "branch_ids": [item["branch_id"] for item in inspector.branches],
                "loop_ids": [item["loop_id"] for item in inspector.loops],
                "importance_candidate": importance,
                "importance_reasons": reasons,
            }
        )
        self._extend_from_inspector(inspector)
        self.function_depth -= 1

    def _extend_from_inspector(self, inspector: ScopeAwareBodyVisitor) -> None:
        self.calls.extend(inspector.calls)
        self.branches.extend(inspector.branches)
        self.loops.extend(inspector.loops)
        self.module_assignments.extend(inspector.module_assignments)
        self.weight_mappings.extend(inspector.weight_mappings)


def _extract_imports(tree: ast.Module) -> list[dict[str, Any]]:
    imports: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "module": alias.name,
                        "name": alias.asname or alias.name,
                        "kind": "import",
                        "line": node.lineno,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append(
                    {
                        "module": node.module or "",
                        "name": alias.name,
                        "asname": alias.asname,
                        "level": node.level,
                        "kind": "from_import",
                        "line": node.lineno,
                    }
                )
    return sorted(imports, key=lambda item: (item["line"], item["module"], item["name"]))


def _module_to_candidate_file(repo_root: Path, current_file: Path, module: str, level: int) -> Path | None:
    models_root = repo_root / "vllm" / "model_executor" / "models"
    if level:
        base = current_file.parent
        for _ in range(max(level - 1, 0)):
            base = base.parent
        parts = module.split(".") if module else []
        return base.joinpath(*parts).with_suffix(".py")
    if module.startswith("vllm.model_executor.models."):
        relative = module.removeprefix("vllm.model_executor.models.").replace(".", "/")
        return models_root / f"{relative}.py"
    if module.startswith("vllm.model_executor.layers."):
        relative = module.removeprefix("vllm.model_executor.layers.").replace(".", "/")
        return repo_root / "vllm" / "model_executor" / "layers" / f"{relative}.py"
    return None


def _related_file_candidates(repo_root: Path, current_file: Path, imports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in imports:
        candidate = _module_to_candidate_file(
            repo_root,
            current_file,
            item.get("module", ""),
            int(item.get("level", 0) or 0),
        )
        if candidate is None:
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "file": key,
                "symbol": item["name"],
                "reason": f"imported from {item.get('module', '')}",
                "priority": 2 if "models" in key else 4,
                "suggested_depth": 1,
                "exists": candidate.exists(),
            }
        )
    for fixed, reason in (
        ("registry.py", "model registry metadata"),
        ("interfaces.py", "model interface capabilities"),
        ("interfaces_base.py", "base model interface capabilities"),
    ):
        candidate = repo_root / "vllm" / "model_executor" / "models" / fixed
        key = str(candidate.resolve())
        if key not in seen:
            seen.add(key)
            candidates.append(
                {
                    "file": key,
                    "symbol": fixed,
                    "reason": reason,
                    "priority": 3,
                    "suggested_depth": 1,
                    "exists": candidate.exists(),
                }
            )
    return sorted(candidates, key=lambda item: (item["priority"], item["file"]))


def _classification(classes: list[dict[str, Any]], source_text: str) -> dict[str, Any]:
    class_names = [item["name"] for item in classes]
    category_candidates = classify_model_source(class_names, source_text)
    status = "helper" if category_candidates == ["helper_or_shared_module"] else "partial"
    if any(
        name.endswith(("ForCausalLM", "Model", "ForConditionalGeneration"))
        for name in class_names
    ):
        status = "full"
    return {
        "category_candidates": category_candidates,
        "status": status,
        "reason": "classified from class names and generic vLLM capability terms",
    }


def _line_evidence(source_text: str, terms: list[str]) -> list[int]:
    lines = source_text.splitlines()
    found: list[int] = []
    for index, line in enumerate(lines, start=1):
        if any(term in line for term in terms):
            found.append(index)
    return found[:20]


def _ast_counts(tree: ast.Module) -> dict[str, int]:
    class_count = 0
    method_count = 0
    module_function_count = 0

    def walk_class(node: ast.ClassDef) -> None:
        nonlocal class_count, method_count
        class_count += 1
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                walk_class(child)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_count += 1
                for nested in child.body:
                    if isinstance(nested, ast.ClassDef):
                        walk_class(nested)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            walk_class(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_function_count += 1
    return {
        "ast_class_count": class_count,
        "ast_method_count": method_count,
        "ast_module_function_count": module_function_count,
    }


def _source_coverage(
    source_sha256: str,
    tree: ast.Module,
    catalog: CatalogVisitor,
) -> tuple[dict[str, Any], list[str]]:
    counts = _ast_counts(tree)
    warnings: list[str] = []
    top_level_class_count = sum(1 for item in catalog.classes if item["parent_class_id"] is None)
    nested_class_count = len(catalog.classes) - top_level_class_count
    coverage = {
        "source_sha256": source_sha256,
        **counts,
        "catalogued_class_count": len(catalog.classes),
        "top_level_class_count": top_level_class_count,
        "nested_class_count": nested_class_count,
        "catalogued_method_count": len(catalog.methods),
        "catalogued_module_function_count": len(catalog.module_functions),
        "branch_count": len(catalog.branches),
        "loop_count": len(catalog.loops),
        "weight_mapping_count": len(catalog.weight_mappings),
        "all_classes_indexed": counts["ast_class_count"] == len(catalog.classes),
        "all_methods_indexed": counts["ast_method_count"] == len(catalog.methods),
        "all_module_functions_indexed": counts["ast_module_function_count"] == len(catalog.module_functions),
        "unindexed_nodes": [],
    }
    for ast_key, catalog_key, label in (
        ("ast_class_count", "catalogued_class_count", "classes"),
        ("ast_method_count", "catalogued_method_count", "methods"),
        ("ast_module_function_count", "catalogued_module_function_count", "module functions"),
    ):
        if coverage[ast_key] != coverage[catalog_key]:
            message = f"source coverage mismatch for {label}: ast={coverage[ast_key]} catalogued={coverage[catalog_key]}"
            coverage["unindexed_nodes"].append(message)
            warnings.append(message)
    return coverage, warnings


def _empty_context(resolution: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "2.1",
        "target": resolution,
        "source_sha256": "",
        "source_coverage": {
            "source_sha256": "",
            "ast_class_count": 0,
            "catalogued_class_count": 0,
            "top_level_class_count": 0,
            "nested_class_count": 0,
            "ast_method_count": 0,
            "catalogued_method_count": 0,
            "ast_module_function_count": 0,
            "catalogued_module_function_count": 0,
            "branch_count": 0,
            "loop_count": 0,
            "weight_mapping_count": 0,
            "all_classes_indexed": True,
            "all_methods_indexed": True,
            "all_module_functions_indexed": True,
            "unindexed_nodes": [],
        },
        "classification": {"status": "unsupported", "category_candidates": []},
        "classes": [],
        "methods": [],
        "module_functions": [],
        "calls": [],
        "branches": [],
        "loops": [],
        "imports": [],
        "module_assignments": [],
        "control_flow_landmarks": [],
        "weight_mappings": [],
        "registry_info": resolution,
        "capability_signals": {},
        "related_file_candidates": [],
        "traversal_defaults": {"max_depth": 3, "default_related_file_count": 20, "max_related_file_count": 30},
        "warnings": warnings,
    }


def _control_flow_landmarks(catalog: CatalogVisitor) -> list[dict[str, Any]]:
    landmarks: list[dict[str, Any]] = []
    method_by_id = {item["method_id"]: item for item in catalog.methods}
    function_by_id = {item["function_id"]: item for item in catalog.module_functions}
    for call in catalog.calls:
        owner = method_by_id.get(call["owner_id"]) or function_by_id.get(call["owner_id"])
        if owner and (owner.get("importance_candidate") == "core" or call["target"].startswith("self.")):
            landmarks.append(
                {
                    "kind": "module_call",
                    "owner_id": call["owner_id"],
                    "summary": call["target"],
                    "line": call["line"],
                }
            )
    for branch in catalog.branches:
        if branch["architecture_relevance_candidate"] in {"high", "medium"}:
            landmarks.append(
                {
                    "kind": "branch",
                    "owner_id": branch["owner_id"],
                    "condition": branch["condition"],
                    "line": branch["start_line"],
                }
            )
    for loop in catalog.loops:
        landmarks.append(
            {
                "kind": "repeated_layer_loop" if "layer" in loop["iter_or_condition"].lower() else "loop",
                "owner_id": loop["owner_id"],
                "summary": loop["iter_or_condition"],
                "line": loop["start_line"],
            }
        )
    return sorted(landmarks, key=lambda item: (item["line"], item["kind"]))


def collect_source_context(repo_root: Path, resolution: dict[str, Any]) -> dict[str, Any]:
    target_file_value = resolution.get("target_file")
    if not target_file_value:
        return _empty_context(resolution, ["target could not be resolved"])
    target_file = Path(target_file_value)
    warnings = list(resolution.get("warnings", []))
    try:
        source_text = target_file.read_text(encoding="utf-8-sig")
        tree = ast.parse(source_text, filename=str(target_file))
    except (OSError, SyntaxError) as exc:
        return _empty_context(resolution, warnings + [f"failed to parse target file: {exc}"])

    lines = source_text.splitlines()
    source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    catalog = CatalogVisitor(lines)
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            catalog.visit(node)
    coverage, coverage_warnings = _source_coverage(source_sha256, tree, catalog)
    warnings.extend(coverage_warnings)

    imports = _extract_imports(tree)
    capabilities = detect_capabilities(source_text)
    for name, signal in capabilities.items():
        signal["capability_id"] = f"capability:{name}"
        signal["evidence_lines"] = _line_evidence(source_text, list(signal["evidence_terms"]))
    classification = _classification(catalog.classes, source_text)
    if warnings and classification["status"] == "full":
        classification["status"] = "partial"
    context = {
        "schema_version": "2.1",
        "target": resolution,
        "source_sha256": source_sha256,
        "source_coverage": coverage,
        "classification": classification,
        "classes": sorted(catalog.classes, key=lambda item: (item["start_line"], item["qualified_name"])),
        "methods": sorted(catalog.methods, key=lambda item: (item["start_line"], item["qualified_name"])),
        "module_functions": sorted(catalog.module_functions, key=lambda item: (item["start_line"], item["qualified_name"])),
        "calls": sorted(catalog.calls, key=lambda item: (item["line"], item["call_id"])),
        "branches": sorted(catalog.branches, key=lambda item: (item["start_line"], item["branch_id"])),
        "loops": sorted(catalog.loops, key=lambda item: (item["start_line"], item["loop_id"])),
        "imports": imports,
        "module_assignments": sorted(catalog.module_assignments, key=lambda item: (item["line"], item["assignment_id"])),
        "control_flow_landmarks": _control_flow_landmarks(catalog),
        "weight_mappings": sorted(catalog.weight_mappings, key=lambda item: (item["start_line"], item["mapping_id"])),
        "registry_info": resolution,
        "capability_signals": capabilities,
        "related_file_candidates": _related_file_candidates(repo_root.resolve(), target_file.resolve(), imports),
        "traversal_defaults": {
            "max_depth": 3,
            "default_related_file_count": 20,
            "max_related_file_count": 30,
            "priority_order": [
                "target model file",
                "local model classes",
                "relative imports",
                "model interfaces and utilities",
                "instantiated vLLM layers",
                "registry metadata",
                "external runtime components",
            ],
        },
        "warnings": warnings,
    }
    if not catalog.classes:
        context["classification"]["status"] = "helper"
        context["warnings"].append("no classes found; treat as helper or shared module")
    return context


def _manifest_item(item_id: str) -> dict[str, Any]:
    return {"item_id": item_id, "status": None, "page_refs": [], "claim_ids": [], "reason": ""}


def plan_template(context: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Return a v2.1 checklist template for Codex to complete."""

    capabilities = [
        _manifest_item(value["capability_id"])
        for value in context.get("capability_signals", {}).values()
        if value.get("detected")
    ]
    return {
        "schema_version": "2.1",
        "model_name": model_name,
        "target_file": context["target"].get("target_file"),
        "source_sha256": context.get("source_sha256", ""),
        "detail_level": "complete",
        "model_category": context.get("classification", {}).get("category_candidates", []),
        "files_read": [],
        "traversal_notes": [],
        "page_budget": {"target_pages": 4, "max_pages": 5},
        "pages": [],
        "class_review": [
            {
                "class_id": item["class_id"],
                "role": None,
                "status": None,
                "page_refs": [],
                "claim_ids": [],
                "reason": "",
            }
            for item in context.get("classes", [])
        ],
        "method_review": [
            {
                "method_id": item["method_id"],
                "importance": None,
                "status": None,
                "page_refs": [],
                "claim_ids": [],
                "reason": "",
            }
            for item in context.get("methods", [])
        ],
        "function_review": [
            {
                "function_id": item["function_id"],
                "importance": None,
                "status": None,
                "page_refs": [],
                "claim_ids": [],
                "reason": "",
            }
            for item in context.get("module_functions", [])
        ],
        "coverage_manifest": {
            "branches": [_manifest_item(item["branch_id"]) for item in context.get("branches", [])],
            "weight_mappings": [_manifest_item(item["mapping_id"]) for item in context.get("weight_mappings", [])],
            "capabilities": capabilities,
            "external_boundaries": [],
        },
        "global_boundaries": [],
        "omissions": [],
        "unresolved": [],
    }


def evidence_template(context: dict[str, Any]) -> dict[str, Any]:
    """Return an empty v2.1 evidence skeleton for Codex to fill."""

    return {
        "schema_version": "2.1",
        "target_file": context["target"].get("target_file"),
        "source_sha256": context.get("source_sha256"),
        "claims": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--architecture")
    parser.add_argument("--resolution", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan-template", type=Path)
    parser.add_argument("--evidence-template", type=Path)
    parser.add_argument("--model-name")
    args = parser.parse_args(argv)

    if args.resolution:
        resolution = json.loads(args.resolution.read_text(encoding="utf-8"))
    else:
        resolution = resolve_model_target(
            args.repo_root,
            input_path=args.input,
            architecture=args.architecture,
        )
    context = collect_source_context(args.repo_root, resolution)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    model_name = args.model_name or args.architecture or Path(str(context["target"].get("target_file", "model"))).stem
    if args.plan_template:
        args.plan_template.parent.mkdir(parents=True, exist_ok=True)
        args.plan_template.write_text(
            json.dumps(plan_template(context, model_name), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.evidence_template:
        args.evidence_template.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_template.write_text(
            json.dumps(evidence_template(context), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"Source context written to {args.output}")
    return 0 if context["classification"]["status"] != "unsupported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
