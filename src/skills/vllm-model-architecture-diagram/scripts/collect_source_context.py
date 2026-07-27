"""Collect lightweight source context for Agent-native diagram design."""

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

IMPORTANT_METHODS = {
    "__init__",
    "forward",
    "compute_logits",
    "load_weights",
    "get_expert_mapping",
    "pooling",
    "embed_input_ids",
    "encoder",
}


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _call_target(node: ast.Call) -> str:
    return _unparse(node.func)


def _decorator_name(node: ast.AST) -> str:
    return _unparse(node)


def _class_end_line(node: ast.ClassDef) -> int:
    return int(getattr(node, "end_lineno", node.lineno))


def _method_summary(owner: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    loops: list[dict[str, Any]] = []
    returns: list[dict[str, Any]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            calls.append({"target": _call_target(child), "line": child.lineno})
        elif isinstance(child, ast.If):
            branches.append({"condition": _unparse(child.test), "line": child.lineno})
        elif isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
            loops.append(
                {
                    "kind": child.__class__.__name__.lower(),
                    "target": _unparse(getattr(child, "target", None)),
                    "iter_or_condition": _unparse(getattr(child, "iter", getattr(child, "test", None))),
                    "line": child.lineno,
                }
            )
        elif isinstance(child, ast.Return):
            returns.append({"expression": _unparse(child.value), "line": child.lineno})
    return {
        "qualified_name": f"{owner}.{node.name}",
        "owner_class": owner,
        "name": node.name,
        "parameters": [arg.arg for arg in node.args.args],
        "start_line": node.lineno,
        "end_line": int(getattr(node, "end_lineno", node.lineno)),
        "is_important": node.name in IMPORTANT_METHODS,
        "calls": sorted(calls, key=lambda item: (item["line"], item["target"])),
        "branches": sorted(branches, key=lambda item: item["line"]),
        "loops": sorted(loops, key=lambda item: item["line"]),
        "returns": sorted(returns, key=lambda item: item["line"]),
    }


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


def _extract_classes(tree: ast.Module) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    classes: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        method_names = [
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        classes.append(
            {
                "name": node.name,
                "bases": [_unparse(base) for base in node.bases],
                "decorators": [_decorator_name(deco) for deco in node.decorator_list],
                "start_line": node.lineno,
                "end_line": _class_end_line(node),
                "methods": method_names,
            }
        )
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(_method_summary(node.name, child))
    return classes, sorted(methods, key=lambda item: (item["start_line"], item["qualified_name"]))


def _target_name(target: ast.AST) -> str:
    return _unparse(target)


def _self_attr_name(target: ast.AST) -> str | None:
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return target.attr
    return None


def _extract_module_assignments(tree: ast.Module) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    class_stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            class_stack.append(node.name)
            self.generic_visit(node)
            class_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
            if not class_stack or not isinstance(node.value, ast.Call):
                return
            for target in node.targets:
                attr = _self_attr_name(target)
                if attr is None:
                    continue
                constructor = _call_target(node.value)
                match = categorize_constructor(constructor)
                assignments.append(
                    {
                        "owner_class": class_stack[-1],
                        "attribute": attr,
                        "constructor": constructor,
                        "args_source": [_unparse(arg) for arg in node.value.args],
                        "kwargs_source": {
                            keyword.arg or "**": _unparse(keyword.value)
                            for keyword in node.value.keywords
                        },
                        "line": node.lineno,
                        "category_candidate": match.category,
                        "confidence": match.confidence,
                        "reason": match.reason,
                    }
                )

    Visitor().visit(tree)
    return sorted(assignments, key=lambda item: (item["line"], item["owner_class"], item["attribute"]))


def _extract_control_flow_landmarks(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    landmarks: list[dict[str, Any]] = []
    for method in methods:
        if not method["is_important"]:
            continue
        for call in method["calls"]:
            if call["target"].startswith("self.") or "layers" in call["target"] or "logits" in call["target"]:
                landmarks.append(
                    {
                        "kind": "module_call",
                        "method": method["qualified_name"],
                        "summary": call["target"],
                        "line": call["line"],
                    }
                )
        for branch in method["branches"]:
            landmarks.append(
                {
                    "kind": "branch",
                    "method": method["qualified_name"],
                    "condition": branch["condition"],
                    "line": branch["line"],
                }
            )
        for loop in method["loops"]:
            loop_summary = loop["iter_or_condition"]
            kind = "repeated_layer_loop" if "layers" in loop_summary else "loop"
            landmarks.append(
                {
                    "kind": kind,
                    "method": method["qualified_name"],
                    "summary": loop_summary,
                    "line": loop["line"],
                }
            )
        for ret in method["returns"]:
            landmarks.append(
                {
                    "kind": "return_boundary",
                    "method": method["qualified_name"],
                    "summary": ret["expression"],
                    "line": ret["line"],
                }
            )
    return sorted(landmarks, key=lambda item: (item["line"], item.get("summary", "")))


def _extract_weight_mappings(tree: ast.Module, methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            expr = _unparse(node.value)
            targets = [_target_name(target) for target in node.targets]
            if "mapping" in " ".join(targets).lower() or "weight" in expr.lower():
                mappings.append(
                    {
                        "kind": "assignment",
                        "targets": targets,
                        "expression": expr,
                        "line": node.lineno,
                    }
                )
    for method in methods:
        if "load_weights" not in method["name"]:
            continue
        for call in method["calls"]:
            mappings.append(
                {
                    "kind": "loader_call",
                    "method": method["qualified_name"],
                    "target": call["target"],
                    "line": call["line"],
                }
            )
    return sorted(mappings, key=lambda item: item["line"])


def _module_to_candidate_file(repo_root: Path, current_file: Path, module: str, level: int) -> Path | None:
    models_root = repo_root / "vllm" / "model_executor" / "models"
    if level:
        base = current_file.parent
        for _ in range(max(level - 1, 0)):
            base = base.parent
        parts = module.split(".") if module else []
        candidate = base.joinpath(*parts).with_suffix(".py")
        return candidate
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


def collect_source_context(repo_root: Path, resolution: dict[str, Any]) -> dict[str, Any]:
    target_file_value = resolution.get("target_file")
    if not target_file_value:
        return {
            "schema_version": "2.0",
            "target": resolution,
            "classification": {"status": "unsupported", "category_candidates": []},
            "classes": [],
            "methods": [],
            "imports": [],
            "module_assignments": [],
            "control_flow_landmarks": [],
            "weight_mappings": [],
            "registry_info": resolution,
            "capability_signals": {},
            "related_file_candidates": [],
            "traversal_defaults": {"max_depth": 3, "default_related_file_count": 20, "max_related_file_count": 30},
            "warnings": ["target could not be resolved"],
        }
    target_file = Path(target_file_value)
    warnings = list(resolution.get("warnings", []))
    try:
        source_text = target_file.read_text(encoding="utf-8-sig")
        tree = ast.parse(source_text, filename=str(target_file))
    except (OSError, SyntaxError) as exc:
        return {
            "schema_version": "2.0",
            "target": resolution,
            "classification": {"status": "unsupported", "category_candidates": []},
            "classes": [],
            "methods": [],
            "imports": [],
            "module_assignments": [],
            "control_flow_landmarks": [],
            "weight_mappings": [],
            "registry_info": resolution,
            "capability_signals": {},
            "related_file_candidates": [],
            "traversal_defaults": {"max_depth": 3, "default_related_file_count": 20, "max_related_file_count": 30},
            "warnings": warnings + [f"failed to parse target file: {exc}"],
        }

    imports = _extract_imports(tree)
    classes, methods = _extract_classes(tree)
    capabilities = detect_capabilities(source_text)
    for name, signal in capabilities.items():
        signal["evidence_lines"] = _line_evidence(source_text, list(signal["evidence_terms"]))
    context = {
        "schema_version": "2.0",
        "target": resolution,
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "classification": _classification(classes, source_text),
        "classes": classes,
        "methods": methods,
        "imports": imports,
        "module_assignments": _extract_module_assignments(tree),
        "control_flow_landmarks": _extract_control_flow_landmarks(methods),
        "weight_mappings": _extract_weight_mappings(tree, methods),
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
    if not classes:
        context["classification"]["status"] = "helper"
        context["warnings"].append("no classes found; treat as helper or shared module")
    return context


def plan_template(context: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Return an empty Agent-authored plan skeleton."""

    return {
        "schema_version": "2.0",
        "model_name": model_name,
        "target_file": context["target"].get("target_file"),
        "model_category": context.get("classification", {}).get("category_candidates", []),
        "files_read": [],
        "pages": [],
        "global_boundaries": [],
        "omissions": [],
        "unresolved": [],
    }


def evidence_template(context: dict[str, Any]) -> dict[str, Any]:
    """Return an empty evidence skeleton for Codex to fill."""

    return {
        "schema_version": "2.0",
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
