"""Resolve vLLM model files or registry architecture names without imports."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegistryEntry:
    architecture: str
    module: str
    class_name: str
    category: str
    source_mapping: str


def _models_root(repo_root: Path) -> Path:
    candidate = repo_root / "vllm" / "model_executor" / "models"
    if candidate.exists():
        return candidate
    if repo_root.name == "models" and (repo_root / "registry.py").exists():
        return repo_root
    return candidate


def _registry_path(repo_root: Path) -> Path:
    return _models_root(repo_root) / "registry.py"


def _literal_str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _category_from_mapping(name: str) -> str:
    cleaned = name.strip("_").lower()
    if cleaned.endswith("_models"):
        cleaned = cleaned[: -len("_models")]
    return cleaned or "unknown"


def _entry_from_value(
    architecture: str,
    value: ast.AST,
    category: str,
    mapping_name: str,
) -> RegistryEntry | None:
    if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) >= 2:
        module = _literal_str(value.elts[0])
        class_name = _literal_str(value.elts[1])
        if module and class_name:
            return RegistryEntry(
                architecture=architecture,
                module=module,
                class_name=class_name,
                category=category,
                source_mapping=mapping_name,
            )
    return None


def _parse_static_dict(
    value: ast.AST,
    mapping_name: str,
    category: str,
) -> tuple[list[RegistryEntry], list[str]]:
    entries: list[RegistryEntry] = []
    warnings: list[str] = []
    if not isinstance(value, ast.Dict):
        warnings.append(f"{mapping_name} is not a static dict")
        return entries, warnings
    for key, item in zip(value.keys, value.values, strict=False):
        if key is None:
            continue
        architecture = _literal_str(key)
        if architecture is None:
            warnings.append(f"{mapping_name} contains a non-string architecture key")
            continue
        entry = _entry_from_value(architecture, item, category, mapping_name)
        if entry is None:
            warnings.append(f"{mapping_name}.{architecture} is not a static (module, class) tuple")
            continue
        entries.append(entry)
    return entries, warnings


def parse_registry(registry_path: Path) -> tuple[list[RegistryEntry], list[str]]:
    """Parse registry mappings while avoiding runtime imports."""

    warnings: list[str] = []
    if not registry_path.exists():
        return [], [f"registry.py not found at {registry_path}"]
    try:
        source = registry_path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(registry_path))
    except (OSError, SyntaxError) as exc:
        return [], [f"failed to parse registry.py: {exc}"]

    mapping_entries: dict[str, list[RegistryEntry]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if not name.startswith("_"):
                continue
            category = _category_from_mapping(name)
            entries, dict_warnings = _parse_static_dict(node.value, name, category)
            if entries:
                mapping_entries[name] = entries
            warnings.extend(dict_warnings)

    merged: list[RegistryEntry] = []
    seen: set[str] = set()
    for name, entries in mapping_entries.items():
        if name == "_VLLM_MODELS":
            continue
        for entry in entries:
            if entry.architecture not in seen:
                merged.append(entry)
                seen.add(entry.architecture)

    # Prefer the explicit aggregate if it can be statically expanded.
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_VLLM_MODELS" for target in node.targets):
            continue
        aggregate: list[RegistryEntry] = []
        if isinstance(node.value, ast.Dict):
            for key, item in zip(node.value.keys, node.value.values, strict=False):
                if key is None and isinstance(item, ast.Name):
                    aggregate.extend(mapping_entries.get(item.id, []))
                elif key is not None:
                    architecture = _literal_str(key)
                    if architecture:
                        entry = _entry_from_value(architecture, item, "aggregate", "_VLLM_MODELS")
                        if entry:
                            aggregate.append(entry)
        if aggregate:
            deduped: list[RegistryEntry] = []
            aggregate_seen: set[str] = set()
            for entry in aggregate:
                if entry.architecture not in aggregate_seen:
                    deduped.append(entry)
                    aggregate_seen.add(entry.architecture)
            return deduped, warnings
        warnings.append("_VLLM_MODELS could not be statically expanded")
    return merged, warnings


def list_registry_models(repo_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    entries, warnings = parse_registry(_registry_path(repo_root))
    return [asdict(entry) for entry in sorted(entries, key=lambda item: item.architecture)], warnings


def _candidate_matches(architecture: str, entries: list[RegistryEntry]) -> list[str]:
    needle = architecture.lower()
    return [
        entry.architecture
        for entry in entries
        if needle in entry.architecture.lower() or entry.architecture.lower() in needle
    ][:10]


def resolve_model_target(
    repo_root: Path,
    *,
    input_path: Path | None = None,
    architecture: str | None = None,
) -> dict[str, Any]:
    """Resolve a target model file and registry metadata."""

    repo_root = repo_root.resolve()
    models_root = _models_root(repo_root).resolve()
    entries, warnings = parse_registry(_registry_path(repo_root))
    result: dict[str, Any] = {
        "repo_root": str(repo_root),
        "models_root": str(models_root),
        "target_file": None,
        "requested_architecture": architecture,
        "registered_architectures": [],
        "registered_classes": [],
        "registry_category": None,
        "warnings": warnings,
        "status": "unresolved",
    }
    if input_path is not None:
        target_file = input_path.resolve()
        result["target_file"] = str(target_file)
        matching_entries = [
            entry
            for entry in entries
            if (models_root / f"{entry.module}.py").resolve() == target_file
        ]
        result["registered_architectures"] = [entry.architecture for entry in matching_entries]
        result["registered_classes"] = sorted({entry.class_name for entry in matching_entries})
        result["registry_category"] = matching_entries[0].category if matching_entries else None
        if not target_file.exists():
            result["status"] = "missing_file"
            result["warnings"].append(f"target file does not exist: {target_file}")
        elif models_root.exists() and not str(target_file).startswith(str(models_root)):
            result["status"] = "partial"
            result["warnings"].append("target file is outside vllm/model_executor/models")
        else:
            result["status"] = "resolved"
        return result

    if architecture is None:
        result["warnings"].append("no input file or architecture was supplied")
        return result

    exact = next((entry for entry in entries if entry.architecture == architecture), None)
    if exact is None:
        result["candidate_architectures"] = _candidate_matches(architecture, entries)
        result["warnings"].append(f"architecture not found in registry: {architecture}")
        return result

    target_file = (models_root / f"{exact.module}.py").resolve()
    result.update(
        {
            "target_file": str(target_file),
            "registered_architectures": [exact.architecture],
            "registered_classes": [exact.class_name],
            "registry_category": exact.category,
            "status": "resolved" if target_file.exists() else "missing_file",
        }
    )
    if not target_file.exists():
        result["warnings"].append(f"registered module file does not exist: {target_file}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--architecture")
    parser.add_argument("--list", action="store_true", help="List registry entries instead of resolving one target.")
    parser.add_argument("--filter", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.list:
        entries, warnings = list_registry_models(args.repo_root)
        if args.filter:
            needle = args.filter.lower()
            entries = [
                entry
                for entry in entries
                if needle in entry["architecture"].lower()
                or needle in entry["module"].lower()
                or needle in entry["class_name"].lower()
            ]
        payload: dict[str, Any] = {"models": entries, "warnings": warnings}
    else:
        payload = resolve_model_target(
            args.repo_root,
            input_path=args.input,
            architecture=args.architecture,
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if payload.get("status", "resolved") in {"resolved", "partial"} or args.list else 2


if __name__ == "__main__":
    raise SystemExit(main())
