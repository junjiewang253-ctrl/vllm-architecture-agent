"""Scan vLLM model files and report v2.0 preparation compatibility."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from collect_source_context import collect_source_context  # noqa: E402
from resolve_model_target import list_registry_models, resolve_model_target  # noqa: E402


def _models_root(repo_root: Path) -> Path:
    return repo_root / "vllm" / "model_executor" / "models"


def _status_from_context(context: dict[str, Any], registered: bool) -> str:
    classification_status = context.get("classification", {}).get("status")
    if classification_status == "unsupported":
        return "unsupported"
    if classification_status == "full":
        return "full"
    if registered and classification_status == "partial":
        return "partial"
    if classification_status == "helper":
        return "helper"
    if context.get("warnings") and not registered:
        return "boundary-only"
    return "unsupported"


def scan_models_directory(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    models_root = _models_root(repo_root)
    entries, registry_warnings = list_registry_models(repo_root)
    entries_by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        path = (models_root / f"{entry['module']}.py").resolve()
        entries_by_file[str(path)].append(entry)

    results: list[dict[str, Any]] = []
    failures = 0
    python_files = sorted(models_root.glob("*.py")) if models_root.exists() else []
    if not models_root.exists():
        return {
            "schema_version": "2.0",
            "repo_root": str(repo_root),
            "models_root": str(models_root),
            "entries": [],
            "summary": {
                "total_python_files": 0,
                "registered_model_modules": 0,
                "helper_modules": 0,
                "full": 0,
                "partial": 0,
                "boundary_only": 0,
                "unsupported": 0,
                "warnings": len(registry_warnings),
                "failures": 1,
                "registry_resolution_rate": 0.0,
                "prepare_success_rate": 0.0,
            },
            "warnings": registry_warnings + [f"models root does not exist: {models_root}"],
        }
    for file_path in python_files:
        try:
            resolution = resolve_model_target(repo_root, input_path=file_path)
            context = collect_source_context(repo_root, resolution)
            registered_entries = entries_by_file.get(str(file_path.resolve()), [])
            status = _status_from_context(context, bool(registered_entries))
            capabilities = {
                key: value
                for key, value in context.get("capability_signals", {}).items()
                if value.get("detected")
            }
            results.append(
                {
                    "file": str(file_path.resolve()),
                    "registered_architectures": [entry["architecture"] for entry in registered_entries],
                    "category": context.get("classification", {}).get("category_candidates", []),
                    "status": status,
                    "detected_roles": sorted(
                        {
                            assignment.get("category_candidate", "unknown")
                            for assignment in context.get("module_assignments", [])
                        }
                    ),
                    "capabilities": sorted(capabilities.keys()),
                    "warnings": context.get("warnings", []),
                    "related_files": context.get("related_file_candidates", [])[:10],
                    "prepare_success": status in {"full", "partial", "boundary-only", "helper"},
                }
            )
        except Exception as exc:  # keep full directory scans resilient
            failures += 1
            results.append(
                {
                    "file": str(file_path.resolve()),
                    "registered_architectures": [
                        entry["architecture"] for entry in entries_by_file.get(str(file_path.resolve()), [])
                    ],
                    "category": [],
                    "status": "unsupported",
                    "detected_roles": [],
                    "capabilities": [],
                    "warnings": [f"prepare failed: {exc}"],
                    "related_files": [],
                    "prepare_success": False,
                }
            )
    registered_modules = {entry["module"] for entry in entries}
    resolvable = [
        entry
        for entry in entries
        if (models_root / f"{entry['module']}.py").exists()
    ]
    prepare_successes = [item for item in results if item["registered_architectures"] and item["prepare_success"]]
    registered_results = [item for item in results if item["registered_architectures"]]
    status_counts: dict[str, int] = defaultdict(int)
    for item in results:
        status_counts[item["status"].replace("-", "_")] += 1
    summary = {
        "total_python_files": len(python_files),
        "registered_model_modules": len(registered_modules),
        "helper_modules": status_counts["helper"],
        "full": status_counts["full"],
        "partial": status_counts["partial"],
        "boundary_only": status_counts["boundary_only"],
        "unsupported": status_counts["unsupported"],
        "warnings": len(registry_warnings) + sum(len(item["warnings"]) for item in results),
        "failures": failures,
        "registry_resolution_rate": round(len(resolvable) / len(entries), 4) if entries else 1.0,
        "prepare_success_rate": round(len(prepare_successes) / len(registered_results), 4) if registered_results else 1.0,
    }
    return {
        "schema_version": "2.0",
        "repo_root": str(repo_root),
        "models_root": str(models_root.resolve()),
        "entries": results,
        "summary": summary,
        "warnings": registry_warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = scan_models_directory(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Compatibility report written to {args.output}")
    return 0 if report["summary"]["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
