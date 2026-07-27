#!/usr/bin/env python3
"""Compatibility wrapper for Architecture View generation.

v1.1 forbids direct Concept -> View projection. This legacy command now routes
through Architecture Design Graph first, then mechanically builds the View
Graph from that design blueprint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_script_module(filename: str, module_name: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to import {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_architecture_view_graph(architecture_concept: dict[str, Any], fact_graph: dict[str, Any]) -> dict[str, Any]:
    """Build a View Graph through the v1.1 Design layer."""

    design_architect = _load_script_module("run_design_architect.py", "run_design_architect_compat")
    view_builder = _load_script_module("build_view_from_design.py", "build_view_from_design_compat")
    design = design_architect.build_architecture_design_graph(architecture_concept, fact_graph)
    return view_builder.build_view_from_design(design)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_concept", type=Path)
    parser.add_argument("source_fact_graph", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        view = build_architecture_view_graph(_load_json(args.architecture_concept), _load_json(args.source_fact_graph))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Architecture View Graph through Architecture Design to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
