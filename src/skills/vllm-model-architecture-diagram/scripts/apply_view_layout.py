#!/usr/bin/env python3
"""Apply deterministic layout to an Architecture View Graph."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_layout_module() -> Any:
    path = Path(__file__).with_name("layout_diagram.py")
    spec = importlib.util.spec_from_file_location("layout_diagram_for_view_graph", path)
    if spec is None or spec.loader is None:
        raise ValueError("unable to import layout_diagram.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def apply_view_layout(view: dict[str, Any]) -> dict[str, Any]:
    if view.get("view_graph_type") != "architecture_view_graph":
        raise ValueError("apply_view_layout expects an Architecture View Graph")
    layout_module = _load_layout_module()
    return layout_module.build_layout_plan(view)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_view", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = apply_view_layout(_load_json(args.architecture_view))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Architecture View Layout to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
