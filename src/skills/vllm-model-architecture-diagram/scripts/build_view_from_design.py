#!/usr/bin/env python3
"""Build Architecture View Graph from Architecture Design Graph.

This script is intentionally mechanical. The design layer decides the story;
this builder turns that blueprint into the renderer-facing View Graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

VIEW_GRAPH_VERSION = "0.1"
DESIGN_GRAPH_VERSION = "0.1"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _view_node(node: dict[str, Any]) -> dict[str, Any]:
    role = str(node.get("role") or node.get("visual_role") or "annotation")
    return {
        "id": node.get("id"),
        "type": node.get("type"),
        "role": role,
        "label": node.get("label"),
        "subtitle": node.get("subtitle") or node.get("implementation", ""),
        "implementation": node.get("implementation", ""),
        "concept_refs": node.get("concept_refs", []),
        "fact_refs": node.get("fact_refs", []),
        "ports": node.get("ports", []),
        "visual_role": role,
        "layout": node.get("layout", {}),
    }


def _view_edge(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": edge.get("id"),
        "type": edge.get("type"),
        "source": edge.get("source"),
        "target": edge.get("target"),
        "source_port": edge.get("source_port", "out"),
        "target_port": edge.get("target_port", "in"),
        "label": edge.get("label", ""),
        "show_label": edge.get("show_label") is True,
        "concept_refs": edge.get("concept_refs", []),
        "fact_refs": edge.get("fact_refs", []),
    }


def build_view_from_design(design: dict[str, Any]) -> dict[str, Any]:
    if design.get("schema_version") != DESIGN_GRAPH_VERSION or design.get("design_graph_type") != "architecture_design_graph":
        raise ValueError("Architecture Design Graph must use schema_version '0.1'")
    pages: list[dict[str, Any]] = []
    for page in design.get("pages", []):
        if not isinstance(page, dict):
            continue
        pages.append(
            {
                "id": page.get("id"),
                "title": page.get("title"),
                "purpose": page.get("purpose"),
                "question": page.get("question"),
                "main_flow": page.get("main_flow", []),
                "branches": page.get("branches", []),
                "external_boundaries": page.get("external_boundaries", []),
                "nodes": [_view_node(node) for node in page.get("nodes", []) if isinstance(node, dict)],
                "edges": [_view_edge(edge) for edge in page.get("edges", []) if isinstance(edge, dict)],
                "groups": page.get("groups", []),
                "lanes": page.get("lanes", []),
                "annotations": page.get("annotations", []),
                "strategies": page.get("strategies", []),
            }
        )
    return {
        "schema_version": VIEW_GRAPH_VERSION,
        "view_graph_type": "architecture_view_graph",
        "source_design_version": design.get("schema_version"),
        "model_name": design.get("model_name", "unknown-model"),
        "pages": pages,
        "boundaries": design.get("boundaries", []),
        "legend": [
            {"type": "runtime_flow", "label": "Runtime tensor/data flow"},
            {"type": "dependency", "label": "Construction or configuration dependency"},
            {"type": "delegation", "label": "Adapter delegation to external component"},
            {"type": "boundary", "label": "External boundary"},
            {"type": "mapping", "label": "Checkpoint mapping"},
            {"type": "parallel", "label": "Parallel strategy"},
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_design", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        view = build_view_from_design(_load_json(args.architecture_design))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Architecture View Graph to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
