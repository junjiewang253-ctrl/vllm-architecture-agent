#!/usr/bin/env python3
"""Compile strict Architecture Design 1.0 into Architecture View 1.0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DESIGN_VERSION = "1.0"
VIEW_VERSION = "1.0"


KIND_MAP = {
    "data": "data",
    "component": "component",
    "process": "process",
    "storage": "storage",
    "external_component": "boundary",
    "strategy": "capability",
    "mapping": "process",
    "annotation": "annotation",
    "merge": "add",
    "boundary": "boundary",
}

EDGE_STYLE_MAP = {
    "tensor_flow": "runtime",
    "residual_flow": "residual",
    "dependency": "dependency",
    "containment": "containment",
    "mapping": "weight_mapping",
    "parallel_effect": "parallel_partition",
    "delegation": "invocation",
    "external_boundary": "dependency",
    "control": "dependency",
}

ROUTE_MAP = {
    "direct": "horizontal_lane",
    "upper_lane": "vertical_branch",
    "lower_lane": "horizontal_lane",
    "branch_top": "vertical_branch",
    "branch_bottom": "local_branch",
    "bus": "horizontal_lane",
    "local": "local_branch",
    "hidden": "hidden_semantic",
}

ROLE_MAP = {
    "primary": "primary_flow",
    "branch": "secondary_flow",
    "auxiliary": "annotation",
    "contained": "secondary_flow",
    "strategy": "strategy",
    "boundary": "boundary",
    "annotation": "annotation",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _visible_node(node: dict[str, Any], container_ids: set[str]) -> dict[str, Any]:
    display = node["display"]
    layout_hint = node["layout_hint"]
    kind = "container" if node["id"] in container_ids else KIND_MAP.get(node["semantic_kind"], "note")
    return {
        "semantic_id": node["id"],
        "id": node["id"],
        "label": display["title"] or node["label"],
        "type": kind,
        "display_label": display["title"] or node["label"],
        "display_subtitle": display["subtitle"],
        "region_id": layout_hint.get("region"),
        "lane_id": layout_hint.get("lane"),
        "kind": kind,
        "semantic_kind": node["semantic_kind"],
        "role": ROLE_MAP.get(node["role"], node["role"]),
        "design_role": node["role"],
        "phase": node["phase"],
        "parent_id": node.get("parent_id"),
        "preferred_size": layout_hint["preferred_size"],
        "ports": node.get("ports", []),
        "badges": display.get("badges", []),
        "layout": {"order": layout_hint.get("order", 0)},
        "concept_refs": node.get("concept_refs", []),
        "fact_refs": node.get("fact_refs", []),
    }


def _visible_edge(edge: dict[str, Any]) -> dict[str, Any]:
    display = edge["display"]
    route_hint = edge.get("route_hint", "direct")
    return {
        "semantic_id": edge["id"],
        "id": edge["id"],
        "type": edge["kind"],
        "source": edge["source"],
        "target": edge["target"],
        "source_port": edge["source_port"],
        "target_port": edge["target_port"],
        "style_kind": EDGE_STYLE_MAP.get(edge["kind"], "runtime"),
        "edge_kind": edge["kind"],
        "phase": edge["phase"],
        "label": display.get("label", ""),
        "label_visible": display.get("show_label") is True,
        "visible": display.get("visible") is True,
        "priority": display.get("priority", "primary"),
        "route_class": ROUTE_MAP.get(route_hint, "horizontal_lane"),
        "route_hint": route_hint,
        "bundle_id": None,
        "concept_refs": edge.get("concept_refs", []),
        "fact_refs": edge.get("fact_refs", []),
    }


def _region(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": group["id"],
        "label": group["label"],
        "kind": group["kind"],
        "member_ids": group.get("member_ids", []),
    }


def _decoration(page: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"decorative_{annotation['id']}",
        "type": "annotation",
        "text": annotation["text"],
        "target_ids": annotation.get("target_ids", []),
        "fact_refs": annotation.get("fact_refs", []),
    }


def compile_architecture_view(design: dict[str, Any]) -> dict[str, Any]:
    if design.get("schema_version") != DESIGN_VERSION or design.get("design_graph_type") != "architecture_design_graph":
        raise ValueError("Architecture Design must use schema_version '1.0'")
    pages: list[dict[str, Any]] = []
    for page in design.get("pages", []):
        if not isinstance(page, dict):
            continue
        container_ids = {
            str(node.get("parent_id"))
            for node in page.get("nodes", [])
            if isinstance(node, dict) and isinstance(node.get("parent_id"), str) and node.get("parent_id")
        }
        regions = [_region(group) for group in page.get("groups", []) if isinstance(group, dict)]
        pages.append(
            {
                "id": page["id"],
                "title": page["title"],
                "question": page["question"],
                "purpose": page["purpose"],
                "view_kind": page["view_kind"],
                "scope": page.get("scope", []),
                "primary_story": page["primary_story"],
                "regions": regions,
                "lanes": [region for region in regions if region.get("kind") == "lane"],
                "visible_nodes": [_visible_node(node, container_ids) for node in page.get("nodes", []) if isinstance(node, dict)],
                "visible_edges": [_visible_edge(edge) for edge in page.get("edges", []) if isinstance(edge, dict)],
                "decorations": [_decoration(page, annotation) for annotation in page.get("annotations", []) if isinstance(annotation, dict)],
                "layout_constraints": page["layout_constraints"],
                "legend_policy": {
                    "show": True,
                    "compact": True,
                    "edge_kinds": sorted({edge.get("kind") for edge in page.get("edges", []) if isinstance(edge, dict)}),
                },
                "coverage": page.get("coverage", []),
            }
        )
    return {
        "schema_version": VIEW_VERSION,
        "view_graph_type": "architecture_view_graph",
        "source_design_version": design.get("schema_version"),
        "model_name": design.get("model_name", "unknown-model"),
        "source_sha256": design.get("source_sha256"),
        "pages": pages,
        "external_boundaries": design.get("external_boundaries", []),
        "legend": [
            {"type": "runtime", "label": "Primary tensor/data flow"},
            {"type": "residual", "label": "Residual lane"},
            {"type": "dependency", "label": "Configuration/dependency"},
            {"type": "weight_mapping", "label": "Checkpoint mapping"},
            {"type": "parallel_partition", "label": "Parallel strategy"},
            {"type": "invocation", "label": "Delegation"},
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_design", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        view = compile_architecture_view(_load_json(args.architecture_design))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Architecture View Graph to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
