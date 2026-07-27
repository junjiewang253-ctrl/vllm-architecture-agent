#!/usr/bin/env python3
"""Apply a constrained visual-only patch to Architecture Design 1.0."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ALLOWED_OPS = {
    "update_display",
    "set_badge_visibility",
    "set_preferred_size",
    "set_route_hint",
    "set_edge_label_visibility",
    "set_region",
    "set_lane",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_by_id(page: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node["id"]): node
        for node in page.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }


def _edge_by_id(page: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(edge["id"]): edge
        for edge in page.get("edges", [])
        if isinstance(edge, dict) and isinstance(edge.get("id"), str)
    }


def _apply_operation(design: dict[str, Any], operation: dict[str, Any]) -> None:
    op = operation.get("op")
    if op not in ALLOWED_OPS:
        raise ValueError(f"unsupported visual patch operation: {op!r}")
    page_id = operation.get("page_id")
    target_id = operation.get("target_id")
    page = next((item for item in design.get("pages", []) if isinstance(item, dict) and item.get("id") == page_id), None)
    if page is None:
        raise ValueError(f"operation {operation.get('op_id')} references missing page {page_id!r}")
    value = operation.get("value")
    nodes = _node_by_id(page)
    edges = _edge_by_id(page)

    if op == "update_display":
        node = nodes.get(str(target_id))
        if node is None or not isinstance(value, dict):
            raise ValueError(f"operation {operation.get('op_id')} must target a node with display object value")
        display = node.setdefault("display", {})
        for key in ("title", "subtitle", "show_implementation", "badges", "max_title_lines", "max_subtitle_lines"):
            if key in value:
                display[key] = value[key]
        return
    if op == "set_badge_visibility":
        node = nodes.get(str(target_id))
        if node is None or not isinstance(value, list):
            raise ValueError(f"operation {operation.get('op_id')} must target a node with badge list value")
        node.setdefault("display", {})["badges"] = value
        return
    if op == "set_preferred_size":
        node = nodes.get(str(target_id))
        if node is None or not isinstance(value, dict):
            raise ValueError(f"operation {operation.get('op_id')} must target a node with size object value")
        width = float(value.get("width", 0))
        height = float(value.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"operation {operation.get('op_id')} has non-positive preferred size")
        node.setdefault("layout_hint", {})["preferred_size"] = {"width": width, "height": height}
        return
    if op == "set_route_hint":
        edge = edges.get(str(target_id))
        if edge is None or not isinstance(value, str):
            raise ValueError(f"operation {operation.get('op_id')} must target an edge with route_hint string")
        edge["route_hint"] = value
        return
    if op == "set_edge_label_visibility":
        edge = edges.get(str(target_id))
        if edge is None or not isinstance(value, bool):
            raise ValueError(f"operation {operation.get('op_id')} must target an edge with boolean value")
        edge.setdefault("display", {})["show_label"] = value
        return
    if op in {"set_region", "set_lane"}:
        node = nodes.get(str(target_id))
        if node is None or not isinstance(value, str):
            raise ValueError(f"operation {operation.get('op_id')} must target a node with string value")
        key = "region" if op == "set_region" else "lane"
        node.setdefault("layout_hint", {})[key] = value
        return


def apply_visual_patch(design: dict[str, Any], patch: dict[str, Any], *, expected_hash: str) -> dict[str, Any]:
    if patch.get("schema_version") != "1.0":
        raise ValueError("visual patch schema_version must be '1.0'")
    if patch.get("base_design_sha256") != expected_hash:
        raise ValueError("visual patch base_design_sha256 does not match design file")
    updated = copy.deepcopy(design)
    applied: list[str] = []
    for operation in patch.get("operations", []):
        if not isinstance(operation, dict):
            raise ValueError("visual patch operations must be objects")
        _apply_operation(updated, operation)
        applied.append(str(operation.get("op_id")))
    updated["visual_patch"] = {
        "base_design_sha256": expected_hash,
        "applied_operations": applied,
    }
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_design", type=Path)
    parser.add_argument("visual_patch", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        expected_hash = _sha256(args.architecture_design)
        updated = apply_visual_patch(_load_json(args.architecture_design), _load_json(args.visual_patch), expected_hash=expected_hash)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote visually patched Architecture Design to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
