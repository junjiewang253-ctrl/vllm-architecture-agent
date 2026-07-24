#!/usr/bin/env python3
"""Apply Diagram View presentation patches without changing semantics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PATCH_VERSION = "0.1"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} root must be an object")
    return data


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _page(view: dict[str, Any], page_id: str) -> dict[str, Any]:
    for page in view.get("pages", []):
        if isinstance(page, dict) and page.get("id") == page_id:
            return page
    raise ValueError(f"page not found: {page_id}")


def _node(page: dict[str, Any], node_id: str) -> dict[str, Any]:
    for node in page.get("visible_nodes", []):
        if isinstance(node, dict) and node.get("semantic_id") == node_id:
            return node
    raise ValueError(f"visible node not found: {node_id}")


def _edge(page: dict[str, Any], edge_id: str) -> dict[str, Any]:
    for edge in page.get("visible_edges", []):
        if isinstance(edge, dict) and edge.get("semantic_id") == edge_id:
            return edge
    raise ValueError(f"visible edge not found: {edge_id}")


def _bundle(page: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    for bundle in page.get("bundles", []):
        if isinstance(bundle, dict) and bundle.get("id") == bundle_id:
            return bundle
    raise ValueError(f"bundle not found: {bundle_id}")


def _apply_operation(view: dict[str, Any], operation: dict[str, Any]) -> None:
    op = operation.get("op")
    page_id = operation.get("page_id")
    if not isinstance(page_id, str):
        raise ValueError(f"{op} requires page_id")
    page = _page(view, page_id)
    if op == "update_display_label":
        _node(page, str(operation.get("semantic_id") or operation.get("node_id")))["display_label"] = str(operation.get("label"))
    elif op == "update_display_subtitle":
        _node(page, str(operation.get("semantic_id") or operation.get("node_id")))["display_subtitle"] = str(operation.get("subtitle"))
    elif op == "set_edge_visible":
        edge = _edge(page, str(operation.get("edge_id") or operation.get("semantic_id")))
        edge["route_class"] = "direct" if operation.get("visible") is True else "hidden_semantic"
    elif op == "set_edge_label_visible":
        _edge(page, str(operation.get("edge_id") or operation.get("semantic_id")))["label_visible"] = bool(operation.get("visible"))
    elif op == "update_edge_label":
        _edge(page, str(operation.get("edge_id") or operation.get("semantic_id")))["label"] = operation.get("label")
    elif op == "set_route_class":
        _edge(page, str(operation.get("edge_id") or operation.get("semantic_id")))["route_class"] = str(operation.get("route_class"))
    elif op == "set_region":
        _node(page, str(operation.get("semantic_id") or operation.get("node_id")))["region_id"] = operation.get("region_id")
    elif op == "set_lane":
        _node(page, str(operation.get("semantic_id") or operation.get("node_id")))["lane_id"] = operation.get("lane_id")
    elif op == "set_preferred_size":
        size = operation.get("preferred_size")
        if not isinstance(size, dict):
            raise ValueError("set_preferred_size requires preferred_size object")
        _node(page, str(operation.get("semantic_id") or operation.get("node_id")))["preferred_size"] = copy.deepcopy(size)
    elif op == "set_badge_visibility":
        _node(page, str(operation.get("semantic_id") or operation.get("node_id")))["badges_visible"] = bool(operation.get("visible"))
    elif op == "add_bundle":
        bundle = operation.get("bundle")
        if not isinstance(bundle, dict) or not isinstance(bundle.get("id"), str):
            raise ValueError("add_bundle requires bundle object with id")
        page.setdefault("bundles", []).append(copy.deepcopy(bundle))
    elif op == "update_bundle":
        bundle = operation.get("bundle")
        if not isinstance(bundle, dict) or not isinstance(bundle.get("id"), str):
            raise ValueError("update_bundle requires bundle object with id")
        _bundle(page, bundle["id"]).update(copy.deepcopy(bundle))
    elif op in {"add_legend", "add_boundary_note", "add_page_note"}:
        annotation = operation.get("annotation")
        if not isinstance(annotation, dict) or not isinstance(annotation.get("id"), str):
            raise ValueError(f"{op} requires annotation object with id")
        page.setdefault("annotations", []).append(copy.deepcopy(annotation))
    elif op == "update_layout_constraint":
        updates = operation.get("updates")
        if not isinstance(updates, dict):
            raise ValueError("update_layout_constraint requires updates object")
        page.setdefault("layout_constraints", {}).update(copy.deepcopy(updates))
    elif op in {"split_visual_region", "merge_visual_region"}:
        # Region splitting/merging is intentionally conservative for v0.9:
        # it records review intent through annotations without semantic impact.
        annotation = {
            "id": f"decorative_note_{operation.get('op_id', 'region')}",
            "type": "page_note",
            "text": operation.get("reason", ""),
            "x": 48,
            "y": 58,
            "width": 300,
            "height": 54,
        }
        page.setdefault("annotations", []).append(annotation)
    else:
        raise ValueError(f"unsupported view patch operation: {op}")


def apply_view_patch(base_view_path: Path, patch_path: Path) -> dict[str, Any]:
    view = _load_json(base_view_path, "baseline Diagram View")
    patch = _load_json(patch_path, "Diagram View patch")
    if patch.get("schema_version") != PATCH_VERSION:
        raise ValueError(f"patch schema_version must be {PATCH_VERSION!r}")
    base_hash = _hash_file(base_view_path)
    if patch.get("base_view_sha256") != base_hash:
        raise ValueError("patch base_view_sha256 does not match baseline Diagram View")
    reviewed = copy.deepcopy(view)
    applied: list[str] = []
    deferred: list[dict[str, Any]] = list(patch.get("deferred_operations", [])) if isinstance(patch.get("deferred_operations"), list) else []
    for operation in patch.get("operations", []):
        if not isinstance(operation, dict):
            raise ValueError("view patch operation must be an object")
        if float(operation.get("confidence", 0)) < 0.70:
            deferred.append(copy.deepcopy(operation))
            continue
        _apply_operation(reviewed, operation)
        applied.append(str(operation.get("op_id")))
    reviewed["review"] = {
        "mode": "agent-guided",
        "base_view_sha256": base_hash,
        "patch_sha256": _hash_file(patch_path),
        "applied_operations": applied,
        "deferred_operations": deferred,
    }
    return reviewed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Diagram View patch.")
    parser.add_argument("baseline_view", type=Path)
    parser.add_argument("diagram_view_patch", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reviewed = apply_view_patch(args.baseline_view, args.diagram_view_patch)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_stable_json(reviewed), encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote reviewed Diagram View to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
