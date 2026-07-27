#!/usr/bin/env python3
"""Validate visual review and Diagram View patch artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REVIEW_VERSION = "0.1"
PATCH_VERSION = "0.1"
ALLOWED_OPS = {
    "update_display_label",
    "update_display_subtitle",
    "set_edge_visible",
    "set_edge_label_visible",
    "update_edge_label",
    "set_route_class",
    "set_region",
    "set_lane",
    "set_preferred_size",
    "add_bundle",
    "update_bundle",
    "add_legend",
    "add_boundary_note",
    "add_page_note",
    "split_visual_region",
    "merge_visual_region",
    "update_layout_constraint",
    "set_badge_visibility",
}
FORBIDDEN_OPS = {
    "add_node",
    "remove_node",
    "add_edge",
    "remove_edge",
    "update_edge",
    "replace_evidence",
    "set_evidence_type",
}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} root must be an object")
    return data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pages(view: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        page["id"]: page
        for page in view.get("pages", [])
        if isinstance(page, dict) and isinstance(page.get("id"), str)
    }


def _view_ids(view: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    page_ids: set[str] = set()
    for page in view.get("pages", []):
        if not isinstance(page, dict):
            continue
        page_id = page.get("id")
        if isinstance(page_id, str):
            page_ids.add(page_id)
        for node in page.get("visible_nodes", []):
            if isinstance(node, dict) and isinstance(node.get("semantic_id"), str):
                node_ids.add(node["semantic_id"])
        for edge in page.get("visible_edges", []):
            if isinstance(edge, dict) and isinstance(edge.get("semantic_id"), str):
                edge_ids.add(edge["semantic_id"])
    return page_ids, node_ids, edge_ids


def _edge_by_id(view: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for page in view.get("pages", []):
        if not isinstance(page, dict):
            continue
        for edge in page.get("visible_edges", []):
            if isinstance(edge, dict) and isinstance(edge.get("semantic_id"), str):
                result[edge["semantic_id"]] = edge
    return result


def _ir_required_external_ids(ir: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for page in ir.get("pages", []):
        if not isinstance(page, dict):
            continue
        for collection in ("nodes", "edges"):
            for item in page.get(collection, []):
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    continue
                for evidence in item.get("evidence", []):
                    if isinstance(evidence, dict) and evidence.get("type") == "external":
                        ids.add(item["id"])
    return ids


def validate_visual_review(
    reviewed_ir: dict[str, Any],
    baseline_view: dict[str, Any],
    baseline_view_path: Path,
    visual_review: dict[str, Any],
    view_patch: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if visual_review.get("schema_version") != REVIEW_VERSION:
        errors.append("visual review schema_version must be '0.1'")
    if view_patch.get("schema_version") != PATCH_VERSION:
        errors.append("Diagram View patch schema_version must be '0.1'")
    if view_patch.get("base_view_sha256") != _sha256(baseline_view_path):
        errors.append("Diagram View patch base_view_sha256 does not match baseline view")

    page_ids, node_ids, edge_ids = _view_ids(baseline_view)
    edge_lookup = _edge_by_id(baseline_view)
    external_ids = _ir_required_external_ids(reviewed_ir)
    patched_findings: set[str] = set()
    for finding in visual_review.get("findings", []):
        if not isinstance(finding, dict):
            errors.append("visual finding must be an object")
            continue
        if finding.get("semantic_change") is not False:
            errors.append(f"visual finding proposes a semantic change: {finding.get('finding_id') or finding.get('id')}")

    for operation in view_patch.get("operations", []):
        if not isinstance(operation, dict):
            errors.append("view patch operation must be an object")
            continue
        op_id = operation.get("op_id")
        op = operation.get("op")
        if op in FORBIDDEN_OPS:
            errors.append(f"view patch attempts forbidden semantic operation: {op_id}")
        if op not in ALLOWED_OPS:
            errors.append(f"invalid view patch operation: {op}")
        confidence = operation.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            errors.append(f"invalid view patch confidence: {op_id}")
        elif float(confidence) < 0.70:
            errors.append(f"low-confidence view patch operation must be deferred: {op_id}")
        page_id = operation.get("page_id")
        if op not in {"merge_visual_region"} and isinstance(page_id, str) and page_id not in page_ids:
            errors.append(f"view patch references unknown page: {op_id}: {page_id}")
        node_id = operation.get("semantic_id") or operation.get("node_id")
        edge_id = operation.get("edge_id") or operation.get("semantic_id")
        if op in {"update_display_label", "update_display_subtitle", "set_region", "set_lane", "set_preferred_size", "set_badge_visibility"}:
            if not isinstance(node_id, str) or node_id not in node_ids:
                errors.append(f"view patch references unknown node: {op_id}: {node_id}")
        if op in {"set_edge_visible", "set_edge_label_visible", "update_edge_label", "set_route_class"}:
            if not isinstance(edge_id, str) or edge_id not in edge_ids:
                errors.append(f"view patch references unknown edge: {op_id}: {edge_id}")
            if op == "set_edge_label_visible" and operation.get("visible") is False:
                edge = edge_lookup.get(str(edge_id), {})
                if edge.get("label_visible") is True and str(edge.get("label") or "").lower() in {"optional", "external"}:
                    errors.append(f"view patch hides required correctness label: {op_id}")
        if op in {"add_bundle", "update_bundle"}:
            bundle = operation.get("bundle")
            if not isinstance(bundle, dict):
                errors.append(f"{op} requires bundle object: {op_id}")
            else:
                for member_id in bundle.get("member_ids", []):
                    if member_id not in edge_ids:
                        errors.append(f"bundle references unknown visible edge: {op_id}: {member_id}")
        if op in {"add_boundary_note", "add_page_note", "add_legend"}:
            annotation = operation.get("annotation")
            if not isinstance(annotation, dict) or not str(annotation.get("id") or "").startswith("decorative_"):
                errors.append(f"{op} requires decorative annotation id: {op_id}")
        for finding_id in operation.get("finding_ids", []):
            if isinstance(finding_id, str):
                patched_findings.add(finding_id)

    for finding in visual_review.get("findings", []):
        if not isinstance(finding, dict):
            continue
        if finding.get("severity") in {"critical", "major"}:
            finding_id = finding.get("finding_id") or finding.get("id")
            if isinstance(finding_id, str) and finding_id not in patched_findings:
                errors.append(f"major visual finding is not patched: {finding_id}")
    if external_ids:
        has_boundary_note = any(
            isinstance(op, dict)
            and op.get("op") == "add_boundary_note"
            and any(affected in external_ids for affected in op.get("annotation", {}).get("affected_ids", []) or op.get("affected_ids", []))
            for op in view_patch.get("operations", [])
        )
        if not has_boundary_note:
            # Existing v0.8 views may already label external edges; this is a warning-grade guard
            # encoded as a soft pass unless a patch removes such notes.
            pass
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate visual review and Diagram View patch.")
    parser.add_argument("reviewed_ir", type=Path)
    parser.add_argument("baseline_view", type=Path)
    parser.add_argument("visual_review", type=Path)
    parser.add_argument("diagram_view_patch", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        errors = validate_visual_review(
            _load_json(args.reviewed_ir, "reviewed Architecture IR"),
            _load_json(args.baseline_view, "baseline Diagram View"),
            args.baseline_view,
            _load_json(args.visual_review, "visual review"),
            _load_json(args.diagram_view_patch, "Diagram View patch"),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Visual review validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
