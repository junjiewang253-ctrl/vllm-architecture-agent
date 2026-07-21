#!/usr/bin/env python3
"""Validate the semantic architecture IR before Draw.io rendering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

NODE_KINDS = {
    "input",
    "embedding",
    "normalization",
    "attention",
    "ffn",
    "moe",
    "repeated_block",
    "head",
    "logits_processor",
    "decision",
    "add",
    "merge",
    "output",
    "container",
    "note",
}

EDGE_KINDS = {
    "runtime",
    "residual",
    "containment",
    "dependency",
    "conditional_true",
    "conditional_false",
    "weight_mapping",
}

NON_MAJOR_NODE_KINDS = {"note", "container"}
PARALLEL_BADGE_LABELS = {"TP", "PP", "EP"}


def _is_non_empty_evidence(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, dict) for item in value
    )


def validate_architecture_ir(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if data.get("schema_version") != "0.1":
        errors.append("schema_version must be '0.1'")
    if not isinstance(data.get("model_name"), str) or not data["model_name"].strip():
        errors.append("model_name must be a non-empty string")
    if data.get("detail_level") not in {"overview", "full"}:
        errors.append("detail_level must be 'overview' or 'full'")

    pages = data.get("pages")
    if not isinstance(pages, list) or not pages:
        errors.append("pages must be a non-empty list")
        return errors

    for page_index, page in enumerate(pages):
        prefix = f"pages[{page_index}]"
        if not isinstance(page, dict):
            errors.append(f"{prefix} must be an object")
            continue

        nodes = page.get("nodes")
        edges = page.get("edges")
        if not isinstance(nodes, list):
            errors.append(f"{prefix}.nodes must be a list")
            continue
        if not isinstance(edges, list):
            errors.append(f"{prefix}.edges must be a list")
            continue

        node_by_id: dict[str, dict[str, Any]] = {}
        for node_index, node in enumerate(nodes):
            node_prefix = f"{prefix}.nodes[{node_index}]"
            if not isinstance(node, dict):
                errors.append(f"{node_prefix} must be an object")
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str) or not node_id:
                errors.append(f"{node_prefix}.id must be a non-empty string")
                continue
            if node_id in node_by_id:
                errors.append(f"{prefix} contains duplicate node id: {node_id}")
            else:
                node_by_id[node_id] = node

            kind = node.get("kind")
            if kind not in NODE_KINDS:
                errors.append(f"{node_prefix}.kind is invalid: {kind!r}")
            label = node.get("label")
            if not isinstance(label, str) or not label.strip():
                errors.append(f"{node_prefix}.label must be a non-empty string")
            elif label.strip().upper() in PARALLEL_BADGE_LABELS and kind != "note":
                errors.append(
                    f"{node_prefix}: TP/PP/EP must be badges or notes, not compute nodes"
                )

            if kind not in NON_MAJOR_NODE_KINDS and not _is_non_empty_evidence(
                node.get("evidence")
            ):
                errors.append(f"{node_prefix} must include source evidence")

        edge_ids: set[str] = set()
        outgoing: dict[str, list[dict[str, Any]]] = {}
        for edge_index, edge in enumerate(edges):
            edge_prefix = f"{prefix}.edges[{edge_index}]"
            if not isinstance(edge, dict):
                errors.append(f"{edge_prefix} must be an object")
                continue
            edge_id = edge.get("id")
            if not isinstance(edge_id, str) or not edge_id:
                errors.append(f"{edge_prefix}.id must be a non-empty string")
            elif edge_id in edge_ids:
                errors.append(f"{prefix} contains duplicate edge id: {edge_id}")
            else:
                edge_ids.add(edge_id)

            source = edge.get("source")
            target = edge.get("target")
            kind = edge.get("kind")
            if source not in node_by_id:
                errors.append(f"{edge_prefix}.source references unknown node: {source!r}")
            if target not in node_by_id:
                errors.append(f"{edge_prefix}.target references unknown node: {target!r}")
            if kind not in EDGE_KINDS:
                errors.append(f"{edge_prefix}.kind is invalid: {kind!r}")
            if not _is_non_empty_evidence(edge.get("evidence")):
                errors.append(f"{edge_prefix} must include source evidence")
            if isinstance(source, str):
                outgoing.setdefault(source, []).append(edge)

            if kind == "residual" and target in node_by_id:
                target_kind = node_by_id[target].get("kind")
                if target_kind not in {"add", "merge"}:
                    errors.append(
                        f"{edge_prefix}: residual edge must target an add or merge node"
                    )

        for node_id, node in node_by_id.items():
            if node.get("kind") != "decision":
                continue
            kinds = {edge.get("kind") for edge in outgoing.get(node_id, [])}
            if "conditional_true" not in kinds or "conditional_false" not in kinds:
                errors.append(
                    f"{prefix}: decision node {node_id!r} requires true and false branches"
                )

        if data.get("detail_level") == "overview":
            major_nodes = [
                node
                for node in nodes
                if isinstance(node, dict)
                and node.get("kind") not in NON_MAJOR_NODE_KINDS
            ]
            if len(major_nodes) > 12:
                errors.append(
                    f"{prefix}: overview contains {len(major_nodes)} major nodes; maximum is 12"
                )

    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate architecture IR JSON.")
    parser.add_argument("input", type=Path, help="Architecture IR JSON file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists() or not args.input.is_file():
        print(f"error: file does not exist: {args.input}", file=sys.stderr)
        return 2
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: unable to read JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(data, dict):
        print("error: architecture IR root must be an object", file=sys.stderr)
        return 2

    errors = validate_architecture_ir(data)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Architecture IR is valid: {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
