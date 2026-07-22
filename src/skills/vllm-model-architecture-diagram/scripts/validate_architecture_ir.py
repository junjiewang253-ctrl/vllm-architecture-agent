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
    "invocation",
    "summary",
}

NON_MAJOR_NODE_KINDS = {"note", "container"}
PARALLEL_BADGE_LABELS = {"TP", "PP", "EP"}
PHASES = {
    "construction",
    "runtime",
    "checkpoint_loading",
    "parallel_partition",
}


def _is_non_empty_evidence(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, dict) for item in value
    )


def validate_architecture_ir(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if data.get("schema_version") != "0.2":
        errors.append("schema_version must be '0.2'")
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
            phase = node.get("phase")
            if phase not in PHASES:
                errors.append(f"{node_prefix}.phase is invalid: {phase!r}")
            label = node.get("label")
            if not isinstance(label, str) or not label.strip():
                errors.append(f"{node_prefix}.label must be a non-empty string")
            elif label.strip().upper() in PARALLEL_BADGE_LABELS and kind != "note":
                errors.append(
                    f"{node_prefix}: TP/PP/EP must be badges or notes, not compute nodes"
                )

            badges = node.get("badges")
            if not isinstance(badges, list):
                errors.append(f"{node_prefix}.badges must be a list")
            else:
                for badge in badges:
                    if badge not in PARALLEL_BADGE_LABELS:
                        errors.append(f"{node_prefix}.badges contains invalid badge: {badge!r}")

            parent_id = node.get("parent_id")
            if parent_id is not None:
                if not isinstance(parent_id, str) or not parent_id:
                    errors.append(f"{node_prefix}.parent_id must be a non-empty string or null")
                elif parent_id == node_id:
                    errors.append(f"{node_prefix}: node cannot be its own parent")

            repetition = node.get("repetition")
            if repetition is not None:
                if not isinstance(repetition, dict):
                    errors.append(f"{node_prefix}.repetition must be an object or null")
                else:
                    count_expression = repetition.get("count_expression")
                    if not isinstance(count_expression, str) or not count_expression.strip():
                        errors.append(
                            f"{node_prefix}.repetition.count_expression must be non-empty"
                        )

            variants = node.get("variants")
            if variants is not None:
                if not isinstance(variants, list):
                    errors.append(f"{node_prefix}.variants must be a list")
                else:
                    for variant_index, variant in enumerate(variants):
                        variant_prefix = f"{node_prefix}.variants[{variant_index}]"
                        if not isinstance(variant, dict):
                            errors.append(f"{variant_prefix} must be an object")
                            continue
                        variant_phase = variant.get("phase")
                        if variant_phase not in PHASES:
                            errors.append(
                                f"{variant_prefix}.phase is invalid: {variant_phase!r}"
                            )
                        component = variant.get("component")
                        if (
                            isinstance(component, str)
                            and ("MoE" in component or "FeedForward" in component or "FFN" in component)
                            and variant_phase != "construction"
                        ):
                            errors.append(
                                f"{variant_prefix}: Dense/MoE variants must use construction phase"
                            )

            if kind not in NON_MAJOR_NODE_KINDS and not _is_non_empty_evidence(
                node.get("evidence")
            ):
                errors.append(f"{node_prefix} must include source evidence")

        for node_index, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            parent_id = node.get("parent_id")
            if isinstance(parent_id, str) and parent_id and parent_id not in node_by_id:
                errors.append(
                    f"{prefix}.nodes[{node_index}].parent_id references unknown node: {parent_id!r}"
                )

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
            phase = edge.get("phase")
            if source not in node_by_id:
                errors.append(f"{edge_prefix}.source references unknown node: {source!r}")
            if target not in node_by_id:
                errors.append(f"{edge_prefix}.target references unknown node: {target!r}")
            if kind not in EDGE_KINDS:
                errors.append(f"{edge_prefix}.kind is invalid: {kind!r}")
            if phase not in PHASES:
                errors.append(f"{edge_prefix}.phase is invalid: {phase!r}")
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
            if kind == "weight_mapping" and phase != "checkpoint_loading":
                errors.append(
                    f"{edge_prefix}: weight_mapping edge must use checkpoint_loading phase"
                )

            if source in node_by_id and target in node_by_id:
                source_node = node_by_id[source]
                target_node = node_by_id[target]
                source_scope = source_node.get("scope")
                target_scope = target_node.get("scope")
                if (
                    kind == "runtime"
                    and isinstance(source_scope, str)
                    and isinstance(target_scope, str)
                    and source_scope
                    and target_scope
                    and source_scope != target_scope
                ):
                    errors.append(
                        f"{edge_prefix}: runtime edge cannot cross scopes "
                        f"{source_scope!r} -> {target_scope!r}"
                    )
                if (
                    isinstance(source_scope, str)
                    and isinstance(target_scope, str)
                    and source_scope
                    and target_scope
                    and source_scope != target_scope
                    and kind not in {"invocation", "summary"}
                ):
                    errors.append(
                        f"{edge_prefix}: cross-scope edge must use invocation or summary"
                    )
                if kind == "runtime" and (
                    source_node.get("phase") == "construction"
                    or target_node.get("phase") == "construction"
                ):
                    errors.append(
                        f"{edge_prefix}: construction phase nodes cannot use runtime edges"
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
