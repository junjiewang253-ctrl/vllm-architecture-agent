#!/usr/bin/env python3
"""Validate semantic Architecture IR before Draw.io rendering."""

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
    "adaptation",
    "parallel_partition",
}
PHASES = {"construction", "runtime", "checkpoint_loading", "parallel_partition"}
NON_MAJOR_NODE_KINDS = {"note", "container"}
PARALLEL_BADGE_LABELS = {"TP", "PP", "EP"}
PAGE_TYPES = {
    "overview",
    "decoder_detail",
    "attention_detail",
    "moe_detail",
    "adapter_integration",
    "parallelism",
    "weight_loading",
}
EVIDENCE_TYPES = {"direct", "derived", "external", "unresolved"}
PORT_DIRECTIONS = {"input", "output", "bidirectional"}
PORT_DATA_KINDS = {"tensor", "config", "weights", "cache", "capability", "control"}
EDGE_DISPLAY_ROUTES = {
    "direct",
    "top_lane",
    "bottom_lane",
    "horizontal_lane",
    "vertical_branch",
    "local_branch",
    "cache_write",
    "cache_read",
    "weight_mapping",
    "hidden_semantic",
}


def _is_non_empty_evidence(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)


def _has_derived_evidence(value: Any) -> bool:
    return isinstance(value, list) and any(isinstance(item, dict) and item.get("type") == "derived" for item in value)


def _fact_ids(evidence: Any) -> list[str]:
    if not isinstance(evidence, list):
        return []
    result: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        for fact_id in item.get("fact_ids", []):
            if isinstance(fact_id, str):
                result.append(fact_id)
    return result


def _evidence_types(evidence: Any) -> set[str]:
    if not isinstance(evidence, list):
        return set()
    return {str(item.get("type")) for item in evidence if isinstance(item, dict)}


def _validate_evidence(value: Any, prefix: str, errors: list[str], known_fact_ids: set[str]) -> None:
    if not _is_non_empty_evidence(value):
        errors.append(f"{prefix} must include source evidence")
        return
    for index, item in enumerate(value):
        item_prefix = f"{prefix}.evidence[{index}]"
        evidence_type = item.get("type")
        if evidence_type not in EVIDENCE_TYPES:
            errors.append(f"{item_prefix}.type is invalid: {evidence_type!r}")
        fact_ids = item.get("fact_ids")
        if fact_ids is not None:
            if not isinstance(fact_ids, list) or not all(isinstance(fact_id, str) and fact_id for fact_id in fact_ids):
                errors.append(f"{item_prefix}.fact_ids must be a list of non-empty strings")
            elif known_fact_ids:
                for fact_id in fact_ids:
                    if fact_id not in known_fact_ids:
                        errors.append(f"{item_prefix}.fact_ids references unknown fact_id: {fact_id}")
        if evidence_type == "direct" and not item.get("fact_ids"):
            errors.append(f"{item_prefix}: direct evidence must include fact_ids")
        if evidence_type == "derived" and not item.get("fact_ids") and not item.get("note") and not item.get("derivation_rule"):
            errors.append(f"{item_prefix}: derived evidence must include fact_ids, note, or derivation_rule")
        if evidence_type == "external" and not item.get("note"):
            errors.append(f"{item_prefix}: external evidence must describe the external boundary")


def _validate_node_display(value: Any, prefix: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f"{prefix}.display must be an object")
        return
    for key in value:
        if key not in {"label", "subtitle", "description", "show_badges"}:
            errors.append(f"{prefix}.display contains unknown key: {key}")
    for key in ("label", "subtitle", "description"):
        if key in value and value[key] is not None and not isinstance(value[key], str):
            errors.append(f"{prefix}.display.{key} must be a string or null")
    if "show_badges" in value and not isinstance(value["show_badges"], bool):
        errors.append(f"{prefix}.display.show_badges must be boolean")


def _validate_edge_display(value: Any, prefix: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f"{prefix}.display must be an object")
        return
    for key in value:
        if key not in {"visible", "label", "show_label", "route"}:
            errors.append(f"{prefix}.display contains unknown key: {key}")
    if "visible" in value and not isinstance(value["visible"], bool):
        errors.append(f"{prefix}.display.visible must be boolean")
    if "label" in value and value["label"] is not None and not isinstance(value["label"], str):
        errors.append(f"{prefix}.display.label must be a string or null")
    if "show_label" in value and not isinstance(value["show_label"], bool):
        errors.append(f"{prefix}.display.show_label must be boolean")
    if "route" in value and value["route"] not in EDGE_DISPLAY_ROUTES:
        errors.append(f"{prefix}.display.route is invalid: {value.get('route')!r}")


def _ports_by_id(node: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ports = node.get("ports")
    if not isinstance(ports, list):
        return {}
    return {
        port["id"]: port
        for port in ports
        if isinstance(port, dict) and isinstance(port.get("id"), str)
    }


def _validate_ports(value: Any, prefix: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        errors.append(f"{prefix}.ports must be a list")
        return
    seen: set[str] = set()
    for index, port in enumerate(value):
        port_prefix = f"{prefix}.ports[{index}]"
        if not isinstance(port, dict):
            errors.append(f"{port_prefix} must be an object")
            continue
        port_id = port.get("id")
        if not isinstance(port_id, str) or not port_id:
            errors.append(f"{port_prefix}.id must be a non-empty string")
        elif port_id in seen:
            errors.append(f"{prefix}.ports contains duplicate port id: {port_id}")
        else:
            seen.add(port_id)
        if not isinstance(port.get("label"), str) or not port["label"].strip():
            errors.append(f"{port_prefix}.label must be a non-empty string")
        if port.get("direction") not in PORT_DIRECTIONS:
            errors.append(f"{port_prefix}.direction is invalid: {port.get('direction')!r}")
        if port.get("data_kind") not in PORT_DATA_KINDS:
            errors.append(f"{port_prefix}.data_kind is invalid: {port.get('data_kind')!r}")


def _edge_port_data_compatible(kind: str, source_kind: str | None, target_kind: str | None) -> bool:
    if kind in {"runtime", "residual", "conditional_true", "conditional_false", "summary"}:
        return source_kind in {"tensor", "cache"} and target_kind in {"tensor", "cache"}
    if kind == "weight_mapping":
        return source_kind == "weights" and target_kind == "weights"
    if kind in {"dependency", "adaptation"}:
        allowed = {"config", "capability", "control", "weights"}
        return source_kind in allowed and target_kind in allowed
    if kind == "parallel_partition":
        allowed = {"capability", "control"}
        return source_kind in allowed and target_kind in allowed
    return True


def validate_architecture_ir(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if data.get("schema_version") != "0.6":
        errors.append("schema_version must be '0.6'")
    if not isinstance(data.get("model_name"), str) or not data["model_name"].strip():
        errors.append("model_name must be a non-empty string")
    if data.get("detail_level") not in {"overview", "full"}:
        errors.append("detail_level must be 'overview' or 'full'")
    source_fact_ids_value = data.get("source_fact_ids")
    if source_fact_ids_value is not None and not isinstance(source_fact_ids_value, list):
        errors.append("source_fact_ids must be a list when present")
        known_fact_ids: set[str] = set()
    else:
        known_fact_ids = {
            fact_id
            for fact_id in (source_fact_ids_value or [])
            if isinstance(fact_id, str)
        }
    entrypoints = data.get("weight_loading_entrypoints")
    if entrypoints is not None:
        if not isinstance(entrypoints, list) or not all(isinstance(item, str) for item in entrypoints):
            errors.append("weight_loading_entrypoints must be a list of strings")
        elif {"HYV3ForCausalLM.load_weights", "HYV3Model.load_weights"} - set(entrypoints):
            errors.append("weight_loading_entrypoints must keep wrapper and model load_weights flows separate")

    pages = data.get("pages")
    if not isinstance(pages, list) or not pages:
        errors.append("pages must be a non-empty list")
        return errors

    global_node_to_page: dict[str, str] = {}
    global_edge_ids: set[str] = set()
    page_nodes: dict[str, dict[str, dict[str, Any]]] = {}

    for page_index, page in enumerate(pages):
        prefix = f"pages[{page_index}]"
        if not isinstance(page, dict):
            errors.append(f"{prefix} must be an object")
            continue
        page_id = page.get("id")
        if not isinstance(page_id, str) or not page_id:
            errors.append(f"{prefix}.id must be a non-empty string")
            page_id = f"<invalid-{page_index}>"
        if not isinstance(page.get("title"), str) or not page["title"].strip():
            errors.append(f"{prefix}.title must be a non-empty string")
        if page.get("page_type") not in PAGE_TYPES:
            errors.append(f"{prefix}.page_type is invalid: {page.get('page_type')!r}")

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
            if node_id in global_node_to_page:
                errors.append(f"node id {node_id!r} appears on multiple pages")
            global_node_to_page[node_id] = page_id
            node_by_id[node_id] = node

            kind = node.get("kind")
            phase = node.get("phase")
            label = node.get("label")
            if kind not in NODE_KINDS:
                errors.append(f"{node_prefix}.kind is invalid: {kind!r}")
            if phase not in PHASES:
                errors.append(f"{node_prefix}.phase is invalid: {phase!r}")
            if not isinstance(label, str) or not label.strip():
                errors.append(f"{node_prefix}.label must be a non-empty string")
            elif label.strip().upper() in PARALLEL_BADGE_LABELS and kind != "note":
                errors.append(f"{node_prefix}: TP/PP/EP must be badges or notes, not compute nodes")
            _validate_node_display(node.get("display"), node_prefix, errors)
            _validate_ports(node.get("ports"), node_prefix, errors)

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
                elif not isinstance(repetition.get("count_expression"), str) or not repetition["count_expression"].strip():
                    errors.append(f"{node_prefix}.repetition.count_expression must be non-empty")

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
                        if variant.get("phase") not in PHASES:
                            errors.append(f"{variant_prefix}.phase is invalid: {variant.get('phase')!r}")
                        component = variant.get("component")
                        if isinstance(component, str) and ("MoE" in component or "FeedForward" in component or "FFN" in component) and variant.get("phase") != "construction":
                            errors.append(f"{variant_prefix}: Dense/MoE variants must use construction phase")

            if kind not in NON_MAJOR_NODE_KINDS:
                _validate_evidence(node.get("evidence"), node_prefix, errors, known_fact_ids)
            elif _is_non_empty_evidence(node.get("evidence")):
                _validate_evidence(node.get("evidence"), node_prefix, errors, known_fact_ids)
            if kind in {"add", "merge"} and str(node.get("label", "")).lower().find("handoff") >= 0 and not _has_derived_evidence(node.get("evidence")):
                errors.append(f"{node_prefix}: derived Add/Handoff node must include derived evidence")

        for node_index, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            parent_id = node.get("parent_id")
            if isinstance(parent_id, str) and parent_id and parent_id not in node_by_id:
                errors.append(f"{prefix}.nodes[{node_index}].parent_id references unknown or cross-page node: {parent_id!r}")

        page_nodes[page_id] = node_by_id

    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        prefix = f"pages[{page_index}]"
        page_id = page.get("id")
        node_by_id = page_nodes.get(page_id, {})
        edges = page.get("edges")
        if not isinstance(edges, list):
            continue
        outgoing: dict[str, list[dict[str, Any]]] = {}
        for edge_index, edge in enumerate(edges):
            edge_prefix = f"{prefix}.edges[{edge_index}]"
            if not isinstance(edge, dict):
                errors.append(f"{edge_prefix} must be an object")
                continue
            edge_id = edge.get("id")
            if not isinstance(edge_id, str) or not edge_id:
                errors.append(f"{edge_prefix}.id must be a non-empty string")
            elif edge_id in global_edge_ids:
                errors.append(f"duplicate edge id: {edge_id}")
            else:
                global_edge_ids.add(edge_id)

            source = edge.get("source")
            target = edge.get("target")
            kind = edge.get("kind")
            phase = edge.get("phase")
            _validate_edge_display(edge.get("display"), edge_prefix, errors)
            if source not in node_by_id:
                errors.append(f"{edge_prefix}.source references unknown or cross-page node: {source!r}")
            if target not in node_by_id:
                errors.append(f"{edge_prefix}.target references unknown or cross-page node: {target!r}")
            if kind not in EDGE_KINDS:
                errors.append(f"{edge_prefix}.kind is invalid: {kind!r}")
            if phase not in PHASES:
                errors.append(f"{edge_prefix}.phase is invalid: {phase!r}")
            _validate_evidence(edge.get("evidence"), edge_prefix, errors, known_fact_ids)
            evidence_types = _evidence_types(edge.get("evidence"))
            fact_ids = _fact_ids(edge.get("evidence"))
            if kind in {"runtime", "weight_mapping"} and "direct" in evidence_types and fact_ids and all(":import:" in fact_id for fact_id in fact_ids):
                errors.append(f"{edge_prefix}: import fact cannot be the only direct behavioral evidence")
            if "external" in evidence_types and "direct" in evidence_types:
                errors.append(f"{edge_prefix}: external behavior cannot also be marked direct")
            if isinstance(source, str):
                outgoing.setdefault(source, []).append(edge)
            if kind == "residual" and target in node_by_id and node_by_id[target].get("kind") not in {"add", "merge", "normalization", "output"}:
                errors.append(f"{edge_prefix}: residual edge must target an add, merge, normalization, or output node")
            if kind == "weight_mapping" and phase != "checkpoint_loading":
                errors.append(f"{edge_prefix}: weight_mapping edge must use checkpoint_loading phase")
            if kind == "residual" and edge.get("source_port") == "hidden_out":
                errors.append(f"{edge_prefix}: residual edge cannot use hidden_out as residual_out")
            if phase == "construction" and kind == "runtime":
                errors.append(f"{edge_prefix}: constructor dependency must not be marked runtime")
            if isinstance(edge.get("condition"), str) and edge["condition"].strip():
                if not any(":branch:" in fact_id for fact_id in fact_ids):
                    errors.append(f"{edge_prefix}: optional/conditional edge must cite a branch fact")

            if source in node_by_id and target in node_by_id:
                source_node = node_by_id[source]
                target_node = node_by_id[target]
                source_port_id = edge.get("source_port")
                target_port_id = edge.get("target_port")
                source_port = None
                target_port = None
                if not isinstance(source_port_id, str) or not source_port_id:
                    errors.append(f"{edge_prefix}.source_port must be a non-empty string")
                else:
                    source_port = _ports_by_id(source_node).get(source_port_id)
                    if source_port is None:
                        errors.append(f"{edge_prefix}.source_port references unknown port {source_port_id!r} on {source!r}")
                if not isinstance(target_port_id, str) or not target_port_id:
                    errors.append(f"{edge_prefix}.target_port must be a non-empty string")
                else:
                    target_port = _ports_by_id(target_node).get(target_port_id)
                    if target_port is None:
                        errors.append(f"{edge_prefix}.target_port references unknown port {target_port_id!r} on {target!r}")
                if source_port is not None:
                    if source_port.get("direction") == "input":
                        errors.append(f"{edge_prefix}: source_port cannot be input-only")
                if target_port is not None:
                    if target_port.get("direction") == "output":
                        errors.append(f"{edge_prefix}: target_port cannot be output-only")
                if source_port is not None and target_port is not None:
                    source_kind = source_port.get("data_kind")
                    target_kind = target_port.get("data_kind")
                    if not _edge_port_data_compatible(str(kind), source_kind, target_kind):
                        errors.append(f"{edge_prefix}: incompatible port data_kind for {kind}: {source_kind!r} -> {target_kind!r}")
                source_scope = source_node.get("scope")
                target_scope = target_node.get("scope")
                if kind == "runtime" and isinstance(source_scope, str) and isinstance(target_scope, str) and source_scope and target_scope and source_scope != target_scope:
                    errors.append(f"{edge_prefix}: runtime edge cannot cross scopes {source_scope!r} -> {target_scope!r}")
                if isinstance(source_scope, str) and isinstance(target_scope, str) and source_scope and target_scope and source_scope != target_scope and kind not in {"invocation", "summary", "dependency", "adaptation", "weight_mapping", "parallel_partition"}:
                    errors.append(f"{edge_prefix}: cross-scope edge must use invocation or summary")
                if kind == "runtime" and (source_node.get("phase") == "construction" or target_node.get("phase") == "construction"):
                    errors.append(f"{edge_prefix}: construction phase nodes cannot use runtime edges")

        for node_id, node in node_by_id.items():
            if node.get("kind") != "decision":
                continue
            kinds = {edge.get("kind") for edge in outgoing.get(node_id, [])}
            if "conditional_true" not in kinds or "conditional_false" not in kinds:
                errors.append(f"{prefix}: decision node {node_id!r} requires true and false branches")

        if page_id == "overview":
            major_nodes = [node for node in node_by_id.values() if node.get("kind") not in NON_MAJOR_NODE_KINDS]
            if len(major_nodes) > 12:
                errors.append(f"{prefix}: overview contains {len(major_nodes)} major nodes; maximum is 12")

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
