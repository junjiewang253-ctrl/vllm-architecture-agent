#!/usr/bin/env python3
"""Validate strict Architecture Design 1.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

DESIGN_VERSION = "1.0"
VIEW_KINDS = {"pipeline", "block_with_residual", "branch_merge", "routed_container", "mapping_dispatch", "strategy_matrix", "boundary_map"}
NODE_KINDS = {"data", "component", "process", "storage", "external_component", "strategy", "mapping", "annotation", "merge", "boundary"}
NODE_ROLES = {"primary", "branch", "auxiliary", "contained", "strategy", "boundary", "annotation"}
PHASES = {"construction", "runtime", "checkpoint_loading", "parallel_partition", "external"}
EDGE_KINDS = {"tensor_flow", "residual_flow", "dependency", "containment", "mapping", "parallel_effect", "delegation", "external_boundary", "control"}
ROUTE_HINTS = {"direct", "upper_lane", "lower_lane", "branch_top", "branch_bottom", "bus", "local", "hidden"}
PORT_DIRECTIONS = {"input", "output", "bidirectional"}
DATA_KINDS = {"tensor", "weights", "config", "cache", "control", "capability"}

ROOT_KEYS = {"schema_version", "design_graph_type", "model_name", "source_sha256", "author", "pages", "external_boundaries", "coverage", "assumptions", "visual_patch"}
PAGE_KEYS = {"id", "title", "question", "purpose", "view_kind", "scope", "primary_story", "nodes", "edges", "groups", "annotations", "layout_constraints", "coverage"}
NODE_KEYS = {"id", "semantic_kind", "role", "label", "implementation", "phase", "parent_id", "concept_refs", "fact_refs", "ports", "display", "layout_hint"}
EDGE_KEYS = {"id", "kind", "source", "target", "source_port", "target_port", "phase", "concept_refs", "fact_refs", "display", "route_hint"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids(items: list[Any]) -> set[str]:
    return {item["id"] for item in items if isinstance(item, dict) and isinstance(item.get("id"), str)}


def _fact_types(fact_graph: dict[str, Any]) -> dict[str, str]:
    return {
        fact["id"]: str(fact.get("type"))
        for fact in fact_graph.get("facts", [])
        if isinstance(fact, dict) and isinstance(fact.get("id"), str)
    }


def validate_architecture_design(
    design: dict[str, Any],
    fact_graph: dict[str, Any],
    concept_graph: dict[str, Any],
    *,
    source_file: Path | None = None,
    allow_baseline_author: bool = False,
) -> list[str]:
    errors: list[str] = []
    _check_keys("design", design, ROOT_KEYS, errors)
    if design.get("schema_version") != DESIGN_VERSION or design.get("design_graph_type") != "architecture_design_graph":
        errors.append("Architecture Design must use schema_version '1.0' and design_graph_type 'architecture_design_graph'")
    if source_file is not None and source_file.exists() and design.get("source_sha256") != _sha256(source_file):
        errors.append("Architecture Design source_sha256 does not match source file")
    author = design.get("author")
    if not isinstance(author, dict):
        errors.append("author must be an object")
    else:
        author_type = author.get("type")
        if author_type != "agent" and not allow_baseline_author:
            errors.append("Architecture Design author.type must be 'agent' for finalize")
        if author_type not in {"agent", "baseline"}:
            errors.append("Architecture Design author.type must be 'agent' or 'baseline'")

    concept_ids = {
        concept["id"]
        for concept in concept_graph.get("concepts", [])
        if isinstance(concept, dict) and isinstance(concept.get("id"), str)
    }
    fact_types = _fact_types(fact_graph)
    fact_ids = set(fact_types)
    pages = design.get("pages")
    if not isinstance(pages, list) or not pages:
        errors.append("pages must be a non-empty list")
        return errors
    page_ids = set()
    for page in pages:
        if not isinstance(page, dict):
            errors.append("page must be an object")
            continue
        _validate_page(page, concept_ids, fact_ids, fact_types, errors)
        page_id = page.get("id")
        if isinstance(page_id, str):
            if page_id in page_ids:
                errors.append(f"duplicate page id: {page_id}")
            page_ids.add(page_id)
    _validate_golden_semantics(pages, errors)
    return errors


def _check_keys(label: str, value: dict[str, Any], allowed: set[str], errors: list[str]) -> None:
    extra = sorted(set(value) - allowed)
    if extra:
        errors.append(f"{label} has unknown properties: {', '.join(extra)}")


def _validate_page(
    page: dict[str, Any],
    concept_ids: set[str],
    fact_ids: set[str],
    fact_types: dict[str, str],
    errors: list[str],
) -> None:
    page_id = str(page.get("id"))
    _check_keys(f"page {page_id}", page, PAGE_KEYS, errors)
    view_kind = page.get("view_kind")
    if view_kind not in VIEW_KINDS:
        errors.append(f"page {page_id} has invalid view_kind {view_kind!r}")
    nodes = [node for node in page.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in page.get("edges", []) if isinstance(edge, dict)]
    node_ids = _ids(nodes)
    if len(node_ids) != len(nodes):
        errors.append(f"page {page_id} has duplicate or invalid node ids")
    edge_ids = _ids(edges)
    if len(edge_ids) != len(edges):
        errors.append(f"page {page_id} has duplicate or invalid edge ids")
    node_by_id = {node["id"]: node for node in nodes if isinstance(node.get("id"), str)}
    for node in nodes:
        _validate_node(page_id, node, node_ids, concept_ids, fact_ids, errors)
    for edge in edges:
        _validate_edge(page_id, edge, node_by_id, concept_ids, fact_ids, fact_types, errors)
    story = page.get("primary_story")
    if not isinstance(story, dict):
        errors.append(f"page {page_id} primary_story must be an object")
    else:
        ordered = story.get("ordered_node_ids")
        if view_kind in {"pipeline", "block_with_residual", "branch_merge", "routed_container", "mapping_dispatch"}:
            if not isinstance(ordered, list) or len(ordered) < 2:
                errors.append(f"page {page_id} must define a primary_story with at least two nodes")
            for node_id in ordered or []:
                if node_id not in node_ids:
                    errors.append(f"page {page_id} primary_story references unknown node {node_id!r}")
    if view_kind == "branch_merge":
        incoming: dict[str, int] = {}
        for edge in edges:
            incoming[str(edge.get("target"))] = incoming.get(str(edge.get("target")), 0) + 1
        if not any(count >= 2 for count in incoming.values()):
            errors.append(f"page {page_id} branch_merge must have a merge node with at least two incoming edges")
    if view_kind == "strategy_matrix":
        for edge in edges:
            if edge.get("kind") == "tensor_flow":
                errors.append(f"page {page_id} strategy_matrix must not contain tensor_flow edges")
    if view_kind == "boundary_map":
        group_labels = {str(group.get("label", "")).lower() for group in page.get("groups", []) if isinstance(group, dict)}
        if not any("local" in label for label in group_labels) or not any("external" in label for label in group_labels):
            errors.append(f"page {page_id} boundary_map must contain local and external groups")
    if view_kind == "mapping_dispatch":
        branch_count = sum(1 for edge in edges if edge.get("source") == "mapping_dispatch" and edge.get("kind") == "mapping")
        if branch_count < 2:
            errors.append(f"page {page_id} mapping_dispatch must have at least two mapping branches")
    if view_kind == "routed_container":
        container_members = [node for node in nodes if node.get("parent_id")]
        if not container_members:
            errors.append(f"page {page_id} routed_container must contain nodes with parent_id")


def _validate_node(
    page_id: str,
    node: dict[str, Any],
    node_ids: set[str],
    concept_ids: set[str],
    fact_ids: set[str],
    errors: list[str],
) -> None:
    node_id = str(node.get("id"))
    _check_keys(f"page {page_id} node {node_id}", node, NODE_KEYS, errors)
    if node.get("semantic_kind") not in NODE_KINDS:
        errors.append(f"page {page_id} node {node_id} has invalid semantic_kind")
    if node.get("role") not in NODE_ROLES:
        errors.append(f"page {page_id} node {node_id} has invalid role")
    if node.get("phase") not in PHASES:
        errors.append(f"page {page_id} node {node_id} has invalid phase")
    parent_id = node.get("parent_id")
    if isinstance(parent_id, str) and parent_id not in node_ids:
        errors.append(f"page {page_id} node {node_id} references missing parent_id {parent_id}")
    concept_refs = node.get("concept_refs")
    fact_refs = node.get("fact_refs")
    if node.get("semantic_kind") != "annotation":
        if not isinstance(concept_refs, list) or not concept_refs:
            errors.append(f"page {page_id} node {node_id} must have concept_refs")
        if not isinstance(fact_refs, list) or not fact_refs:
            errors.append(f"page {page_id} node {node_id} must have fact_refs")
    for ref in concept_refs or []:
        if ref not in concept_ids:
            errors.append(f"page {page_id} node {node_id} references missing concept {ref}")
    for ref in fact_refs or []:
        if ref not in fact_ids:
            errors.append(f"page {page_id} node {node_id} references missing fact {ref}")
    ports = node.get("ports")
    if not isinstance(ports, list):
        errors.append(f"page {page_id} node {node_id} ports must be a list")
        return
    if node.get("semantic_kind") not in {"strategy", "boundary", "annotation"} and not ports:
        errors.append(f"page {page_id} node {node_id} must have at least one port")
    seen_ports: set[str] = set()
    for port in ports:
        if not isinstance(port, dict):
            errors.append(f"page {page_id} node {node_id} port must be an object")
            continue
        port_id = port.get("id")
        if not isinstance(port_id, str) or port_id in seen_ports:
            errors.append(f"page {page_id} node {node_id} has invalid or duplicate port id")
        seen_ports.add(str(port_id))
        if port.get("direction") not in PORT_DIRECTIONS or port.get("data_kind") not in DATA_KINDS:
            errors.append(f"page {page_id} node {node_id} port {port_id} has invalid direction/data_kind")


def _validate_edge(
    page_id: str,
    edge: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    concept_ids: set[str],
    fact_ids: set[str],
    fact_types: dict[str, str],
    errors: list[str],
) -> None:
    edge_id = str(edge.get("id"))
    _check_keys(f"page {page_id} edge {edge_id}", edge, EDGE_KEYS, errors)
    if edge.get("kind") not in EDGE_KINDS:
        errors.append(f"page {page_id} edge {edge_id} has invalid kind")
    if edge.get("phase") not in PHASES:
        errors.append(f"page {page_id} edge {edge_id} has invalid phase")
    if edge.get("route_hint") not in ROUTE_HINTS:
        errors.append(f"page {page_id} edge {edge_id} has invalid route_hint")
    source = node_by_id.get(str(edge.get("source")))
    target = node_by_id.get(str(edge.get("target")))
    if source is None or target is None:
        errors.append(f"page {page_id} edge {edge_id} references unknown source/target")
        return
    source_port = _port_by_id(source, str(edge.get("source_port")))
    target_port = _port_by_id(target, str(edge.get("target_port")))
    if source_port is None or target_port is None:
        errors.append(f"page {page_id} edge {edge_id} references missing ports")
    else:
        if source_port.get("direction") == "input":
            errors.append(f"page {page_id} edge {edge_id} source_port is input-only")
        if target_port.get("direction") == "output":
            errors.append(f"page {page_id} edge {edge_id} target_port is output-only")
        if edge.get("kind") in {"tensor_flow", "residual_flow"} and (
            source_port.get("data_kind") not in {"tensor", "cache"} or target_port.get("data_kind") not in {"tensor", "cache"}
        ):
            errors.append(f"page {page_id} edge {edge_id} tensor/residual flow must use tensor/cache ports")
        if edge.get("kind") == "mapping" and (
            source_port.get("data_kind") != "weights" or target_port.get("data_kind") != "weights"
        ):
            errors.append(f"page {page_id} edge {edge_id} mapping must use weights ports")
    for ref in edge.get("concept_refs", []):
        if ref not in concept_ids:
            errors.append(f"page {page_id} edge {edge_id} references missing concept {ref}")
    fact_refs = edge.get("fact_refs")
    if not isinstance(fact_refs, list) or not fact_refs:
        errors.append(f"page {page_id} edge {edge_id} must have fact_refs")
    else:
        for ref in fact_refs:
            if ref not in fact_ids:
                errors.append(f"page {page_id} edge {edge_id} references missing fact {ref}")
        if edge.get("kind") in {"tensor_flow", "mapping", "delegation", "external_boundary"}:
            if all(fact_types.get(str(ref)) == "import" for ref in fact_refs):
                errors.append(f"page {page_id} edge {edge_id} cannot be supported only by import facts")
    if target.get("semantic_kind") == "external_component" and edge.get("phase") == "runtime" and edge.get("kind") != "tensor_flow":
        errors.append(f"page {page_id} edge {edge_id} external behavior must not be hidden as direct runtime dependency")


def _port_by_id(node: dict[str, Any], port_id: str) -> dict[str, Any] | None:
    for port in node.get("ports", []):
        if isinstance(port, dict) and port.get("id") == port_id:
            return port
    return None


def _validate_golden_semantics(pages: list[Any], errors: list[str]) -> None:
    page_by_id = {page.get("id"): page for page in pages if isinstance(page, dict)}
    attention = page_by_id.get("attention_adaptation")
    if isinstance(attention, dict):
        edges = {(edge.get("source"), edge.get("target"), edge.get("source_port")) for edge in attention.get("edges", []) if isinstance(edge, dict)}
        if ("qkv_projection", "hpc_rope_norm", "qkv_out") not in edges:
            errors.append("attention_adaptation: HPC input must branch from qkv_projection.qkv_out")
        bad_v_targets = {
            edge.get("target")
            for edge in attention.get("edges", [])
            if isinstance(edge, dict) and edge.get("source") == "v_stream"
        } & {"q_norm", "k_norm", "rope"}
        if bad_v_targets:
            errors.append("attention_adaptation: V stream must not pass through Q/K Norm or RoPE")
    moe = page_by_id.get("moe_execution")
    if isinstance(moe, dict):
        edge_pairs = {(edge.get("source"), edge.get("target")) for edge in moe.get("edges", []) if isinstance(edge, dict)}
        if ("flatten_tokens", "fused_moe") not in edge_pairs or ("router_logits", "fused_moe") not in edge_pairs:
            errors.append("moe_execution: hidden states and router logits must both enter FusedMoE")
        parent_by_id = {node.get("id"): node.get("parent_id") for node in moe.get("nodes", []) if isinstance(node, dict)}
        if parent_by_id.get("routed_experts") != "fused_moe":
            errors.append("moe_execution: Routed Experts must be contained by FusedMoE")
    checkpoint = page_by_id.get("checkpoint_weight_loading")
    if isinstance(checkpoint, dict):
        group_labels = {str(group.get("label")) for group in checkpoint.get("groups", []) if isinstance(group, dict)}
        if "HYV3ForCausalLM.load_weights" not in group_labels or "HYV3Model.load_weights" not in group_labels:
            errors.append("checkpoint_weight_loading: must contain wrapper and model load_weights lanes")
    parallel = page_by_id.get("parallel_strategies")
    if isinstance(parallel, dict):
        strategy_ids = {"tensor_parallel_panel", "pipeline_parallel_panel", "expert_parallel_panel"}
        for edge in parallel.get("edges", []):
            if isinstance(edge, dict) and edge.get("source") in strategy_ids and edge.get("target") in strategy_ids:
                errors.append("parallel_strategies: TP/PP/EP panels must not connect to each other")
    boundary = page_by_id.get("adapter_boundary")
    if isinstance(boundary, dict):
        generic_sources = {"adapter_boundary", "local_adapter"}
        for edge in boundary.get("edges", []):
            if isinstance(edge, dict) and edge.get("kind") == "delegation" and edge.get("source") in generic_sources:
                errors.append("adapter_boundary: delegation sources must be concrete class/method nodes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_design", type=Path)
    parser.add_argument("source_fact_graph", type=Path)
    parser.add_argument("architecture_concept", type=Path)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--allow-baseline-author", action="store_true")
    args = parser.parse_args(argv)
    try:
        errors = validate_architecture_design(
            _load_json(args.architecture_design),
            _load_json(args.source_fact_graph),
            _load_json(args.architecture_concept),
            source_file=args.source_file,
            allow_baseline_author=args.allow_baseline_author,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Architecture Design validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
