#!/usr/bin/env python3
"""Validate Architecture View Graph quality.

The View Graph is generated from an Architecture Design Graph, not directly
from concepts. These checks guard against regressions where pages become
concept-card summaries instead of readable architecture views.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

VIEW_VERSIONS = {"0.1", "1.0"}
NODE_TYPES = {"component", "data", "process", "storage", "boundary", "annotation", "capability", "group", "add", "container"}
EDGE_TYPES = {"runtime_flow", "dependency", "mapping", "parallel", "delegation", "boundary", "annotation", "containment"}
ROLE_TYPES = {"primary_flow", "secondary_flow", "boundary", "strategy", "annotation", "mapping"}
CORE_NODE_TYPES = {"component", "process", "storage", "boundary", "data", "container"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _nodes(page: dict[str, Any]) -> list[dict[str, Any]]:
    key = "visible_nodes" if "visible_nodes" in page else "nodes"
    return [node for node in page.get(key, []) if isinstance(node, dict)]


def _edges(page: dict[str, Any]) -> list[dict[str, Any]]:
    key = "visible_edges" if "visible_edges" in page else "edges"
    return [edge for edge in page.get(key, []) if isinstance(edge, dict)]


def _node_id(node: dict[str, Any]) -> Any:
    return node.get("semantic_id") or node.get("id")


def _edge_id(edge: dict[str, Any]) -> Any:
    return edge.get("semantic_id") or edge.get("id")


def _edge_type(edge: dict[str, Any]) -> str:
    return str(edge.get("edge_kind") or edge.get("type") or edge.get("style_kind") or "")


def validate_architecture_view(view: dict[str, Any], architecture_concept: dict[str, Any] | None = None, fact_graph: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    if view.get("schema_version") not in VIEW_VERSIONS or view.get("view_graph_type") != "architecture_view_graph":
        errors.append("Architecture View Graph must use schema_version '0.1'/'1.0' and view_graph_type 'architecture_view_graph'")
    concept_ids = {
        concept.get("id")
        for concept in (architecture_concept or {}).get("concepts", [])
        if isinstance(concept, dict)
    }
    fact_ids = {
        fact.get("id")
        for fact in (fact_graph or {}).get("facts", [])
        if isinstance(fact, dict)
    }
    pages = view.get("pages")
    if not isinstance(pages, list) or not pages:
        errors.append("Architecture View Graph pages must be a non-empty list")
        return errors
    for page in pages:
        if not isinstance(page, dict):
            errors.append("page must be an object")
            continue
        page_id = page.get("id")
        view_kind = str(page.get("view_kind") or page.get("page_type") or page_id or "")
        purpose = page.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            errors.append(f"page {page_id} must have purpose")
        nodes = _nodes(page)
        edges = _edges(page)
        if len(nodes) < 3:
            errors.append(f"page {page_id} must contain at least 3 nodes")
        node_ids = {_node_id(node) for node in nodes}
        roles = [str(node.get("role") or node.get("visual_role") or "") for node in nodes]
        if all(role == "annotation" for role in roles):
            errors.append(f"page {page_id} appears to contain only annotations")
        if all(str(node.get("visual_role", "")).startswith("concept") for node in nodes):
            errors.append(f"page {page_id} appears to contain only concept cards")
        primary_flow_count = sum(1 for role in roles if role == "primary_flow")
        if view_kind not in {"strategy_matrix", "boundary_map"} and primary_flow_count < 3:
            errors.append(f"page {page_id} must contain at least 3 primary_flow nodes")
        if edges and all(edge.get("type") == "annotation" for edge in edges):
            errors.append(f"page {page_id} must contain non-annotation architecture edges")
        connected: set[Any] = set()
        for node in nodes:
            node_id = _node_id(node)
            node_type = node.get("type") or node.get("kind")
            role = node.get("role") or node.get("visual_role")
            if node_type not in NODE_TYPES:
                errors.append(f"page {page_id} node {node_id} has invalid type {node_type!r}")
            if role not in ROLE_TYPES:
                errors.append(f"page {page_id} node {node_id} has invalid role {role!r}")
            concept_refs = node.get("concept_refs")
            fact_refs = node.get("fact_refs")
            if node_type in CORE_NODE_TYPES:
                if not isinstance(concept_refs, list) or not concept_refs:
                    errors.append(f"page {page_id} node {node_id} must have concept_refs")
                if not isinstance(fact_refs, list) or not fact_refs:
                    errors.append(f"page {page_id} node {node_id} must have fact_refs")
            if concept_ids:
                missing = [ref for ref in concept_refs or [] if ref not in concept_ids]
                if missing:
                    errors.append(f"page {page_id} node {node_id} references missing concepts")
            if fact_ids:
                missing_facts = [ref for ref in fact_refs or [] if ref not in fact_ids]
                if missing_facts:
                    errors.append(f"page {page_id} node {node_id} references missing facts")
        for edge in edges:
            edge_id = _edge_id(edge)
            edge_type = _edge_type(edge)
            source = edge.get("source")
            target = edge.get("target")
            if edge_type not in EDGE_TYPES and edge_type not in {"tensor_flow", "residual_flow", "parallel_effect", "external_boundary", "control", "weight_mapping", "runtime", "residual", "invocation", "parallel_partition"}:
                errors.append(f"page {page_id} edge {edge_id} has invalid type {edge_type!r}")
            if source not in node_ids or target not in node_ids:
                errors.append(f"page {page_id} edge {edge_id} references unknown node")
            connected.update([source, target])
            if edge_type in {"runtime_flow", "tensor_flow", "runtime"}:
                source_node = next((node for node in nodes if _node_id(node) == source), {})
                target_node = next((node for node in nodes if _node_id(node) == target), {})
                if (source_node.get("type") or source_node.get("kind")) == "boundary" and (target_node.get("type") or target_node.get("kind")) != "boundary":
                    errors.append(f"page {page_id} edge {edge_id} cannot be external boundary -> runtime_flow")
            if edge_type != "annotation" and (not edge.get("fact_refs") or not edge.get("concept_refs")):
                errors.append(f"page {page_id} edge {edge_id} must have concept_refs and fact_refs")
        for node in nodes:
            if (node.get("type") or node.get("kind")) in {"component", "process"} and _node_id(node) not in connected:
                errors.append(f"page {page_id} component/process node {_node_id(node)} is isolated")
        if page_id in {"model_overview", "attention", "moe", "model_execution_overview", "decoder_block", "attention_adaptation", "moe_execution"}:
            has_data = any((node.get("type") or node.get("kind")) == "data" for node in nodes)
            has_component = any((node.get("type") or node.get("kind")) == "component" for node in nodes)
            has_runtime = any(_edge_type(edge) in {"runtime_flow", "tensor_flow", "runtime"} for edge in edges)
            if not (has_data and has_component and has_runtime):
                errors.append(f"page {page_id} must contain data node, component node, and runtime_flow edge")
        if page_id in {"attention", "attention_adaptation"}:
            labels = {str(node.get("label")) for node in nodes}
            required = {"QKV Projection", "Q", "K", "V"}
            if page_id == "attention":
                required |= {"Q/K/V Split", "HPC Fused Processing", "Optional QK Norm", "KV Cache Boundary", "vLLM Attention Backend"}
            else:
                required |= {"Split Q / K / V", "HPC Fused Processing", "KV Cache", "vLLM Attention Backend"}
            missing = sorted(required - labels)
            if missing:
                errors.append(f"attention page missing required nodes: {', '.join(missing)}")
        if page_id in {"moe", "moe_execution"}:
            labels = {str(node.get("label")) for node in nodes}
            required = {"FusedMoE"}
            if page_id == "moe":
                required |= {"Router", "Top-K Selection", "Experts", "Shared Experts", "Expert Parallel"}
            else:
                required |= {"GateLinear", "Router Logits", "Routed Experts", "Optional Shared Experts"}
            missing = sorted(required - labels)
            if missing:
                errors.append(f"moe page missing required nodes: {', '.join(missing)}")
        if page_id in {"checkpoint", "checkpoint_weight_loading"}:
            labels = {str(node.get("label")) for node in nodes}
            required = {"Weight Name Processing"} if page_id == "checkpoint" else {"Weights", "_filter_weights", "AutoWeightsLoader", "Mapping Dispatch", "loaded_params"}
            missing = sorted(required - labels)
            if missing:
                errors.append(f"checkpoint page missing required nodes: {', '.join(missing)}")
        if page_id in {"vllm_boundary", "adapter_boundary"}:
            labels = {str(node.get("label")) for node in nodes}
            group_labels = {str(group.get("label")) for group in page.get("regions", []) if isinstance(group, dict)}
            required = {"Adapter Boundary"} if page_id == "adapter_boundary" else {"Local Adapter", "Adapter Boundary", "External vLLM Runtime"}
            missing = sorted(item for item in required if item not in labels and item not in group_labels)
            if missing:
                errors.append(f"vllm_boundary page missing local/external boundary nodes: {', '.join(missing)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_view", type=Path)
    parser.add_argument("--architecture-concept", type=Path)
    parser.add_argument("--source-fact-graph", type=Path)
    args = parser.parse_args(argv)
    try:
        errors = validate_architecture_view(
            _load_json(args.architecture_view),
            _load_json(args.architecture_concept) if args.architecture_concept else None,
            _load_json(args.source_fact_graph) if args.source_fact_graph else None,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Architecture View validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
