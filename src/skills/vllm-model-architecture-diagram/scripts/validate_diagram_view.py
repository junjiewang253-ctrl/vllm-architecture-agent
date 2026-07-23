#!/usr/bin/env python3
"""Validate Diagram View against Architecture IR semantics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

VIEW_VERSION = "0.1"
IR_VERSION = "0.5"
REQUIRED_PAGE_IDS = {
    "overview",
    "decoder_layer_detail",
    "attention_detail",
    "moe_detail",
    "adapter_integration",
    "parallelism_weight_loading",
}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} root must be an object")
    return data


def _nodes_by_page(ir: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for page in ir.get("pages", []):
        if not isinstance(page, dict) or not isinstance(page.get("id"), str):
            continue
        result[page["id"]] = {
            node["id"]: node
            for node in page.get("nodes", [])
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }
    return result


def _edges_by_page(ir: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for page in ir.get("pages", []):
        if not isinstance(page, dict) or not isinstance(page.get("id"), str):
            continue
        result[page["id"]] = {
            edge["id"]: edge
            for edge in page.get("edges", [])
            if isinstance(edge, dict) and isinstance(edge.get("id"), str)
        }
    return result


def _port_ids(node: dict[str, Any]) -> set[str]:
    return {
        port["id"]
        for port in node.get("ports", [])
        if isinstance(port, dict) and isinstance(port.get("id"), str)
    }


def validate_diagram_view(ir: dict[str, Any], view: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if ir.get("schema_version") != IR_VERSION:
        errors.append(f"Architecture IR schema_version must be {IR_VERSION!r}")
    if view.get("schema_version") != VIEW_VERSION:
        errors.append(f"Diagram View schema_version must be {VIEW_VERSION!r}")
    if view.get("source_ir_version") != IR_VERSION:
        errors.append(f"Diagram View source_ir_version must be {IR_VERSION!r}")
    pages = view.get("pages")
    if not isinstance(pages, list) or not pages:
        errors.append("Diagram View pages must be a non-empty list")
        return errors

    view_page_ids = {page.get("id") for page in pages if isinstance(page, dict)}
    missing = sorted(REQUIRED_PAGE_IDS - {item for item in view_page_ids if isinstance(item, str)})
    if missing:
        errors.append(f"Diagram View missing required pages: {', '.join(missing)}")
    if "vllm_adaptation_map" in view_page_ids:
        errors.append("vllm_adaptation_map must be split into adapter_integration and parallelism_weight_loading")

    nodes_by_page = _nodes_by_page(ir)
    edges_by_page = _edges_by_page(ir)
    for page in pages:
        if not isinstance(page, dict):
            errors.append("Diagram View page must be an object")
            continue
        page_id = page.get("id")
        if not isinstance(page_id, str):
            errors.append("Diagram View page id must be a string")
            continue
        ir_nodes = nodes_by_page.get(page_id, {})
        ir_edges = edges_by_page.get(page_id, {})
        region_ids = {region.get("id") for region in page.get("regions", []) if isinstance(region, dict)}
        lane_ids = {lane.get("id") for lane in page.get("lanes", []) if isinstance(lane, dict)}
        seen_nodes: set[str] = set()
        for node in page.get("visible_nodes", []):
            if not isinstance(node, dict):
                errors.append(f"page {page_id}: visible node must be an object")
                continue
            semantic_id = node.get("semantic_id")
            if semantic_id not in ir_nodes:
                errors.append(f"page {page_id}: visible node references unknown semantic_id {semantic_id!r}")
                continue
            if semantic_id in seen_nodes:
                errors.append(f"page {page_id}: semantic node appears more than once: {semantic_id}")
            seen_nodes.add(str(semantic_id))
            region_id = node.get("region_id")
            if region_id is not None and region_id not in region_ids:
                errors.append(f"page {page_id}: node {semantic_id} references unknown region {region_id!r}")
            lane_id = node.get("lane_id")
            if lane_id is not None and lane_id not in lane_ids:
                errors.append(f"page {page_id}: node {semantic_id} references unknown lane {lane_id!r}")
        for lane in page.get("lanes", []):
            if not isinstance(lane, dict):
                continue
            for member_id in lane.get("member_ids", []):
                if member_id not in ir_nodes:
                    errors.append(f"page {page_id}: lane {lane.get('id')} references unknown member {member_id!r}")
        for edge in page.get("visible_edges", []):
            if not isinstance(edge, dict):
                errors.append(f"page {page_id}: visible edge must be an object")
                continue
            semantic_id = edge.get("semantic_id")
            ir_edge = ir_edges.get(semantic_id)
            if ir_edge is None:
                errors.append(f"page {page_id}: visible edge references unknown semantic_id {semantic_id!r}")
                continue
            if edge.get("source") != ir_edge.get("source") or edge.get("target") != ir_edge.get("target"):
                errors.append(f"page {page_id}: visible edge {semantic_id} changes source/target")
            source_node = ir_nodes.get(str(ir_edge.get("source")))
            target_node = ir_nodes.get(str(ir_edge.get("target")))
            if source_node and edge.get("source_port") not in _port_ids(source_node):
                errors.append(f"page {page_id}: edge {semantic_id} references invalid source_port {edge.get('source_port')!r}")
            if target_node and edge.get("target_port") not in _port_ids(target_node):
                errors.append(f"page {page_id}: edge {semantic_id} references invalid target_port {edge.get('target_port')!r}")
        _validate_page_semantics(page, errors)
    return errors


def _validate_page_semantics(page: dict[str, Any], errors: list[str]) -> None:
    page_id = page.get("id")
    node_ids = {node.get("semantic_id") for node in page.get("visible_nodes", []) if isinstance(node, dict)}
    edge_pairs = {
        (edge.get("source"), edge.get("target"))
        for edge in page.get("visible_edges", [])
        if isinstance(edge, dict)
    }
    if page_id == "decoder_layer_detail":
        forbidden = {"attention_residual", "ffn_residual"}
        if forbidden & node_ids:
            errors.append("decoder_layer_detail must not contain Residual Handoff semantic nodes")
        lane_ids = {lane.get("id") for lane in page.get("lanes", []) if isinstance(lane, dict)}
        if not {"hidden_states_lane", "residual_lane"}.issubset(lane_ids):
            errors.append("decoder_layer_detail must contain hidden/residual lanes")
    if page_id == "attention_detail":
        if ("hpc_fused_processing", "fallback_q_norm") in edge_pairs or ("fallback_q_norm", "hpc_fused_processing") in edge_pairs:
            errors.append("attention_detail must not serialize HPC and fallback paths")
        if ("v_stream", "fallback_q_norm") in edge_pairs or ("v_stream", "fallback_k_norm") in edge_pairs:
            errors.append("V stream must not enter Q/K norm")
    if page_id == "adapter_integration":
        if any("checkpoint" in str(node_id) or "mapping" in str(node_id) for node_id in node_ids):
            errors.append("adapter_integration must not contain checkpoint mapping nodes")
        if ("supports_pp", "supports_lora") in edge_pairs or ("supports_lora", "mixture_of_experts") in edge_pairs:
            errors.append("Adapter interfaces must not form a fake inheritance chain")
    if page_id == "parallelism_weight_loading":
        region_ids = {region.get("id") for region in page.get("regions", []) if isinstance(region, dict)}
        if not {"parallelism_region", "weight_loading_region"}.issubset(region_ids):
            errors.append("parallelism_weight_loading must contain Parallelism and Weight Loading regions")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Diagram View against Architecture IR.")
    parser.add_argument("ir", type=Path, help="Architecture IR JSON file")
    parser.add_argument("view", type=Path, help="Diagram View JSON file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ir = _load_json(args.ir, "Architecture IR")
        view = _load_json(args.view, "Diagram View")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    errors = validate_diagram_view(ir, view)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Diagram View validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
