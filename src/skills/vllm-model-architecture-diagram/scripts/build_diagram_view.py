#!/usr/bin/env python3
"""Build a deterministic Diagram View from Architecture IR.

Diagram View is a display-planning layer. It keeps semantic node/edge IDs from
Architecture IR and adds regions, lanes, route classes, labels and preferred
sizes for layout without changing source semantics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DIAGRAM_VIEW_VERSION = "0.1"
SOURCE_IR_VERSION = "0.6"

PAGE_REGION_PRESETS: dict[str, list[tuple[str, str, str]]] = {
    "overview": [("overview_flow", "Top-level inference flow", "horizontal")],
    "decoder_detail": [
        ("decoder_main", "Hidden-state flow", "horizontal"),
        ("decoder_residual", "Residual-state flow", "horizontal"),
        ("decoder_variants", "Construction variants", "horizontal"),
    ],
    "attention_detail": [
        ("attention_entry", "QKV entry", "horizontal"),
        ("hpc_lane_region", "HPC fused lane", "horizontal"),
        ("fallback_lane_region", "Fallback Q/K/V lane", "horizontal"),
        ("attention_cache_region", "KV cache", "horizontal"),
        ("attention_exit", "Attention output", "horizontal"),
    ],
    "moe_detail": [
        ("moe_main", "Router and experts", "horizontal"),
        ("moe_aux", "Auxiliary inputs", "horizontal"),
    ],
    "adapter_integration": [
        ("external_inputs", "External Inputs", "vertical"),
        ("vllm_configuration", "vLLM Configuration", "vertical"),
        ("adapter_classes", "Adapter Classes", "vertical"),
        ("interfaces_compilation", "Interfaces & Compilation", "vertical"),
        ("execution_components", "Execution Components", "grid"),
    ],
    "parallelism": [
        ("tensor_parallel_region", "Tensor Parallel", "horizontal"),
        ("pipeline_parallel_region", "Pipeline Parallel", "horizontal"),
        ("expert_parallel_region", "Expert Parallel", "horizontal"),
    ],
    "weight_loading": [
        ("wrapper_weight_flow", "HYV3ForCausalLM.load_weights", "horizontal"),
        ("model_weight_flow", "HYV3Model.load_weights", "horizontal"),
        ("mapping_dispatch_region", "Mapping dispatch", "grid"),
    ],
}

PAGE_LANE_PRESETS: dict[str, list[tuple[str, str, str, list[str]]]] = {
    "decoder_detail": [
        ("hidden_states_lane", "hidden states", "horizontal", ["decoder_input", "input_rmsnorm", "self_attention", "post_attention_rmsnorm", "ffn_stage", "decoder_output"]),
        ("residual_lane", "residual state", "horizontal", ["decoder_input", "input_rmsnorm", "post_attention_rmsnorm", "decoder_output"]),
    ],
    "attention_detail": [
        ("hpc_fused_lane", "HPC fused", "horizontal", ["qkv_projection", "hpc_fused_processing", "kv_cache", "attention_core"]),
        ("fallback_lane", "fallback", "horizontal", ["q_stream", "k_stream", "v_stream", "fallback_q_norm", "fallback_k_norm", "rotary_embedding", "attention_core"]),
        ("q_lane", "Q", "horizontal", ["q_stream", "fallback_q_norm", "rotary_embedding", "attention_core"]),
        ("k_lane", "K", "horizontal", ["k_stream", "fallback_k_norm", "rotary_embedding", "attention_core"]),
        ("v_lane", "V", "horizontal", ["v_stream", "attention_core"]),
    ],
    "parallelism": [
        ("tp_lane", "Tensor Parallel", "horizontal", ["tp_lane", "tp_world_size", "tp_head_partition", "tp_kv_partition", "tp_embedding_linear_components"]),
        ("pp_lane", "Pipeline Parallel", "horizontal", ["pp_lane", "pp_make_layers", "pp_layer_range", "pp_missing_layer", "pp_rank_flows", "pp_missing_filter"]),
        ("ep_lane", "Expert Parallel", "horizontal", ["ep_lane", "ep_group", "ep_rank_size", "ep_physical_local_experts", "ep_fused_moe", "ep_eplb_update"]),
    ],
    "weight_loading": [
        ("wrapper_load_weights_lane", "HYV3ForCausalLM.load_weights", "horizontal", ["wrapper_weights", "wrapper_filter_weights", "wrapper_speculative_filter", "wrapper_tied_lm_head_filter", "wrapper_auto_weights_loader", "wrapper_loaded_set"]),
        ("model_load_weights_lane", "HYV3Model.load_weights", "horizontal", ["model_loaded_weight", "model_fp8_scale_remap", "mapping_dispatch", "pp_missing_parameter_filter", "router_gate_rename", "param_weight_loader", "default_weight_loader_fallback", "model_loaded_params"]),
    ],
}


def _display_label(node: dict[str, Any]) -> tuple[str, str | None]:
    display = node.get("display")
    label = display.get("label") if isinstance(display, dict) else None
    subtitle = display.get("subtitle") if isinstance(display, dict) else None
    return str(label or node.get("label", "")), subtitle if isinstance(subtitle, str) else None


def _measure(label: str, subtitle: str | None, badges: list[Any]) -> dict[str, float]:
    longest = max([len(part) for part in [label, subtitle or ""] if part] or [8])
    width = min(max(128.0, longest * 7.2 + 36.0 + len(badges) * 28.0), 260.0)
    lines = 1 + (1 if subtitle else 0)
    height = max(52.0, 30.0 + lines * 18.0)
    return {"width": round(width, 2), "height": round(height, 2)}


def _region_for_node(page_type: str, node_id: str) -> str | None:
    if page_type == "decoder_detail":
        if node_id in {"dense_ffn", "moe_ffn"}:
            return "decoder_variants"
        if node_id == "ffn_stage":
            return "decoder_main"
        return "decoder_main"
    if page_type == "attention_detail":
        if node_id in {"hpc_fused_processing"}:
            return "hpc_lane_region"
        if node_id in {"q_stream", "k_stream", "v_stream", "fallback_q_norm", "fallback_k_norm", "rotary_embedding"}:
            return "fallback_lane_region"
        if node_id == "kv_cache":
            return "attention_cache_region"
        if node_id in {"attention_core", "output_projection", "attention_output"}:
            return "attention_exit"
        return "attention_entry"
    if page_type == "moe_detail":
        if node_id in {"expert_bias", "eplb_metadata"}:
            return "moe_aux"
        return "moe_main"
    if page_type == "adapter_integration":
        if node_id.startswith("external_"):
            return "external_inputs"
        if node_id in {"vllm_config", "cache_config", "quantization_config", "parallel_config_eplb"}:
            return "vllm_configuration"
        if node_id in {"adapter_for_causal_lm", "adapter_model"}:
            return "adapter_classes"
        if node_id in {"supports_pp", "supports_lora", "mixture_of_experts", "support_torch_compile"}:
            return "interfaces_compilation"
        return "execution_components"
    if page_type == "parallelism":
        if node_id.startswith("tp_"):
            return "tensor_parallel_region"
        if node_id.startswith("pp_"):
            return "pipeline_parallel_region"
        return "expert_parallel_region"
    if page_type == "weight_loading":
        if node_id.startswith("wrapper_"):
            return "wrapper_weight_flow"
        if node_id in {"mapping_dispatch", "packed_modules_mapping", "stacked_params_mapping", "qkv_stacked_target", "gate_up_stacked_target", "expert_params_mapping", "expert_id_shard_id", "regular_parameter"}:
            return "mapping_dispatch_region"
        return "model_weight_flow"
    return "overview_flow"


def _lane_for_node(page_type: str, node_id: str) -> str | None:
    for lane_id, _label, _orientation, members in PAGE_LANE_PRESETS.get(page_type, []):
        if node_id in members:
            return lane_id
    return None


def _route_class(edge: dict[str, Any]) -> str:
    display = edge.get("display")
    if isinstance(display, dict) and display.get("visible") is False:
        return "hidden_semantic"
    route = display.get("route") if isinstance(display, dict) else None
    if isinstance(route, str) and route in {
        "horizontal_lane",
        "vertical_branch",
        "local_branch",
        "cache_write",
        "cache_read",
        "weight_mapping",
        "hidden_semantic",
    }:
        return route
    kind = edge.get("kind")
    if kind == "weight_mapping":
        return "weight_mapping"
    if kind == "residual":
        return "horizontal_lane"
    return "direct"


def _edge_label(edge: dict[str, Any]) -> tuple[str | None, bool]:
    display = edge.get("display")
    if isinstance(display, dict):
        label_visible = display.get("show_label") is True
        label = display.get("label")
        return (label if isinstance(label, str) else None), label_visible
    return None, False


def _visible_node(node: dict[str, Any], page_type: str) -> dict[str, Any]:
    label, subtitle = _display_label(node)
    badges = node.get("badges") if isinstance(node.get("badges"), list) else []
    return {
        "semantic_id": node["id"],
        "display_label": label,
        "display_subtitle": subtitle,
        "region_id": _region_for_node(page_type, str(node["id"])),
        "lane_id": _lane_for_node(page_type, str(node["id"])),
        "preferred_size": _measure(label, subtitle, badges),
        "ports": node.get("ports", []),
        "badges": badges,
        "kind": node.get("kind"),
        "parent_id": node.get("parent_id"),
    }


def _visible_edge(edge: dict[str, Any]) -> dict[str, Any]:
    label, label_visible = _edge_label(edge)
    return {
        "semantic_id": edge["id"],
        "source": edge["source"],
        "target": edge["target"],
        "source_port": edge.get("source_port"),
        "target_port": edge.get("target_port"),
        "style_kind": edge.get("kind"),
        "label": label,
        "label_visible": label_visible,
        "route_class": _route_class(edge),
        "bundle_id": edge.get("bundle_id"),
    }


def _regions(page: dict[str, Any]) -> list[dict[str, Any]]:
    page_type = str(page.get("page_type"))
    members_by_region: dict[str, list[str]] = {}
    for node in page.get("nodes", []):
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            continue
        region = _region_for_node(page_type, node["id"])
        if region:
            members_by_region.setdefault(region, []).append(node["id"])
    return [
        {"id": region_id, "label": label, "orientation": orientation, "member_ids": members_by_region.get(region_id, [])}
        for region_id, label, orientation in PAGE_REGION_PRESETS.get(page_type, [])
    ]


def _lanes(page: dict[str, Any]) -> list[dict[str, Any]]:
    page_type = str(page.get("page_type"))
    return [
        {"id": lane_id, "label": label, "orientation": orientation, "member_ids": members}
        for lane_id, label, orientation, members in PAGE_LANE_PRESETS.get(page_type, [])
    ]


def build_diagram_view(ir: dict[str, Any]) -> dict[str, Any]:
    if ir.get("schema_version") != SOURCE_IR_VERSION:
        raise ValueError(f"Architecture IR schema_version must be {SOURCE_IR_VERSION!r}")
    pages = ir.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Architecture IR pages must be a list")
    view_pages: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        nodes = [node for node in page.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)]
        edges = [edge for edge in page.get("edges", []) if isinstance(edge, dict) and isinstance(edge.get("id"), str)]
        view_pages.append(
            {
                "id": page["id"],
                "title": page["title"],
                "page_type": page["page_type"],
                "regions": _regions(page),
                "lanes": _lanes(page),
                "visible_nodes": [_visible_node(node, str(page["page_type"])) for node in nodes],
                "visible_edges": [_visible_edge(edge) for edge in edges],
                "bundles": [
                    {"id": "qkv_bundle", "member_ids": ["qkv_split_to_q_stream", "qkv_split_to_k_stream", "qkv_split_to_v_stream"]},
                    {"id": "weight_mapping_bundle", "member_ids": [edge["id"] for edge in edges if edge.get("kind") == "weight_mapping"]},
                ],
                "annotations": [],
                "layout_constraints": {"target_aspect_ratio": 1.777, "min_gap": 24, "max_edge_label_length": 32},
            }
        )
    return {
        "schema_version": DIAGRAM_VIEW_VERSION,
        "source_ir_version": SOURCE_IR_VERSION,
        "model_name": ir.get("model_name", "unknown-model"),
        "pages": view_pages,
        "style_profile": "v0.8-clean-routing",
        "unresolved": ir.get("unresolved", []),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Diagram View from Architecture IR.")
    parser.add_argument("input", type=Path, help="Architecture IR JSON file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output diagram-view JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists() or not args.input.is_file():
        print(f"error: file does not exist: {args.input}", file=sys.stderr)
        return 2
    try:
        ir = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: unable to read Architecture IR: {exc}", file=sys.stderr)
        return 2
    try:
        view = build_diagram_view(ir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: unable to build Diagram View: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Diagram View to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
