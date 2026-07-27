#!/usr/bin/env python3
"""Build deterministic layout plans with semantic port anchors.

The router is intentionally small: page-specific placements keep the diagrams
readable, while a deterministic Manhattan fallback produces orthogonal points
from source port anchors to target port anchors.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LAYOUT_PLAN_VERSION = "0.1"
VIEW_VERSION = "0.1"
ARCHITECTURE_VIEW_VERSION = "1.0"


@dataclass(frozen=True)
class LayoutBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class PortAnchor:
    node_id: str
    port_id: str
    x: float
    y: float
    side: str


@dataclass(frozen=True)
class RoutedEdge:
    edge_id: str
    points: list[tuple[float, float]]


def _box(x: float, y: float, width: float, height: float) -> LayoutBox:
    return LayoutBox(x, y, width, height)


def _dynamic_boxes(page: dict[str, Any]) -> dict[str, LayoutBox]:
    """Lay out concept-driven pages without adding new page-specific presets."""

    nodes = [node for node in page.get("visible_nodes", []) if isinstance(node, dict)]
    count = max(1, len(nodes))
    width, height = page.get("layout_constraints", {}).get("page_size", [1280, 720])
    layout_positions = [
        node.get("layout")
        for node in nodes
        if isinstance(node.get("layout"), dict) and "row" in node.get("layout", {}) and "col" in node.get("layout", {})
    ]
    if layout_positions:
        max_col = max(int(pos.get("col", 0)) for pos in layout_positions)
        max_row = max(int(pos.get("row", 0)) for pos in layout_positions)
        columns = max_col + 1
        rows = max_row + 1
    else:
        columns = 3 if count > 4 else max(1, count)
        rows = (count + columns - 1) // columns
    usable_width = max(600.0, float(width) - 160.0)
    usable_height = max(360.0, float(height) - 160.0)
    cell_w = usable_width / columns
    cell_h = usable_height / max(1, rows)
    boxes: dict[str, LayoutBox] = {}
    for index, node in enumerate(nodes):
        node_id = str(node.get("semantic_id"))
        preferred = node.get("preferred_size") if isinstance(node.get("preferred_size"), dict) else {}
        box_w = float(preferred.get("width", min(260.0, cell_w - 36.0)))
        box_h = float(preferred.get("height", 86.0))
        position = node.get("layout") if isinstance(node.get("layout"), dict) else {}
        col = int(position.get("col", index % columns))
        row = int(position.get("row", index // columns))
        x = 80.0 + col * cell_w + max(0.0, (cell_w - box_w) / 2)
        y = 110.0 + row * cell_h + max(0.0, (cell_h - box_h) / 2)
        boxes[node_id] = _box(round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2))
    return boxes


def _edge_style(edge_type: str) -> str:
    return {
        "runtime_flow": "runtime",
        "dependency": "dependency",
        "mapping": "weight_mapping",
        "parallel": "parallel_partition",
        "delegation": "invocation",
        "boundary": "dependency",
        "annotation": "dependency",
    }.get(edge_type, "runtime")


def _route_class(edge_type: str) -> str:
    return {
        "runtime_flow": "horizontal_lane",
        "dependency": "horizontal_lane",
        "mapping": "weight_mapping",
        "parallel": "horizontal_lane",
        "delegation": "horizontal_lane",
        "boundary": "horizontal_lane",
        "annotation": "hidden_semantic",
    }.get(edge_type, "horizontal_lane")


def _normalize_architecture_view_graph(view: dict[str, Any]) -> dict[str, Any]:
    if view.get("view_graph_type") != "architecture_view_graph":
        return view
    if view.get("schema_version") == ARCHITECTURE_VIEW_VERSION:
        pages = []
        for page in view.get("pages", []):
            if not isinstance(page, dict):
                continue
            normalized = dict(page)
            normalized["page_type"] = page.get("view_kind", page.get("id"))
            normalized.setdefault("regions", page.get("regions", []))
            normalized.setdefault("lanes", page.get("lanes", []))
            normalized.setdefault("annotations", page.get("decorations", []))
            normalized.setdefault("layout_constraints", page.get("layout_constraints", {"page_size": [1600, 900]}))
            pages.append(normalized)
        return {
            "schema_version": VIEW_VERSION,
            "source_architecture_view_version": view.get("schema_version"),
            "model_name": view.get("model_name", "unknown-model"),
            "pages": pages,
        }
    pages: list[dict[str, Any]] = []
    for page in view.get("pages", []):
        if not isinstance(page, dict):
            continue
        visible_nodes: list[dict[str, Any]] = []
        for node in page.get("nodes", []):
            if not isinstance(node, dict) or not isinstance(node.get("id"), str):
                continue
            visible_nodes.append(
                {
                    "semantic_id": node["id"],
                    "display_label": node.get("label", node["id"]),
                    "display_subtitle": node.get("subtitle"),
                    "region_id": None,
                    "lane_id": None,
                    "kind": node.get("type", "component"),
                    "preferred_size": node.get("preferred_size", {"width": 170, "height": 74}),
                    "ports": node.get("ports", []),
                    "badges": node.get("badges", []),
                    "layout": node.get("layout", {}),
                    "visual_role": node.get("visual_role"),
                }
            )
        visible_edges: list[dict[str, Any]] = []
        for edge in page.get("edges", []):
            if not isinstance(edge, dict) or not isinstance(edge.get("id"), str):
                continue
            edge_type = str(edge.get("type") or "runtime_flow")
            visible_edges.append(
                {
                    "semantic_id": edge["id"],
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "source_port": edge.get("source_port", "out"),
                    "target_port": edge.get("target_port", "in"),
                    "style_kind": _edge_style(edge_type),
                    "label": edge.get("label", ""),
                    "label_visible": edge.get("show_label") is True,
                    "route_class": _route_class(edge_type),
                    "bundle_id": None,
                }
            )
        pages.append(
            {
                "id": page.get("id"),
                "title": page.get("title"),
                "page_type": page.get("id"),
                "purpose": page.get("purpose"),
                "visible_nodes": visible_nodes,
                "visible_edges": visible_edges,
                "regions": page.get("groups", []),
                "lanes": page.get("lanes", []),
                "annotations": [
                    {
                        "id": f"decorative_purpose_{page.get('id')}",
                        "text": page.get("purpose", ""),
                        "x": 760,
                        "y": 16,
                        "width": 420,
                        "height": 48,
                    }
                ],
                "layout_constraints": {"page_size": [1400, 780]},
            }
        )
    return {
        "schema_version": VIEW_VERSION,
        "source_architecture_view_version": view.get("schema_version"),
        "model_name": view.get("model_name", "unknown-model"),
        "pages": pages,
    }


PAGE_SIZES: dict[str, tuple[int, int]] = {
    "overview": (1280, 720),
    "decoder_detail": (1280, 720),
    "attention_detail": (1500, 820),
    "moe_detail": (1280, 720),
    "adapter_integration": (1600, 900),
    "parallelism": (1500, 840),
    "weight_loading": (1600, 900),
}


LAYOUTS: dict[str, dict[str, LayoutBox]] = {
    "overview": {
        "hyv3_for_causal_lm": _box(40, 58, 1160, 320),
        "hyv3_model": _box(70, 132, 850, 168),
        "input": _box(96, 192, 110, 54),
        "vocab_parallel_embedding": _box(235, 186, 150, 66),
        "hyv3_decoder_layer": _box(418, 176, 190, 88),
        "final_residual_add": _box(615, 187, 140, 64),
        "norm": _box(775, 190, 120, 60),
        "hidden_states_output": _box(902, 190, 128, 60),
        "lm_head": _box(1030, 105, 138, 58),
        "logits_processor": _box(1050, 205, 130, 62),
        "logits_output": _box(1050, 310, 115, 54),
    },
    "decoder_layer_detail": {
        "decoder_input": _box(60, 160, 145, 64),
        "residual_initialization": _box(82, 310, 150, 58),
        "input_rmsnorm": _box(245, 150, 150, 78),
        "self_attention": _box(435, 150, 155, 78),
        "post_attention_rmsnorm": _box(630, 150, 176, 78),
        "ffn_stage": _box(850, 116, 250, 160),
        "dense_ffn": _box(870, 178, 92, 56),
        "moe_ffn": _box(982, 178, 96, 56),
        "decoder_output": _box(1140, 160, 120, 64),
    },
    "attention_detail": {
        "attention_input": _box(45, 360, 120, 56),
        "qkv_projection": _box(210, 350, 150, 70),
        "qkv_split": _box(410, 356, 125, 60),
        "hpc_fused_processing": _box(610, 180, 230, 76),
        "kv_cache": _box(900, 145, 125, 66),
        "q_stream": _box(590, 380, 72, 46),
        "k_stream": _box(590, 475, 72, 46),
        "v_stream": _box(590, 570, 72, 46),
        "fallback_q_norm": _box(720, 372, 118, 62),
        "fallback_k_norm": _box(720, 467, 118, 62),
        "rotary_embedding": _box(900, 412, 142, 78),
        "attention_core": _box(1110, 360, 182, 76),
        "output_projection": _box(1320, 362, 145, 70),
        "attention_output": _box(1330, 500, 125, 58),
    },
    "moe_detail": {
        "moe_input": _box(50, 220, 120, 56),
        "reshape_tokens": _box(205, 220, 130, 56),
        "gate_linear": _box(370, 210, 120, 68),
        "router_logits": _box(525, 214, 125, 60),
        "fused_moe": _box(705, 150, 350, 210),
        "routed_experts": _box(735, 220, 110, 58),
        "shared_experts": _box(850, 220, 172, 58),
        "expert_bias": _box(520, 86, 120, 54),
        "eplb_metadata": _box(690, 86, 130, 54),
        "restore_original_shape": _box(1065, 220, 165, 58),
        "moe_output": _box(1110, 330, 115, 54),
    },
    "adapter_integration": {
        "external_hf_config": _box(65, 210, 135, 58),
        "vllm_config": _box(300, 180, 130, 58),
        "cache_config": _box(510, 95, 130, 54),
        "quantization_config": _box(510, 180, 150, 54),
        "parallel_config_eplb": _box(500, 265, 195, 54),
        "adapter_for_causal_lm": _box(760, 140, 185, 80),
        "adapter_model": _box(760, 295, 185, 80),
        "supports_pp": _box(1010, 95, 118, 48),
        "supports_lora": _box(1010, 155, 118, 48),
        "mixture_of_experts": _box(1010, 300, 150, 48),
        "support_torch_compile": _box(1010, 360, 170, 48),
        "exec_input_group": _box(1190, 70, 210, 112),
        "component_vocab_embedding": _box(1210, 122, 165, 44),
        "exec_attention_group": _box(1190, 200, 210, 225),
        "component_qkv_linear": _box(1210, 250, 160, 38),
        "component_hpc_rope_norm": _box(1210, 294, 160, 38),
        "component_attention": _box(1210, 338, 160, 38),
        "component_row_parallel": _box(1210, 382, 160, 38),
        "exec_ffn_moe_group": _box(1190, 430, 210, 230),
        "component_merged_linear": _box(1210, 480, 185, 38),
        "component_silu_and_mul": _box(1210, 524, 150, 38),
        "component_gate_linear": _box(1210, 568, 150, 38),
        "component_fused_moe": _box(1210, 612, 150, 38),
        "exec_output_group": _box(1190, 685, 210, 150),
        "component_lm_head": _box(1210, 735, 150, 38),
        "component_logits_processor": _box(1210, 779, 160, 38),
    },
    "parallelism": {
        "tp_lane": _box(50, 80, 1380, 160),
        "tp_world_size": _box(80, 150, 135, 48),
        "tp_head_partition": _box(250, 150, 145, 48),
        "tp_kv_partition": _box(430, 150, 150, 48),
        "tp_embedding_linear_components": _box(620, 145, 245, 58),
        "pp_lane": _box(50, 310, 1380, 180),
        "pp_make_layers": _box(80, 390, 120, 48),
        "pp_layer_range": _box(235, 390, 135, 48),
        "pp_missing_layer": _box(405, 390, 140, 48),
        "pp_rank_flows": _box(580, 385, 220, 58),
        "pp_missing_filter": _box(835, 390, 155, 48),
        "ep_lane": _box(50, 560, 1380, 180),
        "ep_group": _box(80, 640, 120, 48),
        "ep_rank_size": _box(235, 640, 135, 48),
        "ep_physical_local_experts": _box(405, 635, 175, 58),
        "ep_fused_moe": _box(620, 635, 140, 58),
        "ep_eplb_update": _box(800, 640, 180, 48),
    },
    "weight_loading": {
        "wrapper_weights": _box(60, 105, 120, 50),
        "wrapper_filter_weights": _box(220, 100, 135, 60),
        "wrapper_speculative_filter": _box(395, 100, 150, 60),
        "wrapper_tied_lm_head_filter": _box(585, 100, 155, 60),
        "wrapper_auto_weights_loader": _box(780, 100, 165, 60),
        "wrapper_loaded_set": _box(985, 105, 120, 50),
        "model_loaded_weight": _box(60, 375, 135, 50),
        "model_fp8_scale_remap": _box(235, 370, 145, 60),
        "mapping_dispatch": _box(430, 250, 520, 335),
        "packed_modules_mapping": _box(455, 310, 150, 46),
        "stacked_params_mapping": _box(455, 370, 150, 46),
        "qkv_stacked_target": _box(630, 345, 145, 46),
        "gate_up_stacked_target": _box(630, 405, 170, 46),
        "expert_params_mapping": _box(455, 485, 170, 46),
        "expert_id_shard_id": _box(650, 485, 150, 46),
        "regular_parameter": _box(815, 405, 130, 46),
        "bias_skip": _box(1020, 300, 105, 48),
        "pp_missing_parameter_filter": _box(1020, 380, 150, 54),
        "router_gate_rename": _box(1210, 380, 145, 54),
        "param_weight_loader": _box(1210, 475, 145, 54),
        "default_weight_loader_fallback": _box(1390, 475, 170, 54),
        "model_loaded_params": _box(1390, 590, 125, 50),
    },
}


def _preferred(node: dict[str, Any], default: tuple[float, float] = (180.0, 74.0)) -> tuple[float, float]:
    preferred = node.get("preferred_size") if isinstance(node.get("preferred_size"), dict) else {}
    return float(preferred.get("width", default[0])), float(preferred.get("height", default[1]))


def _node_order(node: dict[str, Any]) -> int:
    layout = node.get("layout") if isinstance(node.get("layout"), dict) else {}
    return int(layout.get("order", 0))


def _node_role(node: dict[str, Any]) -> str:
    return str(node.get("role") or node.get("visual_role") or "")


def _place(boxes: dict[str, LayoutBox], node: dict[str, Any], x: float, y: float, default: tuple[float, float] = (180.0, 74.0)) -> None:
    node_id = str(node["semantic_id"])
    width, height = _preferred(node, default)
    boxes[node_id] = _box(round(x, 2), round(y, 2), round(width, 2), round(height, 2))


def _story_nodes(page: dict[str, Any]) -> list[str]:
    story = page.get("primary_story") if isinstance(page.get("primary_story"), dict) else {}
    ordered = story.get("ordered_node_ids")
    return [str(item) for item in ordered] if isinstance(ordered, list) else []


def layout_pipeline(page: dict[str, Any]) -> dict[str, LayoutBox]:
    nodes = {str(node["semantic_id"]): node for node in page.get("visible_nodes", []) if isinstance(node, dict) and isinstance(node.get("semantic_id"), str)}
    story = [node_id for node_id in _story_nodes(page) if node_id in nodes]
    boxes: dict[str, LayoutBox] = {}
    x = 50.0
    y = 300.0
    gap = 20.0
    for node_id in story:
        node = nodes[node_id]
        _place(boxes, node, x, y)
        x += boxes[node_id].width + gap
    auxiliaries = [node for node in nodes.values() if node["semantic_id"] not in boxes]
    top_x = 120.0
    bottom_x = 560.0
    for index, node in enumerate(sorted(auxiliaries, key=_node_order)):
        role = _node_role(node)
        if role in {"auxiliary", "strategy"}:
            _place(boxes, node, top_x + index * 260.0, 160.0, (190, 62))
        else:
            _place(boxes, node, bottom_x + index * 260.0, 470.0, (190, 62))
    return boxes


def layout_block_with_residual(page: dict[str, Any]) -> dict[str, LayoutBox]:
    nodes = {str(node["semantic_id"]): node for node in page.get("visible_nodes", []) if isinstance(node, dict) and isinstance(node.get("semantic_id"), str)}
    story = [node_id for node_id in _story_nodes(page) if node_id in nodes]
    boxes: dict[str, LayoutBox] = {}
    x = 70.0
    for node_id in story:
        node = nodes[node_id]
        default = (230.0, 90.0) if node_id == "ffn_stage" else (170.0, 78.0)
        _place(boxes, node, x, 210.0, default)
        x += boxes[node_id].width + 42.0
    if "residual_init" in nodes:
        _place(boxes, nodes["residual_init"], 95.0, 405.0, (180, 58))
    if "dense_ffn" in nodes and "ffn_stage" in boxes:
        parent = boxes["ffn_stage"]
        _place(boxes, nodes["dense_ffn"], parent.x + 24.0, parent.y + 108.0, (120, 56))
    if "moe_ffn" in nodes and "ffn_stage" in boxes:
        parent = boxes["ffn_stage"]
        _place(boxes, nodes["moe_ffn"], parent.x + 200.0, parent.y + 108.0, (120, 56))
    for node in sorted(nodes.values(), key=_node_order):
        if str(node["semantic_id"]) not in boxes:
            _place(boxes, node, 80.0 + _node_order(node) * 160.0, 520.0)
    return boxes


def layout_branch_merge(page: dict[str, Any]) -> dict[str, LayoutBox]:
    nodes = {str(node["semantic_id"]): node for node in page.get("visible_nodes", []) if isinstance(node, dict) and isinstance(node.get("semantic_id"), str)}
    boxes: dict[str, LayoutBox] = {}
    placements = {
        "attention_input": (60, 390),
        "qkv_projection": (260, 380),
        "hpc_rope_norm": (520, 170),
        "qkv_split": (520, 390),
        "q_stream": (740, 340),
        "k_stream": (740, 430),
        "v_stream": (740, 520),
        "q_norm": (880, 330),
        "k_norm": (880, 430),
        "rope": (1070, 380),
        "kv_cache": (1100, 190),
        "attention_backend": (1270, 365),
        "output_projection": (1510, 382),
        "attention_output": (1510, 515),
    }
    for node_id, node in nodes.items():
        x, y = placements.get(node_id, (80.0 + _node_order(node) * 120.0, 610.0))
        default = (210.0, 78.0) if node_id in {"hpc_rope_norm", "attention_backend"} else (150.0, 66.0)
        _place(boxes, node, float(x), float(y), default)
    return boxes


def layout_routed_container(page: dict[str, Any]) -> dict[str, LayoutBox]:
    nodes = {str(node["semantic_id"]): node for node in page.get("visible_nodes", []) if isinstance(node, dict) and isinstance(node.get("semantic_id"), str)}
    boxes: dict[str, LayoutBox] = {}
    placements = {
        "moe_input": (70, 320),
        "flatten_tokens": (280, 320),
        "gate_linear": (500, 230),
        "router_logits": (710, 230),
        "fused_moe": (900, 230),
        "restore_shape": (1280, 320),
        "moe_output": (1510, 320),
        "expert_bias": (610, 100),
        "eplb_metadata": (800, 100),
    }
    for node_id, node in nodes.items():
        if node_id == "fused_moe":
            _place(boxes, node, 900, 220, (390, 220))
        elif node_id == "routed_experts":
            _place(boxes, node, 930, 320, (120, 56))
        elif node_id == "shared_experts":
            _place(boxes, node, 1100, 320, (150, 56))
        else:
            x, y = placements.get(node_id, (80.0 + _node_order(node) * 140.0, 520.0))
            _place(boxes, node, float(x), float(y))
    return boxes


def layout_mapping_dispatch(page: dict[str, Any]) -> dict[str, LayoutBox]:
    nodes = {str(node["semantic_id"]): node for node in page.get("visible_nodes", []) if isinstance(node, dict) and isinstance(node.get("semantic_id"), str)}
    boxes: dict[str, LayoutBox] = {}
    placements = {
        "wrapper_weights": (70, 130),
        "wrapper_filter": (280, 120),
        "speculative_filter": (500, 120),
        "tied_lm_head_filter": (740, 120),
        "auto_weights_loader": (1000, 120),
        "wrapper_loaded_set": (1260, 130),
        "model_loaded_weight": (70, 430),
        "fp8_scale_remap": (260, 420),
        "mapping_dispatch": (470, 375),
        "stacked_mapping": (760, 290),
        "qkv_proj": (980, 250),
        "gate_up_proj": (980, 335),
        "expert_mapping": (760, 470),
        "fused_moe_params": (980, 470),
        "regular_parameter": (760, 610),
        "pp_missing_filter": (560, 610),
        "param_weight_loader": (1010, 610),
        "default_loader": (1210, 610),
        "model_loaded_params": (1420, 430),
    }
    for node_id, node in nodes.items():
        x, y = placements.get(node_id, (80.0 + _node_order(node) * 120.0, 760.0))
        default = (240, 110) if node_id == "mapping_dispatch" else (170, 62)
        _place(boxes, node, float(x), float(y), default)
    return boxes


def layout_strategy_matrix(page: dict[str, Any]) -> dict[str, LayoutBox]:
    nodes = {str(node["semantic_id"]): node for node in page.get("visible_nodes", []) if isinstance(node, dict) and isinstance(node.get("semantic_id"), str)}
    boxes: dict[str, LayoutBox] = {}
    panels = [
        ("tensor_parallel_panel", "tp_components", 80.0),
        ("pipeline_parallel_panel", "pp_components", 560.0),
        ("expert_parallel_panel", "ep_components", 1040.0),
    ]
    for panel_id, annotation_id, x in panels:
        if panel_id in nodes:
            _place(boxes, nodes[panel_id], x, 160.0, (360, 170))
        if annotation_id in nodes:
            _place(boxes, nodes[annotation_id], x + 20.0, 380.0, (320, 150))
    for node in sorted(nodes.values(), key=_node_order):
        if str(node["semantic_id"]) not in boxes:
            _place(boxes, node, 80.0 + _node_order(node) * 180.0, 620.0)
    return boxes


def layout_boundary_map(page: dict[str, Any]) -> dict[str, LayoutBox]:
    nodes = {str(node["semantic_id"]): node for node in page.get("visible_nodes", []) if isinstance(node, dict) and isinstance(node.get("semantic_id"), str)}
    boxes: dict[str, LayoutBox] = {}
    local = ["hyv3_for_causal_lm", "hyv3_model", "hyv3_attention", "hyv3_moe", "local_weight_mapping", "local_parallel_logic", "capabilities"]
    external = ["external_attention", "external_hpc_rope", "external_fused_moe", "external_auto_loader", "external_param_loaders"]
    for index, node_id in enumerate(local):
        if node_id in nodes:
            _place(boxes, nodes[node_id], 90.0, 130.0 + index * 95.0, (240, 66))
    if "adapter_boundary" in nodes:
        _place(boxes, nodes["adapter_boundary"], 690.0, 310.0, (200, 110))
    for index, node_id in enumerate(external):
        if node_id in nodes:
            _place(boxes, nodes[node_id], 1110.0, 150.0 + index * 105.0, (250, 70))
    return boxes


def _pattern_boxes(page: dict[str, Any]) -> dict[str, LayoutBox] | None:
    page_type = str(page.get("page_type") or page.get("view_kind") or "")
    return {
        "pipeline": layout_pipeline,
        "block_with_residual": layout_block_with_residual,
        "branch_merge": layout_branch_merge,
        "routed_container": layout_routed_container,
        "mapping_dispatch": layout_mapping_dispatch,
        "strategy_matrix": layout_strategy_matrix,
        "boundary_map": layout_boundary_map,
    }.get(page_type, lambda _page: None)(page)


def _side_for_port(port_id: str, direction: str) -> str:
    if port_id in {"residual", "updated_residual", "residual_out"}:
        return "bottom" if direction == "output" else "left"
    if port_id in {"write", "weights_in", "config_in", "capability_in", "control_in"}:
        return "left"
    if direction == "input":
        return "left"
    return "right"


def _anchor(box: LayoutBox, node_id: str, port: dict[str, Any]) -> PortAnchor:
    port_id = str(port["id"])
    direction = str(port.get("direction", "output"))
    if node_id == "qkv_split" and port_id in {"q", "k", "v"}:
        offsets = {"q": 16.0, "k": 30.0, "v": 44.0}
        return PortAnchor(node_id, port_id, box.right, box.y + offsets[port_id], "right")
    if node_id == "attention_core" and port_id in {"q", "k", "v", "kv_cache"}:
        offsets = {"q": 16.0, "k": 30.0, "v": 44.0, "kv_cache": 58.0}
        return PortAnchor(node_id, port_id, box.x, box.y + offsets[port_id], "left")
    if node_id == "rotary_embedding" and port_id in {"q_in", "k_in", "q_out", "k_out"}:
        offsets = {"q_in": 24.0, "k_in": 54.0, "q_out": 24.0, "k_out": 54.0}
        side = "left" if port_id.endswith("_in") else "right"
        x = box.x if side == "left" else box.right
        return PortAnchor(node_id, port_id, x, box.y + offsets[port_id], side)
    side = _side_for_port(port_id, direction)
    if side == "left":
        return PortAnchor(node_id, port_id, box.x, box.y + box.height / 2, side)
    if side == "right":
        return PortAnchor(node_id, port_id, box.right, box.y + box.height / 2, side)
    if side == "top":
        return PortAnchor(node_id, port_id, box.x + box.width / 2, box.y, side)
    return PortAnchor(node_id, port_id, box.x + box.width / 2, box.bottom, side)


def _anchors(page: dict[str, Any], boxes: dict[str, LayoutBox]) -> dict[str, dict[str, PortAnchor]]:
    result: dict[str, dict[str, PortAnchor]] = {}
    for node in page.get("visible_nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("semantic_id"))
        box = boxes.get(node_id)
        if box is None:
            continue
        result[node_id] = {
            str(port["id"]): _anchor(box, node_id, port)
            for port in node.get("ports", [])
            if isinstance(port, dict) and isinstance(port.get("id"), str)
        }
    return result


def _route(source: PortAnchor, target: PortAnchor, route_class: str, index: int) -> list[tuple[float, float]]:
    sx, sy, tx, ty = source.x, source.y, target.x, target.y
    if route_class == "hidden_semantic":
        return [(sx, sy), (tx, ty)]
    if source.port_id in {"residual", "updated_residual"} or target.port_id == "residual":
        track_y = max(sy, ty) + 118
        return [(sx, sy), (sx, track_y), (tx, track_y), (tx, ty)]
    if route_class in {"horizontal_lane", "weight_mapping"}:
        mid = sx + (tx - sx) / 2
        return [(sx, sy), (mid, sy), (mid, ty), (tx, ty)]
    if route_class == "vertical_branch":
        mid_y = sy + (ty - sy) / 2
        return [(sx, sy), (sx, mid_y), (tx, mid_y), (tx, ty)]
    if route_class == "cache_write":
        track_y = min(sy, ty) - 28 - index * 8
        return [(sx, sy), (sx, track_y), (tx, track_y), (tx, ty)]
    if route_class == "cache_read":
        track_y = min(sy, ty) - 54 - index * 8
        return [(sx, sy), (sx, track_y), (tx, track_y), (tx, ty)]
    if route_class == "local_branch":
        track_x = sx + 34 + index * 10
        return [(sx, sy), (track_x, sy), (track_x, ty), (tx, ty)]
    mid = sx + (tx - sx) / 2
    return [(sx, sy), (mid, sy), (mid, ty), (tx, ty)]


def _segments(points: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return list(zip(points, points[1:]))


def _segment_intersects_box(segment: tuple[tuple[float, float], tuple[float, float]], box: LayoutBox) -> bool:
    (x1, y1), (x2, y2) = segment
    if abs(y1 - y2) < 0.001:
        if y1 <= box.y or y1 >= box.bottom:
            return False
        return max(min(x1, x2), box.x) < min(max(x1, x2), box.right)
    if abs(x1 - x2) < 0.001:
        if x1 <= box.x or x1 >= box.right:
            return False
        return max(min(y1, y2), box.y) < min(max(y1, y2), box.bottom)
    return False


def _path_clear(points: list[tuple[float, float]], boxes: dict[str, LayoutBox], skip: set[str]) -> bool:
    for segment in _segments(points):
        for node_id, box in boxes.items():
            if node_id in skip:
                continue
            if _segment_intersects_box(segment, box):
                return False
    return True


def _reroute_around_nodes(
    source: PortAnchor,
    target: PortAnchor,
    route_class: str,
    index: int,
    boxes: dict[str, LayoutBox],
) -> list[tuple[float, float]]:
    primary = _route(source, target, route_class, index)
    skip = {source.node_id, target.node_id}
    if _path_clear(primary, boxes, skip):
        return primary
    sx, sy, tx, ty = source.x, source.y, target.x, target.y
    candidates: list[list[tuple[float, float]]] = []
    top_track = min(sy, ty) - 42 - index * 6
    bottom_track = max(sy, ty) + 42 + index * 6
    mid_x = sx + (tx - sx) / 2
    candidates.append([(sx, sy), (sx, top_track), (tx, top_track), (tx, ty)])
    candidates.append([(sx, sy), (sx, bottom_track), (tx, bottom_track), (tx, ty)])
    candidates.append([(sx, sy), (mid_x, sy), (mid_x, ty), (tx, ty)])
    side_track = max(sx, tx) + 36 + index * 6
    candidates.append([(sx, sy), (side_track, sy), (side_track, ty), (tx, ty)])
    for candidate in candidates:
        if _path_clear(candidate, boxes, skip):
            return candidate
    return primary


def _clean_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for x, y in points:
        point = (round(x, 2), round(y, 2))
        if not cleaned or cleaned[-1] != point:
            cleaned.append(point)
    return cleaned


def _regions(page: dict[str, Any], boxes: dict[str, LayoutBox]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for region in page.get("regions", []):
        if not isinstance(region, dict):
            continue
        member_boxes = [boxes[item] for item in region.get("member_ids", []) if item in boxes]
        if not member_boxes:
            continue
        x = min(box.x for box in member_boxes) - 20
        y = min(box.y for box in member_boxes) - 48
        right = max(box.right for box in member_boxes) + 20
        bottom = max(box.bottom for box in member_boxes) + 24
        result.append(
            {
                "id": region["id"],
                "label": region["label"],
                "x": round(x, 2),
                "y": round(y, 2),
                "width": round(right - x, 2),
                "height": round(bottom - y, 2),
                "member_ids": region.get("member_ids", []),
            }
        )
    return result


def _layout_page(page: dict[str, Any]) -> dict[str, Any]:
    page_id = str(page["id"])
    page_type = str(page["page_type"])
    visible_node_ids = {
        str(node.get("semantic_id"))
        for node in page.get("visible_nodes", [])
        if isinstance(node, dict) and isinstance(node.get("semantic_id"), str)
    }
    pattern_boxes = _pattern_boxes(page)
    boxes = dict(pattern_boxes or LAYOUTS.get(page_id, {}))
    if not boxes or not visible_node_ids.issubset(set(boxes)):
        dynamic = _dynamic_boxes(page)
        boxes.update({node_id: box for node_id, box in dynamic.items() if node_id not in boxes})
    anchors = _anchors(page, boxes)
    obstacle_ids = {
        str(node.get("semantic_id"))
        for node in page.get("visible_nodes", [])
        if isinstance(node, dict) and node.get("kind") not in {"container", "repeated_block"}
    }
    obstacle_boxes = {node_id: box for node_id, box in boxes.items() if node_id in obstacle_ids}
    routed_edges: dict[str, dict[str, Any]] = {}
    for index, edge in enumerate(page.get("visible_edges", [])):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        source_port = str(edge.get("source_port"))
        target_port = str(edge.get("target_port"))
        source_anchor = anchors.get(source, {}).get(source_port)
        target_anchor = anchors.get(target, {}).get(target_port)
        if source_anchor is None or target_anchor is None:
            raise ValueError(f"edge {edge.get('semantic_id')} references an unlayoutable port")
        route_class = str(edge.get("route_class") or "direct")
        points = _reroute_around_nodes(source_anchor, target_anchor, route_class, index, obstacle_boxes)
        points = _clean_points(points)
        routed_edges[str(edge["semantic_id"])] = {
            "points": [[x, y] for x, y in points],
            "visible": route_class != "hidden_semantic" and edge.get("visible") is not False,
            "label": edge.get("label"),
            "label_visible": edge.get("label_visible") is True,
            "route_class": route_class,
            "style_kind": edge.get("style_kind"),
        }
    page_size = page.get("layout_constraints", {}).get("page_size")
    if isinstance(page_size, list) and len(page_size) == 2:
        width, height = int(page_size[0]), int(page_size[1])
    else:
        width, height = PAGE_SIZES.get(page_id, (1280, 720))
    return {
        "id": page_id,
        "title": page["title"],
        "page_type": page_type,
        "width": width,
        "height": height,
        "nodes": {
            node_id: {
                "x": box.x,
                "y": box.y,
                "width": box.width,
                "height": box.height,
                "parent_id": next((node.get("parent_id") for node in page.get("visible_nodes", []) if node.get("semantic_id") == node_id), None),
            }
            for node_id, box in boxes.items()
        },
        "ports": {
            node_id: {
                port_id: {"x": anchor.x, "y": anchor.y, "side": anchor.side}
                for port_id, anchor in port_map.items()
            }
            for node_id, port_map in anchors.items()
        },
        "edges": routed_edges,
        "regions": _regions(page, boxes),
        "decorations": [],
    }


def build_layout_plan(view: dict[str, Any]) -> dict[str, Any]:
    view = _normalize_architecture_view_graph(view)
    if view.get("schema_version") not in {VIEW_VERSION, ARCHITECTURE_VIEW_VERSION}:
        raise ValueError(f"View schema_version must be {VIEW_VERSION!r} or {ARCHITECTURE_VIEW_VERSION!r}")
    pages = view.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Diagram View pages must be a non-empty list")
    return {
        "schema_version": LAYOUT_PLAN_VERSION,
        "source_view_version": view.get("schema_version"),
        "model_name": view.get("model_name", "unknown-model"),
        "pages": [_layout_page(page) for page in pages if isinstance(page, dict)],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic layout plan from Diagram View.")
    parser.add_argument("input", type=Path, help="Diagram View JSON file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output layout-plan JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists() or not args.input.is_file():
        print(f"error: file does not exist: {args.input}", file=sys.stderr)
        return 2
    try:
        view = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: unable to read Diagram View: {exc}", file=sys.stderr)
        return 2
    try:
        plan = build_layout_plan(view)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: unable to build layout plan: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Layout Plan to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
