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


PAGE_SIZES: dict[str, tuple[int, int]] = {
    "overview": (1280, 720),
    "decoder_detail": (1280, 720),
    "attention_detail": (1500, 820),
    "moe_detail": (1280, 720),
    "adapter_integration": (1440, 810),
    "parallelism_weight_loading": (1500, 900),
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
        "moe_output": _box(1085, 222, 115, 54),
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
        "exec_input_group": _box(1215, 105, 145, 56),
        "exec_attention_group": _box(1215, 190, 145, 56),
        "exec_ffn_moe_group": _box(1215, 275, 145, 56),
        "exec_output_group": _box(1215, 360, 145, 56),
    },
    "parallelism_weight_loading": {
        "tensor_parallel_lane": _box(55, 120, 170, 54),
        "tp_components": _box(285, 120, 210, 54),
        "pipeline_parallel_lane": _box(55, 205, 170, 54),
        "pp_components": _box(285, 205, 230, 54),
        "expert_parallel_lane": _box(55, 290, 170, 54),
        "ep_components": _box(285, 290, 210, 54),
        "hf_checkpoint": _box(55, 560, 145, 58),
        "qkv_checkpoint_weights": _box(285, 470, 185, 54),
        "qkv_proj_mapping": _box(535, 470, 145, 54),
        "gate_up_checkpoint_weights": _box(285, 555, 185, 54),
        "gate_up_proj_mapping": _box(535, 555, 145, 54),
        "expert_checkpoint_weights": _box(285, 640, 185, 54),
        "fused_moe_parameter_mapping": _box(535, 640, 185, 54),
        "remaining_weights": _box(770, 555, 145, 54),
        "auto_weights_loader": _box(1010, 555, 160, 54),
        "pp_missing_filter": _box(770, 470, 170, 54),
        "fp8_kv_scale_remap": _box(770, 640, 170, 54),
    },
}


def _side_for_port(port_id: str, direction: str) -> str:
    if port_id in {"residual", "updated_residual"}:
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


def _manual_route(page_id: str, edge_id: str) -> list[tuple[float, float]] | None:
    """Return hand-tuned routes for dense local structures.

    These overrides keep the router deterministic while avoiding a few known
    local crossings that a generic Manhattan fallback cannot infer from the
    semantic view alone.
    """
    attention_routes: dict[str, list[tuple[float, float]]] = {
        "qkv_split_to_hpc_fused": [(535, 372), (600, 372), (600, 218), (610, 218)],
        "qkv_split_to_q_stream": [(535, 372), (590, 372), (590, 403)],
        "qkv_split_to_k_stream": [(535, 386), (590, 386), (590, 498)],
        "qkv_split_to_v_stream": [(535, 400), (590, 400), (590, 593)],
        "q_norm_to_rotary": [(838, 403), (870, 403), (870, 436), (900, 436)],
        "k_norm_to_rotary": [(838, 498), (870, 498), (870, 466), (900, 466)],
        "rotary_q_to_attention": [(1042, 436), (1072, 436), (1072, 376), (1110, 376)],
        "rotary_k_to_attention": [(1042, 466), (1080, 466), (1080, 390), (1110, 390)],
        "v_stream_to_attention": [(662, 593), (1090, 593), (1090, 404), (1110, 404)],
    }
    parallel_routes: dict[str, list[tuple[float, float]]] = {
        "checkpoint_to_remaining_weights": [(200, 589), (240, 589), (240, 735), (770, 735), (770, 582)],
    }
    if page_id == "attention_detail":
        return attention_routes.get(edge_id)
    if page_id == "parallelism_weight_loading":
        return parallel_routes.get(edge_id)
    return None


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
    boxes = dict(LAYOUTS.get(page_id, {}))
    if not boxes:
        raise ValueError(f"no layout preset for page {page_id!r}")
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
        points = _manual_route(page_id, str(edge["semantic_id"])) or points
        points = _clean_points(points)
        routed_edges[str(edge["semantic_id"])] = {
            "points": [[x, y] for x, y in points],
            "visible": route_class != "hidden_semantic",
            "label": edge.get("label"),
            "label_visible": edge.get("label_visible") is True,
            "route_class": route_class,
            "style_kind": edge.get("style_kind"),
        }
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
    if view.get("schema_version") != VIEW_VERSION:
        raise ValueError(f"Diagram View schema_version must be {VIEW_VERSION!r}")
    pages = view.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Diagram View pages must be a non-empty list")
    return {
        "schema_version": LAYOUT_PLAN_VERSION,
        "source_view_version": VIEW_VERSION,
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
