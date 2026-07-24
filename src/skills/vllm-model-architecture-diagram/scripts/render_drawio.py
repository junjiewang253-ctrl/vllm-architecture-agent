#!/usr/bin/env python3
"""Render Diagram View or Architecture View to deterministic Draw.io XML."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_PARENT_ID = "1"
TITLE_HEIGHT = 48.0
PADDING = 24.0
GAP = 28.0
SIDE_GAP = 40.0

NODE_STYLES: dict[str, str] = {
    "container": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=16;"
        "fontStyle=1;align=left;verticalAlign=top;spacingLeft=16;spacingTop=12;"
        "fillColor=#F6F8FB;strokeColor=#8091A7;strokeWidth=2;"
        "container=1;recursiveResize=0;collapsible=0;"
    ),
    "input": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;"
        "fillColor=#EAF3FF;strokeColor=#4F83C4;arcSize=12;"
    ),
    "embedding": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;"
        "fillColor=#E9F8EF;strokeColor=#4D9B65;arcSize=12;"
    ),
    "repeated_block": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=15;"
        "fontStyle=1;align=left;verticalAlign=top;spacingLeft=14;spacingTop=12;"
        "fillColor=#FFF7E5;strokeColor=#C9932D;strokeWidth=2;"
        "container=1;recursiveResize=0;collapsible=0;dashed=1;"
    ),
    "attention": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;"
        "fillColor=#F1ECFF;strokeColor=#7963B3;arcSize=12;"
    ),
    "normalization": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;"
        "fillColor=#EEF2F5;strokeColor=#65758B;arcSize=12;"
    ),
    "add": (
        "ellipse;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;"
        "fillColor=#FFEDED;strokeColor=#C65B5B;"
    ),
    "head": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;"
        "fillColor=#EFFFFB;strokeColor=#399885;arcSize=12;"
    ),
    "logits_processor": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;"
        "fillColor=#F4F1FF;strokeColor=#7563B8;arcSize=12;"
    ),
    "output": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;"
        "fillColor=#EAF3FF;strokeColor=#4F83C4;arcSize=12;"
    ),
    "ffn": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;"
        "fillColor=#FFF2E8;strokeColor=#C9773A;arcSize=12;"
    ),
    "moe": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;"
        "fillColor=#FFF0F5;strokeColor=#C35A7D;arcSize=12;"
    ),
    "note": (
        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=12;"
        "fillColor=#FFFFEE;strokeColor=#B8AA5A;dashed=1;arcSize=10;"
    ),
}

EDGE_STYLES: dict[str, str] = {
    "runtime": (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=0;fontFamily=Inter;fontSize=12;endArrow=classic;strokeColor=#334155;"
    ),
    "residual": (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=0;fontFamily=Inter;fontSize=12;endArrow=classic;strokeColor=#B45309;"
        "strokeWidth=2;"
    ),
    "dependency": (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=0;fontFamily=Inter;fontSize=12;endArrow=open;dashed=1;"
        "strokeColor=#64748B;"
    ),
    "invocation": (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=0;fontFamily=Inter;fontSize=12;endArrow=classic;dashed=1;"
        "strokeColor=#475569;"
    ),
    "summary": (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=0;fontFamily=Inter;fontSize=12;endArrow=classic;strokeWidth=2;"
        "strokeColor=#0F766E;"
    ),
    "containment": (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=0;fontFamily=Inter;fontSize=11;endArrow=open;dashed=1;"
        "strokeColor=#CBD5E1;opacity=35;"
    ),
    "weight_mapping": (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=0;fontFamily=Inter;fontSize=12;endArrow=open;dashPattern=8 4 1 4;"
        "strokeColor=#7C3AED;"
    ),
    "conditional_true": (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=0;fontFamily=Inter;fontSize=12;endArrow=classic;strokeColor=#15803D;"
    ),
    "conditional_false": (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=0;fontFamily=Inter;fontSize=12;endArrow=classic;strokeColor=#BE123C;"
    ),
    "adaptation": (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=0;fontFamily=Inter;fontSize=11;endArrow=classic;dashed=1;"
        "strokeColor=#2563EB;"
    ),
    "parallel_partition": (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=0;fontFamily=Inter;fontSize=11;endArrow=classic;strokeWidth=2;"
        "strokeColor=#0E7490;"
    ),
}


@dataclass(frozen=True)
class LayoutBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Decoration:
    cell_id: str
    parent: str
    value: str
    style: str
    box: LayoutBox


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 0.001:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _node_label(node: dict[str, Any]) -> str:
    if node.get("kind") in {"container", "repeated_block"}:
        return ""
    display = node.get("display")
    label: Any = display.get("label") if isinstance(display, dict) else None
    subtitle: Any = display.get("subtitle") if isinstance(display, dict) else None
    if not isinstance(label, str) or not label.strip():
        label = str(node.get("label", ""))
    if not isinstance(subtitle, str):
        subtitle = node.get("subtitle")
    if node.get("kind") == "repeated_block" and not (isinstance(display, dict) and display.get("label")):
        repetition = node.get("repetition")
        if isinstance(repetition, dict):
            count = repetition.get("count_expression")
            if isinstance(count, str) and count.strip():
                label = f"{count} x {label}"
            start = repetition.get("local_start")
            end = repetition.get("local_end")
            if isinstance(start, str) and isinstance(end, str):
                subtitle = f"local range: {start} -> {end}"
    if isinstance(subtitle, str) and subtitle.strip():
        return f"{label}\n{subtitle}"
    return label


def _leaf_size(node: dict[str, Any]) -> tuple[float, float]:
    kind = node.get("kind")
    if kind == "add":
        return 96.0, 72.0
    if kind == "input":
        return 160.0, 74.0
    if kind == "embedding":
        return 190.0, 78.0
    if kind == "attention":
        return 190.0, 78.0
    if kind == "head":
        return 180.0, 76.0
    if kind == "logits_processor":
        return 200.0, 76.0
    if kind == "normalization":
        return 150.0, 74.0
    return 170.0, 74.0


def _variant_lines(node: dict[str, Any]) -> list[str]:
    variants = node.get("variants")
    if not isinstance(variants, list) or not variants:
        return []
    lines = ["Layer construction"] if node.get("id") == "ffn_stage" else ["Layer composition"]
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        component = str(variant.get("component", "")).strip()
        if not component:
            continue
        if "FeedForward" in component or "FFN" in component:
            lines.append("First K layers: Dense FFN")
        elif "MoE" in component:
            lines.append("Remaining layers: MoE")
        else:
            lines.append(component)
    return lines


def _variant_box(node: dict[str, Any]) -> LayoutBox | None:
    lines = _variant_lines(node)
    if not lines:
        return None
    width = max(300.0, max(len(line) for line in lines) * 6.8 + 28.0)
    height = 34.0 + max(0, len(lines) - 1) * 22.0
    return LayoutBox(PADDING, 0.0, width, height)


def _ordered_children(
    node_id: str | None,
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        node
        for node in nodes
        if node.get("parent_id") == node_id
    ]


def _measure_node(
    node: dict[str, Any],
    nodes: list[dict[str, Any]],
    boxes: dict[str, LayoutBox],
    decorations: list[Decoration],
) -> LayoutBox:
    node_id = str(node["id"])
    kind = node.get("kind")
    children = _ordered_children(node_id, nodes)

    if kind not in {"container", "repeated_block"}:
        width, height = _leaf_size(node)
        box = LayoutBox(0.0, 0.0, width, height)
        boxes[node_id] = box
        return box

    child_boxes = [(child, _measure_node(child, nodes, boxes, decorations)) for child in children]
    if kind == "repeated_block":
        box = _layout_repeated_block(node, child_boxes, boxes, decorations)
    else:
        box = _layout_container(node, child_boxes, boxes)
    boxes[node_id] = box
    return box


def _layout_repeated_block(
    node: dict[str, Any],
    child_boxes: list[tuple[dict[str, Any], LayoutBox]],
    boxes: dict[str, LayoutBox],
    decorations: list[Decoration],
) -> LayoutBox:
    node_id = str(node["id"])
    y = TITLE_HEIGHT + PADDING
    max_width = 300.0
    for child, child_box in child_boxes:
        child_id = str(child["id"])
        placed = LayoutBox(PADDING, y, child_box.width, child_box.height)
        boxes[child_id] = placed
        y += child_box.height + GAP
        max_width = max(max_width, child_box.width)

    variant_box = _variant_box(node)
    if variant_box:
        placed_variant = LayoutBox(PADDING, y, variant_box.width, variant_box.height)
        decorations.append(
            Decoration(
                cell_id=f"decorative_variants_{node_id}",
                parent=node_id,
                value="\n".join(_variant_lines(node)),
                style=(
                    "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=12;"
                    "align=left;verticalAlign=top;spacing=10;fillColor=#FFFBEB;"
                    "strokeColor=#D6A63B;dashed=1;"
                ),
                box=placed_variant,
            )
        )
        y += variant_box.height + PADDING
        max_width = max(max_width, variant_box.width)
    else:
        y += PADDING

    width = max_width + PADDING * 2
    height = max(y, 230.0)
    return LayoutBox(0.0, 0.0, width, height)


def _layout_container(
    node: dict[str, Any],
    child_boxes: list[tuple[dict[str, Any], LayoutBox]],
    boxes: dict[str, LayoutBox],
) -> LayoutBox:
    node_id = str(node["id"])
    container_children = [
        (child, box)
        for child, box in child_boxes
        if child.get("kind") in {"container", "repeated_block"}
    ]
    leaf_children = [
        (child, box)
        for child, box in child_boxes
        if child.get("kind") not in {"container", "repeated_block"}
    ]

    if container_children and leaf_children and node.get("parent_id") is None:
        return _layout_root_container(node_id, container_children, leaf_children, boxes)
    return _layout_horizontal_container(node_id, child_boxes, boxes)


def _layout_root_container(
    node_id: str,
    container_children: list[tuple[dict[str, Any], LayoutBox]],
    leaf_children: list[tuple[dict[str, Any], LayoutBox]],
    boxes: dict[str, LayoutBox],
) -> LayoutBox:
    x = PADDING
    y = TITLE_HEIGHT + PADDING
    max_height = 0.0
    total_width = 0.0

    for child, child_box in container_children:
        child_id = str(child["id"])
        boxes[child_id] = LayoutBox(x, y, child_box.width, child_box.height)
        x += child_box.width + SIDE_GAP
        total_width = x
        max_height = max(max_height, child_box.height)

    side_width = max((box.width for _, box in leaf_children), default=0.0)
    side_y = y
    for child, child_box in leaf_children:
        child_id = str(child["id"])
        boxes[child_id] = LayoutBox(x, side_y, child_box.width, child_box.height)
        side_y += child_box.height + GAP
    side_height = side_y - y - GAP if leaf_children else 0.0
    max_height = max(max_height, side_height)
    total_width += side_width

    width = max(total_width + PADDING, 640.0)
    height = TITLE_HEIGHT + PADDING + max_height + PADDING
    return LayoutBox(40.0, 40.0, width, height)


def _layout_horizontal_container(
    node_id: str,
    child_boxes: list[tuple[dict[str, Any], LayoutBox]],
    boxes: dict[str, LayoutBox],
) -> LayoutBox:
    x = PADDING
    y = TITLE_HEIGHT + PADDING
    max_height = 0.0
    for child, child_box in child_boxes:
        child_id = str(child["id"])
        boxes[child_id] = LayoutBox(x, y, child_box.width, child_box.height)
        x += child_box.width + GAP
        max_height = max(max_height, child_box.height)
    content_width = x - GAP if child_boxes else 300.0
    width = max(content_width + PADDING, 360.0)
    height = max(TITLE_HEIGHT + PADDING + max_height + PADDING, 180.0)
    return LayoutBox(0.0, 0.0, width, height)


def compute_layout(page: dict[str, Any]) -> tuple[dict[str, LayoutBox], list[Decoration]]:
    nodes = page.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("page.nodes must be a list")
    boxes: dict[str, LayoutBox] = {}
    decorations: list[Decoration] = []
    page_type = page.get("page_type") or page.get("id")
    if page_type == "overview":
        return _layout_overview(page, boxes, decorations)
    if page_type == "decoder_detail" or page.get("id") == "decoder_layer_detail":
        return _layout_decoder_detail(page, boxes, decorations)
    if page_type == "attention_detail":
        return _layout_attention_detail(page, boxes, decorations)
    if page_type == "adaptation_map":
        return _layout_adaptation_map(page, boxes, decorations)
    roots = _ordered_children(None, nodes)
    if not roots:
        raise ValueError("page must contain at least one root node")
    x = 40.0
    for root in roots:
        measured = _measure_node(root, nodes, boxes, decorations)
        root_id = str(root["id"])
        boxes[root_id] = LayoutBox(x, 40.0, measured.width, measured.height)
        x += measured.width + SIDE_GAP
    return boxes, decorations


def _layout_decoder_detail(
    page: dict[str, Any],
    boxes: dict[str, LayoutBox],
    decorations: list[Decoration],
) -> tuple[dict[str, LayoutBox], list[Decoration]]:
    nodes = page.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("page.nodes must be a list")
    node_by_id = {
        str(node["id"]): node
        for node in nodes
        if isinstance(node, dict) and "id" in node
    }
    sizes = {
        node_id: LayoutBox(0.0, 0.0, *_leaf_size(node))
        for node_id, node in node_by_id.items()
        if node.get("kind") not in {"container", "repeated_block"}
    }

    for node_id, x in zip(
        ["decoder_input", "input_layernorm", "self_attention", "attention_residual"],
        [40.0, 250.0, 470.0, 690.0],
    ):
        if node_id in sizes:
            size = sizes[node_id]
            boxes[node_id] = LayoutBox(x, 84.0, size.width, size.height)

    if "ffn_stage" in node_by_id:
        boxes["ffn_stage"] = LayoutBox(250.0, 284.0, 448.0, 220.0)
        boxes["dense_ffn"] = LayoutBox(24.0, 72.0, 180.0, 76.0)
        boxes["moe_ffn"] = LayoutBox(228.0, 72.0, 188.0, 76.0)
        decorations.append(
            Decoration(
                cell_id="decorative_variants_ffn_stage",
                parent="ffn_stage",
                value="\n".join(_variant_lines(node_by_id["ffn_stage"])),
                style=(
                    "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=12;"
                    "align=left;verticalAlign=top;spacing=8;fillColor=#FFFBEB;"
                    "strokeColor=#D6A63B;dashed=1;"
                ),
                box=LayoutBox(24.0, 156.0, 392.0, 48.0),
            )
        )

    for node_id, x in {
        "post_attention_layernorm": 40.0,
        "ffn_residual": 740.0,
        "decoder_output": 950.0,
    }.items():
        if node_id in sizes:
            size = sizes[node_id]
            boxes[node_id] = LayoutBox(x, 300.0, size.width, size.height)

    for node_id, size in sizes.items():
        if node_id not in boxes:
            boxes[node_id] = size
    return boxes, decorations


def _layout_overview(
    page: dict[str, Any],
    boxes: dict[str, LayoutBox],
    decorations: list[Decoration],
) -> tuple[dict[str, LayoutBox], list[Decoration]]:
    boxes.update(
        {
            "hyv3_for_causal_lm": LayoutBox(40.0, 52.0, 1160.0, 360.0),
            "hyv3_model": LayoutBox(24.0, 68.0, 830.0, 250.0),
            "input": LayoutBox(24.0, 84.0, 130.0, 62.0),
            "vocab_parallel_embedding": LayoutBox(182.0, 78.0, 170.0, 74.0),
            "hyv3_decoder_layer": LayoutBox(380.0, 66.0, 190.0, 112.0),
            "final_residual_add": LayoutBox(610.0, 78.0, 78.0, 72.0),
            "norm": LayoutBox(720.0, 78.0, 96.0, 72.0),
            "lm_head": LayoutBox(910.0, 106.0, 180.0, 70.0),
            "logits_processor": LayoutBox(910.0, 240.0, 200.0, 70.0),
        }
    )
    nodes = {str(node["id"]): node for node in page.get("nodes", []) if isinstance(node, dict) and "id" in node}
    if "hyv3_decoder_layer" in nodes:
        decorations.append(
            Decoration(
                cell_id="decorative_variants_hyv3_decoder_layer",
                parent="hyv3_decoder_layer",
                value="\n".join(_variant_lines(nodes["hyv3_decoder_layer"])),
                style=(
                    "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=11;"
                    "align=left;verticalAlign=top;spacing=6;fillColor=#FFFBEB;"
                    "strokeColor=#D6A63B;dashed=1;"
                ),
                box=LayoutBox(12.0, 54.0, 166.0, 48.0),
            )
        )
    return boxes, decorations


def _layout_attention_detail(
    page: dict[str, Any],
    boxes: dict[str, LayoutBox],
    decorations: list[Decoration],
) -> tuple[dict[str, LayoutBox], list[Decoration]]:
    boxes.update(
        {
            "attention_input": LayoutBox(40.0, 334.0, 160.0, 68.0),
            "qkv_projection": LayoutBox(250.0, 330.0, 180.0, 76.0),
            "split_qkv": LayoutBox(480.0, 334.0, 170.0, 68.0),
            "attention_path": LayoutBox(700.0, 182.0, 360.0, 360.0),
            "hpc_rope_norm": LayoutBox(34.0, 78.0, 292.0, 76.0),
            "fallback_qk_norm": LayoutBox(34.0, 234.0, 144.0, 72.0),
            "rotary_embedding": LayoutBox(202.0, 234.0, 124.0, 72.0),
            "attention_core": LayoutBox(1120.0, 330.0, 190.0, 76.0),
            "output_projection": LayoutBox(1360.0, 330.0, 180.0, 76.0),
            "attention_output": LayoutBox(1580.0, 334.0, 150.0, 68.0),
        }
    )
    decorations.append(
        Decoration(
            cell_id="decorative_attention_branch_note",
            parent="attention_path",
            value="HPC and fallback paths are mutually exclusive",
            style=(
                "text;html=0;strokeColor=none;fillColor=none;fontFamily=Inter;"
                "fontSize=11;align=left;verticalAlign=middle;fontColor=#475569;"
            ),
            box=LayoutBox(34.0, 44.0, 292.0, 24.0),
        )
    )
    return boxes, decorations


def _layout_adaptation_map(
    page: dict[str, Any],
    boxes: dict[str, LayoutBox],
    decorations: list[Decoration],
) -> tuple[dict[str, LayoutBox], list[Decoration]]:
    columns = {
        "region_hf_inputs": (35.0, ["hyv3_config", "hf_checkpoint"]),
        "region_vllm_config": (365.0, ["vllm_config", "cache_config", "quantization_config", "parallel_config_eplb_config"]),
        "region_adapter_interfaces": (
            695.0,
            [
                "adapter_hyv3_for_causal_lm",
                "adapter_hyv3_model",
                "supports_pp",
                "supports_lora",
                "mixture_of_experts",
                "support_torch_compile",
            ],
        ),
        "region_execution_components": (
            1025.0,
            [
                "adapt_vocab_parallel_embedding",
                "adapt_qkv_parallel_linear",
                "adapt_row_parallel_linear",
                "adapt_vllm_attention",
                "adapt_fused_moe",
                "adapt_parallel_lm_head",
                "adapt_logits_processor",
            ],
        ),
        "region_weight_parallel": (
            1355.0,
            [
                "packed_modules_mapping",
                "stacked_params_mapping",
                "expert_parameter_mapping",
                "auto_weights_loader",
                "tensor_parallel",
                "pipeline_parallel",
                "expert_parallel",
            ],
        ),
    }
    for region_id, (x, child_ids) in columns.items():
        boxes[region_id] = LayoutBox(x, 72.0, 300.0, 760.0)
        y = 70.0
        for child_id in child_ids:
            boxes[child_id] = LayoutBox(28.0, y, 244.0, 46.0)
            y += 62.0
    return boxes, decorations


def _page_size(page: dict[str, Any]) -> tuple[int, int]:
    if page.get("page_type") in {"attention_detail", "adaptation_map"}:
        return 1780, 900
    return 1280, 720


def _make_cell(parent: ET.Element, attrs: dict[str, str]) -> ET.Element:
    return ET.SubElement(parent, "mxCell", attrs)


def _add_geometry(cell: ET.Element, box: LayoutBox, *, relative: bool = False) -> None:
    attrs = {"as": "geometry"}
    if relative:
        attrs["relative"] = "1"
    else:
        attrs.update(
            {
                "x": _fmt(box.x),
                "y": _fmt(box.y),
                "width": _fmt(box.width),
                "height": _fmt(box.height),
            }
        )
    ET.SubElement(cell, "mxGeometry", attrs)


def _absolute_box(node_id: str, nodes_by_id: dict[str, dict[str, Any]], boxes: dict[str, LayoutBox]) -> LayoutBox:
    box = boxes[node_id]
    parent_id = nodes_by_id.get(node_id, {}).get("parent_id")
    while isinstance(parent_id, str) and parent_id:
        parent_box = boxes[parent_id]
        box = LayoutBox(box.x + parent_box.x, box.y + parent_box.y, box.width, box.height)
        parent_id = nodes_by_id.get(parent_id, {}).get("parent_id")
    return box


def _edge_points(edge: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]], boxes: dict[str, LayoutBox]) -> list[tuple[float, float]]:
    display = edge.get("display")
    route = display.get("route") if isinstance(display, dict) else None
    if route not in {"top_lane", "bottom_lane"} and edge.get("kind") != "residual":
        return []
    source = edge.get("source")
    target = edge.get("target")
    if not isinstance(source, str) or not isinstance(target, str) or source not in boxes or target not in boxes:
        return []
    source_box = _absolute_box(source, nodes_by_id, boxes)
    target_box = _absolute_box(target, nodes_by_id, boxes)
    sx = source_box.x + source_box.width / 2
    tx = target_box.x + target_box.width / 2
    if route == "bottom_lane":
        lane_y = max(source_box.y + source_box.height, target_box.y + target_box.height) + 52.0
    else:
        lane_y = min(source_box.y, target_box.y) - 52.0
    return [(sx, lane_y), (tx, lane_y)]


def _add_edge_geometry(cell: ET.Element, points: list[tuple[float, float]]) -> None:
    geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if not points:
        return
    points_element = ET.SubElement(geometry, "Array", {"as": "points"})
    for x, y in points:
        ET.SubElement(points_element, "mxPoint", {"x": _fmt(x), "y": _fmt(y)})


def _node_parent(node: dict[str, Any]) -> str:
    parent_id = node.get("parent_id")
    return str(parent_id) if isinstance(parent_id, str) and parent_id else ROOT_PARENT_ID


def _edge_label(edge: dict[str, Any]) -> str:
    display = edge.get("display")
    if isinstance(display, dict):
        if display.get("visible") is False:
            return ""
        if display.get("show_label") is False:
            return ""
        display_label = display.get("label")
        if isinstance(display_label, str):
            return display_label
    if edge.get("kind") == "runtime":
        return ""
    label = edge.get("label")
    if isinstance(label, str) and label:
        return label
    kind = edge.get("kind")
    if kind == "residual":
        return "residual"
    if kind == "conditional_true":
        return "true"
    if kind == "conditional_false":
        return "false"
    return ""


def _add_badges(root: ET.Element, node: dict[str, Any], box: LayoutBox) -> list[Decoration]:
    decorations: list[Decoration] = []
    display = node.get("display")
    if isinstance(display, dict) and display.get("show_badges") is False:
        return decorations
    badges = node.get("badges")
    if not isinstance(badges, list):
        return decorations
    for index, badge in enumerate(badges):
        if not isinstance(badge, str) or not badge:
            continue
        decorations.append(
            Decoration(
                cell_id=f"decorative_badge_{node['id']}_{badge}",
                parent=str(node["id"]),
                value=badge,
                style=(
                    "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=10;"
                    "fontStyle=1;align=center;verticalAlign=middle;fillColor=#EEF2FF;"
                    "strokeColor=#64748B;arcSize=20;"
                ),
                box=LayoutBox(box.width - 30.0 - index * 34.0, 8.0, 28.0, 18.0),
            )
        )
    return decorations


def _view_node_by_id(page: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node["semantic_id"]): node
        for node in page.get("visible_nodes", [])
        if isinstance(node, dict) and isinstance(node.get("semantic_id"), str)
    }


def _view_edge_by_id(page: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(edge["semantic_id"]): edge
        for edge in page.get("visible_edges", [])
        if isinstance(edge, dict) and isinstance(edge.get("semantic_id"), str)
    }


def _layout_page_by_id(layout_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(page["id"]): page
        for page in layout_plan.get("pages", [])
        if isinstance(page, dict) and isinstance(page.get("id"), str)
    }


def _label_from_view(node: dict[str, Any]) -> str:
    label = str(node.get("display_label") or "")
    subtitle = node.get("display_subtitle")
    if isinstance(subtitle, str) and subtitle.strip():
        return f"{label}\n{subtitle}"
    return label


def _box_from_plan(value: dict[str, Any]) -> LayoutBox:
    return LayoutBox(float(value["x"]), float(value["y"]), float(value["width"]), float(value["height"]))


def _relative_box(box: LayoutBox, parent_id: str | None, plan_nodes: dict[str, Any]) -> LayoutBox:
    if parent_id and parent_id in plan_nodes:
        parent = _box_from_plan(plan_nodes[parent_id])
        return LayoutBox(box.x - parent.x, box.y - parent.y, box.width, box.height)
    return box


def _add_points_geometry(cell: ET.Element, points: list[Any]) -> None:
    geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    points_element = ET.SubElement(geometry, "Array", {"as": "points"})
    for point in points:
        if isinstance(point, list) and len(point) == 2:
            ET.SubElement(points_element, "mxPoint", {"x": _fmt(float(point[0])), "y": _fmt(float(point[1]))})


def _render_from_view_plan(view: dict[str, Any], layout_plan: dict[str, Any]) -> str:
    pages = view.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Diagram View must contain at least one page")
    layout_pages = _layout_page_by_id(layout_plan)
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-07-23T00:00:00.000Z",
            "agent": "vllm-architecture-agent",
            "version": "v0.9",
            "type": "device",
        },
    )
    for page in pages:
        page_id = str(page["id"])
        layout_page = layout_pages.get(page_id)
        if layout_page is None:
            raise ValueError(f"layout plan missing page {page_id!r}")
        diagram = ET.SubElement(mxfile, "diagram", {"id": page_id, "name": str(page["title"])})
        model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1200",
                "dy": "700",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(layout_page["width"]),
                "pageHeight": str(layout_page["height"]),
                "math": "0",
                "shadow": "0",
            },
        )
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": ROOT_PARENT_ID, "parent": "0"})
        title = _make_cell(root, {"id": f"decorative_title_{page_id}", "value": str(page["title"]), "style": "text;html=0;strokeColor=none;fillColor=none;fontFamily=Inter;fontSize=22;fontStyle=1;align=left;verticalAlign=middle;", "vertex": "1", "parent": ROOT_PARENT_ID})
        _add_geometry(title, LayoutBox(40, 12, 460, 28))

        for region in layout_page.get("regions", []):
            if not isinstance(region, dict):
                continue
            cell = _make_cell(root, {"id": f"decorative_region_{page_id}_{region['id']}", "value": str(region["label"]), "style": "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=12;fontStyle=1;align=left;verticalAlign=top;spacingLeft=10;spacingTop=8;fillColor=#F8FAFC;strokeColor=#CBD5E1;dashed=1;", "vertex": "1", "parent": ROOT_PARENT_ID})
            _add_geometry(cell, LayoutBox(float(region["x"]), float(region["y"]), float(region["width"]), float(region["height"])))

        for annotation in page.get("annotations", []):
            if not isinstance(annotation, dict):
                continue
            annotation_id = str(annotation.get("id") or "")
            if not annotation_id.startswith("decorative_"):
                continue
            value = str(annotation.get("text") or annotation.get("label") or "")
            x = float(annotation.get("x", 48))
            y = float(annotation.get("y", 58))
            width = float(annotation.get("width", 300))
            height = float(annotation.get("height", 54))
            note = _make_cell(
                root,
                {
                    "id": annotation_id,
                    "value": value,
                    "style": (
                        "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=11;"
                        "align=left;verticalAlign=middle;spacing=8;fillColor=#F8FAFC;"
                        "strokeColor=#94A3B8;dashed=1;"
                    ),
                    "vertex": "1",
                    "parent": ROOT_PARENT_ID,
                },
            )
            _add_geometry(note, LayoutBox(x, y, width, height))

        view_nodes = _view_node_by_id(page)
        plan_nodes = layout_page.get("nodes", {})
        for node_id, view_node in view_nodes.items():
            plan_node = plan_nodes.get(node_id)
            if not isinstance(plan_node, dict):
                continue
            box = _box_from_plan(plan_node)
            parent_id = plan_node.get("parent_id")
            parent = str(parent_id) if isinstance(parent_id, str) and parent_id else ROOT_PARENT_ID
            rel_box = _relative_box(box, parent_id if isinstance(parent_id, str) else None, plan_nodes)
            kind = str(view_node.get("kind") or "note")
            value = "" if kind in {"container", "repeated_block"} else _label_from_view(view_node)
            cell = _make_cell(root, {"id": node_id, "value": value, "style": NODE_STYLES.get(kind, NODE_STYLES["note"]), "vertex": "1", "parent": parent})
            _add_geometry(cell, rel_box)
            if kind in {"container", "repeated_block"}:
                header = _make_cell(root, {"id": f"decorative_header_{node_id}", "value": _label_from_view(view_node), "style": "rounded=0;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;fontStyle=1;align=left;verticalAlign=middle;spacingLeft=10;fillColor=#E2E8F0;strokeColor=none;", "vertex": "1", "parent": node_id})
                _add_geometry(header, LayoutBox(0, 0, rel_box.width, 40))
            for index, badge in enumerate(view_node.get("badges", [])):
                if not isinstance(badge, str):
                    continue
                badge_y = 44.0 if kind in {"container", "repeated_block"} else 8.0
                badge_cell = _make_cell(root, {"id": f"decorative_badge_{node_id}_{badge}", "value": badge, "style": "rounded=1;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=10;fontStyle=1;align=center;verticalAlign=middle;fillColor=#EEF2FF;strokeColor=#64748B;arcSize=20;", "vertex": "1", "parent": node_id})
                _add_geometry(badge_cell, LayoutBox(rel_box.width - 34 - index * 34, badge_y, 28, 18))

        view_edges = _view_edge_by_id(page)
        for edge_id, view_edge in view_edges.items():
            plan_edge = layout_page.get("edges", {}).get(edge_id)
            if not isinstance(plan_edge, dict):
                continue
            style_kind = str(view_edge.get("style_kind") or "runtime")
            style = EDGE_STYLES.get(style_kind, EDGE_STYLES["runtime"])
            if plan_edge.get("visible") is False:
                style += "opacity=0;"
            value = str(plan_edge.get("label") or "") if plan_edge.get("label_visible") is True else ""
            attrs = {"id": edge_id, "value": value, "style": style, "edge": "1", "parent": ROOT_PARENT_ID, "source": str(view_edge["source"]), "target": str(view_edge["target"]), "sourcePort": str(view_edge.get("source_port") or ""), "targetPort": str(view_edge.get("target_port") or "")}
            cell = _make_cell(root, attrs)
            _add_points_geometry(cell, plan_edge.get("points", []))
    return ET.tostring(mxfile, encoding="unicode", short_empty_elements=True) + "\n"


def render_drawio(ir: dict[str, Any], layout_plan: dict[str, Any] | None = None) -> str:
    if layout_plan is not None:
        return _render_from_view_plan(ir, layout_plan)
    pages = ir.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("IR must contain at least one page")

    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-07-22T00:00:00.000Z",
            "agent": "vllm-architecture-agent",
            "version": "v0.7",
            "type": "device",
        },
    )

    for page in pages:
        page_id = str(page.get("id", "overview"))
        diagram = ET.SubElement(
            mxfile,
            "diagram",
            {"id": page_id, "name": str(page.get("title", page_id))},
        )
        page_width, page_height = _page_size(page)
        model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1200",
                "dy": "700",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(page_width),
                "pageHeight": str(page_height),
                "math": "0",
                "shadow": "0",
            },
        )
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": ROOT_PARENT_ID, "parent": "0"})

        boxes, layout_decorations = compute_layout(page)
        nodes = page.get("nodes", [])
        edges = page.get("edges", [])
        nodes_by_id = {
            str(node["id"]): node
            for node in nodes
            if isinstance(node, dict) and "id" in node
        }

        title_cell = _make_cell(
            root,
            {
                "id": f"decorative_title_{page_id}",
                "value": str(page.get("title", page_id)),
                "style": (
                    "text;html=0;strokeColor=none;fillColor=none;fontFamily=Inter;"
                    "fontSize=22;fontStyle=1;align=left;verticalAlign=middle;"
                ),
                "vertex": "1",
                "parent": ROOT_PARENT_ID,
            },
        )
        _add_geometry(title_cell, LayoutBox(40.0, 8.0, 400.0, 28.0))

        badge_decorations: list[Decoration] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node["id"])
            box = boxes[node_id]
            style = NODE_STYLES.get(str(node.get("kind")), NODE_STYLES["note"])
            cell = _make_cell(
                root,
                {
                    "id": node_id,
                    "value": _node_label(node),
                    "style": style,
                    "vertex": "1",
                    "parent": _node_parent(node),
                },
            )
            _add_geometry(cell, box)
            if node.get("kind") in {"container", "repeated_block"}:
                display = node.get("display")
                label = display.get("label") if isinstance(display, dict) else node.get("label")
                subtitle = display.get("subtitle") if isinstance(display, dict) else node.get("subtitle")
                header_value = str(label or node_id)
                if isinstance(subtitle, str) and subtitle.strip():
                    header_value = f"{header_value}\n{subtitle}"
                layout_decorations.append(
                    Decoration(
                        cell_id=f"decorative_header_{node_id}",
                        parent=node_id,
                        value=header_value,
                        style=(
                            "rounded=0;whiteSpace=wrap;html=0;fontFamily=Inter;fontSize=13;"
                            "fontStyle=1;align=left;verticalAlign=middle;spacingLeft=10;"
                            "fillColor=#E2E8F0;strokeColor=none;"
                        ),
                        box=LayoutBox(0.0, 0.0, box.width, 42.0),
                    )
                )
            badge_decorations.extend(_add_badges(root, node, box))

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            kind = str(edge.get("kind"))
            edge_style = EDGE_STYLES.get(kind, EDGE_STYLES["runtime"])
            display = edge.get("display")
            if isinstance(display, dict) and display.get("visible") is False:
                edge_style += "opacity=0;"
            cell = _make_cell(
                root,
                {
                    "id": str(edge["id"]),
                    "value": _edge_label(edge),
                    "style": edge_style,
                    "edge": "1",
                    "parent": ROOT_PARENT_ID,
                    "source": str(edge["source"]),
                    "target": str(edge["target"]),
                },
            )
            _add_edge_geometry(cell, _edge_points(edge, nodes_by_id, boxes))

        for decoration in layout_decorations + badge_decorations:
            cell = _make_cell(
                root,
                {
                    "id": decoration.cell_id,
                    "value": decoration.value,
                    "style": decoration.style,
                    "vertex": "1",
                    "parent": decoration.parent,
                },
            )
            _add_geometry(cell, decoration.box)

    return ET.tostring(mxfile, encoding="unicode", short_empty_elements=True) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Diagram View/Layout Plan to Draw.io XML.")
    parser.add_argument("input", type=Path, help="Diagram View JSON file, or Architecture IR for legacy rendering")
    parser.add_argument("--layout-plan", type=Path, help="Layout Plan JSON file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output .drawio path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists() or not args.input.is_file():
        print(f"error: file does not exist: {args.input}", file=sys.stderr)
        return 2
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: unable to read input JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(data, dict):
        print("error: Architecture IR root must be an object", file=sys.stderr)
        return 2
    try:
        layout_plan = None
        if args.layout_plan is not None:
            if not args.layout_plan.exists() or not args.layout_plan.is_file():
                print(f"error: layout plan does not exist: {args.layout_plan}", file=sys.stderr)
                return 2
            layout_plan = json.loads(args.layout_plan.read_text(encoding="utf-8"))
            if not isinstance(layout_plan, dict):
                print("error: layout plan root must be an object", file=sys.stderr)
                return 2
        xml = render_drawio(data, layout_plan)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(xml, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: unable to render Draw.io XML: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Draw.io XML to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
