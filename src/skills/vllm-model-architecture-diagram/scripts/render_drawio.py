#!/usr/bin/env python3
"""Render Architecture IR 0.2 to deterministic Draw.io XML."""

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
    label = str(node.get("label", ""))
    subtitle = node.get("subtitle")
    if node.get("kind") == "repeated_block":
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
    lines = ["Construction variants"]
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        condition = str(variant.get("condition", "")).strip()
        component = str(variant.get("component", "")).strip()
        if condition and component:
            lines.append(f"{condition} -> {component}")
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


def _node_parent(node: dict[str, Any]) -> str:
    parent_id = node.get("parent_id")
    return str(parent_id) if isinstance(parent_id, str) and parent_id else ROOT_PARENT_ID


def _edge_label(edge: dict[str, Any]) -> str:
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


def render_drawio(ir: dict[str, Any]) -> str:
    pages = ir.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("IR must contain at least one page")

    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-07-22T00:00:00.000Z",
            "agent": "vllm-architecture-agent",
            "version": "v0.5",
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
                "pageWidth": "1280",
                "pageHeight": "720",
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
            badge_decorations.extend(_add_badges(root, node, box))

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            kind = str(edge.get("kind"))
            edge_style = EDGE_STYLES.get(kind, EDGE_STYLES["runtime"])
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
            _add_geometry(cell, LayoutBox(0.0, 0.0, 0.0, 0.0), relative=True)

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
    parser = argparse.ArgumentParser(description="Render Architecture IR to Draw.io XML.")
    parser.add_argument("input", type=Path, help="Architecture IR JSON file")
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
        print(f"error: unable to read Architecture IR: {exc}", file=sys.stderr)
        return 2
    if not isinstance(data, dict):
        print("error: Architecture IR root must be an object", file=sys.stderr)
        return 2
    try:
        xml = render_drawio(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(xml, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: unable to render Draw.io XML: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Draw.io XML to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
