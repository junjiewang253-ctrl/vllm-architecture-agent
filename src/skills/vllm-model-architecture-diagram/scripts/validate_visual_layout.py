#!/usr/bin/env python3
"""Validate deterministic Draw.io layout quality for Architecture IR pages."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_PARENT_ID = "1"
DECORATIVE_PREFIX = "decorative_"
CONTAINER_KINDS = {"container", "repeated_block"}
REGION_IDS = {
    "region_hf_inputs",
    "region_vllm_config",
    "region_adapter_interfaces",
    "region_execution_components",
    "region_weight_parallel",
}


@dataclass(frozen=True)
class Box:
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
class Page:
    page_id: str
    title: str
    width: float
    height: float
    cells: dict[str, ET.Element]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read Architecture IR: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Architecture IR root must be an object")
    return data


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _geometry(cell: ET.Element) -> Box | None:
    geometry = cell.find("mxGeometry")
    if geometry is None:
        return None
    return Box(
        _parse_float(geometry.get("x")),
        _parse_float(geometry.get("y")),
        _parse_float(geometry.get("width")),
        _parse_float(geometry.get("height")),
    )


def _pages(path: Path) -> tuple[dict[str, Page], list[str]]:
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError) as exc:
        return {}, [f"Draw.io XML is not parseable: {exc}"]
    root = tree.getroot()
    if root.tag != "mxfile":
        return {}, [f"unexpected Draw.io root element: {root.tag}"]
    pages: dict[str, Page] = {}
    errors: list[str] = []
    for diagram in root.findall("diagram"):
        page_id = diagram.get("id")
        model = diagram.find("mxGraphModel")
        model_root = model.find("root") if model is not None else None
        if not page_id or model is None or model_root is None:
            errors.append("diagram is missing id or mxGraphModel/root")
            continue
        cells = {cell.get("id", ""): cell for cell in model_root.findall("mxCell") if cell.get("id")}
        pages[page_id] = Page(
            page_id=page_id,
            title=diagram.get("name") or "",
            width=_parse_float(model.get("pageWidth"), 1280.0),
            height=_parse_float(model.get("pageHeight"), 720.0),
            cells=cells,
        )
    return pages, errors


def _absolute_box(cell_id: str, page: Page, node_by_id: dict[str, dict[str, Any]]) -> Box | None:
    cell = page.cells.get(cell_id)
    if cell is None:
        return None
    box = _geometry(cell)
    if box is None:
        return None
    parent = cell.get("parent")
    while parent and parent not in {"0", ROOT_PARENT_ID}:
        parent_cell = page.cells.get(parent)
        if parent_cell is None:
            break
        parent_box = _geometry(parent_cell)
        if parent_box is None:
            break
        box = Box(box.x + parent_box.x, box.y + parent_box.y, box.width, box.height)
        parent = parent_cell.get("parent")
    return box


def _overlaps(a: Box, b: Box, *, margin: float = 0.0) -> bool:
    return not (
        a.right + margin <= b.x
        or b.right + margin <= a.x
        or a.bottom + margin <= b.y
        or b.bottom + margin <= a.y
    )


def _has_route_points(cell: ET.Element) -> bool:
    geometry = cell.find("mxGeometry")
    return geometry is not None and geometry.find("Array/mxPoint") is not None


def _edge_visible(edge: dict[str, Any]) -> bool:
    display = edge.get("display")
    return not (isinstance(display, dict) and display.get("visible") is False)


def _edge_show_label(edge: dict[str, Any]) -> bool:
    display = edge.get("display")
    return isinstance(display, dict) and display.get("show_label") is True


def validate_visual_layout(ir: dict[str, Any], drawio_path: Path) -> list[str]:
    pages, errors = _pages(drawio_path)
    if errors:
        return errors
    ir_pages = ir.get("pages")
    if not isinstance(ir_pages, list):
        return ["IR pages must be a list"]
    for page in ir_pages:
        if not isinstance(page, dict):
            continue
        page_id = page.get("id")
        if not isinstance(page_id, str) or page_id not in pages:
            errors.append(f"missing Draw.io page for visual validation: {page_id!r}")
            continue
        _validate_page(page, pages[page_id], errors)
    return errors


def _validate_page(ir_page: dict[str, Any], page: Page, errors: list[str]) -> None:
    if page.width <= 0 or page.height <= 0:
        errors.append(f"page {page.page_id}: page dimensions must be positive")
    elif page.width / page.height > 2.4:
        errors.append(f"page {page.page_id}: page aspect ratio exceeds 2.4")

    nodes = [node for node in ir_page.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)]
    node_by_id = {node["id"]: node for node in nodes}
    boxes: dict[str, Box] = {}
    for node in nodes:
        node_id = node["id"]
        box = _absolute_box(node_id, page, node_by_id)
        if box is None:
            continue
        boxes[node_id] = box
        if box.x < 0 or box.y < 0 or box.right > page.width or box.bottom > page.height:
            errors.append(f"page {page.page_id}: node {node_id} is outside page bounds")

        parent_id = node.get("parent_id")
        if isinstance(parent_id, str) and parent_id:
            child_cell = page.cells.get(node_id)
            parent_cell = page.cells.get(parent_id)
            child_rel = _geometry(child_cell) if child_cell is not None else None
            parent_box = _geometry(parent_cell) if parent_cell is not None else None
            if child_rel and parent_box and (
                child_rel.x < 0
                or child_rel.y < 0
                or child_rel.right > parent_box.width
                or child_rel.bottom > parent_box.height
            ):
                errors.append(f"page {page.page_id}: node {node_id} is outside parent container {parent_id}")

    leaf_nodes = [node for node in nodes if node.get("kind") not in CONTAINER_KINDS]
    for index, left in enumerate(leaf_nodes):
        left_box = boxes.get(left["id"])
        if left_box is None:
            continue
        for right in leaf_nodes[index + 1 :]:
            right_box = boxes.get(right["id"])
            if right_box is not None and _overlaps(left_box, right_box, margin=0.0):
                errors.append(f"page {page.page_id}: semantic nodes overlap: {left['id']} and {right['id']}")

    _validate_decorations(page, boxes, node_by_id, errors)
    _validate_edges(ir_page, page, errors)
    if ir_page.get("page_type") == "attention_detail":
        _validate_attention_branches(ir_page, errors)
    if ir_page.get("page_type") == "adaptation_map":
        present = {node["id"] for node in nodes}
        missing = sorted(REGION_IDS - present)
        if missing:
            errors.append(f"page {page.page_id}: adaptation map is missing visual regions: {', '.join(missing)}")


def _validate_decorations(
    page: Page,
    boxes: dict[str, Box],
    node_by_id: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    semantic_ids = set(boxes)
    for cell_id, cell in page.cells.items():
        if not cell_id.startswith(DECORATIVE_PREFIX) or cell.get("vertex") != "1":
            continue
        box = _absolute_box(cell_id, page, {}) or _geometry(cell)
        if box is None:
            continue
        parent = cell.get("parent")
        for node_id, semantic_box in boxes.items():
            if node_by_id.get(node_id, {}).get("kind") in CONTAINER_KINDS:
                continue
            if parent == node_id:
                continue
            if node_id not in semantic_ids:
                continue
            if _overlaps(box, semantic_box, margin=0.0):
                errors.append(f"page {page.page_id}: decorative cell {cell_id} overlaps semantic node {node_id}")


def _validate_edges(ir_page: dict[str, Any], page: Page, errors: list[str]) -> None:
    for edge in ir_page.get("edges", []):
        if not isinstance(edge, dict):
            continue
        edge_id = edge.get("id")
        if not isinstance(edge_id, str):
            continue
        cell = page.cells.get(edge_id)
        if cell is None:
            continue
        value = cell.get("value") or ""
        if _edge_visible(edge) and len(value) > 32:
            errors.append(f"page {page.page_id}: edge label too long on {edge_id}")
        if edge.get("kind") == "runtime" and value:
            errors.append(f"page {page.page_id}: runtime edge {edge_id} should not show a label")
        if edge.get("kind") == "runtime" and _edge_show_label(edge):
            errors.append(f"page {page.page_id}: runtime edge {edge_id} has show_label=true")
        if edge.get("kind") == "residual" and not _has_route_points(cell):
            errors.append(f"page {page.page_id}: residual edge {edge_id} must include route waypoints")


def _validate_attention_branches(ir_page: dict[str, Any], errors: list[str]) -> None:
    edges = {
        (edge.get("source"), edge.get("target"))
        for edge in ir_page.get("edges", [])
        if isinstance(edge, dict)
    }
    required = {
        ("split_qkv", "hpc_rope_norm"),
        ("hpc_rope_norm", "attention_core"),
        ("split_qkv", "fallback_qk_norm"),
        ("fallback_qk_norm", "rotary_embedding"),
        ("rotary_embedding", "attention_core"),
    }
    missing = sorted(required - edges)
    if missing:
        errors.append(f"page {ir_page.get('id')}: attention detail missing branch edges: {missing}")
    if ("hpc_rope_norm", "fallback_qk_norm") in edges or ("fallback_qk_norm", "hpc_rope_norm") in edges:
        errors.append(f"page {ir_page.get('id')}: HPC and fallback paths must not be serially connected")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate visual Draw.io layout constraints.")
    parser.add_argument("ir", type=Path, help="Architecture IR JSON file")
    parser.add_argument("drawio", type=Path, help="Draw.io XML file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.ir.exists() or not args.ir.is_file():
        print(f"error: IR file does not exist: {args.ir}", file=sys.stderr)
        return 2
    if not args.drawio.exists() or not args.drawio.is_file():
        print(f"error: Draw.io file does not exist: {args.drawio}", file=sys.stderr)
        return 2
    try:
        ir = _load_json(args.ir)
        errors = validate_visual_layout(ir, args.drawio)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Visual layout validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
