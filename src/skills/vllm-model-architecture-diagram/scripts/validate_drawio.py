#!/usr/bin/env python3
"""Validate Draw.io XML against Architecture IR semantics."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_PARENT_ID = "1"
DECORATIVE_PREFIX = "decorative_"
SENTINEL_IDS = {"0", ROOT_PARENT_ID}


@dataclass(frozen=True)
class Geometry:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class DiagramPage:
    page_id: str
    title: str
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


def _geometry(cell: ET.Element) -> Geometry | None:
    geometry = cell.find("mxGeometry")
    if geometry is None:
        return None
    return Geometry(
        x=_parse_float(geometry.get("x")),
        y=_parse_float(geometry.get("y")),
        width=_parse_float(geometry.get("width")),
        height=_parse_float(geometry.get("height")),
    )


def _drawio_pages(path: Path) -> tuple[dict[str, DiagramPage], list[str]]:
    errors: list[str] = []
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError) as exc:
        return {}, [f"Draw.io XML is not parseable: {exc}"]

    document = tree.getroot()
    pages: dict[str, DiagramPage] = {}
    if document.tag == "mxGraphModel":
        root = document.find("root")
        if root is None:
            return {}, ["Draw.io XML does not contain mxGraphModel/root"]
        pages["overview"] = DiagramPage("overview", "Overview", _collect_cells(root, errors))
        return pages, errors

    if document.tag != "mxfile":
        return {}, [f"unexpected Draw.io root element: {document.tag}"]

    for diagram in document.findall("diagram"):
        page_id = diagram.get("id")
        title = diagram.get("name")
        if not page_id:
            errors.append("diagram without id")
            continue
        if page_id in pages:
            errors.append(f"duplicate Draw.io page id: {page_id}")
            continue
        model = diagram.find("mxGraphModel")
        root = model.find("root") if model is not None else None
        if root is None:
            errors.append(f"diagram {page_id!r} does not contain mxGraphModel/root")
            continue
        pages[page_id] = DiagramPage(page_id, title or "", _collect_cells(root, errors))
    if not pages and not errors:
        errors.append("Draw.io XML does not contain any diagram pages")
    return pages, errors


def _collect_cells(root: ET.Element, errors: list[str]) -> dict[str, ET.Element]:
    cells: dict[str, ET.Element] = {}
    for cell in root.findall("mxCell"):
        cell_id = cell.get("id")
        if not cell_id:
            errors.append("mxCell without id")
            continue
        if cell_id in cells and cell_id not in SENTINEL_IDS:
            errors.append(f"duplicate Draw.io cell id within page: {cell_id}")
        cells[cell_id] = cell
    return cells


def _is_decorative(cell_id: str) -> bool:
    return cell_id.startswith(DECORATIVE_PREFIX)


def _node_id(node: dict[str, Any]) -> str | None:
    node_id = node.get("id") or node.get("semantic_id")
    return node_id if isinstance(node_id, str) else None


def _edge_id(edge: dict[str, Any]) -> str | None:
    edge_id = edge.get("id") or edge.get("semantic_id")
    return edge_id if isinstance(edge_id, str) else None


def _page_nodes(page: dict[str, Any]) -> list[dict[str, Any]]:
    raw_nodes = page.get("nodes")
    if not isinstance(raw_nodes, list):
        raw_nodes = page.get("visible_nodes", [])
    return [node for node in raw_nodes if isinstance(node, dict) and _node_id(node)]


def _page_edges(page: dict[str, Any]) -> list[dict[str, Any]]:
    raw_edges = page.get("edges")
    if not isinstance(raw_edges, list):
        raw_edges = page.get("visible_edges", [])
    return [edge for edge in raw_edges if isinstance(edge, dict) and _edge_id(edge)]


def _page_node_ids(page: dict[str, Any]) -> set[str]:
    return {node_id for node in _page_nodes(page) for node_id in [_node_id(node)] if node_id is not None}


def _page_edge_ids(page: dict[str, Any]) -> set[str]:
    return {edge_id for edge in _page_edges(page) for edge_id in [_edge_id(edge)] if edge_id is not None}


def validate_drawio(ir: dict[str, Any], drawio_path: Path) -> list[str]:
    errors: list[str] = []
    draw_pages, parse_errors = _drawio_pages(drawio_path)
    errors.extend(parse_errors)
    if parse_errors:
        return errors

    ir_pages = ir.get("pages")
    if not isinstance(ir_pages, list):
        return ["IR pages must be a list"]

    ir_page_ids = {
        page.get("id")
        for page in ir_pages
        if isinstance(page, dict) and isinstance(page.get("id"), str)
    }
    for page_id in sorted(ir_page_ids):
        if page_id not in draw_pages:
            errors.append(f"missing Draw.io page for IR page: {page_id}")
    for page_id in sorted(draw_pages):
        if page_id not in ir_page_ids:
            errors.append(f"extra Draw.io page not present in IR: {page_id}")

    all_ir_node_ids: set[str] = set()
    for page in ir_pages:
        if not isinstance(page, dict):
            continue
        all_ir_node_ids.update(_page_node_ids(page))

    for page in ir_pages:
        if not isinstance(page, dict):
            continue
        page_id = page.get("id")
        if not isinstance(page_id, str) or page_id not in draw_pages:
            continue
        draw_page = draw_pages[page_id]
        if draw_page.title != page.get("title"):
            errors.append(
                f"page {page_id} title mismatch: expected {page.get('title')!r}, got {draw_page.title!r}"
            )
        _validate_page(page, draw_page, all_ir_node_ids, errors)

    return errors


def validate_drawio_with_view(
    ir: dict[str, Any],
    drawio_path: Path,
    view: dict[str, Any] | None = None,
    layout_plan: dict[str, Any] | None = None,
) -> list[str]:
    errors = validate_drawio(ir, drawio_path)
    if view is None:
        return errors
    draw_pages, parse_errors = _drawio_pages(drawio_path)
    errors.extend(parse_errors)
    if parse_errors:
        return errors
    layout_pages = {
        page.get("id"): page
        for page in (layout_plan or {}).get("pages", [])
        if isinstance(page, dict)
    }
    for page in view.get("pages", []):
        if not isinstance(page, dict) or not isinstance(page.get("id"), str):
            continue
        page_id = page["id"]
        draw_page = draw_pages.get(page_id)
        if draw_page is None:
            continue
        layout_page = layout_pages.get(page_id, {})
        layout_edges = layout_page.get("edges", {}) if isinstance(layout_page, dict) else {}
        for edge in page.get("visible_edges", []):
            if not isinstance(edge, dict) or not isinstance(edge.get("semantic_id"), str):
                continue
            edge_id = edge["semantic_id"]
            cell = draw_page.cells.get(edge_id)
            if cell is None:
                continue
            if cell.get("sourcePort") != edge.get("source_port"):
                errors.append(f"page {page_id}: edge {edge_id} sourcePort mismatch")
            if cell.get("targetPort") != edge.get("target_port"):
                errors.append(f"page {page_id}: edge {edge_id} targetPort mismatch")
            plan_edge = layout_edges.get(edge_id) if isinstance(layout_edges, dict) else None
            visible = not (isinstance(plan_edge, dict) and plan_edge.get("visible") is False)
            geometry = cell.find("mxGeometry")
            if visible and (geometry is None or geometry.find("Array/mxPoint") is None):
                errors.append(f"page {page_id}: visible edge {edge_id} must include routed waypoints")
    return errors


def _validate_page(
    ir_page: dict[str, Any],
    draw_page: DiagramPage,
    all_ir_node_ids: set[str],
    errors: list[str],
) -> None:
    node_ids = _page_node_ids(ir_page)
    edge_ids = _page_edge_ids(ir_page)
    cells = draw_page.cells

    for node_id in sorted(node_ids):
        if node_id not in cells:
            errors.append(f"page {draw_page.page_id}: missing Draw.io node for IR node: {node_id}")
    for edge_id in sorted(edge_ids):
        if edge_id not in cells:
            errors.append(f"page {draw_page.page_id}: missing Draw.io edge for IR edge: {edge_id}")

    for cell_id, cell in sorted(cells.items()):
        if cell_id in SENTINEL_IDS:
            continue
        if _is_decorative(cell_id):
            parent = cell.get("parent")
            if parent not in SENTINEL_IDS and parent not in node_ids:
                errors.append(
                    f"page {draw_page.page_id}: decorative cell {cell_id} references non-page parent {parent!r}"
                )
            continue
        if cell.get("vertex") == "1" and cell_id not in node_ids:
            if cell_id in all_ir_node_ids:
                errors.append(f"page {draw_page.page_id}: semantic node from another page is present: {cell_id}")
            else:
                errors.append(f"page {draw_page.page_id}: extra semantic Draw.io node not present in IR: {cell_id}")
        if cell.get("edge") == "1" and cell_id not in edge_ids:
            errors.append(f"page {draw_page.page_id}: extra semantic Draw.io edge not present in IR: {cell_id}")

    node_by_id = {
        node_id: node
        for node in _page_nodes(ir_page)
        for node_id in [_node_id(node)]
        if node_id is not None
    }
    edge_by_id = {
        edge_id: edge
        for edge in _page_edges(ir_page)
        for edge_id in [_edge_id(edge)]
        if edge_id is not None
    }

    for node_id, node in sorted(node_by_id.items()):
        cell = cells.get(node_id)
        if cell is None:
            continue
        expected_parent = node.get("parent_id") or ROOT_PARENT_ID
        actual_parent = cell.get("parent")
        if actual_parent != expected_parent:
            errors.append(
                f"page {draw_page.page_id}: node {node_id} parent mismatch: expected {expected_parent!r}, got {actual_parent!r}"
            )
        if node.get("parent_id") and actual_parent == ROOT_PARENT_ID:
            errors.append(f"page {draw_page.page_id}: node {node_id} has parent_id but still uses root parent")
        geometry = _geometry(cell)
        if geometry is None:
            errors.append(f"page {draw_page.page_id}: node {node_id} is missing geometry")
            continue
        if geometry.width <= 0 or geometry.height <= 0:
            errors.append(f"page {draw_page.page_id}: node {node_id} geometry width/height must be greater than 0")

    for edge_id, edge in sorted(edge_by_id.items()):
        cell = cells.get(edge_id)
        if cell is None:
            continue
        if cell.get("source") != edge.get("source"):
            errors.append(
                f"page {draw_page.page_id}: edge {edge_id} source mismatch: expected {edge.get('source')!r}, got {cell.get('source')!r}"
            )
        if cell.get("target") != edge.get("target"):
            errors.append(
                f"page {draw_page.page_id}: edge {edge_id} target mismatch: expected {edge.get('target')!r}, got {cell.get('target')!r}"
            )

    for node_id, node in sorted(node_by_id.items()):
        parent_id = node.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id:
            continue
        child_cell = cells.get(node_id)
        parent_cell = cells.get(parent_id)
        if child_cell is None or parent_cell is None:
            continue
        child_geometry = _geometry(child_cell)
        parent_geometry = _geometry(parent_cell)
        if child_geometry is None or parent_geometry is None:
            continue
        if (
            child_geometry.x < 0
            or child_geometry.y < 0
            or child_geometry.x + child_geometry.width > parent_geometry.width
            or child_geometry.y + child_geometry.height > parent_geometry.height
        ):
            errors.append(f"page {draw_page.page_id}: node {node_id} is outside parent container {parent_id}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Draw.io XML against Architecture IR.")
    parser.add_argument("ir", type=Path, help="Architecture IR JSON file")
    parser.add_argument("drawio", type=Path, help="Draw.io XML file")
    parser.add_argument("--view", type=Path, help="Diagram View JSON file for v0.8 port validation")
    parser.add_argument("--layout-plan", type=Path, help="Layout Plan JSON file for v0.8 route validation")
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
        view = _load_json(args.view) if args.view else None
        layout_plan = _load_json(args.layout_plan) if args.layout_plan else None
        errors = validate_drawio_with_view(ir, args.drawio, view, layout_plan)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Draw.io validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
