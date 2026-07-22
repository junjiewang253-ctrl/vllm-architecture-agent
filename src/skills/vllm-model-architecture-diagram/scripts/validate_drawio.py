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


def _ir_semantics(ir: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    pages = ir.get("pages")
    if not isinstance(pages, list):
        raise ValueError("IR pages must be a list")
    for page in pages:
        if not isinstance(page, dict):
            continue
        for node in page.get("nodes", []):
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                nodes[node["id"]] = node
        for edge in page.get("edges", []):
            if isinstance(edge, dict) and isinstance(edge.get("id"), str):
                edges[edge["id"]] = edge
    return nodes, edges


def _drawio_cells(path: Path) -> tuple[dict[str, ET.Element], list[str]]:
    errors: list[str] = []
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError) as exc:
        return {}, [f"Draw.io XML is not parseable: {exc}"]

    document = tree.getroot()
    roots: list[ET.Element] = []
    if document.tag == "mxGraphModel":
        root = document.find("root")
        if root is not None:
            roots.append(root)
    elif document.tag == "mxfile":
        for diagram in document.findall("diagram"):
            model = diagram.find("mxGraphModel")
            root = model.find("root") if model is not None else None
            if root is not None:
                roots.append(root)
    else:
        errors.append(f"unexpected Draw.io root element: {document.tag}")

    if not roots and not errors:
        errors.append("Draw.io XML does not contain mxGraphModel/root")

    cells: dict[str, ET.Element] = {}
    for root in roots:
        for cell in root.findall("mxCell"):
            cell_id = cell.get("id")
            if not cell_id:
                errors.append("mxCell without id")
                continue
            if cell_id in cells and cell_id not in SENTINEL_IDS:
                errors.append(f"duplicate Draw.io cell id: {cell_id}")
            cells[cell_id] = cell
    return cells, errors


def _is_decorative(cell_id: str) -> bool:
    return cell_id.startswith(DECORATIVE_PREFIX)


def validate_drawio(ir: dict[str, Any], drawio_path: Path) -> list[str]:
    errors: list[str] = []
    ir_nodes, ir_edges = _ir_semantics(ir)
    cells, parse_errors = _drawio_cells(drawio_path)
    errors.extend(parse_errors)
    if parse_errors:
        return errors

    for node_id in sorted(ir_nodes):
        if node_id not in cells:
            errors.append(f"missing Draw.io node for IR node: {node_id}")
    for edge_id in sorted(ir_edges):
        if edge_id not in cells:
            errors.append(f"missing Draw.io edge for IR edge: {edge_id}")

    for cell_id, cell in sorted(cells.items()):
        if cell_id in SENTINEL_IDS or _is_decorative(cell_id):
            continue
        if cell.get("vertex") == "1" and cell_id not in ir_nodes:
            errors.append(f"extra semantic Draw.io node not present in IR: {cell_id}")
        if cell.get("edge") == "1" and cell_id not in ir_edges:
            errors.append(f"extra semantic Draw.io edge not present in IR: {cell_id}")

    for node_id, node in sorted(ir_nodes.items()):
        cell = cells.get(node_id)
        if cell is None:
            continue
        expected_parent = node.get("parent_id") or ROOT_PARENT_ID
        actual_parent = cell.get("parent")
        if actual_parent != expected_parent:
            errors.append(
                f"node {node_id} parent mismatch: expected {expected_parent!r}, got {actual_parent!r}"
            )
        if node.get("parent_id") and actual_parent == ROOT_PARENT_ID:
            errors.append(f"node {node_id} has parent_id but still uses root parent")

        geometry = _geometry(cell)
        if geometry is None:
            errors.append(f"node {node_id} is missing geometry")
            continue
        if geometry.width <= 0 or geometry.height <= 0:
            errors.append(f"node {node_id} geometry width/height must be greater than 0")

    for edge_id, edge in sorted(ir_edges.items()):
        cell = cells.get(edge_id)
        if cell is None:
            continue
        actual_source = cell.get("source")
        actual_target = cell.get("target")
        if actual_source != edge.get("source"):
            errors.append(
                f"edge {edge_id} source mismatch: expected {edge.get('source')!r}, got {actual_source!r}"
            )
        if actual_target != edge.get("target"):
            errors.append(
                f"edge {edge_id} target mismatch: expected {edge.get('target')!r}, got {actual_target!r}"
            )

    for node_id, node in sorted(ir_nodes.items()):
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
            errors.append(f"node {node_id} is outside parent container {parent_id}")

    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Draw.io XML against Architecture IR.")
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
        errors = validate_drawio(ir, args.drawio)
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
