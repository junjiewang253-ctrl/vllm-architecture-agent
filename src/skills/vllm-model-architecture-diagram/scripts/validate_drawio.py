"""Validate Draw.io structure against a lightweight Architecture Plan."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _load_plan(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _diagram_pages(root: ET.Element) -> list[ET.Element]:
    if root.tag == "mxGraphModel":
        return [root]
    return list(root.findall("diagram"))


def _page_name(page: ET.Element) -> str:
    return page.attrib.get("name") or page.attrib.get("id") or "Page-1"


def _graph_model(page: ET.Element) -> ET.Element | None:
    if page.tag == "mxGraphModel":
        return page
    model = page.find("mxGraphModel")
    if model is not None:
        return model
    for child in page:
        if child.tag == "mxGraphModel":
            return child
    return None


def _visible_semantic_nodes(model: ET.Element) -> list[ET.Element]:
    nodes: list[ET.Element] = []
    for cell in model.findall(".//mxCell"):
        cell_id = cell.attrib.get("id", "")
        if cell.attrib.get("vertex") == "1" and not cell_id.startswith("decorative_") and cell_id not in {"0", "1"}:
            nodes.append(cell)
    return nodes


def validate_drawio(drawio_path: Path, plan: dict[str, Any] | None = None, images: list[Path] | None = None) -> list[str]:
    errors: list[str] = []
    if not drawio_path.exists():
        return [f"drawio file does not exist: {drawio_path}"]
    try:
        root = ET.parse(drawio_path).getroot()
    except ET.ParseError as exc:
        return [f"drawio XML cannot be parsed: {exc}"]
    pages = _diagram_pages(root)
    if not pages:
        errors.append("drawio contains no pages")
        return errors
    page_names = [_page_name(page) for page in pages]
    if len(page_names) != len(set(page_names)):
        errors.append("drawio page names must be unique")
    page_ids = [page.attrib.get("id", name) for page, name in zip(pages, page_names, strict=False)]
    if len(page_ids) != len(set(page_ids)):
        errors.append("drawio page IDs must be unique")

    if plan:
        expected_titles = [page["title"] for page in plan.get("pages", [])]
        missing = sorted(set(expected_titles) - set(page_names))
        extra = sorted(set(page_names) - set(expected_titles))
        for title in missing:
            errors.append(f"plan page missing from drawio: {title}")
        for title in extra:
            errors.append(f"drawio contains page not present in plan: {title}")

    all_cell_ids: set[str] = set()
    for page in pages:
        model = _graph_model(page)
        if model is None:
            errors.append(f"{_page_name(page)}: missing mxGraphModel")
            continue
        background = model.attrib.get("background") or model.attrib.get("pageBackgroundColor")
        if background and background.lower() not in {"#ffffff", "white"}:
            errors.append(f"{_page_name(page)}: page background must be white")
        semantic_nodes = _visible_semantic_nodes(model)
        if not semantic_nodes:
            errors.append(f"{_page_name(page)}: page has no visible semantic nodes")
        elif len(semantic_nodes) < 3 and plan:
            matching = next((item for item in plan.get("pages", []) if item.get("title") == _page_name(page)), {})
            if matching.get("view_pattern") not in {"boundary_map", "component_map"}:
                errors.append(f"{_page_name(page)}: page has fewer than three visible semantic nodes")
        for cell in model.findall(".//mxCell"):
            cell_id = cell.attrib.get("id")
            if not cell_id:
                errors.append(f"{_page_name(page)}: mxCell without id")
                continue
            scoped_id = f"{_page_name(page)}::{cell_id}"
            if scoped_id in all_cell_ids:
                errors.append(f"{_page_name(page)}: duplicate cell id {cell_id}")
            all_cell_ids.add(scoped_id)
            style = cell.attrib.get("style", "")
            if "html=1" in style:
                errors.append(f"{_page_name(page)}.{cell_id}: html=1 is not allowed")
            if cell.attrib.get("vertex") == "1":
                geometry = cell.find("mxGeometry")
                if geometry is None:
                    errors.append(f"{_page_name(page)}.{cell_id}: vertex is missing geometry")
                    continue
                try:
                    width = float(geometry.attrib.get("width", "0"))
                    height = float(geometry.attrib.get("height", "0"))
                except ValueError:
                    errors.append(f"{_page_name(page)}.{cell_id}: geometry width/height must be numeric")
                    continue
                if width <= 0 or height <= 0:
                    errors.append(f"{_page_name(page)}.{cell_id}: geometry width/height must be greater than zero")
    for image in images or []:
        if not image.exists():
            errors.append(f"referenced export does not exist: {image}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drawio", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--image", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    errors = validate_drawio(args.drawio, plan=_load_plan(args.plan), images=args.image)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Draw.io validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
