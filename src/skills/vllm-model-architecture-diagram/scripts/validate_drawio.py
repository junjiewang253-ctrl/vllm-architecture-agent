"""Validate Draw.io structure against a lightweight Architecture Plan."""

from __future__ import annotations

import argparse
import json
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MIN_EXPORT_WIDTH = 320
MIN_EXPORT_HEIGHT = 200
FLOW_VIEW_PATTERNS = {
    "pipeline",
    "block",
    "branch_merge",
    "routed_container",
    "mapping_flow",
    "multimodal_pipeline",
    "state_machine",
}


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


def _visible_edges(model: ET.Element) -> list[ET.Element]:
    return [
        cell
        for cell in model.findall(".//mxCell")
        if cell.attrib.get("edge") == "1" and not cell.attrib.get("id", "").startswith("decorative_")
    ]


def _page_text(model: ET.Element) -> str:
    values: list[str] = []
    for cell in model.findall(".//mxCell"):
        value = cell.attrib.get("value")
        if value:
            values.append(value)
    return "\n".join(values)


def _has_similar_visible_text(model: ET.Element, expected: str, threshold: float = 0.8) -> bool:
    expected_tokens = set(re.findall(r"[a-z0-9]+", expected.casefold()))
    if not expected_tokens:
        return False
    for cell in model.findall(".//mxCell"):
        value = cell.attrib.get("value", "")
        value_tokens = set(re.findall(r"[a-z0-9]+", value.casefold()))
        if len(expected_tokens & value_tokens) / len(expected_tokens) >= threshold:
            return True
    return False


def _style_signature(cell: ET.Element) -> tuple[str, ...]:
    properties: dict[str, str] = {}
    for item in cell.attrib.get("style", "").split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            properties[key] = value
    return tuple(
        properties.get(key, "")
        for key in ("shape", "fillColor", "strokeColor", "fontColor", "dashed")
    )


def _validate_png_export(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"PNG export does not exist: {path}"]
    if not path.is_file():
        return [f"PNG export is not a file: {path}"]
    data = path.read_bytes()
    if not data:
        return [f"PNG export is empty: {path}"]
    if not data.startswith(PNG_SIGNATURE):
        return [f"PNG export has an invalid PNG signature: {path}"]

    offset = len(PNG_SIGNATURE)
    width: int | None = None
    height: int | None = None
    saw_idat = False
    saw_iend = False
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            errors.append(f"PNG export contains a truncated chunk: {path}")
            break
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : chunk_end])[0]
        actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            errors.append(f"PNG export contains an invalid chunk checksum: {path}")
            break
        if chunk_type == b"IHDR":
            if length != 13:
                errors.append(f"PNG export has an invalid IHDR chunk: {path}")
                break
            width, height = struct.unpack(">II", chunk_data[:8])
        elif chunk_type == b"IDAT":
            saw_idat = saw_idat or bool(chunk_data)
        elif chunk_type == b"IEND":
            saw_iend = True
            break
        offset = chunk_end

    if width is None or height is None:
        errors.append(f"PNG export is missing image dimensions: {path}")
    elif width < MIN_EXPORT_WIDTH or height < MIN_EXPORT_HEIGHT:
        errors.append(
            f"PNG export is too small for an architecture page: {path} "
            f"({width}x{height}, minimum {MIN_EXPORT_WIDTH}x{MIN_EXPORT_HEIGHT})"
        )
    if not saw_idat:
        errors.append(f"PNG export has no image data: {path}")
    if not saw_iend:
        errors.append(f"PNG export is incomplete: {path}")
    return errors


def _validate_visual_review(path: Path, page_titles: list[str]) -> list[str]:
    if not path.exists():
        return [f"visual review does not exist: {path}"]
    if not path.is_file():
        return [f"visual review is not a file: {path}"]
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"visual review must be UTF-8 text: {path}"]
    if len(text.strip()) < 300:
        return [f"visual review is too short to document a real review: {path}"]

    lowered = text.casefold()
    required_markers = {
        "Draft 1 findings": ("draft", "首稿", "第一稿"),
        "revision": ("revision", "修改", "修订"),
        "final review": ("final", "最终"),
    }
    errors: list[str] = []
    for label, markers in required_markers.items():
        if not any(marker.casefold() in lowered for marker in markers):
            errors.append(f"visual review is missing {label}: {path}")
    for title in page_titles:
        if title.casefold() not in lowered:
            errors.append(f"visual review does not discuss page: {title}")
    return errors


def validate_drawio(
    drawio_path: Path,
    plan: dict[str, Any] | None = None,
    images: list[Path] | None = None,
    images_dir: Path | None = None,
    visual_review: Path | None = None,
) -> list[str]:
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

    plan_pages_by_title: dict[str, dict[str, Any]] = {}
    if plan:
        max_pages = plan.get("page_budget", {}).get("max_pages")
        if isinstance(max_pages, int) and len(pages) > max_pages:
            errors.append("drawio page count exceeds plan.page_budget.max_pages")
        expected_titles = [page["title"] for page in plan.get("pages", [])]
        plan_pages_by_title = {page["title"]: page for page in plan.get("pages", [])}
        missing = sorted(set(expected_titles) - set(page_names))
        extra = sorted(set(page_names) - set(expected_titles))
        for title in missing:
            errors.append(f"plan page missing from drawio: {title}")
        for title in extra:
            errors.append(f"drawio contains page not present in plan: {title}")
        for page in plan.get("pages", []):
            if not page.get("export_name"):
                errors.append(f"{page.get('id')}: plan page is missing export_name")
        if images_dir:
            if not images_dir.exists():
                errors.append(f"PNG export directory does not exist: {images_dir}")
            elif not images_dir.is_dir():
                errors.append(f"PNG export path is not a directory: {images_dir}")
            else:
                for page in plan.get("pages", []):
                    export_name = page.get("export_name")
                    expected_png = images_dir / f"{export_name}.png" if export_name else None
                    if expected_png and not expected_png.exists():
                        errors.append(f"{page.get('id')}: expected PNG export is missing: {expected_png}")

    all_cell_ids: set[str] = set()
    png_exports: set[Path] = set(images or [])
    complete_full_model = bool(
        plan
        and plan.get("detail_level") == "complete"
        and len(plan.get("pages", [])) >= 3
    )
    for page in pages:
        model = _graph_model(page)
        if model is None:
            errors.append(f"{_page_name(page)}: missing mxGraphModel")
            continue
        page_title = _page_name(page)
        matching = plan_pages_by_title.get(page_title, {}) if plan else {}
        background = model.attrib.get("background") or model.attrib.get("pageBackgroundColor")
        if not background:
            errors.append(f"{page_title}: page background must be explicitly white")
        elif background.lower() not in {"#ffffff", "white"}:
            errors.append(f"{_page_name(page)}: page background must be white")
        semantic_nodes = _visible_semantic_nodes(model)
        visible_edges = _visible_edges(model)
        excluded_labels = {
            str(matching.get("title", "")).strip(),
            str(matching.get("question", "")).strip(),
            *{
                str(region.get("title", "")).strip()
                for region in matching.get("detail_regions", [])
            },
        }
        content_nodes = [
            node
            for node in semantic_nodes
            if str(node.attrib.get("value", "")).strip() not in excluded_labels
        ]
        if not semantic_nodes:
            errors.append(f"{_page_name(page)}: page has no visible semantic nodes")
        elif len(semantic_nodes) < 3 and plan:
            matching = next((item for item in plan.get("pages", []) if item.get("title") == _page_name(page)), {})
            if matching.get("view_pattern") not in {"boundary_map", "component_map"}:
                errors.append(f"{_page_name(page)}: page has fewer than three visible semantic nodes")
        if plan and page_title in plan_pages_by_title:
            if matching.get("view_pattern") not in {"boundary_map", "component_map"} and len(semantic_nodes) < 5:
                errors.append(f"{page_title}: full model page should contain at least five visible semantic nodes")
            if complete_full_model and len(content_nodes) < 8:
                errors.append(
                    f"{page_title}: complete full-model page has only {len(content_nodes)} "
                    "semantic elements after excluding headers; minimum is 8"
                )
            if complete_full_model:
                style_signatures = {
                    _style_signature(node)
                    for node in content_nodes
                }
                if len(style_signatures) < 2:
                    errors.append(
                        f"{page_title}: complete full-model page must use at least two "
                        "visual treatments for semantic distinction"
                    )
            minimum_edges = max(
                4,
                min(8, len(matching.get("main_story", [])) - 1),
            )
            if matching.get("view_pattern") in FLOW_VIEW_PATTERNS and len(visible_edges) < minimum_edges:
                errors.append(
                    f"{page_title}: flow-oriented page has {len(visible_edges)} visible edges; "
                    f"minimum is {minimum_edges}"
                )
            page_text = _page_text(model)
            if matching.get("title") not in page_text:
                errors.append(f"{page_title}: page title is not present in visible cell text")
            if not _has_similar_visible_text(model, str(matching.get("question", ""))):
                errors.append(f"{page_title}: engineering question is not present in visible cell text")
            regions = matching.get("detail_regions", [])
            if not regions:
                errors.append(f"{page_title}: plan page has no detail regions")
            for region in regions:
                region_title = region.get("title")
                if region_title and region_title not in page_text:
                    errors.append(f"{page_title}: detail region title is missing from drawio text: {region_title}")
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
            elif cell.attrib.get("edge") == "1":
                if not cell.attrib.get("source") or not cell.attrib.get("target"):
                    errors.append(f"{_page_name(page)}.{cell_id}: visible edge requires source and target")
    if plan and images_dir and images_dir.is_dir():
        for page in plan.get("pages", []):
            export_name = page.get("export_name")
            if export_name:
                expected_png = images_dir / f"{export_name}.png"
                if expected_png.exists():
                    png_exports.add(expected_png)
    for image in sorted(png_exports, key=lambda item: item.as_posix()):
        errors.extend(_validate_png_export(image))
    if images_dir:
        review_path = visual_review or images_dir.parent / "visual-review.md"
        errors.extend(_validate_visual_review(review_path, page_names))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drawio", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--image", action="append", type=Path, default=[])
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--visual-review", type=Path)
    args = parser.parse_args(argv)
    errors = validate_drawio(
        args.drawio,
        plan=_load_plan(args.plan),
        images=args.image,
        images_dir=args.images_dir,
        visual_review=args.visual_review,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Draw.io validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
