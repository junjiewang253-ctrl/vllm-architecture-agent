"""Validate Draw.io structure against a lightweight Architecture Plan."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MIN_EXPORT_WIDTH = 320
MIN_EXPORT_HEIGHT = 200
INTEGRATED_MIN_PAGE_WIDTH = 3200
INTEGRATED_MIN_PAGE_HEIGHT = 1600
INTEGRATED_MIN_CONTENT_WIDTH_RATIO = 0.75
INTEGRATED_MIN_CONTENT_HEIGHT_RATIO = 0.55
INTEGRATED_MIN_ANCHOR_AREA_RATIO = 0.02
INTEGRATED_MIN_GRID_COVERAGE_WITH_LOADING = 0.50
INTEGRATED_MIN_GRID_COVERAGE_WITHOUT_LOADING = 0.30
INTEGRATED_MAX_LOADING_SPAN_RATIO = 0.60
INTEGRATED_MAX_OTHER_SPAN_RATIO = 0.75
INTEGRATED_MAX_ASPECT_RATIO = 2.50
INTEGRATED_MAX_RUNTIME_LOADING_GAP_RATIO = 0.20
INTEGRATED_MAX_LOADING_HORIZONTAL_SPAN_RATIO = 0.32
INTEGRATED_MIN_REGION_ANCHOR_DENSITY = 0.08
INTEGRATED_MIN_CHANGED_ANCHOR_RATIO = 0.15
INTEGRATED_MIN_CHANGED_ANCHORS = 4
INTEGRATED_MIN_CHANGED_EDGES = 2
FLOW_VIEW_PATTERNS = {
    "pipeline",
    "block",
    "branch_merge",
    "routed_container",
    "mapping_flow",
    "multimodal_pipeline",
    "state_machine",
}

BLUE_COLORS = {"#2563EB", "#2B7DBD", "#1D4ED8", "#3B82F6", "#60A5FA"}
CYAN_COLORS = {"#0EA5E9", "#06B6D4", "#0891B2", "#0284C7"}
PURPLE_COLORS = {"#7C3AED", "#8B5CF6", "#6D28D9", "#9333EA"}
WARM_COLORS = {"#D97706", "#F59E0B", "#EA580C", "#F97316", "#DC2626", "#B91C1C"}
METADATA_COLORS = WARM_COLORS | {"#64748B", "#94A3B8", "#475569"}


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
    properties = _style_properties(cell)
    return tuple(
        properties.get(key, "")
        for key in ("shape", "fillColor", "strokeColor", "fontColor", "dashed")
    )


def _style_properties(cell: ET.Element) -> dict[str, str]:
    properties: dict[str, str] = {}
    for item in cell.attrib.get("style", "").split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            properties[key] = value
        elif item:
            properties[item] = "1"
    return properties


def _numeric(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _cell_bounds(
    cell: ET.Element,
    cells_by_id: dict[str, ET.Element],
    cache: dict[str, tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    cell_id = cell.attrib.get("id", "")
    if cell_id in cache:
        return cache[cell_id]
    geometry = cell.find("mxGeometry")
    if geometry is None:
        return None
    x = _numeric(geometry.attrib.get("x"))
    y = _numeric(geometry.attrib.get("y"))
    width = _numeric(geometry.attrib.get("width"))
    height = _numeric(geometry.attrib.get("height"))
    parent_id = cell.attrib.get("parent")
    parent = cells_by_id.get(parent_id or "")
    if parent is not None and parent.attrib.get("vertex") == "1":
        parent_bounds = _cell_bounds(parent, cells_by_id, cache)
        if parent_bounds is not None:
            x += parent_bounds[0]
            y += parent_bounds[1]
    bounds = (x, y, width, height)
    cache[cell_id] = bounds
    return bounds


def _contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    tolerance: float = 4.0,
) -> bool:
    outer_x, outer_y, outer_width, outer_height = outer
    inner_x, inner_y, inner_width, inner_height = inner
    return (
        inner_x >= outer_x - tolerance
        and inner_y >= outer_y - tolerance
        and inner_x + inner_width <= outer_x + outer_width + tolerance
        and inner_y + inner_height <= outer_y + outer_height + tolerance
    )


def _color(cell: ET.Element) -> str:
    return _style_properties(cell).get("strokeColor", "").upper()


def _is_dashed(cell: ET.Element) -> bool:
    return _style_properties(cell).get("dashed") == "1"


def _label_matches(cell: ET.Element, expected: str) -> bool:
    expected_tokens = set(re.findall(r"[a-z0-9]+", expected.casefold()))
    actual_tokens = set(re.findall(r"[a-z0-9]+", cell.attrib.get("value", "").casefold()))
    if not expected_tokens:
        return False
    return len(expected_tokens & actual_tokens) / len(expected_tokens) >= 0.5


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _exact_text_cells(model: ET.Element, expected: str) -> list[ET.Element]:
    normalized = _normalized_text(expected)
    return [
        cell
        for cell in model.findall(".//mxCell")
        if cell.attrib.get("vertex") == "1"
        and _normalized_text(cell.attrib.get("value", "")) == normalized
    ]


def _center(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    return bounds[0] + bounds[2] / 2, bounds[1] + bounds[3] / 2


def _bounds_union(
    bounds: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not bounds:
        return None
    left = min(item[0] for item in bounds)
    top = min(item[1] for item in bounds)
    right = max(item[0] + item[2] for item in bounds)
    bottom = max(item[1] + item[3] for item in bounds)
    return left, top, right - left, bottom - top


def _edge_points(
    edge: ET.Element,
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    points = [_center(source_bounds)]
    geometry = edge.find("mxGeometry")
    if geometry is not None:
        array = geometry.find("Array[@as='points']")
        if array is not None:
            for point in array.findall("mxPoint"):
                points.append(
                    (
                        _numeric(point.attrib.get("x")),
                        _numeric(point.attrib.get("y")),
                    )
                )
    points.append(_center(target_bounds))
    return points


def _polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(
        math.dist(source, target)
        for source, target in zip(points, points[1:], strict=False)
    )


def _edge_style_matches(cell: ET.Element, kind: str) -> bool:
    color = _color(cell)
    dashed = _is_dashed(cell)
    width = _numeric(_style_properties(cell).get("strokeWidth"), 1.0)
    if kind == "runtime_flow":
        return not dashed and color in BLUE_COLORS and width >= 2.0
    if kind == "residual_flow":
        return not dashed and color in CYAN_COLORS and width >= 1.5
    if kind == "loading_mapping":
        return not dashed and color in PURPLE_COLORS and width >= 2.0
    if kind == "construction_dependency":
        return dashed and color in WARM_COLORS
    if kind == "metadata_dependency":
        return dashed and color in METADATA_COLORS
    if kind == "external_delegation":
        return dashed and color in WARM_COLORS
    return False


def _validate_anchor_representation(
    *,
    page_title: str,
    anchor: dict[str, Any],
    cell: ET.Element,
) -> list[str]:
    errors: list[str] = []
    anchor_id = anchor.get("id")
    representation = anchor.get("representation")
    role = anchor.get("role")
    style_role = anchor.get("style_role")
    properties = _style_properties(cell)
    if cell.attrib.get("vertex") != "1":
        return [f"{page_title}.{anchor_id}: required visual anchor must be a vertex"]
    if (
        cell.attrib.get("visible") == "0"
        or properties.get("opacity") == "0"
        or cell.attrib.get("id", "").startswith("decorative_")
    ):
        errors.append(f"{page_title}.{anchor_id}: required visual anchor must be visible")
    if not _label_matches(cell, str(anchor.get("label", ""))):
        errors.append(f"{page_title}.{anchor_id}: visible label does not match the Plan anchor")
    if representation == "container" and "swimlane" not in properties and properties.get("shape") != "swimlane":
        errors.append(f"{page_title}.{anchor_id}: container anchor must use a container/swimlane style")
    if representation == "storage" and "cylinder" not in properties.get("shape", ""):
        errors.append(f"{page_title}.{anchor_id}: storage anchor must use a cylinder shape")
    if representation == "badge" and properties.get("rounded") != "1":
        errors.append(f"{page_title}.{anchor_id}: badge anchor must use a rounded style")
    if representation == "merge" and properties.get("shape") not in {"ellipse", "rhombus"}:
        errors.append(f"{page_title}.{anchor_id}: merge anchor must use ellipse or rhombus")
    if style_role == "containment" and representation != "container":
        errors.append(f"{page_title}.{anchor_id}: containment style_role requires a container")
    if role == "merge" and representation != "merge":
        errors.append(f"{page_title}.{anchor_id}: merge role requires merge representation")
    if role == "external_boundary" or style_role == "external":
        if not _is_dashed(cell) or _color(cell) not in WARM_COLORS:
            errors.append(
                f"{page_title}.{anchor_id}: external boundary must use a warm dashed outline"
            )
    if style_role == "loading" and _color(cell) not in PURPLE_COLORS:
        errors.append(f"{page_title}.{anchor_id}: loading anchor must use the purple loading style")
    if style_role == "construction" and (not _is_dashed(cell) or _color(cell) not in WARM_COLORS):
        errors.append(
            f"{page_title}.{anchor_id}: construction anchor must use a warm dashed style"
        )
    if style_role == "metadata" and not _is_dashed(cell):
        errors.append(f"{page_title}.{anchor_id}: metadata anchor must use a dashed style")
    font_size = _numeric(properties.get("fontSize"), 0.0)
    if font_size <= 0:
        errors.append(f"{page_title}.{anchor_id}: required anchor must declare fontSize")
    return errors


def _validate_integrated_contract(
    *,
    page_title: str,
    model: ET.Element,
    page_plan: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    contract = page_plan.get("visual_contract", {})
    canvas = contract.get("canvas", {})
    page_width = _numeric(model.attrib.get("pageWidth"))
    page_height = _numeric(model.attrib.get("pageHeight"))
    min_width = max(INTEGRATED_MIN_PAGE_WIDTH, int(canvas.get("min_width", 0)))
    min_height = max(INTEGRATED_MIN_PAGE_HEIGHT, int(canvas.get("min_height", 0)))
    if page_width < min_width or page_height < min_height:
        errors.append(
            f"{page_title}: integrated canvas is {page_width:g}x{page_height:g}; "
            f"minimum is {min_width}x{min_height}"
        )
    if page_width <= page_height:
        errors.append(f"{page_title}: integrated canvas must be landscape")
    if page_height > 0 and page_width / page_height > INTEGRATED_MAX_ASPECT_RATIO:
        errors.append(
            f"{page_title}: integrated canvas aspect ratio "
            f"{page_width / page_height:.2f} exceeds "
            f"{INTEGRATED_MAX_ASPECT_RATIO:.2f}; compact the runtime spine or "
            "increase useful vertical composition"
        )

    cells = list(model.findall(".//mxCell"))
    cells_by_id = {
        cell.attrib["id"]: cell
        for cell in cells
        if cell.attrib.get("id")
    }
    cells_by_anchor: dict[str, list[ET.Element]] = {}
    for cell in cells:
        anchor_id = cell.attrib.get("dataAnchor")
        if anchor_id:
            cells_by_anchor.setdefault(anchor_id, []).append(cell)
    for anchor_id, anchored_cells in cells_by_anchor.items():
        if len(anchored_cells) > 1:
            errors.append(f"{page_title}: dataAnchor must be unique: {anchor_id}")

    heading_cells: dict[str, ET.Element] = {}
    for heading_name, expected in (
        ("title", page_plan.get("title", "")),
        ("question", page_plan.get("question", "")),
    ):
        matches = _exact_text_cells(model, str(expected))
        if len(matches) != 1:
            errors.append(
                f"{page_title}: {heading_name} must appear once as a separate "
                "visible text cell"
            )
            continue
        heading_cells[heading_name] = matches[0]
    title_cell = heading_cells.get("title")
    question_cell = heading_cells.get("question")
    if title_cell is not None and question_cell is not None:
        title_bounds = _cell_bounds(title_cell, cells_by_id, {})
        question_bounds = _cell_bounds(question_cell, cells_by_id, {})
        title_font = _numeric(_style_properties(title_cell).get("fontSize"))
        question_font = _numeric(_style_properties(question_cell).get("fontSize"))
        if title_font < 20:
            errors.append(f"{page_title}: title fontSize must be at least 20")
        if not (12 <= question_font <= 18):
            errors.append(
                f"{page_title}: engineering question fontSize must be from 12 to 18"
            )
        if title_bounds and question_bounds:
            if title_bounds[1] > page_height * 0.08:
                errors.append(f"{page_title}: title must remain in the top header band")
            if question_bounds[1] <= title_bounds[1]:
                errors.append(
                    f"{page_title}: engineering question must appear below the title"
                )
            if question_bounds[1] + question_bounds[3] > page_height * 0.14:
                errors.append(
                    f"{page_title}: engineering question must remain in the compact "
                    "top header band"
                )

    plan_anchors = {
        anchor.get("id"): anchor
        for anchor in contract.get("required_anchors", [])
        if anchor.get("id")
    }
    resolved_anchors: dict[str, ET.Element] = {}
    for anchor_id, anchor in plan_anchors.items():
        matches = cells_by_anchor.get(anchor_id, [])
        if not matches:
            errors.append(f"{page_title}: required visual anchor is missing: {anchor_id}")
            continue
        if len(matches) != 1:
            continue
        cell = matches[0]
        resolved_anchors[anchor_id] = cell
        errors.extend(
            _validate_anchor_representation(
                page_title=page_title,
                anchor=anchor,
                cell=cell,
            )
        )
        font_size = _numeric(_style_properties(cell).get("fontSize"), 0.0)
        if font_size and font_size < _numeric(str(canvas.get("min_font_size")), 10.0):
            errors.append(
                f"{page_title}.{anchor_id}: fontSize {font_size:g} is below "
                f"the contract minimum {canvas.get('min_font_size')}"
            )
        bounds = _cell_bounds(cell, cells_by_id, {})
        if bounds is not None:
            x, y, width, height = bounds
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > page_width
                or y + height > page_height
            ):
                errors.append(f"{page_title}.{anchor_id}: required anchor lies outside the canvas")

    bounds_cache: dict[str, tuple[float, float, float, float]] = {}
    for anchor_id, anchor in plan_anchors.items():
        parent_anchor_id = anchor.get("parent_anchor_id")
        if not parent_anchor_id:
            continue
        child_cell = resolved_anchors.get(anchor_id)
        parent_cell = resolved_anchors.get(parent_anchor_id)
        if child_cell is None or parent_cell is None:
            continue
        if child_cell.attrib.get("parent") == parent_cell.attrib.get("id"):
            continue
        child_bounds = _cell_bounds(child_cell, cells_by_id, bounds_cache)
        parent_bounds = _cell_bounds(parent_cell, cells_by_id, bounds_cache)
        if child_bounds and parent_bounds and not _contains(parent_bounds, child_bounds):
            errors.append(
                f"{page_title}.{anchor_id}: visual child is outside container {parent_anchor_id}"
            )

    for region in page_plan.get("detail_regions", []):
        region_id = region.get("id", "<unknown>")
        region_title = str(region.get("title", ""))
        title_matches = [
            cell
            for cell in _exact_text_cells(model, region_title)
            if not cell.attrib.get("dataAnchor")
        ]
        if len(title_matches) != 1:
            errors.append(
                f"{page_title}.{region_id}: detail region title must appear once "
                "as a separate visible text cell"
            )
            continue
        region_anchor_ids = region.get("integration", {}).get("anchor_ids", [])
        region_bounds = [
            bounds
            for anchor_id in region_anchor_ids
            if plan_anchors.get(anchor_id, {}).get("representation")
            not in {"container", "annotation", "badge"}
            for cell in [resolved_anchors.get(anchor_id)]
            if cell is not None
            for bounds in [_cell_bounds(cell, cells_by_id, bounds_cache)]
            if bounds is not None
        ]
        region_title_anchor_bounds = [
            bounds
            for anchor_id in region_anchor_ids
            for cell in [resolved_anchors.get(anchor_id)]
            if cell is not None
            for bounds in [_cell_bounds(cell, cells_by_id, bounds_cache)]
            if bounds is not None
        ]
        if region.get("type") == "mapping_tree":
            loading_ids = {
                anchor_id
                for anchor_id in region_anchor_ids
                if plan_anchors.get(anchor_id, {}).get("phase") == "loading"
            }
            loading_bounds = [
                bounds
                for anchor_id in loading_ids
                for cell in [resolved_anchors.get(anchor_id)]
                if cell is not None
                for bounds in [_cell_bounds(cell, cells_by_id, bounds_cache)]
                if bounds is not None
            ]
            if loading_bounds:
                region_bounds = loading_bounds
                region_title_anchor_bounds = loading_bounds
        region_union = _bounds_union(region_title_anchor_bounds)
        density_union = _bounds_union(region_bounds)
        title_bounds = _cell_bounds(title_matches[0], cells_by_id, bounds_cache)
        if region_union and title_bounds:
            region_left, region_top, region_width, region_height = region_union
            title_center_x = title_bounds[0] + title_bounds[2] / 2
            if (
                title_bounds[1] > region_top + max(80.0, region_height * 0.20)
                or title_center_x < region_left - 100
                or title_center_x > region_left + region_width + 100
            ):
                errors.append(
                    f"{page_title}.{region_id}: detail region title is detached "
                    "from its visible subgraph"
                )
            density_area = max(
                1.0,
                (density_union[2] * density_union[3]) if density_union else 1.0,
            )
            anchor_area = sum(
                max(0.0, width) * max(0.0, height)
                for _, _, width, height in region_bounds
            )
            density = anchor_area / density_area
            if len(region_bounds) >= 3 and density < INTEGRATED_MIN_REGION_ANCHOR_DENSITY:
                errors.append(
                    f"{page_title}.{region_id}: visible anchor density {density:.2f} "
                    f"is below {INTEGRATED_MIN_REGION_ANCHOR_DENSITY:.2f}; "
                    "compact the region instead of surrounding sparse nodes with "
                    "empty space"
                )

    resolved_relationships: list[tuple[dict[str, Any], ET.Element]] = []
    for relationship in contract.get("required_relationships", []):
        relationship_id = relationship.get("id")
        matches = cells_by_anchor.get(relationship_id, [])
        if not matches:
            errors.append(f"{page_title}: required visual relationship is missing: {relationship_id}")
            continue
        if len(matches) != 1:
            continue
        edge = matches[0]
        if edge.attrib.get("edge") != "1":
            errors.append(f"{page_title}.{relationship_id}: visual relationship must be an edge")
            continue
        edge_properties = _style_properties(edge)
        if (
            edge.attrib.get("visible") == "0"
            or edge_properties.get("opacity") == "0"
            or edge.attrib.get("id", "").startswith("decorative_")
        ):
            errors.append(f"{page_title}.{relationship_id}: required relationship must be visible")
        expected_source = resolved_anchors.get(relationship.get("source_anchor_id"))
        expected_target = resolved_anchors.get(relationship.get("target_anchor_id"))
        if expected_source is not None and edge.attrib.get("source") != expected_source.attrib.get("id"):
            errors.append(f"{page_title}.{relationship_id}: edge source does not match Plan anchor")
        if expected_target is not None and edge.attrib.get("target") != expected_target.attrib.get("id"):
            errors.append(f"{page_title}.{relationship_id}: edge target does not match Plan anchor")
        if not _edge_style_matches(edge, str(relationship.get("kind", ""))):
            errors.append(
                f"{page_title}.{relationship_id}: edge style does not match "
                f"{relationship.get('kind')}"
            )
        resolved_relationships.append((relationship, edge))

    gates = contract.get("quality_gates", {})
    visible_required_anchors = [
        cell
        for anchor_id, cell in resolved_anchors.items()
        if plan_anchors.get(anchor_id, {}).get("representation") != "annotation"
    ]
    if len(visible_required_anchors) < int(gates.get("min_semantic_nodes", 0)):
        errors.append(
            f"{page_title}: integrated page has {len(visible_required_anchors)} "
            "required semantic anchors; "
            f"minimum is {gates.get('min_semantic_nodes')}"
        )
    if len(resolved_relationships) < int(gates.get("min_visible_edges", 0)):
        errors.append(
            f"{page_title}: integrated page has {len(resolved_relationships)} "
            "required visible relationships; "
            f"minimum is {gates.get('min_visible_edges')}"
        )
    style_signatures = {
        _style_signature(cell)
        for cell in visible_required_anchors
    }
    if len(style_signatures) < int(gates.get("min_visual_styles", 0)):
        errors.append(
            f"{page_title}: integrated page has {len(style_signatures)} visual styles; "
            f"minimum is {gates.get('min_visual_styles')}"
        )

    measurable_anchor_ids = {
        anchor_id
        for anchor_id, anchor in plan_anchors.items()
        if anchor.get("representation")
        not in {"container", "annotation", "badge"}
    }
    anchor_bounds = [
        bounds
        for anchor_id, cell in resolved_anchors.items()
        if anchor_id in measurable_anchor_ids
        if cell.attrib.get("vertex") == "1"
        for bounds in [_cell_bounds(cell, cells_by_id, bounds_cache)]
        if bounds is not None
    ]
    occupied_bounds = [
        bounds
        for cell in _visible_semantic_nodes(model)
        if cell.attrib.get("visible") != "0"
        for bounds in [_cell_bounds(cell, cells_by_id, bounds_cache)]
        if bounds is not None
    ]
    occupied_union = _bounds_union(occupied_bounds)
    if anchor_bounds and page_width > 0 and page_height > 0:
        left = min(item[0] for item in anchor_bounds)
        top = min(item[1] for item in anchor_bounds)
        right = max(item[0] + item[2] for item in anchor_bounds)
        bottom = max(item[1] + item[3] for item in anchor_bounds)
        width_ratio = max(0.0, right - left) / page_width
        height_ratio = max(0.0, bottom - top) / page_height
        fill_ratio = width_ratio * height_ratio
        minimum_fill = float(canvas.get("min_content_fill_ratio", 0.0))
        if fill_ratio < minimum_fill:
            errors.append(
                f"{page_title}: integrated content fill ratio {fill_ratio:.2f} is below "
                f"{minimum_fill:.2f}"
            )
        occupied_left = 0.0
        occupied_top = 0.0
        occupied_width = page_width
        occupied_height = page_height
        if occupied_union is not None:
            (
                occupied_left,
                occupied_top,
                occupied_width,
                occupied_height,
            ) = occupied_union
        semantic_width_ratio = max(0.0, right - left) / max(1.0, occupied_width)
        semantic_height_ratio = max(0.0, bottom - top) / max(1.0, occupied_height)
        if semantic_width_ratio < INTEGRATED_MIN_CONTENT_WIDTH_RATIO:
            errors.append(
                f"{page_title}: required anchors cover only "
                f"{semantic_width_ratio:.2f} of the occupied content width; minimum is "
                f"{INTEGRATED_MIN_CONTENT_WIDTH_RATIO:.2f}"
            )
        if semantic_height_ratio < INTEGRATED_MIN_CONTENT_HEIGHT_RATIO:
            errors.append(
                f"{page_title}: required anchors cover only "
                f"{semantic_height_ratio:.2f} of the occupied content height; minimum is "
                f"{INTEGRATED_MIN_CONTENT_HEIGHT_RATIO:.2f}"
            )

        anchor_area_ratio = sum(
            max(0.0, width) * max(0.0, height)
            for _, _, width, height in anchor_bounds
        ) / (page_width * page_height)
        if anchor_area_ratio < INTEGRATED_MIN_ANCHOR_AREA_RATIO:
            errors.append(
                f"{page_title}: required anchor area ratio {anchor_area_ratio:.3f} is "
                f"below {INTEGRATED_MIN_ANCHOR_AREA_RATIO:.3f}"
            )

        grid_columns = 8
        grid_rows = 4
        occupied_grid_cells: set[tuple[int, int]] = set()
        for x, y, width, height in anchor_bounds:
            center_x = min(
                occupied_width - 1,
                max(0.0, x + width / 2 - occupied_left),
            )
            center_y = min(
                occupied_height - 1,
                max(0.0, y + height / 2 - occupied_top),
            )
            occupied_grid_cells.add(
                (
                    min(
                        grid_columns - 1,
                        int(center_x / max(1.0, occupied_width) * grid_columns),
                    ),
                    min(
                        grid_rows - 1,
                        int(center_y / max(1.0, occupied_height) * grid_rows),
                    ),
                )
            )
        grid_coverage = len(occupied_grid_cells) / (grid_columns * grid_rows)
        minimum_grid_coverage = (
            INTEGRATED_MIN_GRID_COVERAGE_WITH_LOADING
            if gates.get("require_loading_plane")
            else INTEGRATED_MIN_GRID_COVERAGE_WITHOUT_LOADING
        )
        if grid_coverage < minimum_grid_coverage:
            errors.append(
                f"{page_title}: required anchors occupy only {grid_coverage:.2f} of the "
                f"{grid_columns}x{grid_rows} occupied-content grid; minimum is "
                f"{minimum_grid_coverage:.2f}"
            )

    runtime_bounds = [
        bounds
        for anchor_id, anchor in plan_anchors.items()
        if anchor.get("phase") == "runtime"
        and anchor.get("representation") not in {"container", "annotation", "badge"}
        for cell in [resolved_anchors.get(anchor_id)]
        if cell is not None
        for bounds in [_cell_bounds(cell, cells_by_id, bounds_cache)]
        if bounds is not None
    ]
    loading_bounds = [
        bounds
        for anchor_id, anchor in plan_anchors.items()
        if anchor.get("phase") == "loading"
        and anchor.get("representation") not in {"container", "annotation", "badge"}
        for cell in [resolved_anchors.get(anchor_id)]
        if cell is not None
        for bounds in [_cell_bounds(cell, cells_by_id, bounds_cache)]
        if bounds is not None
    ]
    if gates.get("require_loading_plane") and runtime_bounds and loading_bounds:
        runtime_bottom = max(y + height for _, y, _, height in runtime_bounds)
        loading_top = min(y for _, y, _, _ in loading_bounds)
        gap_ratio = max(0.0, loading_top - runtime_bottom) / max(1.0, page_height)
        if loading_top <= min(y for _, y, _, _ in runtime_bounds):
            errors.append(
                f"{page_title}: loading plane must sit below the runtime system"
            )
        if gap_ratio > INTEGRATED_MAX_RUNTIME_LOADING_GAP_RATIO:
            errors.append(
                f"{page_title}: runtime-to-loading vertical gap is {gap_ratio:.2f} "
                f"of canvas height; maximum is "
                f"{INTEGRATED_MAX_RUNTIME_LOADING_GAP_RATIO:.2f}. Move loading "
                "dispatch directly below its receiving components."
            )

    canvas_diagonal = (page_width**2 + page_height**2) ** 0.5
    if canvas_diagonal > 0:
        for relationship, edge in resolved_relationships:
            source_cell = resolved_anchors.get(
                relationship.get("source_anchor_id")
            )
            target_cell = resolved_anchors.get(
                relationship.get("target_anchor_id")
            )
            if source_cell is None or target_cell is None:
                continue
            source_bounds = _cell_bounds(source_cell, cells_by_id, bounds_cache)
            target_bounds = _cell_bounds(target_cell, cells_by_id, bounds_cache)
            if source_bounds is None or target_bounds is None:
                continue
            source_center = (
                source_bounds[0] + source_bounds[2] / 2,
                source_bounds[1] + source_bounds[3] / 2,
            )
            target_center = (
                target_bounds[0] + target_bounds[2] / 2,
                target_bounds[1] + target_bounds[3] / 2,
            )
            span_ratio = (
                (
                    (source_center[0] - target_center[0]) ** 2
                    + (source_center[1] - target_center[1]) ** 2
                )
                ** 0.5
            ) / canvas_diagonal
            maximum_span = (
                INTEGRATED_MAX_LOADING_SPAN_RATIO
                if relationship.get("kind") == "loading_mapping"
                else INTEGRATED_MAX_OTHER_SPAN_RATIO
            )
            if span_ratio > maximum_span:
                errors.append(
                    f"{page_title}.{relationship.get('id')}: relationship spans "
                    f"{span_ratio:.2f} of the canvas diagonal; maximum is "
                    f"{maximum_span:.2f}. Reposition the source below its destination "
                    "or route through a meaningful local dispatch anchor."
                )
            horizontal_span_ratio = (
                abs(source_center[0] - target_center[0]) / max(1.0, page_width)
            )
            if (
                relationship.get("kind") == "loading_mapping"
                and horizontal_span_ratio
                > INTEGRATED_MAX_LOADING_HORIZONTAL_SPAN_RATIO
            ):
                errors.append(
                    f"{page_title}.{relationship.get('id')}: loading mapping spans "
                    f"{horizontal_span_ratio:.2f} of canvas width; maximum is "
                    f"{INTEGRATED_MAX_LOADING_HORIZONTAL_SPAN_RATIO:.2f}. "
                    "Place dispatch below the destination or introduce an "
                    "evidence-backed local mapping stage."
                )
            route_points = _edge_points(edge, source_bounds, target_bounds)
            route_length = _polyline_length(route_points)
            direct_length = max(1.0, math.dist(source_center, target_center))
            detour_ratio = route_length / direct_length
            if len(route_points) > 2 and detour_ratio > 2.5:
                errors.append(
                    f"{page_title}.{relationship.get('id')}: routed edge length is "
                    f"{detour_ratio:.2f} times its endpoint distance; remove the "
                    "large detour or regroup the connected components"
                )
    return errors


def _validate_png_export(
    path: Path,
    *,
    min_width: int = MIN_EXPORT_WIDTH,
    min_height: int = MIN_EXPORT_HEIGHT,
) -> list[str]:
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
    elif width < min_width or height < min_height:
        errors.append(
            f"PNG export is too small for an architecture page: {path} "
            f"({width}x{height}, minimum {min_width}x{min_height})"
        )
    if not saw_idat:
        errors.append(f"PNG export has no image data: {path}")
    if not saw_iend:
        errors.append(f"PNG export is incomplete: {path}")
    return errors


def _anchored_cells(path: Path) -> dict[str, ET.Element]:
    root = ET.parse(path).getroot()
    return {
        cell.attrib["dataAnchor"]: cell
        for cell in root.findall(".//mxCell[@dataAnchor]")
        if cell.attrib.get("dataAnchor")
    }


def _local_geometry(cell: ET.Element) -> tuple[float, float, float, float] | None:
    geometry = cell.find("mxGeometry")
    if geometry is None:
        return None
    return (
        _numeric(geometry.attrib.get("x")),
        _numeric(geometry.attrib.get("y")),
        _numeric(geometry.attrib.get("width")),
        _numeric(geometry.attrib.get("height")),
    )


def _edge_geometry_signature(cell: ET.Element) -> tuple[str, str]:
    geometry = cell.find("mxGeometry")
    return (
        cell.attrib.get("style", ""),
        ET.tostring(geometry, encoding="unicode") if geometry is not None else "",
    )


def _validate_visual_revision(
    *,
    draft_path: Path,
    final_path: Path,
) -> list[str]:
    try:
        draft_cells = _anchored_cells(draft_path)
        final_cells = _anchored_cells(final_path)
    except ET.ParseError as exc:
        return [f"unable to compare visual revision XML: {exc}"]

    shared_anchor_ids = set(draft_cells) & set(final_cells)
    vertex_anchor_ids = {
        anchor_id
        for anchor_id in shared_anchor_ids
        if draft_cells[anchor_id].attrib.get("vertex") == "1"
        and final_cells[anchor_id].attrib.get("vertex") == "1"
        and anchor_id.startswith("visual:")
    }
    changed_anchors = 0
    for anchor_id in vertex_anchor_ids:
        before = _local_geometry(draft_cells[anchor_id])
        after = _local_geometry(final_cells[anchor_id])
        if before is None or after is None:
            continue
        if any(abs(old - new) >= 15 for old, new in zip(before, after, strict=True)):
            changed_anchors += 1

    edge_anchor_ids = {
        anchor_id
        for anchor_id in shared_anchor_ids
        if draft_cells[anchor_id].attrib.get("edge") == "1"
        and final_cells[anchor_id].attrib.get("edge") == "1"
        and anchor_id.startswith("visual-rel:")
    }
    changed_edges = sum(
        _edge_geometry_signature(draft_cells[anchor_id])
        != _edge_geometry_signature(final_cells[anchor_id])
        for anchor_id in edge_anchor_ids
    )
    required_changed_anchors = max(
        INTEGRATED_MIN_CHANGED_ANCHORS,
        math.ceil(len(vertex_anchor_ids) * INTEGRATED_MIN_CHANGED_ANCHOR_RATIO),
    )
    errors: list[str] = []
    if changed_anchors < required_changed_anchors:
        errors.append(
            f"visual revision changed only {changed_anchors} required anchor "
            f"geometries; minimum is {required_changed_anchors}"
        )
    if changed_edges < INTEGRATED_MIN_CHANGED_EDGES:
        errors.append(
            f"visual revision rerouted only {changed_edges} required relationships; "
            f"minimum is {INTEGRATED_MIN_CHANGED_EDGES}"
        )
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
        "Draft 1 findings": ("draft 1", "首稿", "第一稿"),
        "revision": ("revision", "修改", "修订"),
        "final review": ("final review", "最终复查", "最终检查"),
    }
    errors: list[str] = []
    for label, markers in required_markers.items():
        if not any(marker.casefold() in lowered for marker in markers):
            errors.append(f"visual review is missing {label}: {path}")
    for title in page_titles:
        if title.casefold() not in lowered:
            errors.append(f"visual review does not discuss page: {title}")
    geometry_markers = (
        "geometry",
        "layout",
        "moved",
        "resized",
        "rerouted",
        "regrouped",
        "spacing",
        "布局",
        "移动",
        "缩放",
        "调整大小",
        "重路由",
        "重新分组",
        "间距",
    )
    if not any(marker.casefold() in lowered for marker in geometry_markers):
        errors.append(
            f"visual review documents no substantive geometry or routing revision: {path}"
        )
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
    png_minimums: dict[Path, tuple[int, int]] = {}
    complete_full_model = bool(
        plan
        and plan.get("detail_level") == "complete"
        and (
            plan.get("diagram_strategy") == "integrated_single_canvas"
            or len(plan.get("pages", [])) >= 3
        )
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
            if (
                plan.get("schema_version") == "2.2"
                and plan.get("diagram_strategy") == "integrated_single_canvas"
            ):
                errors.extend(
                    _validate_integrated_contract(
                        page_title=page_title,
                        model=model,
                        page_plan=matching,
                    )
                )
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
                    canvas = page.get("visual_contract", {}).get("canvas", {})
                    png_minimums[expected_png] = (
                        int(canvas.get("min_export_width", MIN_EXPORT_WIDTH)),
                        int(canvas.get("min_export_height", MIN_EXPORT_HEIGHT)),
                    )
    for image in sorted(png_exports, key=lambda item: item.as_posix()):
        min_width, min_height = png_minimums.get(
            image,
            (MIN_EXPORT_WIDTH, MIN_EXPORT_HEIGHT),
        )
        errors.extend(
            _validate_png_export(
                image,
                min_width=min_width,
                min_height=min_height,
            )
        )
    if images_dir:
        review_path = visual_review or images_dir.parent / "visual-review.md"
        errors.extend(_validate_visual_review(review_path, page_names))
        if (
            plan
            and plan.get("schema_version") == "2.2"
            and plan.get("diagram_strategy") == "integrated_single_canvas"
        ):
            draft_candidates = (
                drawio_path.parent / "architecture-draft1.drawio",
                drawio_path.parent / "architecture.draft-input.drawio",
                drawio_path.parent / "architecture-draft.drawio",
            )
            draft_path = next(
                (candidate for candidate in draft_candidates if candidate.exists()),
                None,
            )
            if draft_path is None:
                errors.append(
                    "integrated visual review requires a saved Draft 1 Draw.io "
                    "artifact for geometry comparison"
                )
            else:
                errors.extend(
                    _validate_visual_revision(
                        draft_path=draft_path,
                        final_path=drawio_path,
                    )
                )
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
