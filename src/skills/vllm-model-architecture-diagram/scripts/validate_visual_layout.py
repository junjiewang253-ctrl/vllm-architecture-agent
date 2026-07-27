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
EDGE_CROSSING_THRESHOLDS = {
    "overview": 1,
    "decoder_detail": 1,
    "attention_detail": 10,
    "moe_detail": 2,
    "adapter_integration": 4,
    "parallelism": 4,
    "weight_loading": 10,
    "pipeline": 1,
    "block_with_residual": 2,
    "branch_merge": 3,
    "routed_container": 2,
    "mapping_dispatch": 4,
    "strategy_matrix": 0,
    "boundary_map": 4,
    "model_execution_overview": 1,
    "decoder_block": 2,
    "attention_adaptation": 3,
    "moe_execution": 2,
    "checkpoint_weight_loading": 4,
    "parallel_strategies": 0,
    "adapter_boundary": 4,
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


def _edge_points(cell: ET.Element) -> list[tuple[float, float]]:
    geometry = cell.find("mxGeometry")
    if geometry is None:
        return []
    return [
        (_parse_float(point.get("x")), _parse_float(point.get("y")))
        for point in geometry.findall("Array/mxPoint")
    ]


def _segments(points: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return list(zip(points, points[1:]))


def _segment_intersects_box(segment: tuple[tuple[float, float], tuple[float, float]], box: Box) -> bool:
    (x1, y1), (x2, y2) = segment
    if abs(y1 - y2) < 0.001:
        y = y1
        if y <= box.y or y >= box.bottom:
            return False
        return max(min(x1, x2), box.x) < min(max(x1, x2), box.right)
    if abs(x1 - x2) < 0.001:
        x = x1
        if x <= box.x or x >= box.right:
            return False
        return max(min(y1, y2), box.y) < min(max(y1, y2), box.bottom)
    return False


def _segments_cross(a: tuple[tuple[float, float], tuple[float, float]], b: tuple[tuple[float, float], tuple[float, float]]) -> bool:
    (ax1, ay1), (ax2, ay2) = a
    (bx1, by1), (bx2, by2) = b
    a_horizontal = abs(ay1 - ay2) < 0.001
    b_horizontal = abs(by1 - by2) < 0.001
    if a_horizontal == b_horizontal:
        return False
    if a_horizontal:
        hx1, hx2, hy = sorted([ax1, ax2])[0], sorted([ax1, ax2])[1], ay1
        vx, vy1, vy2 = bx1, sorted([by1, by2])[0], sorted([by1, by2])[1]
    else:
        hx1, hx2, hy = sorted([bx1, bx2])[0], sorted([bx1, bx2])[1], by1
        vx, vy1, vy2 = ax1, sorted([ay1, ay2])[0], sorted([ay1, ay2])[1]
    return hx1 < vx < hx2 and vy1 < hy < vy2


def _estimated_text_fits(value: str, box: Box) -> bool:
    if not value:
        return True
    lines = value.split("\n")
    max_chars = max(len(line) for line in lines)
    needed_width = max_chars * 5.2 + 22
    needed_height = len(lines) * 15 + 16
    return needed_width <= box.width + 0.001 and needed_height <= box.height + 0.001


def _edge_visible(edge: dict[str, Any]) -> bool:
    if edge.get("visible") is False:
        return False
    display = edge.get("display")
    return not (isinstance(display, dict) and display.get("visible") is False)


def _edge_show_label(edge: dict[str, Any]) -> bool:
    display = edge.get("display")
    return isinstance(display, dict) and display.get("show_label") is True


def _node_id(node: dict[str, Any]) -> str | None:
    node_id = node.get("id") or node.get("semantic_id")
    return node_id if isinstance(node_id, str) else None


def _edge_id(edge: dict[str, Any]) -> str | None:
    edge_id = edge.get("id") or edge.get("semantic_id")
    return edge_id if isinstance(edge_id, str) else None


def _node_kind(node: dict[str, Any]) -> str:
    return str(node.get("kind") or node.get("semantic_kind") or node.get("type") or "")


def _edge_kind(edge: dict[str, Any]) -> str:
    return str(edge.get("kind") or edge.get("style_kind") or edge.get("edge_type") or "")


def _edge_source(edge: dict[str, Any]) -> str | None:
    source = edge.get("source")
    return source if isinstance(source, str) else None


def _edge_target(edge: dict[str, Any]) -> str | None:
    target = edge.get("target")
    return target if isinstance(target, str) else None


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


def _page_type(page: dict[str, Any]) -> str:
    return str(page.get("page_type") or page.get("view_kind") or page.get("id") or "")


def _edge_is_runtime(edge: dict[str, Any]) -> bool:
    return _edge_kind(edge) in {"runtime", "tensor_flow", "runtime_flow"}


def _edge_is_residual(edge: dict[str, Any]) -> bool:
    return _edge_kind(edge) in {"residual", "residual_flow"}


def build_layout_metrics(ir: dict[str, Any], drawio_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    pages, errors = _pages(drawio_path)
    if errors:
        return [], errors
    ir_pages = ir.get("pages")
    if not isinstance(ir_pages, list):
        return [], ["IR pages must be a list"]
    metrics: list[dict[str, Any]] = []
    for page in ir_pages:
        if not isinstance(page, dict):
            continue
        page_id = page.get("id")
        if not isinstance(page_id, str) or page_id not in pages:
            errors.append(f"missing Draw.io page for visual validation: {page_id!r}")
            continue
        metrics.append(_page_metrics(page, pages[page_id], errors))
    return metrics, errors


def validate_visual_layout(ir: dict[str, Any], drawio_path: Path) -> list[str]:
    metrics, errors = build_layout_metrics(ir, drawio_path)
    for item in metrics:
        page_id = str(item["page_id"])
        page_type = str(item["page_type"])
        if item["node_overlap_count"] != 0:
            errors.append(f"page {page_id}: node_overlap_count must be 0")
        if item["edge_node_intersection_count"] != 0:
            errors.append(f"page {page_id}: edge_node_intersection_count must be 0")
        if item["text_overflow_count"] != 0:
            errors.append(f"page {page_id}: text_overflow_count must be 0")
        min_fill = 0.18 if page_type in {"strategy_matrix", "pipeline", "block_with_residual"} else 0.25
        if item["content_fill_ratio"] < min_fill or item["content_fill_ratio"] > 0.80:
            errors.append(f"page {page_id}: content_fill_ratio out of range")
        threshold = EDGE_CROSSING_THRESHOLDS.get(page_type, 4)
        if item["edge_crossing_count"] > threshold:
            errors.append(f"page {page_id}: edge_crossing_count exceeds threshold {threshold}")
        if item["page_aspect_ratio"] < 1.65 or item["page_aspect_ratio"] > 1.95:
            errors.append(f"page {page_id}: page aspect ratio outside 1.65-1.95")
    return errors


def _page_metrics(ir_page: dict[str, Any], page: Page, errors: list[str]) -> dict[str, Any]:
    base_error_count = len(errors)
    _validate_page(ir_page, page, errors)
    nodes = _page_nodes(ir_page)
    node_by_id = {
        node_id: node
        for node in nodes
        for node_id in [_node_id(node)]
        if node_id is not None
    }
    boxes = {
        node_id: box
        for node in nodes
        for node_id in [_node_id(node)]
        if node_id is not None
        for box in [_absolute_box(node_id, page, node_by_id)]
        if box is not None
    }
    leaf_nodes = [node for node in nodes if _node_kind(node) not in CONTAINER_KINDS]
    node_overlap_count = 0
    for index, left in enumerate(leaf_nodes):
        left_id = _node_id(left)
        for right in leaf_nodes[index + 1 :]:
            right_id = _node_id(right)
            if left_id in boxes and right_id in boxes and _overlaps(boxes[left_id], boxes[right_id]):
                node_overlap_count += 1
    edge_node_intersection_count = 0
    visible_edge_segments: list[tuple[str, tuple[tuple[float, float], tuple[float, float]]]] = []
    text_overflow_count = 0
    for node in nodes:
        node_id = _node_id(node)
        if node_id is None:
            continue
        cell = page.cells.get(node_id)
        box = boxes.get(node_id)
        if cell is not None and box is not None and _node_kind(node) not in CONTAINER_KINDS:
            if not _estimated_text_fits(cell.get("value") or "", box):
                text_overflow_count += 1
    all_edge_points: list[tuple[float, float]] = []
    for edge in _page_edges(ir_page):
        edge_id = _edge_id(edge)
        if edge_id is None:
            continue
        cell = page.cells.get(edge_id)
        if cell is None:
            continue
        display = edge.get("display")
        if (isinstance(display, dict) and display.get("visible") is False) or edge.get("visible") is False:
            continue
        points = _edge_points(cell)
        if len(points) < 2:
            continue
        all_edge_points.extend(points)
        for segment in _segments(points):
            visible_edge_segments.append((edge_id, segment))
            for node_id, box in boxes.items():
                if _node_kind(node_by_id.get(node_id, {})) in CONTAINER_KINDS:
                    continue
                if node_id in {_edge_source(edge), _edge_target(edge)}:
                    continue
                if _segment_intersects_box(segment, box):
                    edge_node_intersection_count += 1
    edge_crossing_count = 0
    for index, (left_id, left_segment) in enumerate(visible_edge_segments):
        for right_id, right_segment in visible_edge_segments[index + 1 :]:
            if left_id == right_id:
                continue
            if _segments_cross(left_segment, right_segment):
                edge_crossing_count += 1
    if boxes:
        xs = [box.x for box in boxes.values()] + [point[0] for point in all_edge_points]
        ys = [box.y for box in boxes.values()] + [point[1] for point in all_edge_points]
        rights = [box.right for box in boxes.values()] + [point[0] for point in all_edge_points]
        bottoms = [box.bottom for box in boxes.values()] + [point[1] for point in all_edge_points]
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(rights)
        max_y = max(bottoms)
        content_fill_ratio = ((max_x - min_x) * (max_y - min_y)) / (page.width * page.height)
    else:
        content_fill_ratio = 0.0
    return {
        "page_id": ir_page.get("id"),
        "page_type": _page_type(ir_page),
        "node_count": len(nodes),
        "visible_edge_count": sum(
            1
            for edge in _page_edges(ir_page)
            if not (
                (isinstance(edge.get("display"), dict) and edge["display"].get("visible") is False)
                or edge.get("visible") is False
            )
        ),
        "node_overlap_count": node_overlap_count,
        "edge_node_intersection_count": edge_node_intersection_count,
        "edge_crossing_count": edge_crossing_count,
        "text_overflow_count": text_overflow_count,
        "page_aspect_ratio": round(page.width / page.height, 4) if page.height else 0,
        "content_fill_ratio": round(content_fill_ratio, 4),
        "cross_region_edge_count": 0,
        "new_error_count": len(errors) - base_error_count,
    }


def _validate_page(ir_page: dict[str, Any], page: Page, errors: list[str]) -> None:
    if page.width <= 0 or page.height <= 0:
        errors.append(f"page {page.page_id}: page dimensions must be positive")
    elif page.width / page.height > 2.4:
        errors.append(f"page {page.page_id}: page aspect ratio exceeds 2.4")

    nodes = _page_nodes(ir_page)
    node_by_id = {
        node_id: node
        for node in nodes
        for node_id in [_node_id(node)]
        if node_id is not None
    }
    boxes: dict[str, Box] = {}
    for node in nodes:
        node_id = _node_id(node)
        if node_id is None:
            continue
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

    leaf_nodes = [node for node in nodes if _node_kind(node) not in CONTAINER_KINDS]
    for index, left in enumerate(leaf_nodes):
        left_id = _node_id(left)
        if left_id is None:
            continue
        left_box = boxes.get(left_id)
        if left_box is None:
            continue
        for right in leaf_nodes[index + 1 :]:
            right_id = _node_id(right)
            if right_id is None:
                continue
            right_box = boxes.get(right_id)
            if right_box is not None and _overlaps(left_box, right_box, margin=0.0):
                errors.append(f"page {page.page_id}: semantic nodes overlap: {left_id} and {right_id}")

    _validate_decorations(page, boxes, node_by_id, errors)
    _validate_edges(ir_page, page, errors)
    if _page_type(ir_page) in {"attention_detail", "attention_adaptation", "branch_merge"}:
        _validate_attention_branches(ir_page, errors)
    if _page_type(ir_page) == "adaptation_map":
        present = {node_id for node in nodes for node_id in [_node_id(node)] if node_id is not None}
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
        if cell_id.startswith("decorative_region_"):
            continue
        box = _absolute_box(cell_id, page, {}) or _geometry(cell)
        if box is None:
            continue
        parent = cell.get("parent")
        for node_id, semantic_box in boxes.items():
            if _node_kind(node_by_id.get(node_id, {})) in CONTAINER_KINDS:
                continue
            if parent == node_id:
                continue
            if node_id not in semantic_ids:
                continue
            if _overlaps(box, semantic_box, margin=0.0):
                errors.append(f"page {page.page_id}: decorative cell {cell_id} overlaps semantic node {node_id}")


def _validate_edges(ir_page: dict[str, Any], page: Page, errors: list[str]) -> None:
    for edge in _page_edges(ir_page):
        edge_id = _edge_id(edge)
        if edge_id is None:
            continue
        cell = page.cells.get(edge_id)
        if cell is None:
            continue
        value = cell.get("value") or ""
        if _edge_visible(edge) and len(value) > 32:
            errors.append(f"page {page.page_id}: edge label too long on {edge_id}")
        if _edge_is_runtime(edge) and value:
            errors.append(f"page {page.page_id}: runtime edge {edge_id} should not show a label")
        if _edge_is_runtime(edge) and _edge_show_label(edge):
            errors.append(f"page {page.page_id}: runtime edge {edge_id} has show_label=true")
        if _edge_is_residual(edge) and not _has_route_points(cell):
            errors.append(f"page {page.page_id}: residual edge {edge_id} must include route waypoints")


def _validate_attention_branches(ir_page: dict[str, Any], errors: list[str]) -> None:
    edges = {
        (_edge_source(edge), _edge_target(edge))
        for edge in _page_edges(ir_page)
    }
    required = (
        {
            ("qkv_projection", "hpc_fused_processing"),
            ("hpc_fused_processing", "attention_core"),
            ("qkv_split", "q_stream"),
            ("q_stream", "fallback_q_norm"),
            ("fallback_q_norm", "rotary_embedding"),
            ("rotary_embedding", "attention_core"),
        }
        if any("attention_core" in pair for pair in edges)
        else {
            ("qkv_projection", "hpc_rope_norm"),
            ("hpc_rope_norm", "attention_backend"),
            ("qkv_split", "q_stream"),
            ("q_stream", "q_norm"),
            ("q_norm", "rope"),
            ("rope", "attention_backend"),
            ("v_stream", "attention_backend"),
        }
    )
    missing = sorted(required - edges)
    if missing:
        errors.append(f"page {ir_page.get('id')}: attention detail missing branch edges: {missing}")
    if (
        ("hpc_fused_processing", "fallback_q_norm") in edges
        or ("fallback_q_norm", "hpc_fused_processing") in edges
        or ("hpc_fused_processing", "q_norm") in edges
        or ("q_norm", "hpc_fused_processing") in edges
    ):
        errors.append(f"page {ir_page.get('id')}: HPC and fallback paths must not be serially connected")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate visual Draw.io layout constraints.")
    parser.add_argument("ir", type=Path, help="Architecture IR JSON file")
    parser.add_argument("drawio", type=Path, help="Draw.io XML file")
    parser.add_argument("--metrics-output", type=Path, help="Write layout metrics JSON")
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
        metrics, metric_errors = build_layout_metrics(ir, args.drawio)
        if args.metrics_output is not None:
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_output.write_text(json.dumps({"schema_version": "0.1", "pages": metrics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        errors = validate_visual_layout(ir, args.drawio)
        errors = metric_errors + [error for error in errors if error not in metric_errors]
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
