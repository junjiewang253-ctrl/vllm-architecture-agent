from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "src" / "skills" / "vllm-model-architecture-diagram"
EXTRACTOR_PATH = SKILL_ROOT / "scripts" / "extract_architecture.py"
BUILDER_PATH = SKILL_ROOT / "scripts" / "build_architecture_ir.py"
VIEW_BUILDER_PATH = SKILL_ROOT / "scripts" / "build_diagram_view.py"
LAYOUT_PATH = SKILL_ROOT / "scripts" / "layout_diagram.py"
RENDERER_PATH = SKILL_ROOT / "scripts" / "render_drawio.py"
VISUAL_VALIDATOR_PATH = SKILL_ROOT / "scripts" / "validate_visual_layout.py"
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def artifacts() -> tuple[dict[str, Any], ET.ElementTree]:
    extractor = load_module(EXTRACTOR_PATH, "extract_architecture_for_visual_tests")
    builder = load_module(BUILDER_PATH, "build_architecture_ir_for_visual_tests")
    view_builder = load_module(VIEW_BUILDER_PATH, "build_diagram_view_for_visual_tests")
    layout_builder = load_module(LAYOUT_PATH, "layout_diagram_for_visual_tests")
    renderer = load_module(RENDERER_PATH, "render_drawio_for_visual_tests")
    ir = builder.build_architecture_ir(extractor.extract_architecture(HY_V3_PATH))
    view = view_builder.build_diagram_view(ir)
    layout = layout_builder.build_layout_plan(view)
    return ir, ET.ElementTree(ET.fromstring(renderer.render_drawio(view, layout)))


def write_tree(path: Path, tree: ET.ElementTree) -> Path:
    tree.write(path, encoding="unicode", short_empty_elements=True)
    return path


def validate(ir: dict[str, Any], path: Path) -> list[str]:
    validator = load_module(VISUAL_VALIDATOR_PATH, "validate_visual_layout_for_tests")
    return validator.validate_visual_layout(ir, path)


def metrics(ir: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    validator = load_module(VISUAL_VALIDATOR_PATH, "validate_visual_metrics_for_tests")
    result, errors = validator.build_layout_metrics(ir, path)
    assert errors == []
    return result


def find_cell(tree: ET.ElementTree, cell_id: str) -> ET.Element:
    cell = tree.getroot().find(f".//mxCell[@id='{cell_id}']")
    assert cell is not None
    return cell


def remove_edge_points(tree: ET.ElementTree, edge_id: str) -> None:
    geometry = find_cell(tree, edge_id).find("mxGeometry")
    assert geometry is not None
    points = geometry.find("Array")
    assert points is not None
    geometry.remove(points)


def test_correct_hy_v3_visual_layout_passes_and_metrics_are_clean(tmp_path: Path):
    ir, tree = artifacts()
    path = write_tree(tmp_path / "valid.drawio", tree)
    assert validate(ir, path) == []
    for item in metrics(ir, path):
        assert item["node_overlap_count"] == 0
        assert item["edge_node_intersection_count"] == 0
        assert item["text_overflow_count"] == 0
        assert 0.25 <= item["content_fill_ratio"] <= 0.80
        assert 1.65 <= item["page_aspect_ratio"] <= 1.95


def test_node_overlap_fails(tmp_path: Path):
    ir, tree = artifacts()
    src_geom = find_cell(tree, "input").find("mxGeometry")
    dst_geom = find_cell(tree, "vocab_parallel_embedding").find("mxGeometry")
    assert src_geom is not None and dst_geom is not None
    dst_geom.set("x", src_geom.get("x", "0"))
    dst_geom.set("y", src_geom.get("y", "0"))
    errors = validate(ir, write_tree(tmp_path / "overlap.drawio", tree))
    assert any("semantic nodes overlap" in error for error in errors)


def test_runtime_edge_label_fails(tmp_path: Path):
    ir, tree = artifacts()
    find_cell(tree, "input_to_embedding").set("value", "hidden states")
    errors = validate(ir, write_tree(tmp_path / "runtime-label.drawio", tree))
    assert any("runtime edge" in error and "label" in error for error in errors)


def test_missing_residual_waypoints_fails(tmp_path: Path):
    ir, tree = artifacts()
    remove_edge_points(tree, "decoder_residual_to_input_norm")
    errors = validate(ir, write_tree(tmp_path / "missing-points.drawio", tree))
    assert any("residual edge" in error and "waypoints" in error for error in errors)


def test_attention_branch_validation_fails_when_branch_removed(tmp_path: Path):
    ir, tree = artifacts()
    page = next(page for page in ir["pages"] if page["id"] == "attention_detail")
    page["edges"] = [edge for edge in page["edges"] if edge["id"] != "qkv_split_to_q_stream"]
    errors = validate(ir, write_tree(tmp_path / "missing-branch.drawio", tree))
    assert any("attention detail missing branch edges" in error for error in errors)


def test_edge_node_intersection_fails(tmp_path: Path):
    ir, tree = artifacts()
    geometry = find_cell(tree, "qkv_projection_to_hpc_fused").find("mxGeometry")
    assert geometry is not None
    points = geometry.find("Array")
    assert points is not None
    points.clear()
    for x, y in [(360, 385), (520, 385), (520, 218), (610, 218)]:
        ET.SubElement(points, "mxPoint", {"x": str(x), "y": str(y)})
    errors = validate(ir, write_tree(tmp_path / "intersects.drawio", tree))
    assert any("edge_node_intersection_count must be 0" in error for error in errors)


def test_text_overflow_fails(tmp_path: Path):
    ir, tree = artifacts()
    find_cell(tree, "input").set("value", "This label is intentionally far too long to fit inside the tiny input box")
    errors = validate(ir, write_tree(tmp_path / "overflow.drawio", tree))
    assert any("text_overflow_count must be 0" in error for error in errors)
