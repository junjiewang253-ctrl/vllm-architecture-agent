from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "src" / "skills" / "vllm-model-architecture-diagram"
EXTRACTOR_PATH = SKILL_ROOT / "scripts" / "extract_architecture.py"
BUILDER_PATH = SKILL_ROOT / "scripts" / "build_architecture_ir.py"
RENDERER_PATH = SKILL_ROOT / "scripts" / "render_drawio.py"
VISUAL_VALIDATOR_PATH = SKILL_ROOT / "scripts" / "validate_visual_layout.py"
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hy_v3_ir() -> dict[str, Any]:
    extractor = load_module(EXTRACTOR_PATH, "extract_architecture_for_visual_tests")
    builder = load_module(BUILDER_PATH, "build_architecture_ir_for_visual_tests")
    return builder.build_architecture_ir(extractor.extract_architecture(HY_V3_PATH))


def render_tree(ir: dict[str, Any]) -> ET.ElementTree:
    renderer = load_module(RENDERER_PATH, "render_drawio_for_visual_tests")
    return ET.ElementTree(ET.fromstring(renderer.render_drawio(ir)))


def write_tree(path: Path, tree: ET.ElementTree) -> Path:
    tree.write(path, encoding="unicode", short_empty_elements=True)
    return path


def validate(ir: dict[str, Any], path: Path) -> list[str]:
    validator = load_module(VISUAL_VALIDATOR_PATH, "validate_visual_layout_for_tests")
    return validator.validate_visual_layout(ir, path)


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


def test_correct_hy_v3_visual_layout_passes(tmp_path: Path):
    ir = hy_v3_ir()
    path = write_tree(tmp_path / "valid.drawio", render_tree(ir))
    assert validate(ir, path) == []


def test_node_overlap_fails(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    src_geom = find_cell(tree, "input").find("mxGeometry")
    dst_geom = find_cell(tree, "vocab_parallel_embedding").find("mxGeometry")
    assert src_geom is not None and dst_geom is not None
    dst_geom.set("x", src_geom.get("x", "0"))
    dst_geom.set("y", src_geom.get("y", "0"))
    errors = validate(ir, write_tree(tmp_path / "overlap.drawio", tree))
    assert any("semantic nodes overlap" in error for error in errors)


def test_runtime_edge_label_fails(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    find_cell(tree, "input_to_embedding").set("value", "hidden states")
    errors = validate(ir, write_tree(tmp_path / "runtime-label.drawio", tree))
    assert any("runtime edge" in error and "label" in error for error in errors)


def test_missing_residual_waypoints_fails(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    remove_edge_points(tree, "decoder_input_residual_to_attention_residual")
    errors = validate(ir, write_tree(tmp_path / "missing-points.drawio", tree))
    assert any("residual edge" in error and "waypoints" in error for error in errors)


def test_attention_branch_validation_fails_when_branch_removed(tmp_path: Path):
    ir = hy_v3_ir()
    page = next(page for page in ir["pages"] if page["id"] == "attention_detail")
    page["edges"] = [edge for edge in page["edges"] if edge["id"] != "split_qkv_to_fallback_path"]
    tree = render_tree(ir)
    errors = validate(ir, write_tree(tmp_path / "missing-branch.drawio", tree))
    assert any("attention detail missing branch edges" in error for error in errors)


def test_adaptation_region_validation_fails_when_region_removed(tmp_path: Path):
    ir = hy_v3_ir()
    page = next(page for page in ir["pages"] if page["id"] == "vllm_adaptation_map")
    page["nodes"] = [node for node in page["nodes"] if node["id"] != "region_weight_parallel"]
    tree = render_tree(ir)
    errors = validate(ir, write_tree(tmp_path / "missing-region.drawio", tree))
    assert any("adaptation map is missing visual regions" in error for error in errors)
