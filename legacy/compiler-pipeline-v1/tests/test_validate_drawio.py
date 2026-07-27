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
VALIDATOR_PATH = SKILL_ROOT / "scripts" / "validate_drawio.py"
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], ET.ElementTree]:
    extractor = load_module(EXTRACTOR_PATH, "extract_architecture_for_drawio_tests")
    builder = load_module(BUILDER_PATH, "build_architecture_ir_for_drawio_tests")
    view_builder = load_module(VIEW_BUILDER_PATH, "build_diagram_view_for_drawio_tests")
    layout_builder = load_module(LAYOUT_PATH, "layout_diagram_for_drawio_tests")
    renderer = load_module(RENDERER_PATH, "render_drawio_for_validator_tests")
    ir = builder.build_architecture_ir(extractor.extract_architecture(HY_V3_PATH))
    view = view_builder.build_diagram_view(ir)
    layout = layout_builder.build_layout_plan(view)
    return ir, view, layout, ET.ElementTree(ET.fromstring(renderer.render_drawio(view, layout)))


def write_tree(path: Path, tree: ET.ElementTree) -> Path:
    tree.write(path, encoding="unicode", short_empty_elements=True)
    return path


def root_cell_parent(tree: ET.ElementTree) -> ET.Element:
    root = tree.getroot().find(".//root")
    assert root is not None
    return root


def find_cell(tree: ET.ElementTree, cell_id: str) -> ET.Element:
    cell = tree.getroot().find(f".//mxCell[@id='{cell_id}']")
    assert cell is not None
    return cell


def remove_cell(tree: ET.ElementTree, cell_id: str) -> None:
    for root in tree.getroot().findall(".//root"):
        cell = root.find(f"mxCell[@id='{cell_id}']")
        if cell is not None:
            root.remove(cell)
            return
    raise AssertionError(f"cell not found: {cell_id}")


def validate(ir: dict[str, Any], path: Path, view: dict[str, Any] | None = None, layout: dict[str, Any] | None = None) -> list[str]:
    validator = load_module(VALIDATOR_PATH, "validate_drawio_for_tests")
    return validator.validate_drawio_with_view(ir, path, view=view, layout_plan=layout)


def valid_drawio_path(tmp_path: Path, tree: ET.ElementTree) -> Path:
    return write_tree(tmp_path / "architecture.drawio", tree)


def test_missing_node_fails(tmp_path: Path):
    ir, view, layout, tree = artifacts()
    remove_cell(tree, "self_attention")
    errors = validate(ir, write_tree(tmp_path / "missing-node.drawio", tree), view, layout)
    assert any("missing Draw.io node" in error for error in errors)


def test_missing_edge_fails(tmp_path: Path):
    ir, view, layout, tree = artifacts()
    remove_cell(tree, "input_norm_to_attention")
    errors = validate(ir, write_tree(tmp_path / "missing-edge.drawio", tree), view, layout)
    assert any("missing Draw.io edge" in error for error in errors)


def test_extra_semantic_node_fails(tmp_path: Path):
    ir, view, layout, tree = artifacts()
    root = root_cell_parent(tree)
    cell = ET.SubElement(root, "mxCell", {"id": "extra_node", "value": "Extra", "vertex": "1", "parent": "1"})
    ET.SubElement(cell, "mxGeometry", {"x": "1", "y": "1", "width": "10", "height": "10", "as": "geometry"})
    errors = validate(ir, write_tree(tmp_path / "extra-node.drawio", tree), view, layout)
    assert any("extra semantic Draw.io node" in error for error in errors)


def test_extra_semantic_edge_fails(tmp_path: Path):
    ir, view, layout, tree = artifacts()
    root = root_cell_parent(tree)
    cell = ET.SubElement(root, "mxCell", {"id": "extra_edge", "value": "", "edge": "1", "parent": "1", "source": "input", "target": "norm"})
    ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    errors = validate(ir, write_tree(tmp_path / "extra-edge.drawio", tree), view, layout)
    assert any("extra semantic Draw.io edge" in error for error in errors)


def test_decorative_cell_does_not_fail(tmp_path: Path):
    ir, view, layout, tree = artifacts()
    root = root_cell_parent(tree)
    cell = ET.SubElement(root, "mxCell", {"id": "decorative_extra_note", "value": "Note", "vertex": "1", "parent": "1"})
    ET.SubElement(cell, "mxGeometry", {"x": "1", "y": "1", "width": "10", "height": "10", "as": "geometry"})
    assert validate(ir, write_tree(tmp_path / "decorative.drawio", tree), view, layout) == []


def test_wrong_parent_fails(tmp_path: Path):
    ir, view, layout, tree = artifacts()
    find_cell(tree, "input").set("parent", "1")
    errors = validate(ir, write_tree(tmp_path / "wrong-parent.drawio", tree), view, layout)
    assert any("parent mismatch" in error for error in errors)
    assert any("still uses root parent" in error for error in errors)


def test_edge_source_target_and_ports_wrong_fail(tmp_path: Path):
    ir, view, layout, tree = artifacts()
    find_cell(tree, "input_to_embedding").set("source", "norm")
    find_cell(tree, "lm_head_to_logits_processor").set("targetPort", "hidden_states")
    errors = validate(ir, write_tree(tmp_path / "wrong-edge.drawio", tree), view, layout)
    assert any("source mismatch" in error for error in errors)
    assert any("targetPort mismatch" in error for error in errors)


def test_child_outside_container_fails(tmp_path: Path):
    ir, view, layout, tree = artifacts()
    geom = find_cell(tree, "input").find("mxGeometry")
    assert geom is not None
    geom.set("x", "9999")
    errors = validate(ir, write_tree(tmp_path / "outside.drawio", tree), view, layout)
    assert any("outside parent container" in error for error in errors)


def test_visible_edge_without_waypoints_fails(tmp_path: Path):
    ir, view, layout, tree = artifacts()
    geometry = find_cell(tree, "qkv_split_to_q_stream").find("mxGeometry")
    assert geometry is not None
    points = geometry.find("Array")
    assert points is not None
    geometry.remove(points)
    errors = validate(ir, write_tree(tmp_path / "missing-points.drawio", tree), view, layout)
    assert any("visible edge" in error and "waypoints" in error for error in errors)


def test_illegal_xml_fails(tmp_path: Path):
    ir, view, layout, _tree = artifacts()
    path = tmp_path / "invalid.drawio"
    path.write_text("<mxfile><diagram>", encoding="utf-8")
    errors = validate(ir, path, view, layout)
    assert any("not parseable" in error for error in errors)


def test_correct_hy_v3_drawio_passes(tmp_path: Path):
    ir, view, layout, tree = artifacts()
    assert validate(ir, valid_drawio_path(tmp_path, tree), view, layout) == []
