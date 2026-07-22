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
VALIDATOR_PATH = SKILL_ROOT / "scripts" / "validate_drawio.py"
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hy_v3_ir() -> dict[str, Any]:
    extractor = load_module(EXTRACTOR_PATH, "extract_architecture_for_drawio_tests")
    builder = load_module(BUILDER_PATH, "build_architecture_ir_for_drawio_tests")
    return builder.build_architecture_ir(extractor.extract_architecture(HY_V3_PATH))


def render_tree(ir: dict[str, Any]) -> ET.ElementTree:
    renderer = load_module(RENDERER_PATH, "render_drawio_for_validator_tests")
    return ET.ElementTree(ET.fromstring(renderer.render_drawio(ir)))


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
    root = root_cell_parent(tree)
    cell = find_cell(tree, cell_id)
    root.remove(cell)


def validate(ir: dict[str, Any], path: Path) -> list[str]:
    validator = load_module(VALIDATOR_PATH, "validate_drawio_for_tests")
    return validator.validate_drawio(ir, path)


def valid_drawio_path(tmp_path: Path, ir: dict[str, Any]) -> Path:
    return write_tree(tmp_path / "overview.drawio", render_tree(ir))


def test_missing_node_fails(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    remove_cell(tree, "hyv3_attention")
    errors = validate(ir, write_tree(tmp_path / "missing-node.drawio", tree))
    assert any("missing Draw.io node" in error for error in errors)


def test_missing_edge_fails(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    remove_cell(tree, "decoder_invokes_attention")
    errors = validate(ir, write_tree(tmp_path / "missing-edge.drawio", tree))
    assert any("missing Draw.io edge" in error for error in errors)


def test_extra_semantic_node_fails(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    root = root_cell_parent(tree)
    cell = ET.SubElement(
        root,
        "mxCell",
        {"id": "extra_node", "value": "Extra", "vertex": "1", "parent": "1"},
    )
    ET.SubElement(cell, "mxGeometry", {"x": "1", "y": "1", "width": "10", "height": "10", "as": "geometry"})
    errors = validate(ir, write_tree(tmp_path / "extra-node.drawio", tree))
    assert any("extra semantic Draw.io node" in error for error in errors)


def test_extra_semantic_edge_fails(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    root = root_cell_parent(tree)
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": "extra_edge",
            "value": "",
            "edge": "1",
            "parent": "1",
            "source": "input",
            "target": "norm",
        },
    )
    ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    errors = validate(ir, write_tree(tmp_path / "extra-edge.drawio", tree))
    assert any("extra semantic Draw.io edge" in error for error in errors)


def test_decorative_cell_does_not_fail(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    root = root_cell_parent(tree)
    cell = ET.SubElement(
        root,
        "mxCell",
        {"id": "decorative_extra_note", "value": "Note", "vertex": "1", "parent": "1"},
    )
    ET.SubElement(cell, "mxGeometry", {"x": "1", "y": "1", "width": "10", "height": "10", "as": "geometry"})
    assert validate(ir, write_tree(tmp_path / "decorative.drawio", tree)) == []


def test_wrong_parent_fails(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    find_cell(tree, "input").set("parent", "1")
    errors = validate(ir, write_tree(tmp_path / "wrong-parent.drawio", tree))
    assert any("parent mismatch" in error for error in errors)
    assert any("still uses root parent" in error for error in errors)


def test_edge_source_wrong_fails(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    find_cell(tree, "input_to_embedding").set("source", "norm")
    errors = validate(ir, write_tree(tmp_path / "wrong-source.drawio", tree))
    assert any("source mismatch" in error for error in errors)


def test_edge_target_wrong_fails(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    find_cell(tree, "input_to_embedding").set("target", "norm")
    errors = validate(ir, write_tree(tmp_path / "wrong-target.drawio", tree))
    assert any("target mismatch" in error for error in errors)


def test_child_outside_container_fails(tmp_path: Path):
    ir = hy_v3_ir()
    tree = render_tree(ir)
    geom = find_cell(tree, "input").find("mxGeometry")
    assert geom is not None
    geom.set("x", "9999")
    errors = validate(ir, write_tree(tmp_path / "outside.drawio", tree))
    assert any("outside parent container" in error for error in errors)


def test_illegal_xml_fails(tmp_path: Path):
    ir = hy_v3_ir()
    path = tmp_path / "invalid.drawio"
    path.write_text("<mxfile><diagram>", encoding="utf-8")
    errors = validate(ir, path)
    assert any("not parseable" in error for error in errors)


def test_correct_hy_v3_overview_passes(tmp_path: Path):
    ir = hy_v3_ir()
    path = valid_drawio_path(tmp_path, ir)
    assert validate(ir, path) == []
