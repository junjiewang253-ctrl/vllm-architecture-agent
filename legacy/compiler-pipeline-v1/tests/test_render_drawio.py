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
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    extractor = load_module(EXTRACTOR_PATH, "extract_architecture_for_render_tests")
    builder = load_module(BUILDER_PATH, "build_architecture_ir_for_render_tests")
    view_builder = load_module(VIEW_BUILDER_PATH, "build_diagram_view_for_render_tests")
    layout_builder = load_module(LAYOUT_PATH, "layout_diagram_for_render_tests")
    renderer = load_module(RENDERER_PATH, "render_drawio_for_tests")
    ir = builder.build_architecture_ir(extractor.extract_architecture(HY_V3_PATH))
    view = view_builder.build_diagram_view(ir)
    layout = layout_builder.build_layout_plan(view)
    xml = renderer.render_drawio(view, layout)
    return ir, view, layout, xml


def cells_by_id(xml: str) -> dict[str, ET.Element]:
    root = ET.fromstring(xml)
    return {cell.attrib["id"]: cell for cell in root.findall(".//mxCell") if "id" in cell.attrib}


def cells_by_page(xml: str) -> dict[str, set[str]]:
    root = ET.fromstring(xml)
    return {
        str(diagram.get("id")): {cell.get("id") for cell in diagram.findall(".//mxCell") if cell.get("id")}
        for diagram in root.findall("diagram")
    }


def diagrams(xml: str) -> list[ET.Element]:
    return ET.fromstring(xml).findall("diagram")


def geometry(cell: ET.Element) -> dict[str, float]:
    geom = cell.find("mxGeometry")
    assert geom is not None
    return {
        "x": float(geom.get("x", "0")),
        "y": float(geom.get("y", "0")),
        "width": float(geom.get("width", "0")),
        "height": float(geom.get("height", "0")),
    }


def semantic_node_ids(ir: dict[str, Any]) -> set[str]:
    return {node["id"] for page in ir["pages"] for node in page["nodes"]}


def semantic_edge_ids(ir: dict[str, Any]) -> set[str]:
    return {edge["id"] for page in ir["pages"] for edge in page["edges"]}


def test_renderer_generates_legal_xml_and_seven_pages():
    _ir, _view, _layout, xml = artifacts()
    assert [(diagram.get("id"), diagram.get("name")) for diagram in diagrams(xml)] == [
        ("overview", "Model Overview"),
        ("decoder_layer_detail", "HYV3DecoderLayer Detail"),
        ("attention_detail", "HYV3Attention Detail"),
        ("moe_detail", "HYV3MoEFused Detail"),
        ("adapter_integration", "vLLM Adapter Integration"),
        ("parallelism", "Parallelism"),
        ("weight_loading", "Weight Loading"),
    ]


def test_ir_node_and_edge_ids_all_exist():
    ir, _view, _layout, xml = artifacts()
    cells = cells_by_id(xml)
    assert semantic_node_ids(ir).issubset(cells)
    assert semantic_edge_ids(ir).issubset(cells)


def test_parent_id_maps_to_drawio_parent_and_children_not_all_rooted():
    ir, _view, _layout, xml = artifacts()
    cells = cells_by_id(xml)
    for ir_page in ir["pages"]:
        for node in ir_page["nodes"]:
            expected = node.get("parent_id") or "1"
            assert cells[node["id"]].get("parent") == expected
    parents = {cells[node_id].get("parent") for node_id in semantic_node_ids(ir)}
    assert parents != {"1"}
    assert "hyv3_model" in parents
    assert "ffn_stage" in parents


def test_container_dimensions_exceed_internal_node_range():
    _ir, _view, _layout, xml = artifacts()
    cells = cells_by_id(xml)
    parent = geometry(cells["hyv3_model"])
    child_ids = ["input", "vocab_parallel_embedding", "hyv3_decoder_layer", "final_residual_add", "norm"]
    max_right = max(geometry(cells[child_id])["x"] + geometry(cells[child_id])["width"] for child_id in child_ids)
    max_bottom = max(geometry(cells[child_id])["y"] + geometry(cells[child_id])["height"] for child_id in child_ids)
    assert parent["width"] > max_right
    assert parent["height"] > max_bottom


def test_decorative_repetition_variants_and_badges_exist():
    _ir, _view, _layout, xml = artifacts()
    cells = cells_by_id(xml)
    assert "decorative_header_hyv3_decoder_layer" in cells
    assert "N x Decoder Layers" in cells["decorative_header_hyv3_decoder_layer"].get("value", "")
    assert "decorative_badge_hyv3_decoder_layer_EP" in cells
    assert "decorative_badge_output_projection_TP" in cells


def test_hidden_edge_still_exists_with_opacity_zero_and_ports():
    _ir, _view, _layout, xml = artifacts()
    edge = cells_by_id(xml)["top_invokes_model"]
    assert edge.get("edge") == "1"
    assert "opacity=0" in edge.get("style", "")
    assert edge.get("source") == "hyv3_for_causal_lm"
    assert edge.get("target") == "hyv3_model"
    assert edge.get("sourcePort") == "model_out"
    assert edge.get("targetPort") == "hidden_in"


def test_runtime_edges_default_to_no_visible_labels():
    _ir, _view, _layout, xml = artifacts()
    cells = cells_by_id(xml)
    assert cells["input_to_embedding"].get("value", "") == ""
    assert cells["decoder_hidden_to_input_norm"].get("value", "") == ""


def test_visible_edges_have_waypoints_and_ports():
    _ir, view, _layout, xml = artifacts()
    cells = cells_by_id(xml)
    for page in view["pages"]:
        for edge in page["visible_edges"]:
            if edge["route_class"] == "hidden_semantic":
                continue
            cell = cells[edge["semantic_id"]]
            assert cell.get("sourcePort") == edge["source_port"]
            assert cell.get("targetPort") == edge["target_port"]
            geometry_element = cell.find("mxGeometry")
            assert geometry_element is not None
            assert geometry_element.find("Array/mxPoint") is not None


def test_qkv_nodes_are_only_on_attention_page():
    _ir, _view, _layout, xml = artifacts()
    by_page = cells_by_page(xml)
    assert "qkv_projection" in by_page["attention_detail"]
    assert "qkv_projection" not in by_page["overview"]
    assert "q_stream" in by_page["attention_detail"]
    assert "q_stream" not in by_page["decoder_layer_detail"]


def test_renderer_does_not_generate_extra_semantic_nodes_or_edges():
    ir, _view, _layout, xml = artifacts()
    cells = cells_by_id(xml)
    actual_nodes = {
        cell_id
        for cell_id, cell in cells.items()
        if cell.get("vertex") == "1" and not cell_id.startswith("decorative_") and cell_id not in {"0", "1"}
    }
    actual_edges = {
        cell_id
        for cell_id, cell in cells.items()
        if cell.get("edge") == "1" and not cell_id.startswith("decorative_") and cell_id not in {"0", "1"}
    }
    assert actual_nodes == semantic_node_ids(ir)
    assert actual_edges == semantic_edge_ids(ir)


def test_same_input_renders_identical_xml_and_layout_plan():
    ir, view, layout, xml = artifacts()
    layout_builder = load_module(LAYOUT_PATH, "layout_diagram_for_render_determinism")
    renderer = load_module(RENDERER_PATH, "render_drawio_for_render_determinism")
    assert layout == layout_builder.build_layout_plan(view)
    assert xml == renderer.render_drawio(view, layout)
