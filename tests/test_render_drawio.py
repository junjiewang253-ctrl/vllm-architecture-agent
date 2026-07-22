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
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hy_v3_ir() -> dict[str, Any]:
    extractor = load_module(EXTRACTOR_PATH, "extract_architecture_for_render_tests")
    builder = load_module(BUILDER_PATH, "build_architecture_ir_for_render_tests")
    return builder.build_architecture_ir(extractor.extract_architecture(HY_V3_PATH))


def render_xml(ir: dict[str, Any]) -> str:
    renderer = load_module(RENDERER_PATH, "render_drawio_for_tests")
    return renderer.render_drawio(ir)


def cells_by_id(xml: str) -> dict[str, ET.Element]:
    root = ET.fromstring(xml)
    return {
        cell.attrib["id"]: cell
        for cell in root.findall(".//mxCell")
        if "id" in cell.attrib
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


def test_renderer_generates_legal_xml():
    ET.fromstring(render_xml(hy_v3_ir()))


def test_ir_node_ids_all_exist():
    ir = hy_v3_ir()
    cells = cells_by_id(render_xml(ir))
    assert semantic_node_ids(ir).issubset(cells)


def test_ir_edge_ids_all_exist():
    ir = hy_v3_ir()
    cells = cells_by_id(render_xml(ir))
    assert semantic_edge_ids(ir).issubset(cells)


def test_parent_id_maps_to_drawio_parent():
    ir = hy_v3_ir()
    cells = cells_by_id(render_xml(ir))
    for page in ir["pages"]:
        for node in page["nodes"]:
            expected = node.get("parent_id") or "1"
            assert cells[node["id"]].get("parent") == expected


def test_child_nodes_are_not_all_root_parented():
    ir = hy_v3_ir()
    cells = cells_by_id(render_xml(ir))
    semantic_nodes = semantic_node_ids(ir)
    parents = {cells[node_id].get("parent") for node_id in semantic_nodes}
    assert parents != {"1"}
    assert "hyv3_model" in parents
    assert "ffn_stage" in parents


def test_container_dimensions_exceed_internal_node_range():
    cells = cells_by_id(render_xml(hy_v3_ir()))
    parent = geometry(cells["hyv3_model"])
    child_ids = ["input", "vocab_parallel_embedding", "hyv3_decoder_layer", "final_residual_add", "norm"]
    max_right = max(geometry(cells[child_id])["x"] + geometry(cells[child_id])["width"] for child_id in child_ids)
    max_bottom = max(geometry(cells[child_id])["y"] + geometry(cells[child_id])["height"] for child_id in child_ids)
    assert parent["width"] > max_right
    assert parent["height"] > max_bottom


def test_repeated_block_displays_repetition():
    cells = cells_by_id(render_xml(hy_v3_ir()))
    value = cells["hyv3_decoder_layer"].get("value", "")
    assert "N x Decoder Layers" in value
    assert "Pipeline-local transformer stack" in value


def test_variants_generate_decorative_cell():
    cells = cells_by_id(render_xml(hy_v3_ir()))
    variants = cells["decorative_variants_hyv3_decoder_layer"]
    assert variants.get("parent") == "hyv3_decoder_layer"
    assert "Layer composition" in variants.get("value", "")
    assert "Remaining layers: MoE" in variants.get("value", "")


def test_badges_generate_decorative_cells():
    cells = cells_by_id(render_xml(hy_v3_ir()))
    assert "decorative_badge_self_attention_TP" in cells
    assert "decorative_badge_hyv3_decoder_layer_EP" in cells


def test_renderer_generates_two_drawio_pages():
    xml = render_xml(hy_v3_ir())
    assert [(diagram.get("id"), diagram.get("name")) for diagram in diagrams(xml)] == [
        ("overview", "Model Overview"),
        ("decoder_layer_detail", "HYV3DecoderLayer Detail"),
    ]


def test_hidden_edge_still_exists_with_opacity_zero():
    cells = cells_by_id(render_xml(hy_v3_ir()))
    edge = cells["top_invokes_model"]
    assert edge.get("edge") == "1"
    assert "opacity=0" in edge.get("style", "")
    assert edge.get("source") == "hyv3_for_causal_lm"
    assert edge.get("target") == "hyv3_model"


def test_same_input_renders_identical_xml():
    ir = hy_v3_ir()
    assert render_xml(ir) == render_xml(ir)


def test_renderer_does_not_generate_extra_semantic_nodes():
    ir = hy_v3_ir()
    cells = cells_by_id(render_xml(ir))
    actual = {
        cell_id
        for cell_id, cell in cells.items()
        if cell.get("vertex") == "1"
        and not cell_id.startswith("decorative_")
        and cell_id not in {"0", "1"}
    }
    assert actual == semantic_node_ids(ir)


def test_renderer_does_not_generate_extra_semantic_edges():
    ir = hy_v3_ir()
    cells = cells_by_id(render_xml(ir))
    actual = {
        cell_id
        for cell_id, cell in cells.items()
        if cell.get("edge") == "1"
        and not cell_id.startswith("decorative_")
        and cell_id not in {"0", "1"}
    }
    assert actual == semantic_edge_ids(ir)
