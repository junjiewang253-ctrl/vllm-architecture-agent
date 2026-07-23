from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "src" / "skills" / "vllm-model-architecture-diagram"
EXTRACTOR_PATH = SKILL_ROOT / "scripts" / "extract_architecture.py"
BUILDER_PATH = SKILL_ROOT / "scripts" / "build_architecture_ir.py"
VIEW_BUILDER_PATH = SKILL_ROOT / "scripts" / "build_diagram_view.py"
VIEW_VALIDATOR_PATH = SKILL_ROOT / "scripts" / "validate_diagram_view.py"
LAYOUT_PATH = SKILL_ROOT / "scripts" / "layout_diagram.py"
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    extractor = load_module(EXTRACTOR_PATH, "extract_architecture_for_view_tests")
    builder = load_module(BUILDER_PATH, "build_architecture_ir_for_view_tests")
    view_builder = load_module(VIEW_BUILDER_PATH, "build_diagram_view_for_tests")
    layout_builder = load_module(LAYOUT_PATH, "layout_diagram_for_tests")
    ir = builder.build_architecture_ir(extractor.extract_architecture(HY_V3_PATH))
    view = view_builder.build_diagram_view(ir)
    layout = layout_builder.build_layout_plan(view)
    return ir, view, layout


def page(root: dict[str, Any], page_id: str) -> dict[str, Any]:
    return next(item for item in root["pages"] if item["id"] == page_id)


def test_diagram_view_generates_required_pages_and_validates():
    ir, view, _layout = artifacts()
    validator = load_module(VIEW_VALIDATOR_PATH, "validate_diagram_view_for_tests")
    assert [item["id"] for item in view["pages"]] == [
        "overview",
        "decoder_layer_detail",
        "attention_detail",
        "moe_detail",
        "adapter_integration",
        "parallelism_weight_loading",
    ]
    assert "vllm_adaptation_map" not in {item["id"] for item in view["pages"]}
    assert validator.validate_diagram_view(ir, view) == []


def test_decoder_view_has_hidden_and_residual_lanes_without_handoff_nodes():
    _ir, view, _layout = artifacts()
    decoder = page(view, "decoder_layer_detail")
    assert {lane["id"] for lane in decoder["lanes"]} >= {"hidden_states_lane", "residual_lane"}
    assert "attention_residual" not in {node["semantic_id"] for node in decoder["visible_nodes"]}
    assert "ffn_residual" not in {node["semantic_id"] for node in decoder["visible_nodes"]}


def test_attention_view_has_hpc_fallback_and_qkv_lanes():
    _ir, view, _layout = artifacts()
    attention = page(view, "attention_detail")
    lanes = {lane["id"]: lane for lane in attention["lanes"]}
    assert {"hpc_fused_lane", "fallback_lane", "q_lane", "k_lane", "v_lane"} <= set(lanes)
    assert "attention_core" in lanes["hpc_fused_lane"]["member_ids"]
    assert "attention_core" in lanes["fallback_lane"]["member_ids"]


def test_moe_view_contains_gate_fused_moe_and_shared_experts():
    _ir, view, _layout = artifacts()
    moe_nodes = {node["semantic_id"] for node in page(view, "moe_detail")["visible_nodes"]}
    assert {"gate_linear", "fused_moe", "shared_experts"} <= moe_nodes


def test_adapter_and_weight_pages_are_split_by_concern():
    _ir, view, _layout = artifacts()
    adapter_nodes = {node["semantic_id"] for node in page(view, "adapter_integration")["visible_nodes"]}
    weight_nodes = {node["semantic_id"] for node in page(view, "parallelism_weight_loading")["visible_nodes"]}
    assert "qkv_proj_mapping" not in adapter_nodes
    assert "qkv_proj_mapping" in weight_nodes
    assert {region["id"] for region in page(view, "parallelism_weight_loading")["regions"]} == {
        "parallelism_region",
        "weight_loading_region",
    }


def test_layout_plan_is_deterministic_and_all_visible_edges_have_waypoints():
    _ir, view, layout = artifacts()
    layout_builder = load_module(LAYOUT_PATH, "layout_diagram_for_determinism_tests")
    assert layout == layout_builder.build_layout_plan(view)
    for item in layout["pages"]:
        for edge in item["edges"].values():
            if edge["visible"]:
                assert len(edge["points"]) >= 2


def test_qkv_and_residual_routes_use_expected_ports_and_tracks():
    _ir, view, layout = artifacts()
    attention_edges = {edge["semantic_id"]: edge for edge in page(view, "attention_detail")["visible_edges"]}
    assert attention_edges["qkv_split_to_q_stream"]["source_port"] == "q"
    assert attention_edges["qkv_split_to_k_stream"]["source_port"] == "k"
    assert attention_edges["qkv_split_to_v_stream"]["source_port"] == "v"
    assert attention_edges["v_stream_to_attention"]["target_port"] == "v"
    decoder_plan = page(layout, "decoder_layer_detail")
    residual_points = decoder_plan["edges"]["decoder_residual_to_input_norm"]["points"]
    hidden_points = decoder_plan["edges"]["decoder_hidden_to_input_norm"]["points"]
    assert max(point[1] for point in residual_points) > max(point[1] for point in hidden_points)
