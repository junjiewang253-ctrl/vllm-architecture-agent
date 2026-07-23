from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "src" / "skills" / "vllm-model-architecture-diagram"
EXTRACTOR_PATH = SKILL_ROOT / "scripts" / "extract_architecture.py"
BUILDER_PATH = SKILL_ROOT / "scripts" / "build_architecture_ir.py"
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_hy_v3_ir() -> dict[str, Any]:
    extractor = load_module(EXTRACTOR_PATH, "extract_architecture_for_builder_tests")
    builder = load_module(BUILDER_PATH, "build_architecture_ir_for_tests")
    analysis = extractor.extract_architecture(HY_V3_PATH)
    return builder.build_architecture_ir(analysis)


def page(ir: dict[str, Any], page_id: str) -> dict[str, Any]:
    return next(item for item in ir["pages"] if item["id"] == page_id)


def nodes(ir: dict[str, Any], page_id: str) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in page(ir, page_id)["nodes"]}


def edges(ir: dict[str, Any], page_id: str) -> dict[str, dict[str, Any]]:
    return {edge["id"]: edge for edge in page(ir, page_id)["edges"]}


def port_ids(node: dict[str, Any]) -> set[str]:
    return {port["id"] for port in node.get("ports", [])}


def test_schema_version_and_six_pages():
    ir = build_hy_v3_ir()
    assert ir["schema_version"] == "0.5"
    assert [item["id"] for item in ir["pages"]] == [
        "overview",
        "decoder_layer_detail",
        "attention_detail",
        "moe_detail",
        "adapter_integration",
        "parallelism_weight_loading",
    ]


def test_overview_keeps_top_level_flow_small_and_ported():
    ir = build_hy_v3_ir()
    overview_nodes = nodes(ir, "overview")
    assert len(overview_nodes) == 11
    assert "hidden_states_output" in overview_nodes
    assert "logits_output" in overview_nodes
    assert "hyv3_attention" not in overview_nodes
    assert overview_nodes["logits_processor"]["display"]["label"] == "Logits Processor"
    assert {"hidden_states", "lm_head", "logits"} <= port_ids(overview_nodes["logits_processor"])
    assert edges(ir, "overview")["top_invokes_model"]["display"]["visible"] is False
    assert edges(ir, "overview")["input_to_embedding"]["display"]["show_label"] is False


def test_repeated_decoder_variants_remain_construction_semantics():
    decoder = nodes(build_hy_v3_ir(), "overview")["hyv3_decoder_layer"]
    assert decoder["kind"] == "repeated_block"
    assert decoder["repetition"]["count_expression"] == "config.num_hidden_layers"
    assert decoder["repetition"]["local_start"] == "self.start_layer"
    assert decoder["repetition"]["local_end"] == "self.end_layer"
    assert {(item["component"], item["phase"]) for item in decoder["variants"]} == {
        ("HYV3FeedForward", "construction"),
        ("HYV3MoEFused", "construction"),
    }
    assert decoder["badges"] == ["TP", "PP", "EP"]


def test_decoder_detail_uses_hidden_and_residual_ports_without_handoff_nodes():
    ir = build_hy_v3_ir()
    decoder_nodes = nodes(ir, "decoder_layer_detail")
    assert "attention_residual" not in decoder_nodes
    assert "ffn_residual" not in decoder_nodes
    assert {"hidden_states", "residual"} <= port_ids(decoder_nodes["decoder_input"])
    assert {"hidden_states", "residual", "normalized_hidden", "updated_residual"} <= port_ids(decoder_nodes["input_rmsnorm"])
    assert {"attention_output", "residual", "normalized_hidden", "updated_residual"} <= port_ids(decoder_nodes["post_attention_rmsnorm"])
    assert decoder_nodes["dense_ffn"]["phase"] == "construction"
    assert decoder_nodes["moe_ffn"]["phase"] == "construction"
    assert {edge["kind"] for edge in page(ir, "decoder_layer_detail")["edges"]}.isdisjoint({"conditional_true", "conditional_false"})


def test_attention_detail_has_qkv_ports_and_cache_ports():
    attention_nodes = nodes(build_hy_v3_ir(), "attention_detail")
    assert attention_nodes["qkv_projection"]["display"]["label"] == "QKV Projection"
    assert {"q", "k", "v"} <= port_ids(attention_nodes["qkv_split"])
    assert {"q", "k", "v", "kv_cache", "output"} <= port_ids(attention_nodes["attention_core"])
    assert {"write", "read"} <= port_ids(attention_nodes["kv_cache"])
    assert "TP" in attention_nodes["output_projection"]["badges"]


def test_attention_hpc_and_fallback_are_independent_and_v_bypasses_qk_norm():
    attention_edges = {(edge["source"], edge["target"]) for edge in page(build_hy_v3_ir(), "attention_detail")["edges"]}
    assert ("qkv_split", "hpc_fused_processing") in attention_edges
    assert ("hpc_fused_processing", "attention_core") in attention_edges
    assert ("qkv_split", "q_stream") in attention_edges
    assert ("qkv_split", "k_stream") in attention_edges
    assert ("qkv_split", "v_stream") in attention_edges
    assert ("v_stream", "fallback_q_norm") not in attention_edges
    assert ("v_stream", "fallback_k_norm") not in attention_edges
    assert ("v_stream", "attention_core") in attention_edges
    assert ("hpc_fused_processing", "fallback_q_norm") not in attention_edges


def test_moe_detail_contains_router_experts_shared_experts_and_ep():
    moe_nodes = nodes(build_hy_v3_ir(), "moe_detail")
    for node_id in ["gate_linear", "router_logits", "fused_moe", "routed_experts", "shared_experts", "eplb_metadata"]:
        assert node_id in moe_nodes
    assert "EP" in moe_nodes["fused_moe"]["badges"]
    assert moe_nodes["fused_moe"]["display"]["label"] == "FusedMoE"


def test_adapter_integration_scope_excludes_weight_loading_and_vllm_engine():
    adapter_nodes = nodes(build_hy_v3_ir(), "adapter_integration")
    for node_id in ["supports_pp", "supports_lora", "mixture_of_experts", "support_torch_compile"]:
        assert node_id in adapter_nodes
    forbidden = {"Scheduler", "Worker", "EngineCore", "Request Batching"}
    assert forbidden.isdisjoint({node["label"] for node in adapter_nodes.values()})
    assert not any("mapping" in node_id or "checkpoint" in node_id for node_id in adapter_nodes)


def test_parallelism_and_weight_loading_contains_expected_mapping_ports():
    ir = build_hy_v3_ir()
    parallel_nodes = nodes(ir, "parallelism_weight_loading")
    for node_id in ["tensor_parallel_lane", "pipeline_parallel_lane", "expert_parallel_lane"]:
        assert node_id in parallel_nodes
    for node_id in ["qkv_checkpoint_weights", "qkv_proj_mapping", "gate_up_checkpoint_weights", "gate_up_proj_mapping", "fused_moe_parameter_mapping"]:
        assert "weights_in" in port_ids(parallel_nodes[node_id])
        assert "weights_out" in port_ids(parallel_nodes[node_id])
    weight_edges = [edge for edge in page(ir, "parallelism_weight_loading")["edges"] if edge["kind"] == "weight_mapping"]
    assert weight_edges
    assert all(edge["source_port"] == "weights_out" and edge["target_port"] == "weights_in" for edge in weight_edges)


def test_external_parameters_enter_unresolved():
    unresolved = build_hy_v3_ir()["unresolved"]
    config_item = next(item for item in unresolved if item["item"] == "external_config_values")
    assert "config.num_hidden_layers" in config_item["expressions"]
    assert "config.first_k_dense_replace" in config_item["expressions"]


def test_all_major_nodes_and_edges_have_evidence():
    ir = build_hy_v3_ir()
    for ir_page in ir["pages"]:
        for node in ir_page["nodes"]:
            if node["kind"] in {"container", "note"}:
                continue
            assert node["evidence"], node["id"]
        for edge in ir_page["edges"]:
            assert edge["evidence"], edge["id"]
