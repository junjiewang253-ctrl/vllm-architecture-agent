from __future__ import annotations

import importlib.util
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
    spec.loader.exec_module(module)
    return module


def build_hy_v3_ir() -> dict[str, Any]:
    extractor = load_module(EXTRACTOR_PATH, "extract_architecture_for_builder_tests")
    builder = load_module(BUILDER_PATH, "build_architecture_ir_for_tests")
    analysis = extractor.extract_architecture(HY_V3_PATH)
    return builder.build_architecture_ir(analysis)


def overview(ir: dict[str, Any]) -> dict[str, Any]:
    return next(page for page in ir["pages"] if page["id"] == "overview")


def decoder_detail(ir: dict[str, Any]) -> dict[str, Any]:
    return next(page for page in ir["pages"] if page["id"] == "decoder_layer_detail")


def nodes_by_id(ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in overview(ir)["nodes"]}


def detail_nodes_by_id(ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in decoder_detail(ir)["nodes"]}


def test_builds_overview_ir_for_hy_v3():
    ir = build_hy_v3_ir()
    assert ir["schema_version"] == "0.3"
    assert ir["detail_level"] == "overview"
    assert overview(ir)["id"] == "overview"


def test_builder_generates_two_pages():
    ir = build_hy_v3_ir()
    assert [page["id"] for page in ir["pages"]] == ["overview", "decoder_layer_detail"]


def test_identifies_top_level_model():
    nodes = nodes_by_id(build_hy_v3_ir())
    assert nodes["hyv3_for_causal_lm"]["label"] == "HYV3ForCausalLM"


def test_identifies_hyv3_model():
    nodes = nodes_by_id(build_hy_v3_ir())
    assert nodes["hyv3_model"]["label"] == "HYV3Model"
    assert nodes["hyv3_model"]["parent_id"] == "hyv3_for_causal_lm"


def test_identifies_vocab_parallel_embedding():
    nodes = nodes_by_id(build_hy_v3_ir())
    assert nodes["vocab_parallel_embedding"]["label"] == "VocabParallelEmbedding"
    assert "TP" in nodes["vocab_parallel_embedding"]["badges"]


def test_identifies_repeated_decoder_layer_with_repetition():
    nodes = nodes_by_id(build_hy_v3_ir())
    decoder = nodes["hyv3_decoder_layer"]
    assert decoder["label"] == "HYV3DecoderLayer"
    assert decoder["kind"] == "repeated_block"
    assert decoder["repetition"]["count_expression"] == "config.num_hidden_layers"
    assert decoder["repetition"]["local_start"] == "self.start_layer"
    assert decoder["repetition"]["local_end"] == "self.end_layer"


def test_dense_moe_variants_are_construction_phase():
    nodes = nodes_by_id(build_hy_v3_ir())
    variants = nodes["hyv3_decoder_layer"]["variants"]
    assert {
        (variant["component"], variant["phase"]) for variant in variants
    } == {
        ("HYV3FeedForward", "construction"),
        ("HYV3MoEFused", "construction"),
    }
    assert variants[0]["condition"] == "layer_idx < config.first_k_dense_replace"


def test_overview_no_longer_contains_independent_attention_node():
    nodes = nodes_by_id(build_hy_v3_ir())
    assert "hyv3_attention" not in nodes
    assert "self_attention" not in nodes


def test_decoder_detail_contains_self_attention_marked_tp():
    nodes = detail_nodes_by_id(build_hy_v3_ir())
    attention = nodes["self_attention"]
    assert attention["label"] == "HYV3Attention"
    assert "TP" in attention["badges"]


def test_decoder_detail_contains_required_stages():
    nodes = detail_nodes_by_id(build_hy_v3_ir())
    assert nodes["input_layernorm"]["display"]["label"] == "Input RMSNorm"
    assert nodes["post_attention_layernorm"]["display"]["label"] == "Post-Attention RMSNorm"
    assert nodes["ffn_stage"]["display"]["label"] == "Feed-Forward Stage"
    assert nodes["attention_residual"]["kind"] == "add"
    assert nodes["ffn_residual"]["kind"] == "add"


def test_moe_and_decoder_parallel_badges_are_conservative():
    nodes = nodes_by_id(build_hy_v3_ir())
    decoder = nodes["hyv3_decoder_layer"]
    assert "PP" in decoder["badges"]
    assert "EP" in decoder["badges"]


def test_decoder_detail_dense_moe_are_construction_phase():
    nodes = detail_nodes_by_id(build_hy_v3_ir())
    assert nodes["dense_ffn"]["phase"] == "construction"
    assert nodes["moe_ffn"]["phase"] == "construction"
    edges = {edge["kind"] for edge in decoder_detail(build_hy_v3_ir())["edges"]}
    assert "conditional_true" not in edges
    assert "conditional_false" not in edges


def test_external_parameters_enter_unresolved():
    ir = build_hy_v3_ir()
    unresolved = ir["unresolved"]
    config_item = next(item for item in unresolved if item["item"] == "external_config_values")
    assert "config.num_hidden_layers" in config_item["expressions"]
    assert "config.first_k_dense_replace" in config_item["expressions"]


def test_all_major_nodes_have_evidence():
    ir = build_hy_v3_ir()
    for page in ir["pages"]:
        for node in page["nodes"]:
            if node["kind"] in {"container", "note"}:
                continue
            assert node["evidence"], node["id"]


def test_display_label_and_hidden_overview_edge_are_generated():
    ir = build_hy_v3_ir()
    nodes = nodes_by_id(ir)
    assert nodes["vocab_parallel_embedding"]["display"]["label"] == "Token Embedding"
    edges = {edge["id"]: edge for edge in overview(ir)["edges"]}
    assert edges["top_invokes_model"]["display"]["visible"] is False


def test_logits_processor_edges_use_ports():
    edges = {edge["id"]: edge for edge in overview(build_hy_v3_ir())["edges"]}
    assert edges["model_hidden_states_to_logits_processor"]["target_port"] == "hidden_states"
    assert edges["lm_head_to_logits_processor"]["target_port"] == "lm_head"
