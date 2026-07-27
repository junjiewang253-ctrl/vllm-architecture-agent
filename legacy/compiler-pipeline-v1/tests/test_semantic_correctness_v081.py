from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "src" / "skills" / "vllm-model-architecture-diagram"
EXTRACTOR_PATH = SKILL_ROOT / "scripts" / "extract_architecture.py"
INVENTORY_PATH = SKILL_ROOT / "scripts" / "build_semantic_inventory.py"
BUILDER_PATH = SKILL_ROOT / "scripts" / "build_architecture_ir.py"
COVERAGE_PATH = SKILL_ROOT / "scripts" / "validate_semantic_coverage.py"
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"
GOLDEN_PATH = ROOT / "tests" / "golden" / "hy_v3-semantic-contract.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    extractor = load_module(EXTRACTOR_PATH, "extract_architecture_for_v081_tests")
    inventory_builder = load_module(INVENTORY_PATH, "build_semantic_inventory_for_v081_tests")
    builder = load_module(BUILDER_PATH, "build_architecture_ir_for_v081_tests")
    coverage_validator = load_module(COVERAGE_PATH, "validate_semantic_coverage_for_v081_tests")
    analysis = extractor.extract_architecture(HY_V3_PATH)
    inventory = inventory_builder.build_semantic_inventory(analysis)
    ir = builder.build_architecture_ir(analysis)
    coverage, errors = coverage_validator.validate_semantic_coverage(analysis, inventory, ir)
    assert errors == []
    return analysis, inventory, ir, coverage


def page(ir: dict[str, Any], page_id: str) -> dict[str, Any]:
    return next(item for item in ir["pages"] if item["id"] == page_id)


def nodes(ir: dict[str, Any], page_id: str) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in page(ir, page_id)["nodes"]}


def edges(ir: dict[str, Any], page_id: str) -> dict[str, dict[str, Any]]:
    return {edge["id"]: edge for edge in page(ir, page_id)["edges"]}


def fact_ids(item: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for evidence in item.get("evidence", []):
        if isinstance(evidence, dict):
            result.update(str(fact_id) for fact_id in evidence.get("fact_ids", []) if isinstance(fact_id, str))
    return result


def test_source_analysis_03_extracts_structured_semantic_facts():
    analysis, _inventory, _ir, _coverage = artifacts()
    assert analysis["schema_version"] == "0.3"
    methods = {(item["owner_class"], item["method"]) for item in analysis["method_control_flows"]}
    assert ("HYV3ForCausalLM", "_filter_weights") in methods
    assert ("HYV3Model", "load_weights") in methods
    hpc_call = next(item for item in analysis["calls"] if item.get("target") == "self.hpc_rope_norm")
    assert hpc_call["args"][0]["expression"] == "qkv"
    fused_moe_call = next(item for item in analysis["calls"] if item.get("target") == "FusedMoE")
    assert {"num_experts", "top_k", "enable_eplb", "num_redundant_experts", "shared_experts"} <= set(fused_moe_call["kwargs"])
    assert any(item.get("target") == "config" and item.get("path") == "model_config.hf_config" for item in analysis["config_accesses"])
    assert any("router.gate" in item.get("condition", "") for item in analysis["branches"])
    assert any("default_weight_loader" in item.get("summary", "") for flow in analysis["weight_loading_flows"] for item in flow["stages"])


def test_weight_loading_flows_are_separate_and_ordered():
    analysis, _inventory, ir, _coverage = artifacts()
    assert ir["weight_loading_entrypoints"] == ["HYV3ForCausalLM.load_weights", "HYV3Model.load_weights"]
    flows = {flow["entrypoint"]: flow["stages"] for flow in analysis["weight_loading_flows"]}
    assert set(flows) == {"HYV3ForCausalLM.load_weights", "HYV3Model.load_weights"}
    model_stages = flows["HYV3Model.load_weights"]
    assert any("maybe_remap_kv_scale_name" in item["summary"] for item in model_stages)
    ir_edges = edges(ir, "weight_loading")
    assert ir_edges["model_weight_to_fp8_remap"]["target"] == "model_fp8_scale_remap"
    assert ir_edges["fp8_remap_to_mapping_dispatch"]["target"] == "mapping_dispatch"
    assert sum(1 for item in model_stages if "is_pp_missing_parameter" in item["summary"]) >= 3
    wrapper_stages = flows["HYV3ForCausalLM.load_weights"]
    assert any("AutoWeightsLoader" in item["summary"] for item in wrapper_stages)


def test_golden_semantic_contract_is_satisfied():
    _analysis, _inventory, ir, _coverage = artifacts()
    contract = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["assertions"]
    attention_edges = edges(ir, "attention_detail")
    hpc_edge = attention_edges["qkv_projection_to_hpc_fused"]
    assert hpc_edge["source"] == contract["hpc_input_source"]
    assert hpc_edge["source_port"] == contract["hpc_input_port"]
    assert attention_edges["q_stream_to_q_norm"]["condition"] == "self.use_qk_norm"
    assert attention_edges["k_stream_to_k_norm"]["condition"] == "self.use_qk_norm"
    assert ("v_stream", "fallback_q_norm") not in {(edge["source"], edge["target"]) for edge in attention_edges.values()}
    assert attention_edges["kv_cache_read_to_attention"]["evidence"][0]["type"] == "external"
    assert "direct" not in {item["type"] for item in attention_edges["kv_cache_read_to_attention"]["evidence"]}


def test_decoder_moe_adapter_and_weight_pages_have_corrected_semantics():
    _analysis, _inventory, ir, _coverage = artifacts()
    decoder = nodes(ir, "overview")["hyv3_decoder_layer"]
    assert {"hidden_out", "residual_out"} <= {port["id"] for port in decoder["ports"]}
    assert "residual_initialization" in nodes(ir, "decoder_layer_detail")
    moe_nodes = nodes(ir, "moe_detail")
    assert moe_nodes["expert_bias"]["phase"] == "construction"
    assert moe_nodes["eplb_metadata"]["phase"] in {"construction", "parallel_partition"}
    assert "restore_original_shape" in moe_nodes
    adapter_edges = edges(ir, "adapter_integration")
    assert adapter_edges["vllm_config_to_hf_config"]["source"] == "vllm_config"
    assert adapter_edges["causal_lm_contains_model"]["target"] == "adapter_model"
    adapter_nodes = nodes(ir, "adapter_integration")
    assert {"component_qkv_linear", "component_hpc_rope_norm", "component_fused_moe", "component_logits_processor"} <= set(adapter_nodes)
    weight_edges = edges(ir, "weight_loading")
    assert weight_edges["fp8_remap_to_mapping_dispatch"]["source"] == "model_fp8_scale_remap"
    pp_filter_edges = [edge for edge in weight_edges.values() if edge["target"] == "pp_missing_parameter_filter"]
    assert len(pp_filter_edges) >= 3


def test_coverage_has_no_orphaned_required_facts_and_no_import_only_behavior():
    _analysis, _inventory, ir, coverage = artifacts()
    summary = coverage["summary"]
    assert summary["required_fact_count"] > 0
    assert summary["orphaned_fact_count"] == 0
    for ir_page in ir["pages"]:
        for edge in ir_page["edges"]:
            ids = fact_ids(edge)
            if edge["kind"] in {"runtime", "weight_mapping"} and ids:
                assert not all(":import:" in fact_id for fact_id in ids)
