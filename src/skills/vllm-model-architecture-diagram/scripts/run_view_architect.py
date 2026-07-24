#!/usr/bin/env python3
"""Generate an Architecture View Graph from architecture concepts.

The View Graph is the human expression layer: it turns concepts into components,
data nodes, process nodes, storage boundaries and readable architecture flows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

VIEW_GRAPH_VERSION = "0.1"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _concepts(design: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(concept["id"]): concept
        for concept in design.get("concepts", [])
        if isinstance(concept, dict) and isinstance(concept.get("id"), str)
    }


def _evidence(concepts: dict[str, dict[str, Any]], *concept_ids: str) -> list[str]:
    refs: list[str] = []
    for concept_id in concept_ids:
        concept = concepts.get(concept_id)
        if isinstance(concept, dict):
            refs.extend(str(item) for item in concept.get("evidence", []) if isinstance(item, str))
    return sorted(set(refs))


def _ports(*ports: tuple[str, str, str]) -> list[dict[str, str]]:
    return [{"id": port_id, "label": label, "direction": direction, "data_kind": "tensor"} for port_id, label, direction in ports]


def _node(
    node_id: str,
    node_type: str,
    label: str,
    subtitle: str,
    concept_refs: list[str],
    fact_refs: list[str],
    visual_role: str,
    row: int,
    col: int,
    *,
    ports: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "label": label,
        "subtitle": subtitle,
        "concept_refs": concept_refs,
        "fact_refs": fact_refs,
        "ports": ports or _ports(("in", "in", "input"), ("out", "out", "output")),
        "visual_role": visual_role,
        "layout": {"row": row, "col": col},
    }


def _edge(
    edge_id: str,
    edge_type: str,
    source: str,
    target: str,
    label: str,
    concept_refs: list[str],
    fact_refs: list[str],
    *,
    source_port: str = "out",
    target_port: str = "in",
    show_label: bool = False,
) -> dict[str, Any]:
    return {
        "id": edge_id,
        "type": edge_type,
        "source": source,
        "target": target,
        "source_port": source_port,
        "target_port": target_port,
        "label": label,
        "show_label": show_label,
        "concept_refs": concept_refs,
        "fact_refs": fact_refs,
    }


def _page(page_id: str, title: str, purpose: str, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], groups: list[dict[str, Any]] | None = None, lanes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": page_id,
        "title": title,
        "purpose": purpose,
        "nodes": nodes,
        "edges": edges,
        "groups": groups or [],
        "lanes": lanes or [],
    }


def build_architecture_view_graph(architecture_concept: dict[str, Any], fact_graph: dict[str, Any]) -> dict[str, Any]:
    concepts = _concepts(architecture_concept)
    c_adapter = "concept:hyv3_vllm_adapter"
    c_exec = "concept:transformer_execution_flow"
    c_attn = "concept:tensor_parallel_attention_projection"
    c_qk = "concept:qk_positional_processing"
    c_kv = "concept:kv_cache_boundary"
    c_backend = "concept:vllm_attention_backend_boundary"
    c_moe = "concept:moe_execution_strategy"
    c_parallel = "concept:parallel_execution_strategy"
    c_checkpoint = "concept:checkpoint_adaptation"
    c_caps = "concept:adapter_capabilities"
    c_boundary = "concept:vllm_integration_boundary"

    exec_facts = _evidence(concepts, c_exec, c_adapter)
    attn_facts = _evidence(concepts, c_attn, c_qk, c_kv, c_backend)
    moe_facts = _evidence(concepts, c_moe, c_parallel)
    parallel_facts = _evidence(concepts, c_parallel)
    checkpoint_facts = _evidence(concepts, c_checkpoint)
    boundary_facts = _evidence(concepts, c_boundary, c_backend, c_moe, c_checkpoint)

    pages = [
        _page(
            "model_overview",
            "Model Overview",
            "Show complete inference flow through the vLLM adapter.",
            [
                _node("input", "data", "Input IDs / Embeds", "model inputs", [c_exec], exec_facts, "runtime_data", 0, 0),
                _node("embedding", "component", "Token Embedding", "VocabParallelEmbedding", [c_exec, c_parallel], exec_facts, "model_component", 0, 1),
                _node("decoder_stack", "component", "Decoder Stack", "pipeline-local layers", [c_exec, c_parallel], exec_facts, "model_component", 0, 2),
                _node("final_norm", "process", "Final Norm", "RMSNorm", [c_exec], exec_facts, "normalization", 0, 3),
                _node("hidden_states", "data", "Hidden States", "model output tensor", [c_exec], exec_facts, "runtime_data", 0, 4),
                _node("lm_head", "component", "LM Head", "ParallelLMHead", [c_exec, c_parallel], exec_facts, "model_component", 1, 4),
                _node("logits_processor", "component", "Logits Processor", "compute logits", [c_exec], exec_facts, "model_component", 0, 5),
                _node("logits", "data", "Logits", "adapter output", [c_exec], exec_facts, "runtime_data", 0, 6),
            ],
            [
                _edge("overview_input_to_embedding", "runtime_flow", "input", "embedding", "", [c_exec], exec_facts),
                _edge("overview_embedding_to_decoder", "runtime_flow", "embedding", "decoder_stack", "", [c_exec], exec_facts),
                _edge("overview_decoder_to_norm", "runtime_flow", "decoder_stack", "final_norm", "", [c_exec], exec_facts),
                _edge("overview_norm_to_hidden", "runtime_flow", "final_norm", "hidden_states", "", [c_exec], exec_facts),
                _edge("overview_hidden_to_logits_processor", "runtime_flow", "hidden_states", "logits_processor", "", [c_exec], exec_facts),
                _edge("overview_lm_head_to_logits_processor", "dependency", "lm_head", "logits_processor", "weights", [c_exec], exec_facts, show_label=True),
                _edge("overview_logits_processor_to_logits", "runtime_flow", "logits_processor", "logits", "", [c_exec], exec_facts),
            ],
        ),
        _page(
            "attention",
            "Attention Implementation",
            "Explain how HYV3 attention is adapted to vLLM attention.",
            [
                _node("attention_hidden_states", "data", "Hidden States", "attention input", [c_exec], attn_facts, "runtime_data", 1, 0),
                _node("qkv_projection", "component", "QKV Projection", "QKVParallelLinear", [c_attn], attn_facts, "attention_component", 1, 1),
                _node("qkv_split", "process", "Q/K/V Split", "separate streams", [c_attn], attn_facts, "attention_process", 1, 2, ports=_ports(("in", "qkv", "input"), ("q", "Q", "output"), ("k", "K", "output"), ("v", "V", "output"))),
                _node("hpc_fused_processing", "process", "HPC Fused Processing", "qkv -> fused RoPE/QK path", [c_qk], attn_facts, "hpc_path", 0, 3),
                _node("q_path", "data", "Q Path", "query stream", [c_attn], attn_facts, "attention_data", 1, 3),
                _node("k_path", "data", "K Path", "key stream", [c_attn], attn_facts, "attention_data", 2, 3),
                _node("v_path", "data", "V Path", "value stream", [c_attn], attn_facts, "attention_data", 3, 3),
                _node("optional_qk_norm", "process", "Optional QK Norm", "self.use_qk_norm", [c_qk], attn_facts, "fallback_path", 1, 4),
                _node("rope", "process", "RoPE Processing", "rotary embedding", [c_qk], attn_facts, "fallback_path", 2, 4),
                _node("kv_cache_boundary", "storage", "KV Cache Boundary", "external cache behavior", [c_kv], attn_facts, "storage_boundary", 3, 4),
                _node("attention_backend", "boundary", "vLLM Attention Backend", "external backend", [c_backend], attn_facts, "external_boundary", 1, 5),
                _node("output_projection", "component", "Output Projection", "RowParallelLinear", [c_attn], attn_facts, "attention_component", 1, 6),
                _node("attention_output", "data", "Attention Output", "projected states", [c_exec], attn_facts, "runtime_data", 1, 7),
            ],
            [
                _edge("attention_input_to_qkv", "runtime_flow", "attention_hidden_states", "qkv_projection", "", [c_attn], attn_facts),
                _edge("attention_qkv_to_hpc", "runtime_flow", "qkv_projection", "hpc_fused_processing", "HPC path", [c_qk], attn_facts, show_label=True),
                _edge("attention_hpc_to_backend", "delegation", "hpc_fused_processing", "attention_backend", "HPC fused", [c_qk, c_backend], attn_facts, show_label=True),
                _edge("attention_qkv_to_split", "runtime_flow", "qkv_projection", "qkv_split", "fallback path", [c_attn], attn_facts, target_port="in", show_label=True),
                _edge("attention_split_to_q", "runtime_flow", "qkv_split", "q_path", "Q", [c_attn], attn_facts, source_port="q", show_label=True),
                _edge("attention_split_to_k", "runtime_flow", "qkv_split", "k_path", "K", [c_attn], attn_facts, source_port="k", show_label=True),
                _edge("attention_split_to_v", "runtime_flow", "qkv_split", "v_path", "V", [c_attn], attn_facts, source_port="v", show_label=True),
                _edge("attention_q_to_norm", "runtime_flow", "q_path", "optional_qk_norm", "Q/K", [c_qk], attn_facts, show_label=True),
                _edge("attention_k_to_norm", "runtime_flow", "k_path", "optional_qk_norm", "", [c_qk], attn_facts),
                _edge("attention_norm_to_rope", "runtime_flow", "optional_qk_norm", "rope", "", [c_qk], attn_facts),
                _edge("attention_rope_to_backend", "delegation", "rope", "attention_backend", "fallback", [c_backend], attn_facts, show_label=True),
                _edge("attention_v_to_backend", "delegation", "v_path", "attention_backend", "V", [c_backend], attn_facts, show_label=True),
                _edge("attention_kv_cache_to_backend", "boundary", "kv_cache_boundary", "attention_backend", "KV cache boundary", [c_kv, c_backend], attn_facts, show_label=True),
                _edge("attention_backend_to_output_projection", "delegation", "attention_backend", "output_projection", "backend output", [c_attn, c_backend], attn_facts, show_label=True),
                _edge("attention_output_projection_to_output", "runtime_flow", "output_projection", "attention_output", "", [c_attn], attn_facts),
            ],
        ),
        _page(
            "moe",
            "MoE Execution Strategy",
            "Show router, experts, shared experts and expert parallel adaptation.",
            [
                _node("moe_hidden_states", "data", "Hidden States", "MoE input", [c_moe], moe_facts, "runtime_data", 0, 0),
                _node("router", "process", "Router", "GateLinear", [c_moe], moe_facts, "moe_process", 0, 1),
                _node("topk_routing", "process", "Top-K Routing", "select experts", [c_moe], moe_facts, "moe_process", 0, 2),
                _node("fused_moe", "component", "FusedMoE", "external fused experts", [c_moe], moe_facts, "moe_component", 0, 3),
                _node("routed_experts", "component", "Routed Experts", "local / physical experts", [c_moe], moe_facts, "moe_component", 0, 4),
                _node("shared_experts", "component", "Shared Experts", "optional shared MLP", [c_moe], moe_facts, "moe_component", 1, 4),
                _node("expert_parallel", "capability", "Expert Parallel", "EP group + EPLB", [c_parallel], moe_facts, "parallel_capability", 1, 3),
                _node("moe_output", "data", "Output", "restored hidden states", [c_moe], moe_facts, "runtime_data", 0, 5),
            ],
            [
                _edge("moe_input_to_router", "runtime_flow", "moe_hidden_states", "router", "", [c_moe], moe_facts),
                _edge("moe_router_to_topk", "runtime_flow", "router", "topk_routing", "router logits", [c_moe], moe_facts, show_label=True),
                _edge("moe_topk_to_fused", "runtime_flow", "topk_routing", "fused_moe", "top-k", [c_moe], moe_facts, show_label=True),
                _edge("moe_fused_to_routed_experts", "runtime_flow", "fused_moe", "routed_experts", "", [c_moe], moe_facts),
                _edge("moe_shared_to_fused", "dependency", "shared_experts", "fused_moe", "optional", [c_moe], moe_facts, show_label=True),
                _edge("moe_ep_to_fused", "parallel", "expert_parallel", "fused_moe", "EP", [c_parallel], moe_facts, show_label=True),
                _edge("moe_fused_to_output", "runtime_flow", "fused_moe", "moe_output", "", [c_moe], moe_facts),
            ],
        ),
        _page(
            "parallel_strategy",
            "Parallel Strategy",
            "Explain TP, PP and EP as independent strategies, not one runtime chain.",
            [
                _node("tensor_parallel", "capability", "Tensor Parallel", "QKV, rows, vocab and LM head", [c_parallel, c_attn], parallel_facts, "parallel_strategy", 0, 0),
                _node("pipeline_parallel", "capability", "Pipeline Parallel", "make_layers and rank-local flow", [c_parallel], parallel_facts, "parallel_strategy", 0, 1),
                _node("expert_parallel", "capability", "Expert Parallel", "EP group and physical experts", [c_parallel, c_moe], parallel_facts, "parallel_strategy", 0, 2),
                _node("affected_components", "annotation", "Affected Components", "Embedding, attention, FFN/MoE, LM head", [c_parallel], parallel_facts, "annotation", 1, 1),
            ],
            [
                _edge("parallel_tp_to_components", "parallel", "tensor_parallel", "affected_components", "affects", [c_parallel], parallel_facts, show_label=True),
                _edge("parallel_pp_to_components", "parallel", "pipeline_parallel", "affected_components", "partitions", [c_parallel], parallel_facts, show_label=True),
                _edge("parallel_ep_to_components", "parallel", "expert_parallel", "affected_components", "experts", [c_parallel], parallel_facts, show_label=True),
            ],
        ),
        _page(
            "checkpoint",
            "Checkpoint Adaptation",
            "Show how HF checkpoint weights map into vLLM parameters.",
            [
                _node("hf_checkpoint", "storage", "HF Checkpoint", "source weights", [c_checkpoint], checkpoint_facts, "checkpoint_source", 0, 0),
                _node("weight_name_processing", "process", "Weight Name Processing", "filter / rename", [c_checkpoint], checkpoint_facts, "weight_process", 0, 1),
                _node("packed_mapping", "process", "Packed Mapping", "packed + stacked dispatch", [c_checkpoint], checkpoint_facts, "weight_process", 0, 2),
                _node("qkv_weights", "data", "q_proj / k_proj / v_proj", "packed attention weights", [c_checkpoint], checkpoint_facts, "weight_data", 0, 3),
                _node("qkv_proj", "component", "qkv_proj", "packed vLLM parameter", [c_checkpoint, c_attn], checkpoint_facts, "weight_target", 0, 4),
                _node("gate_up_weights", "data", "gate_proj / up_proj", "stacked FFN weights", [c_checkpoint], checkpoint_facts, "weight_data", 1, 3),
                _node("gate_up_proj", "component", "gate_up_proj", "stacked vLLM parameter", [c_checkpoint], checkpoint_facts, "weight_target", 1, 4),
                _node("expert_weights", "data", "Expert Weights", "expert checkpoint tensors", [c_checkpoint, c_moe], checkpoint_facts, "weight_data", 2, 3),
                _node("expert_params", "component", "Expert Params", "FusedMoE parameters", [c_checkpoint, c_moe], checkpoint_facts, "weight_target", 2, 4),
                _node("loader", "process", "Loader Dispatch", "weight_loader / default", [c_checkpoint], checkpoint_facts, "weight_process", 1, 5),
                _node("vllm_params", "storage", "vLLM Parameters", "loaded model state", [c_checkpoint], checkpoint_facts, "checkpoint_target", 1, 6),
            ],
            [
                _edge("checkpoint_hf_to_processing", "mapping", "hf_checkpoint", "weight_name_processing", "filter / rename", [c_checkpoint], checkpoint_facts, show_label=True),
                _edge("checkpoint_processing_to_mapping", "mapping", "weight_name_processing", "packed_mapping", "mapping", [c_checkpoint], checkpoint_facts, show_label=True),
                _edge("checkpoint_mapping_to_qkv", "mapping", "packed_mapping", "qkv_weights", "", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_mapping_to_gate_up", "mapping", "packed_mapping", "gate_up_weights", "", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_mapping_to_expert", "mapping", "packed_mapping", "expert_weights", "", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_qkv_to_qkv_proj", "mapping", "qkv_weights", "qkv_proj", "packed mapping", [c_checkpoint], checkpoint_facts, show_label=True),
                _edge("checkpoint_gate_up_to_target", "mapping", "gate_up_weights", "gate_up_proj", "stacked mapping", [c_checkpoint], checkpoint_facts, show_label=True),
                _edge("checkpoint_expert_to_params", "mapping", "expert_weights", "expert_params", "expert mapping", [c_checkpoint], checkpoint_facts, show_label=True),
                _edge("checkpoint_qkv_to_loader", "mapping", "qkv_proj", "loader", "", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_gate_up_to_loader", "mapping", "gate_up_proj", "loader", "", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_expert_params_to_loader", "mapping", "expert_params", "loader", "", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_loader_to_params", "mapping", "loader", "vllm_params", "vLLM params", [c_checkpoint], checkpoint_facts, show_label=True),
            ],
        ),
        _page(
            "vllm_boundary",
            "vLLM Integration Boundary",
            "Separate local adapter behavior from imported vLLM component behavior.",
            [
                _node("local_adapter", "group", "Local Model Adapter", "hy_v3.py", [c_adapter, c_boundary], boundary_facts, "local_boundary", 0, 0),
                _node("hyv3_for_causal_lm", "component", "HYV3ForCausalLM", "wrapper + logits", [c_adapter], boundary_facts, "local_component", 1, 0),
                _node("hyv3_model", "component", "HYV3Model", "transformer body", [c_adapter], boundary_facts, "local_component", 2, 0),
                _node("local_weight_mapping", "process", "Weight Mapping", "local load_weights logic", [c_checkpoint], boundary_facts, "local_process", 3, 0),
                _node("local_parallel_logic", "process", "Parallel Logic", "TP / PP / EP setup", [c_parallel], boundary_facts, "local_process", 4, 0),
                _node("vllm_attention", "boundary", "vLLM Attention Backend", "external runtime backend", [c_backend], boundary_facts, "external_boundary", 1, 2),
                _node("vllm_fused_moe", "boundary", "FusedMoE Runtime", "external fused experts", [c_moe], boundary_facts, "external_boundary", 2, 2),
                _node("vllm_auto_loader", "boundary", "AutoWeightsLoader", "external loader", [c_checkpoint], boundary_facts, "external_boundary", 3, 2),
            ],
            [
                _edge("boundary_adapter_to_attention", "delegation", "hyv3_model", "vllm_attention", "delegates", [c_backend], boundary_facts, show_label=True),
                _edge("boundary_wrapper_contains_model", "dependency", "hyv3_for_causal_lm", "hyv3_model", "contains", [c_adapter], boundary_facts, show_label=True),
                _edge("boundary_adapter_to_moe", "delegation", "hyv3_model", "vllm_fused_moe", "delegates", [c_moe], boundary_facts, show_label=True),
                _edge("boundary_weights_to_loader", "delegation", "local_weight_mapping", "vllm_auto_loader", "delegates", [c_checkpoint], boundary_facts, show_label=True),
                _edge("boundary_parallel_local", "dependency", "local_parallel_logic", "hyv3_model", "configures", [c_parallel], boundary_facts, show_label=True),
            ],
        ),
    ]

    return {
        "schema_version": VIEW_GRAPH_VERSION,
        "view_graph_type": "architecture_view_graph",
        "source_concept_version": architecture_concept.get("schema_version"),
        "source_fact_graph_version": fact_graph.get("schema_version"),
        "model_name": architecture_concept.get("model_name", "unknown-model"),
        "pages": pages,
        "boundaries": architecture_concept.get("boundaries", []),
        "legend": [
            {"type": "runtime_flow", "label": "Runtime tensor/data flow"},
            {"type": "dependency", "label": "Construction or configuration dependency"},
            {"type": "boundary", "label": "External component boundary"},
            {"type": "mapping", "label": "Checkpoint parameter mapping"},
            {"type": "parallel", "label": "Parallel execution strategy"},
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_concept", type=Path)
    parser.add_argument("source_fact_graph", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        view = build_architecture_view_graph(_load_json(args.architecture_concept), _load_json(args.source_fact_graph))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Architecture View Graph to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
