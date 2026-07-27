#!/usr/bin/env python3
"""Generate an Architecture Design Graph from architecture concepts.

The Design Graph is a blueprint for drawing: page questions, primary flows,
branches, external boundaries, strategy annotations and mapping lanes.
It is not a Draw.io graph and it is not a direct concept-card projection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DESIGN_GRAPH_VERSION = "0.1"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _concepts(concept_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(concept["id"]): concept
        for concept in concept_graph.get("concepts", [])
        if isinstance(concept, dict) and isinstance(concept.get("id"), str)
    }


def _evidence(concepts: dict[str, dict[str, Any]], *concept_ids: str) -> list[str]:
    result: list[str] = []
    for concept_id in concept_ids:
        concept = concepts.get(concept_id)
        if isinstance(concept, dict):
            result.extend(str(item) for item in concept.get("evidence", []) if isinstance(item, str))
    return sorted(set(result))


def _node(
    node_id: str,
    node_type: str,
    role: str,
    label: str,
    implementation: str,
    concept_refs: list[str],
    fact_refs: list[str],
    row: int,
    col: int,
    *,
    subtitle: str | None = None,
    ports: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "role": role,
        "label": label,
        "subtitle": subtitle or implementation,
        "implementation": implementation,
        "concept_refs": concept_refs,
        "fact_refs": fact_refs,
        "ports": ports or [
            {"id": "in", "label": "in", "direction": "input", "data_kind": "tensor"},
            {"id": "out", "label": "out", "direction": "output", "data_kind": "tensor"},
        ],
        "layout": {"row": row, "col": col},
    }


def _edge(
    edge_id: str,
    edge_type: str,
    source: str,
    target: str,
    concept_refs: list[str],
    fact_refs: list[str],
    *,
    label: str = "",
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


def _ports(*ports: tuple[str, str, str]) -> list[dict[str, str]]:
    return [{"id": port_id, "label": label, "direction": direction, "data_kind": "tensor"} for port_id, label, direction in ports]


def build_architecture_design_graph(concept_graph: dict[str, Any], fact_graph: dict[str, Any]) -> dict[str, Any]:
    concepts = _concepts(concept_graph)
    c_adapter = "concept:hyv3_vllm_adapter"
    c_exec = "concept:transformer_execution_flow"
    c_attn = "concept:tensor_parallel_attention_projection"
    c_qk = "concept:qk_positional_processing"
    c_kv = "concept:kv_cache_boundary"
    c_backend = "concept:vllm_attention_backend_boundary"
    c_moe = "concept:moe_execution_strategy"
    c_parallel = "concept:parallel_execution_strategy"
    c_checkpoint = "concept:checkpoint_adaptation"
    c_boundary = "concept:vllm_integration_boundary"

    exec_facts = _evidence(concepts, c_exec, c_adapter)
    attn_facts = _evidence(concepts, c_attn, c_qk, c_kv, c_backend)
    moe_facts = _evidence(concepts, c_moe, c_parallel)
    parallel_facts = _evidence(concepts, c_parallel)
    checkpoint_facts = _evidence(concepts, c_checkpoint)
    boundary_facts = _evidence(concepts, c_boundary, c_backend, c_moe, c_checkpoint, c_parallel)

    pages = [
        {
            "id": "model_overview",
            "title": "Model Overview",
            "question": "How does HYV3 run inside vLLM?",
            "purpose": "Show the primary inference path and keep TP/PP/EP as auxiliary strategy annotations.",
            "main_flow": ["input", "embedding", "decoder_stack", "final_norm", "hidden_states", "logits_processor", "logits"],
            "branches": [{"name": "LM Head dependency", "from": "lm_head", "to": "logits_processor", "edge_type": "dependency"}],
            "external_boundaries": [],
            "nodes": [
                _node("input", "data", "primary_flow", "Input IDs / Embeds", "model inputs", [c_exec], exec_facts, 0, 0),
                _node("embedding", "component", "primary_flow", "Embedding", "VocabParallelEmbedding", [c_exec, c_parallel], exec_facts, 0, 1),
                _node("decoder_stack", "component", "primary_flow", "Decoder Stack", "pipeline-local layers", [c_exec, c_parallel], exec_facts, 0, 2),
                _node("final_norm", "process", "primary_flow", "Final Norm", "RMSNorm", [c_exec], exec_facts, 0, 3),
                _node("hidden_states", "data", "primary_flow", "Hidden States", "model output tensor", [c_exec], exec_facts, 0, 4),
                _node("logits_processor", "component", "primary_flow", "Logits Processor", "compute logits", [c_exec], exec_facts, 0, 5),
                _node("logits", "data", "primary_flow", "Logits", "adapter output", [c_exec], exec_facts, 0, 6),
                _node("lm_head", "component", "secondary_flow", "LM Head", "ParallelLMHead", [c_exec, c_parallel], exec_facts, 1, 5),
                _node("parallel_badges", "capability", "strategy", "TP / PP / EP", "parallel strategy badges", [c_parallel], parallel_facts, 1, 2),
            ],
            "edges": [
                _edge("overview_input_to_embedding", "runtime_flow", "input", "embedding", [c_exec], exec_facts),
                _edge("overview_embedding_to_decoder", "runtime_flow", "embedding", "decoder_stack", [c_exec], exec_facts),
                _edge("overview_decoder_to_norm", "runtime_flow", "decoder_stack", "final_norm", [c_exec], exec_facts),
                _edge("overview_norm_to_hidden", "runtime_flow", "final_norm", "hidden_states", [c_exec], exec_facts),
                _edge("overview_hidden_to_logits_processor", "runtime_flow", "hidden_states", "logits_processor", [c_exec], exec_facts),
                _edge("overview_logits_processor_to_logits", "runtime_flow", "logits_processor", "logits", [c_exec], exec_facts),
                _edge("overview_lm_head_to_logits_processor", "dependency", "lm_head", "logits_processor", [c_exec], exec_facts, label="weights", show_label=True),
                _edge("overview_parallel_to_decoder", "parallel", "parallel_badges", "decoder_stack", [c_parallel], parallel_facts, label="strategy", show_label=True),
            ],
            "groups": [],
            "lanes": [],
            "annotations": ["TP/PP/EP are strategy annotations, not runtime flow nodes."],
            "strategies": ["Tensor Parallel", "Pipeline Parallel", "Expert Parallel"],
        },
        {
            "id": "attention",
            "title": "Attention Implementation",
            "question": "How does HYV3 attention adapt HF attention to the vLLM backend?",
            "purpose": "Show the main hidden-state path, HPC fused branch, fallback Q/K/V branch, KV cache boundary and output projection.",
            "main_flow": ["attention_hidden_states", "qkv_projection", "qkv_split", "attention_backend", "output_projection", "attention_output"],
            "branches": [
                {"name": "HPC fused path", "from": "qkv_projection", "to": "attention_backend", "via": ["hpc_fused_processing"], "edge_type": "delegation"},
                {"name": "fallback path", "from": "qkv_split", "to": "attention_backend", "via": ["q_path", "k_path", "optional_qk_norm", "rope", "v_path"], "edge_type": "runtime_flow"},
            ],
            "external_boundaries": ["vLLM Attention Backend", "KV Cache Boundary"],
            "nodes": [
                _node("attention_hidden_states", "data", "primary_flow", "Hidden States", "attention input", [c_exec], attn_facts, 1, 0),
                _node("qkv_projection", "component", "primary_flow", "QKV Projection", "QKVParallelLinear", [c_attn], attn_facts, 1, 1),
                _node("qkv_split", "process", "primary_flow", "Q/K/V Split", "separate streams", [c_attn], attn_facts, 1, 2, ports=_ports(("in", "qkv", "input"), ("q", "Q", "output"), ("k", "K", "output"), ("v", "V", "output"))),
                _node("hpc_fused_processing", "process", "secondary_flow", "HPC Fused Processing", "qkv fused path", [c_qk], attn_facts, 0, 3),
                _node("q_path", "data", "secondary_flow", "Q", "query stream", [c_attn], attn_facts, 1, 3),
                _node("k_path", "data", "secondary_flow", "K", "key stream", [c_attn], attn_facts, 2, 3),
                _node("v_path", "data", "secondary_flow", "V", "value stream", [c_attn], attn_facts, 3, 3),
                _node("optional_qk_norm", "process", "secondary_flow", "Optional QK Norm", "self.use_qk_norm", [c_qk], attn_facts, 1, 4),
                _node("rope", "process", "secondary_flow", "RoPE Processing", "rotary embedding", [c_qk], attn_facts, 2, 4),
                _node("kv_cache_boundary", "storage", "boundary", "KV Cache Boundary", "external cache behavior", [c_kv], attn_facts, 3, 4),
                _node("attention_backend", "boundary", "boundary", "vLLM Attention Backend", "external backend", [c_backend], attn_facts, 1, 5),
                _node("output_projection", "component", "primary_flow", "Output Projection", "RowParallelLinear", [c_attn], attn_facts, 1, 6),
                _node("attention_output", "data", "primary_flow", "Attention Output", "projected states", [c_exec], attn_facts, 1, 7),
            ],
            "edges": [
                _edge("attention_input_to_qkv", "runtime_flow", "attention_hidden_states", "qkv_projection", [c_attn], attn_facts),
                _edge("attention_qkv_to_split", "runtime_flow", "qkv_projection", "qkv_split", [c_attn], attn_facts, label="fallback", target_port="in", show_label=True),
                _edge("attention_qkv_to_hpc", "delegation", "qkv_projection", "hpc_fused_processing", [c_qk], attn_facts, label="HPC path", show_label=True),
                _edge("attention_hpc_to_backend", "delegation", "hpc_fused_processing", "attention_backend", [c_qk, c_backend], attn_facts, label="HPC fused", show_label=True),
                _edge("attention_split_to_q", "runtime_flow", "qkv_split", "q_path", [c_attn], attn_facts, label="Q", source_port="q", show_label=True),
                _edge("attention_split_to_k", "runtime_flow", "qkv_split", "k_path", [c_attn], attn_facts, label="K", source_port="k", show_label=True),
                _edge("attention_split_to_v", "runtime_flow", "qkv_split", "v_path", [c_attn], attn_facts, label="V", source_port="v", show_label=True),
                _edge("attention_q_to_norm", "runtime_flow", "q_path", "optional_qk_norm", [c_qk], attn_facts),
                _edge("attention_k_to_norm", "runtime_flow", "k_path", "optional_qk_norm", [c_qk], attn_facts),
                _edge("attention_norm_to_rope", "runtime_flow", "optional_qk_norm", "rope", [c_qk], attn_facts),
                _edge("attention_rope_to_backend", "delegation", "rope", "attention_backend", [c_backend], attn_facts, label="fallback", show_label=True),
                _edge("attention_v_to_backend", "delegation", "v_path", "attention_backend", [c_backend], attn_facts, label="V", show_label=True),
                _edge("attention_kv_cache_to_backend", "boundary", "kv_cache_boundary", "attention_backend", [c_kv, c_backend], attn_facts, label="KV boundary", show_label=True),
                _edge("attention_backend_to_output_projection", "delegation", "attention_backend", "output_projection", [c_attn, c_backend], attn_facts, label="backend output", show_label=True),
                _edge("attention_output_projection_to_output", "runtime_flow", "output_projection", "attention_output", [c_attn], attn_facts),
            ],
            "groups": [{"id": "external_attention_boundary", "label": "External vLLM runtime", "member_ids": ["attention_backend", "kv_cache_boundary"]}],
            "lanes": [{"id": "hpc_path", "label": "HPC path"}, {"id": "fallback_path", "label": "Fallback path"}],
            "annotations": ["KV Cache and vLLM Attention internals are external boundaries."],
            "strategies": ["Tensor Parallel"],
        },
    ]

    pages.extend(_remaining_pages(c_moe, c_parallel, c_checkpoint, c_adapter, c_boundary, c_backend, moe_facts, parallel_facts, checkpoint_facts, boundary_facts))

    return {
        "schema_version": DESIGN_GRAPH_VERSION,
        "design_graph_type": "architecture_design_graph",
        "source_concept_version": concept_graph.get("schema_version"),
        "source_fact_graph_version": fact_graph.get("schema_version"),
        "model_name": concept_graph.get("model_name", "unknown-model"),
        "pages": pages,
        "boundaries": concept_graph.get("boundaries", []),
        "assumptions": [
            "Concepts are not rendered directly; the View Graph is generated from this design blueprint.",
            "External vLLM internals are represented only as boundaries.",
        ],
    }


def _remaining_pages(
    c_moe: str,
    c_parallel: str,
    c_checkpoint: str,
    c_adapter: str,
    c_boundary: str,
    c_backend: str,
    moe_facts: list[str],
    parallel_facts: list[str],
    checkpoint_facts: list[str],
    boundary_facts: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "id": "moe",
            "title": "MoE Execution",
            "question": "How does HYV3 route tokens through experts?",
            "purpose": "Show token routing, top-k expert selection, fused MoE execution, shared experts and EP.",
            "main_flow": ["moe_hidden_states", "router", "topk_routing", "fused_moe", "routed_experts", "moe_output"],
            "branches": [{"name": "optional shared experts", "from": "shared_experts", "to": "fused_moe"}, {"name": "expert parallel strategy", "from": "expert_parallel", "to": "fused_moe"}],
            "external_boundaries": ["FusedMoE Runtime"],
            "nodes": [
                _node("moe_hidden_states", "data", "primary_flow", "Hidden States", "MoE input", [c_moe], moe_facts, 0, 0),
                _node("router", "process", "primary_flow", "Router", "GateLinear", [c_moe], moe_facts, 0, 1),
                _node("topk_routing", "process", "primary_flow", "Top-K Selection", "select experts", [c_moe], moe_facts, 0, 2),
                _node("fused_moe", "component", "primary_flow", "FusedMoE", "external fused experts", [c_moe], moe_facts, 0, 3),
                _node("routed_experts", "component", "primary_flow", "Experts", "routed experts", [c_moe], moe_facts, 0, 4),
                _node("shared_experts", "component", "secondary_flow", "Shared Experts", "optional shared MLP", [c_moe], moe_facts, 1, 4),
                _node("expert_parallel", "capability", "strategy", "Expert Parallel", "EP group + EPLB", [c_parallel], parallel_facts, 1, 3),
                _node("moe_output", "data", "primary_flow", "Output", "restored hidden states", [c_moe], moe_facts, 0, 5),
            ],
            "edges": [
                _edge("moe_input_to_router", "runtime_flow", "moe_hidden_states", "router", [c_moe], moe_facts),
                _edge("moe_router_to_topk", "runtime_flow", "router", "topk_routing", [c_moe], moe_facts, label="router logits", show_label=True),
                _edge("moe_topk_to_fused", "runtime_flow", "topk_routing", "fused_moe", [c_moe], moe_facts, label="top-k", show_label=True),
                _edge("moe_fused_to_routed_experts", "runtime_flow", "fused_moe", "routed_experts", [c_moe], moe_facts),
                _edge("moe_fused_to_output", "runtime_flow", "fused_moe", "moe_output", [c_moe], moe_facts),
                _edge("moe_shared_to_fused", "dependency", "shared_experts", "fused_moe", [c_moe], moe_facts, label="optional", show_label=True),
                _edge("moe_ep_to_fused", "parallel", "expert_parallel", "fused_moe", [c_parallel], parallel_facts, label="EP", show_label=True),
            ],
            "groups": [],
            "lanes": [],
            "annotations": ["FusedMoE internals are external to the input file."],
            "strategies": ["Expert Parallel", "EPLB"],
        },
        {
            "id": "checkpoint",
            "title": "Weight Adaptation",
            "question": "How do HF checkpoint weights become vLLM parameters?",
            "purpose": "Show filter/rename, packed mapping, expert mapping and loader dispatch.",
            "main_flow": ["hf_checkpoint", "weight_name_processing", "packed_mapping", "loader", "vllm_params"],
            "branches": [{"name": "qkv packed mapping", "from": "packed_mapping", "to": "qkv_proj"}, {"name": "gate/up stacked mapping", "from": "packed_mapping", "to": "gate_up_proj"}, {"name": "expert mapping", "from": "packed_mapping", "to": "expert_params"}],
            "external_boundaries": ["AutoWeightsLoader"],
            "nodes": [
                _node("hf_checkpoint", "storage", "primary_flow", "HF Checkpoint", "source weights", [c_checkpoint], checkpoint_facts, 0, 0),
                _node("weight_name_processing", "process", "primary_flow", "Weight Name Processing", "filter / rename", [c_checkpoint], checkpoint_facts, 0, 1),
                _node("packed_mapping", "process", "primary_flow", "Mapping", "packed + expert dispatch", [c_checkpoint], checkpoint_facts, 0, 2),
                _node("qkv_weights", "data", "mapping", "q_proj / k_proj / v_proj", "HF attention weights", [c_checkpoint], checkpoint_facts, 0, 3),
                _node("qkv_proj", "component", "mapping", "qkv_proj", "vLLM packed parameter", [c_checkpoint], checkpoint_facts, 0, 4),
                _node("gate_up_weights", "data", "mapping", "gate_proj / up_proj", "HF FFN weights", [c_checkpoint], checkpoint_facts, 1, 3),
                _node("gate_up_proj", "component", "mapping", "gate_up_proj", "vLLM stacked parameter", [c_checkpoint], checkpoint_facts, 1, 4),
                _node("expert_weights", "data", "mapping", "Expert Weights", "HF expert tensors", [c_checkpoint], checkpoint_facts, 2, 3),
                _node("expert_params", "component", "mapping", "FusedMoE Parameters", "expert params", [c_checkpoint], checkpoint_facts, 2, 4),
                _node("loader", "process", "primary_flow", "Loader", "weight_loader / default", [c_checkpoint], checkpoint_facts, 1, 5),
                _node("vllm_params", "storage", "primary_flow", "vLLM Parameters", "loaded model state", [c_checkpoint], checkpoint_facts, 1, 6),
            ],
            "edges": [
                _edge("checkpoint_hf_to_processing", "mapping", "hf_checkpoint", "weight_name_processing", [c_checkpoint], checkpoint_facts, label="filter / rename", show_label=True),
                _edge("checkpoint_processing_to_mapping", "mapping", "weight_name_processing", "packed_mapping", [c_checkpoint], checkpoint_facts, label="mapping", show_label=True),
                _edge("checkpoint_mapping_to_qkv", "mapping", "packed_mapping", "qkv_weights", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_mapping_to_gate_up", "mapping", "packed_mapping", "gate_up_weights", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_mapping_to_expert", "mapping", "packed_mapping", "expert_weights", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_qkv_to_qkv_proj", "mapping", "qkv_weights", "qkv_proj", [c_checkpoint], checkpoint_facts, label="packed", show_label=True),
                _edge("checkpoint_gate_up_to_target", "mapping", "gate_up_weights", "gate_up_proj", [c_checkpoint], checkpoint_facts, label="stacked", show_label=True),
                _edge("checkpoint_expert_to_params", "mapping", "expert_weights", "expert_params", [c_checkpoint], checkpoint_facts, label="expert", show_label=True),
                _edge("checkpoint_qkv_to_loader", "mapping", "qkv_proj", "loader", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_gate_up_to_loader", "mapping", "gate_up_proj", "loader", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_expert_params_to_loader", "mapping", "expert_params", "loader", [c_checkpoint], checkpoint_facts),
                _edge("checkpoint_loader_to_params", "mapping", "loader", "vllm_params", [c_checkpoint], checkpoint_facts, label="load", show_label=True),
            ],
            "groups": [],
            "lanes": [],
            "annotations": ["Checkpoint adaptation is mapping flow, not runtime tensor flow."],
            "strategies": ["packed mapping", "expert mapping"],
        },
        {
            "id": "parallel_strategy",
            "title": "Parallel Strategy",
            "question": "How are TP, PP and EP organized independently?",
            "purpose": "Show three independent strategy regions instead of a fake TP -> PP -> EP flow.",
            "main_flow": ["tensor_parallel", "pipeline_parallel", "expert_parallel"],
            "branches": [],
            "external_boundaries": [],
            "nodes": [
                _node("tensor_parallel", "capability", "primary_flow", "Tensor Parallel", "QKVLinear / RowLinear / Embedding / LM Head", [c_parallel], parallel_facts, 0, 0),
                _node("pipeline_parallel", "capability", "primary_flow", "Pipeline Parallel", "make_layers / layer partition / intermediate tensors", [c_parallel], parallel_facts, 0, 1),
                _node("expert_parallel", "capability", "primary_flow", "Expert Parallel", "expert group / local experts / FusedMoE", [c_parallel], parallel_facts, 0, 2),
                _node("strategy_notes", "annotation", "annotation", "Independent Regions", "parallel strategies are not a runtime chain", [c_parallel], parallel_facts, 1, 1),
            ],
            "edges": [
                _edge("parallel_tp_to_notes", "parallel", "tensor_parallel", "strategy_notes", [c_parallel], parallel_facts, label="components", show_label=True),
                _edge("parallel_pp_to_notes", "parallel", "pipeline_parallel", "strategy_notes", [c_parallel], parallel_facts, label="partitions", show_label=True),
                _edge("parallel_ep_to_notes", "parallel", "expert_parallel", "strategy_notes", [c_parallel], parallel_facts, label="experts", show_label=True),
            ],
            "groups": [{"id": "tp_region", "label": "Tensor Parallel"}, {"id": "pp_region", "label": "Pipeline Parallel"}, {"id": "ep_region", "label": "Expert Parallel"}],
            "lanes": [],
            "annotations": ["No TP -> PP -> EP serial relationship is implied."],
            "strategies": ["TP", "PP", "EP"],
        },
        {
            "id": "vllm_boundary",
            "title": "vLLM Boundary",
            "question": "Where does local adapter code stop and vLLM runtime behavior begin?",
            "purpose": "Separate local adapter modules and mapping logic from external vLLM runtime components.",
            "main_flow": ["local_adapter", "adapter_boundary", "external_runtime"],
            "branches": [],
            "external_boundaries": ["vLLM Attention Backend", "FusedMoE Runtime", "AutoWeightsLoader"],
            "nodes": [
                _node("local_adapter", "group", "primary_flow", "Local Adapter", "hy_v3.py", [c_adapter, c_boundary], boundary_facts, 0, 0),
                _node("hyv3_for_causal_lm", "component", "primary_flow", "HYV3ForCausalLM", "wrapper + logits", [c_adapter], boundary_facts, 1, 0),
                _node("hyv3_model", "component", "primary_flow", "HYV3Model", "transformer body", [c_adapter], boundary_facts, 2, 0),
                _node("local_weight_mapping", "process", "primary_flow", "Weight Mapping", "local load_weights logic", [c_checkpoint], boundary_facts, 3, 0),
                _node("local_parallel_logic", "process", "primary_flow", "Parallel Logic", "TP / PP / EP setup", [c_parallel], boundary_facts, 4, 0),
                _node("adapter_boundary", "boundary", "boundary", "Adapter Boundary", "local evidence stops here", [c_boundary], boundary_facts, 2, 1),
                _node("external_runtime", "group", "boundary", "External vLLM Runtime", "imported components", [c_boundary], boundary_facts, 0, 2),
                _node("vllm_attention", "boundary", "boundary", "Attention Backend", "external runtime backend", [c_backend], boundary_facts, 1, 2),
                _node("vllm_fused_moe", "boundary", "boundary", "FusedMoE Runtime", "external fused experts", [c_moe], boundary_facts, 2, 2),
                _node("vllm_auto_loader", "boundary", "boundary", "AutoWeightsLoader", "external loader", [c_checkpoint], boundary_facts, 3, 2),
            ],
            "edges": [
                _edge("boundary_wrapper_contains_model", "dependency", "hyv3_for_causal_lm", "hyv3_model", [c_adapter], boundary_facts, label="contains", show_label=True),
                _edge("boundary_model_to_adapter_boundary", "delegation", "hyv3_model", "adapter_boundary", [c_boundary], boundary_facts, label="delegates", show_label=True),
                _edge("boundary_weight_to_adapter_boundary", "delegation", "local_weight_mapping", "adapter_boundary", [c_checkpoint], boundary_facts, label="loads", show_label=True),
                _edge("boundary_parallel_to_model", "dependency", "local_parallel_logic", "hyv3_model", [c_parallel], boundary_facts, label="configures", show_label=True),
                _edge("boundary_to_attention", "boundary", "adapter_boundary", "vllm_attention", [c_backend], boundary_facts, label="external", show_label=True),
                _edge("boundary_to_moe", "boundary", "adapter_boundary", "vllm_fused_moe", [c_moe], boundary_facts, label="external", show_label=True),
                _edge("boundary_to_loader", "boundary", "adapter_boundary", "vllm_auto_loader", [c_checkpoint], boundary_facts, label="external", show_label=True),
            ],
            "groups": [{"id": "local_region", "label": "Local Adapter"}, {"id": "external_region", "label": "External vLLM Runtime"}],
            "lanes": [],
            "annotations": ["External runtime internals are not direct claims from hy_v3.py."],
            "strategies": [],
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_concept", type=Path)
    parser.add_argument("source_fact_graph", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        design = build_architecture_design_graph(_load_json(args.architecture_concept), _load_json(args.source_fact_graph))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Architecture Design Graph to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
