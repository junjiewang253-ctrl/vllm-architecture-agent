#!/usr/bin/env python3
"""Build a deterministic Architecture Design template.

This is a fallback/template generator, not the architect default. Finalize mode
requires an Agent-authored design unless explicitly allowed by tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DESIGN_VERSION = "1.0"


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
    unique = sorted(set(result))
    return sorted(unique, key=lambda item: (":import:" in item or item.startswith("fact:<module>"), item))


def _display(title: str, subtitle: str = "", badges: list[str] | None = None) -> dict[str, Any]:
    return {
        "title": title,
        "subtitle": subtitle,
        "show_implementation": bool(subtitle),
        "badges": badges or [],
        "max_title_lines": 2,
        "max_subtitle_lines": 2,
    }


def _port(
    port_id: str,
    label: str,
    direction: str,
    data_kind: str,
    side: str,
    order: int,
) -> dict[str, Any]:
    return {
        "id": port_id,
        "label": label,
        "direction": direction,
        "data_kind": data_kind,
        "preferred_side": side,
        "order": order,
    }


def _ports(data_kind: str = "tensor") -> list[dict[str, Any]]:
    return [
        _port("in", "in", "input", data_kind, "left", 0),
        _port("out", "out", "output", data_kind, "right", 0),
    ]


def _node(
    node_id: str,
    semantic_kind: str,
    role: str,
    label: str,
    implementation: str,
    phase: str,
    concept_refs: list[str],
    fact_refs: list[str],
    order: int,
    *,
    parent_id: str | None = None,
    ports: list[dict[str, Any]] | None = None,
    region: str | None = None,
    lane: str | None = None,
    size: tuple[float, float] = (180.0, 74.0),
    badges: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "semantic_kind": semantic_kind,
        "role": role,
        "label": label,
        "implementation": implementation,
        "phase": phase,
        "parent_id": parent_id,
        "concept_refs": concept_refs,
        "fact_refs": fact_refs,
        "ports": ports if ports is not None else _ports("tensor"),
        "display": _display(label, implementation, badges),
        "layout_hint": {
            "region": region,
            "lane": lane,
            "order": order,
            "preferred_size": {"width": size[0], "height": size[1]},
        },
    }


def _edge(
    edge_id: str,
    kind: str,
    source: str,
    target: str,
    phase: str,
    concept_refs: list[str],
    fact_refs: list[str],
    *,
    source_port: str = "out",
    target_port: str = "in",
    label: str = "",
    show_label: bool = False,
    priority: str = "primary",
    route_hint: str = "direct",
    visible: bool = True,
) -> dict[str, Any]:
    return {
        "id": edge_id,
        "kind": kind,
        "source": source,
        "target": target,
        "source_port": source_port,
        "target_port": target_port,
        "phase": phase,
        "concept_refs": concept_refs,
        "fact_refs": fact_refs,
        "display": {
            "visible": visible,
            "show_label": show_label,
            "label": label,
            "priority": priority,
        },
        "route_hint": route_hint,
    }


def _group(group_id: str, label: str, kind: str, member_ids: list[str]) -> dict[str, Any]:
    return {"id": group_id, "label": label, "kind": kind, "member_ids": member_ids}


def _annotation(annotation_id: str, text: str, target_ids: list[str], fact_refs: list[str]) -> dict[str, Any]:
    return {"id": annotation_id, "text": text, "target_ids": target_ids, "fact_refs": fact_refs}


def _coverage(topic: str, concept_refs: list[str], fact_refs: list[str], status: str = "covered") -> dict[str, Any]:
    return {"topic": topic, "concept_refs": concept_refs, "fact_refs": fact_refs, "status": status}


def _constraints(pattern: str, width: float = 1900, height: float = 1068) -> dict[str, Any]:
    return {"page_size": [width, height], "orientation": "landscape", "pattern": pattern}


def _page(
    page_id: str,
    title: str,
    question: str,
    purpose: str,
    view_kind: str,
    scope: list[str],
    ordered: list[str],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    groups: list[dict[str, Any]] | None = None,
    annotations: list[dict[str, Any]] | None = None,
    coverage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": page_id,
        "title": title,
        "question": question,
        "purpose": purpose,
        "view_kind": view_kind,
        "scope": scope,
        "primary_story": {
            "start_node_id": ordered[0] if ordered else "",
            "end_node_id": ordered[-1] if ordered else "",
            "ordered_node_ids": ordered,
            "reading_direction": "left_to_right",
        },
        "nodes": nodes,
        "edges": edges,
        "groups": groups or [],
        "annotations": annotations or [],
        "layout_constraints": _constraints(view_kind),
        "coverage": coverage or [],
    }


def _external_boundaries(boundary_report: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(boundary_report.get("external_components", [])):
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "id": f"external_boundary_{index}",
                "name": str(item.get("name", f"external-{index}")),
                "reason": str(item.get("reason", "Imported component is outside the analyzed file.")),
                "allowed_claim": str(item.get("allowed_claim", "External component boundary.")),
                "fact_refs": [str(ref) for ref in item.get("evidence", []) if isinstance(ref, str)],
            }
        )
    return result


def _has_profile(brief: dict[str, Any], key: str) -> bool:
    value = brief.get("model_profile", {}).get(key)
    return bool(value)


def _cluster_facts(brief: dict[str, Any], *cluster_ids: str) -> list[str]:
    facts: list[str] = []
    wanted = set(cluster_ids)
    for cluster in brief.get("fact_clusters", []):
        if not isinstance(cluster, dict) or cluster.get("id") not in wanted:
            continue
        facts.extend(str(item) for item in cluster.get("fact_ids", []) if isinstance(item, str))
    unique = sorted(set(facts))
    return sorted(unique, key=lambda item: (":import:" in item or item.startswith("fact:<module>"), item))


def build_baseline_design(
    architect_brief: dict[str, Any],
    concept_graph: dict[str, Any],
    boundary_report: dict[str, Any],
    *,
    author_type: str = "baseline",
) -> dict[str, Any]:
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
    c_caps = "concept:adapter_capabilities"

    exec_facts = _evidence(concepts, c_exec, c_adapter) + _cluster_facts(architect_brief, "model_execution")
    decoder_facts = _evidence(concepts, c_exec) + _cluster_facts(architect_brief, "decoder_block")
    attn_facts = _evidence(concepts, c_attn, c_qk, c_kv, c_backend) + _cluster_facts(architect_brief, "attention")
    moe_facts = _evidence(concepts, c_moe, c_parallel) + _cluster_facts(architect_brief, "moe")
    parallel_facts = _evidence(concepts, c_parallel) + _cluster_facts(architect_brief, "tensor_parallel", "pipeline_parallel", "expert_parallel")
    checkpoint_facts = _evidence(concepts, c_checkpoint) + _cluster_facts(architect_brief, "checkpoint_wrapper_loader", "checkpoint_model_loader")
    boundary_facts = _evidence(concepts, c_boundary, c_backend, c_moe, c_checkpoint, c_parallel) + _cluster_facts(architect_brief, "external_boundaries")
    cap_facts = _evidence(concepts, c_caps)

    pages: list[dict[str, Any]] = []
    pages.append(_model_overview_page(c_exec, c_parallel, exec_facts, parallel_facts))
    if _has_profile(architect_brief, "decoder_class"):
        pages.append(_decoder_page(c_exec, c_moe, decoder_facts, moe_facts))
    if _has_profile(architect_brief, "attention_class"):
        pages.append(_attention_page(c_exec, c_attn, c_qk, c_kv, c_backend, attn_facts))
    if _has_profile(architect_brief, "moe_class"):
        pages.append(_moe_page(c_moe, c_parallel, moe_facts, parallel_facts))
    if architect_brief.get("model_profile", {}).get("load_weights_entrypoints"):
        pages.append(_checkpoint_page(c_checkpoint, checkpoint_facts))
    pages.append(_parallel_page(c_parallel, parallel_facts, include_ep=_has_profile(architect_brief, "moe_class")))
    pages.append(_boundary_page(c_adapter, c_boundary, c_backend, c_moe, c_checkpoint, c_parallel, c_caps, boundary_facts, cap_facts, include_moe=_has_profile(architect_brief, "moe_class")))

    return {
        "schema_version": DESIGN_VERSION,
        "design_graph_type": "architecture_design_graph",
        "model_name": str(architect_brief.get("model_name", concept_graph.get("model_name", "model"))),
        "source_sha256": str(architect_brief["source_sha256"]),
        "author": {
            "type": author_type,
            "agent": "codex" if author_type == "agent" else "deterministic-template",
            "created_at": "2026-07-27T00:00:00Z",
            "prompt_version": "architect-design-v1",
        },
        "pages": pages,
        "external_boundaries": _external_boundaries(boundary_report),
        "coverage": [_coverage("architect brief topics", [c_exec, c_parallel], exec_facts + parallel_facts)],
        "assumptions": [
            "This baseline/template is source-grounded but not a substitute for an Agent-authored Design in architect finalize mode.",
            "External vLLM internals remain external boundaries.",
        ],
    }


def _model_overview_page(c_exec: str, c_parallel: str, exec_facts: list[str], parallel_facts: list[str]) -> dict[str, Any]:
    nodes = [
        _node("input_ids", "data", "primary", "Input IDs / Inputs Embeds", "model inputs", "runtime", [c_exec], exec_facts, 0),
        _node("embedding", "component", "primary", "Token Embedding", "VocabParallelEmbedding", "runtime", [c_exec, c_parallel], exec_facts, 1, badges=["TP"]),
        _node("decoder_stack", "component", "primary", "Pipeline-local Decoder Stack", "make_layers range", "runtime", [c_exec, c_parallel], exec_facts + parallel_facts, 2, badges=["TP", "PP", "EP"], size=(235, 84)),
        _node("final_add", "merge", "primary", "Final Residual Add", "hidden + residual", "runtime", [c_exec], exec_facts, 3, ports=[_port("hidden", "hidden", "input", "tensor", "left", 0), _port("residual", "residual", "input", "tensor", "bottom", 1), _port("out", "hidden", "output", "tensor", "right", 0)]),
        _node("final_norm", "process", "primary", "Final RMSNorm", "RMSNorm", "runtime", [c_exec], exec_facts, 4),
        _node("hidden_states", "data", "primary", "Hidden States", "model body output", "runtime", [c_exec], exec_facts, 5),
        _node("logits_processor", "component", "primary", "LogitsProcessor", "hidden + LM head", "runtime", [c_exec], exec_facts, 6, ports=[_port("hidden", "hidden", "input", "tensor", "left", 0), _port("lm_head", "lm head", "input", "weights", "top", 1), _port("logits", "logits", "output", "tensor", "right", 0)]),
        _node("logits", "data", "primary", "Logits", "output", "runtime", [c_exec], exec_facts, 7),
        _node("lm_head", "component", "auxiliary", "LM Head", "ParallelLMHead", "construction", [c_exec, c_parallel], exec_facts + parallel_facts, 8, ports=[_port("weights", "weights", "output", "weights", "bottom", 0)], badges=["TP"]),
        _node("pp_non_first_input", "data", "auxiliary", "Non-first PP Rank Input", "IntermediateTensors", "runtime", [c_parallel], parallel_facts, 9),
        _node("pp_non_last_output", "data", "auxiliary", "Non-last PP Rank Output", "IntermediateTensors", "runtime", [c_parallel], parallel_facts, 10),
    ]
    edges = [
        _edge("overview_input_to_embedding", "tensor_flow", "input_ids", "embedding", "runtime", [c_exec], exec_facts),
        _edge("overview_embedding_to_decoder", "tensor_flow", "embedding", "decoder_stack", "runtime", [c_exec], exec_facts),
        _edge("overview_decoder_to_add_hidden", "tensor_flow", "decoder_stack", "final_add", "runtime", [c_exec], exec_facts, target_port="hidden"),
        _edge("overview_decoder_to_add_residual", "residual_flow", "decoder_stack", "final_add", "runtime", [c_exec], exec_facts, target_port="residual", route_hint="lower_lane"),
        _edge("overview_add_to_norm", "tensor_flow", "final_add", "final_norm", "runtime", [c_exec], exec_facts),
        _edge("overview_norm_to_hidden", "tensor_flow", "final_norm", "hidden_states", "runtime", [c_exec], exec_facts),
        _edge("overview_hidden_to_logits_processor", "tensor_flow", "hidden_states", "logits_processor", "runtime", [c_exec], exec_facts, target_port="hidden"),
        _edge("overview_lm_head_to_logits_processor", "dependency", "lm_head", "logits_processor", "construction", [c_exec], exec_facts, source_port="weights", target_port="lm_head", label="head weights", show_label=True, priority="secondary", route_hint="local"),
        _edge("overview_logits_processor_to_logits", "tensor_flow", "logits_processor", "logits", "runtime", [c_exec], exec_facts, source_port="logits"),
        _edge("overview_pp_non_first_to_decoder", "tensor_flow", "pp_non_first_input", "decoder_stack", "runtime", [c_parallel], parallel_facts, priority="secondary", route_hint="branch_top", visible=False),
        _edge("overview_decoder_to_pp_non_last", "tensor_flow", "decoder_stack", "pp_non_last_output", "runtime", [c_parallel], parallel_facts, priority="secondary", route_hint="branch_bottom", visible=False),
    ]
    return _page(
        "model_execution_overview",
        "Model Execution Overview",
        "How does HY V3 run inside vLLM?",
        "Show hidden-state production, logits computation and PP side branches without turning TP/PP/EP into runtime nodes.",
        "pipeline",
        ["HYV3ForCausalLM.forward", "HYV3ForCausalLM.compute_logits"],
        ["input_ids", "embedding", "decoder_stack", "final_add", "final_norm", "hidden_states", "logits_processor", "logits"],
        nodes,
        edges,
        annotations=[_annotation("overview_pp_note", "Main line represents last-rank completion; side branches show PP rank handoff.", ["decoder_stack"], parallel_facts)],
        coverage=[_coverage("model execution", [c_exec, c_parallel], exec_facts + parallel_facts)],
    )


def _decoder_page(c_exec: str, c_moe: str, decoder_facts: list[str], moe_facts: list[str]) -> dict[str, Any]:
    rms_ports = [
        _port("hidden_in", "hidden", "input", "tensor", "left", 0),
        _port("residual_in", "residual", "input", "tensor", "bottom", 1),
        _port("hidden_out", "normalized", "output", "tensor", "right", 0),
        _port("residual_out", "residual", "output", "tensor", "bottom", 1),
    ]
    nodes = [
        _node("decoder_input", "data", "primary", "Decoder Input", "hidden + residual", "runtime", [c_exec], decoder_facts, 0, ports=[_port("hidden", "hidden", "output", "tensor", "right", 0), _port("residual", "residual", "output", "tensor", "bottom", 1)]),
        _node("input_rmsnorm", "process", "primary", "Input RMSNorm", "fused residual handoff", "runtime", [c_exec], decoder_facts, 1, ports=rms_ports),
        _node("self_attention", "component", "primary", "Self Attention", "HYV3Attention", "runtime", [c_exec], decoder_facts, 2),
        _node("post_attention_rmsnorm", "process", "primary", "Post-Attention RMSNorm", "fused residual handoff", "runtime", [c_exec], decoder_facts, 3, ports=rms_ports),
        _node("ffn_stage", "component", "primary", "FFN Stage", "Dense or MoE per layer", "runtime", [c_exec, c_moe], decoder_facts + moe_facts, 4, size=(380, 210)),
        _node("dense_ffn", "component", "contained", "Dense FFN", "HYV3FeedForward", "construction", [c_exec], decoder_facts, 5, parent_id="ffn_stage", size=(140, 58)),
        _node("moe_ffn", "component", "contained", "MoE Variant", "HYV3MoEFused", "construction", [c_moe], moe_facts, 6, parent_id="ffn_stage", size=(140, 58)),
        _node("decoder_output", "data", "primary", "Decoder Output", "hidden + residual", "runtime", [c_exec], decoder_facts, 7, ports=[_port("hidden", "hidden", "input", "tensor", "left", 0), _port("residual", "residual", "input", "tensor", "bottom", 1)]),
        _node("residual_init", "process", "auxiliary", "Residual Init", "residual is None -> hidden_states", "runtime", [c_exec], decoder_facts, 8, size=(245, 74)),
    ]
    edges = [
        _edge("decoder_hidden_to_input_norm", "tensor_flow", "decoder_input", "input_rmsnorm", "runtime", [c_exec], decoder_facts, source_port="hidden", target_port="hidden_in"),
        _edge("decoder_input_norm_to_attention", "tensor_flow", "input_rmsnorm", "self_attention", "runtime", [c_exec], decoder_facts, source_port="hidden_out"),
        _edge("decoder_attention_to_post_norm", "tensor_flow", "self_attention", "post_attention_rmsnorm", "runtime", [c_exec], decoder_facts, target_port="hidden_in"),
        _edge("decoder_post_norm_to_ffn", "tensor_flow", "post_attention_rmsnorm", "ffn_stage", "runtime", [c_exec], decoder_facts, source_port="hidden_out"),
        _edge("decoder_ffn_to_output", "tensor_flow", "ffn_stage", "decoder_output", "runtime", [c_exec], decoder_facts, target_port="hidden"),
        _edge("decoder_residual_to_input_norm", "residual_flow", "decoder_input", "input_rmsnorm", "runtime", [c_exec], decoder_facts, source_port="residual", target_port="residual_in", route_hint="lower_lane"),
        _edge("decoder_residual_init_to_lane", "control", "residual_init", "input_rmsnorm", "runtime", [c_exec], decoder_facts, label="if residual is None", show_label=True, priority="secondary", target_port="residual_in", route_hint="local"),
        _edge("decoder_input_norm_residual_to_post_norm", "residual_flow", "input_rmsnorm", "post_attention_rmsnorm", "runtime", [c_exec], decoder_facts, source_port="residual_out", target_port="residual_in", route_hint="lower_lane"),
        _edge("decoder_post_norm_residual_to_output", "residual_flow", "post_attention_rmsnorm", "decoder_output", "runtime", [c_exec], decoder_facts, source_port="residual_out", target_port="residual", route_hint="lower_lane"),
        _edge("decoder_dense_contained", "containment", "ffn_stage", "dense_ffn", "construction", [c_exec], decoder_facts, label="First K", show_label=True, priority="auxiliary", route_hint="hidden"),
        _edge("decoder_moe_contained", "containment", "ffn_stage", "moe_ffn", "construction", [c_moe], moe_facts, label="Remaining", show_label=True, priority="auxiliary", route_hint="hidden"),
    ]
    return _page(
        "decoder_block",
        "Decoder Block",
        "How does one decoder layer move hidden states and residual state?",
        "Show hidden-state and residual lanes while keeping Dense/MoE as construction variants.",
        "block_with_residual",
        ["HYV3DecoderLayer.forward"],
        ["decoder_input", "input_rmsnorm", "self_attention", "post_attention_rmsnorm", "ffn_stage", "decoder_output"],
        nodes,
        edges,
        groups=[_group("ffn_stage_group", "FFN construction variants", "container", ["dense_ffn", "moe_ffn"])],
        annotations=[_annotation("decoder_variant_note", "First K layers use Dense FFN; remaining layers use MoE. They are not runtime branches.", ["ffn_stage"], decoder_facts + moe_facts)],
        coverage=[_coverage("decoder block", [c_exec, c_moe], decoder_facts + moe_facts)],
    )


def _attention_page(c_exec: str, c_attn: str, c_qk: str, c_kv: str, c_backend: str, attn_facts: list[str]) -> dict[str, Any]:
    nodes = [
        _node("attention_input", "data", "primary", "Hidden States", "attention input", "runtime", [c_exec, c_attn], attn_facts, 0),
        _node("qkv_projection", "component", "primary", "QKV Projection", "QKVParallelLinear", "runtime", [c_attn], attn_facts, 1, ports=[_port("hidden_in", "hidden", "input", "tensor", "left", 0), _port("qkv_out", "qkv", "output", "tensor", "right", 0)], badges=["TP"]),
        _node("hpc_rope_norm", "process", "branch", "HPC Fused Processing", "HpcRopeNorm(qkv)", "runtime", [c_qk], attn_facts, 2, ports=[_port("qkv_in", "qkv", "input", "tensor", "left", 0), _port("processed_q", "processed Q", "output", "tensor", "right", 0), _port("cache_write", "KV write", "output", "cache", "bottom", 1)]),
        _node("qkv_split", "process", "branch", "Split Q / K / V", "fallback path", "runtime", [c_attn], attn_facts, 3, ports=[_port("qkv_in", "qkv", "input", "tensor", "left", 0), _port("q", "Q", "output", "tensor", "right", 0), _port("k", "K", "output", "tensor", "right", 1), _port("v", "V", "output", "tensor", "right", 2)]),
        _node("q_stream", "data", "branch", "Q", "query stream", "runtime", [c_attn], attn_facts, 4, size=(96, 56)),
        _node("k_stream", "data", "branch", "K", "key stream", "runtime", [c_attn], attn_facts, 5, size=(96, 56)),
        _node("v_stream", "data", "branch", "V", "value stream", "runtime", [c_attn], attn_facts, 6, size=(96, 56)),
        _node("q_norm", "process", "branch", "Optional Q RMSNorm", "self.use_qk_norm", "runtime", [c_qk], attn_facts, 7),
        _node("k_norm", "process", "branch", "Optional K RMSNorm", "self.use_qk_norm", "runtime", [c_qk], attn_facts, 8),
        _node("rope", "process", "branch", "RoPE", "rotary embedding", "runtime", [c_qk], attn_facts, 9),
        _node("kv_cache", "storage", "boundary", "KV Cache", "external cache boundary", "external", [c_kv, c_backend], attn_facts, 10, ports=[_port("write", "write", "input", "cache", "left", 0), _port("read", "read", "output", "cache", "right", 1)]),
        _node("attention_backend", "external_component", "boundary", "vLLM Attention Backend", "Attention", "external", [c_backend], attn_facts, 11, ports=[_port("q", "Q", "input", "tensor", "left", 0), _port("k", "K", "input", "tensor", "left", 1), _port("v", "V", "input", "tensor", "left", 2), _port("kv_cache", "KV cache", "input", "cache", "top", 3), _port("out", "attention", "output", "tensor", "right", 0)]),
        _node("output_projection", "component", "primary", "Output Projection", "RowParallelLinear", "runtime", [c_attn], attn_facts, 12, badges=["TP"]),
        _node("attention_output", "data", "primary", "Attention Output", "projected states", "runtime", [c_exec, c_attn], attn_facts, 13),
    ]
    edges = [
        _edge("attention_input_to_qkv", "tensor_flow", "attention_input", "qkv_projection", "runtime", [c_attn], attn_facts, target_port="hidden_in"),
        _edge("attention_qkv_to_hpc", "tensor_flow", "qkv_projection", "hpc_rope_norm", "runtime", [c_qk], attn_facts, source_port="qkv_out", target_port="qkv_in", route_hint="branch_top"),
        _edge("attention_hpc_to_backend_q", "tensor_flow", "hpc_rope_norm", "attention_backend", "runtime", [c_qk, c_backend], attn_facts, source_port="processed_q", target_port="q", route_hint="branch_top"),
        _edge("attention_hpc_to_kv_cache", "external_boundary", "hpc_rope_norm", "kv_cache", "external", [c_kv], attn_facts, source_port="cache_write", target_port="write", label="KV write", show_label=True, priority="secondary", route_hint="local"),
        _edge("attention_qkv_to_split", "tensor_flow", "qkv_projection", "qkv_split", "runtime", [c_attn], attn_facts, source_port="qkv_out", target_port="qkv_in", route_hint="branch_bottom"),
        _edge("attention_split_to_q", "tensor_flow", "qkv_split", "q_stream", "runtime", [c_attn], attn_facts, source_port="q", route_hint="local"),
        _edge("attention_split_to_k", "tensor_flow", "qkv_split", "k_stream", "runtime", [c_attn], attn_facts, source_port="k", route_hint="local"),
        _edge("attention_split_to_v", "tensor_flow", "qkv_split", "v_stream", "runtime", [c_attn], attn_facts, source_port="v", route_hint="local"),
        _edge("attention_q_to_norm", "tensor_flow", "q_stream", "q_norm", "runtime", [c_qk], attn_facts),
        _edge("attention_k_to_norm", "tensor_flow", "k_stream", "k_norm", "runtime", [c_qk], attn_facts),
        _edge("attention_q_norm_to_rope", "tensor_flow", "q_norm", "rope", "runtime", [c_qk], attn_facts),
        _edge("attention_k_norm_to_rope", "tensor_flow", "k_norm", "rope", "runtime", [c_qk], attn_facts),
        _edge("attention_rope_to_backend_q", "tensor_flow", "rope", "attention_backend", "runtime", [c_backend], attn_facts, target_port="q", route_hint="local"),
        _edge("attention_v_to_backend", "tensor_flow", "v_stream", "attention_backend", "runtime", [c_backend], attn_facts, target_port="v", route_hint="local"),
        _edge("attention_kv_cache_read", "external_boundary", "kv_cache", "attention_backend", "external", [c_kv, c_backend], attn_facts, source_port="read", target_port="kv_cache", label="external read", show_label=True, priority="secondary", route_hint="local"),
        _edge("attention_backend_to_output_projection", "delegation", "attention_backend", "output_projection", "external", [c_backend, c_attn], attn_facts, label="backend output", show_label=True, priority="secondary"),
        _edge("attention_projection_to_output", "tensor_flow", "output_projection", "attention_output", "runtime", [c_attn], attn_facts),
    ]
    return _page(
        "attention_adaptation",
        "Attention Adaptation",
        "How does HY V3 attention adapt to the vLLM attention backend?",
        "Show QKV projection, HPC/fallback branches, KV cache boundary and external backend merge.",
        "branch_merge",
        ["HYV3Attention.forward"],
        ["attention_input", "qkv_projection", "attention_backend", "output_projection", "attention_output"],
        nodes,
        edges,
        groups=[_group("local_attention", "Local Adapter", "region", ["attention_input", "qkv_projection", "hpc_rope_norm", "qkv_split", "q_stream", "k_stream", "v_stream", "q_norm", "k_norm", "rope"]), _group("external_attention", "External vLLM Runtime", "boundary", ["kv_cache", "attention_backend"])],
        annotations=[_annotation("attention_external_note", "KV cache read and vLLM Attention internals are external boundaries, not direct local behavior.", ["attention_backend", "kv_cache"], attn_facts)],
        coverage=[_coverage("attention branch merge", [c_attn, c_qk, c_kv, c_backend], attn_facts)],
    )


def _moe_page(c_moe: str, c_parallel: str, moe_facts: list[str], parallel_facts: list[str]) -> dict[str, Any]:
    nodes = [
        _node("moe_input", "data", "primary", "Hidden States", "MoE input", "runtime", [c_moe], moe_facts, 0),
        _node("flatten_tokens", "process", "primary", "Flatten Tokens", "reshape for routing", "runtime", [c_moe], moe_facts, 1),
        _node("gate_linear", "component", "primary", "GateLinear", "router", "runtime", [c_moe], moe_facts, 2),
        _node("router_logits", "data", "primary", "Router Logits", "routing scores", "runtime", [c_moe], moe_facts, 3),
        _node(
            "fused_moe",
            "component",
            "primary",
            "FusedMoE",
            "external fused experts",
            "runtime",
            [c_moe],
            moe_facts,
            4,
            ports=[
                _port("hidden", "hidden", "input", "tensor", "left", 0),
                _port("router", "router logits", "input", "tensor", "top", 1),
                _port("expert_bias", "expert bias", "input", "config", "bottom", 2),
                _port("eplb", "EPLB metadata", "input", "control", "bottom", 3),
                _port("out", "output", "output", "tensor", "right", 0),
            ],
            badges=["EP"],
            size=(390, 220),
        ),
        _node("routed_experts", "component", "contained", "Routed Experts", "inside FusedMoE", "external", [c_moe], moe_facts, 5, parent_id="fused_moe", size=(120, 56)),
        _node("shared_experts", "component", "contained", "Optional Shared Experts", "shared_mlp", "construction", [c_moe], moe_facts, 6, parent_id="fused_moe", size=(150, 56)),
        _node("expert_bias", "data", "auxiliary", "Expert Bias", "construction dependency", "construction", [c_moe], moe_facts, 7, ports=[_port("out", "bias", "output", "config", "right", 0)]),
        _node("eplb_metadata", "strategy", "strategy", "EPLB / EP Metadata", "expert parallel", "parallel_partition", [c_parallel], parallel_facts, 8, ports=[_port("out", "metadata", "output", "control", "right", 0)]),
        _node("restore_shape", "process", "primary", "Restore Original Shape", "view(orig_shape)", "runtime", [c_moe], moe_facts, 9),
        _node("moe_output", "data", "primary", "Output", "hidden states", "runtime", [c_moe], moe_facts, 10),
    ]
    edges = [
        _edge("moe_input_to_flatten", "tensor_flow", "moe_input", "flatten_tokens", "runtime", [c_moe], moe_facts),
        _edge("moe_flatten_to_gate", "tensor_flow", "flatten_tokens", "gate_linear", "runtime", [c_moe], moe_facts),
        _edge("moe_gate_to_logits", "tensor_flow", "gate_linear", "router_logits", "runtime", [c_moe], moe_facts),
        _edge("moe_hidden_to_fused", "tensor_flow", "flatten_tokens", "fused_moe", "runtime", [c_moe], moe_facts, target_port="hidden", route_hint="branch_bottom"),
        _edge("moe_logits_to_fused", "tensor_flow", "router_logits", "fused_moe", "runtime", [c_moe], moe_facts, target_port="router", route_hint="branch_top"),
        _edge("moe_expert_bias_dependency", "dependency", "expert_bias", "fused_moe", "construction", [c_moe], moe_facts, target_port="expert_bias", label="bias", show_label=True, priority="secondary", visible=False),
        _edge("moe_eplb_dependency", "parallel_effect", "eplb_metadata", "fused_moe", "parallel_partition", [c_parallel], parallel_facts, target_port="eplb", label="EP", show_label=True, priority="secondary"),
        _edge("moe_routed_containment", "containment", "fused_moe", "routed_experts", "external", [c_moe], moe_facts, route_hint="hidden"),
        _edge("moe_shared_containment", "containment", "fused_moe", "shared_experts", "construction", [c_moe], moe_facts, route_hint="hidden"),
        _edge("moe_fused_to_restore", "tensor_flow", "fused_moe", "restore_shape", "runtime", [c_moe], moe_facts),
        _edge("moe_restore_to_output", "tensor_flow", "restore_shape", "moe_output", "runtime", [c_moe], moe_facts),
    ]
    return _page(
        "moe_execution",
        "MoE Execution",
        "How are tokens routed into fused experts?",
        "Show router logits and hidden states entering FusedMoE, with experts contained inside the FusedMoE boundary.",
        "routed_container",
        ["HYV3MoEFused.forward"],
        ["moe_input", "flatten_tokens", "gate_linear", "router_logits", "fused_moe", "restore_shape", "moe_output"],
        nodes,
        edges,
        groups=[_group("fused_moe_container", "FusedMoE internals", "container", ["routed_experts", "shared_experts"])],
        annotations=[_annotation("moe_container_note", "Experts are internal members of FusedMoE, not a stage after FusedMoE.", ["fused_moe"], moe_facts)],
        coverage=[_coverage("moe execution", [c_moe, c_parallel], moe_facts + parallel_facts)],
    )


def _checkpoint_page(c_checkpoint: str, checkpoint_facts: list[str]) -> dict[str, Any]:
    weights_ports = _ports("weights")
    nodes = [
        _node("wrapper_weights", "data", "primary", "Weights", "wrapper input", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 0, ports=weights_ports),
        _node("wrapper_filter", "process", "primary", "_filter_weights", "nested filter", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 1, ports=weights_ports),
        _node("speculative_filter", "process", "primary", "Speculative Layer Filter", "skip unmatched layers", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 2, ports=weights_ports),
        _node("tied_lm_head_filter", "process", "primary", "Tied LM Head Filter", "skip tied lm_head", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 3, ports=weights_ports),
        _node("auto_weights_loader", "external_component", "primary", "AutoWeightsLoader", "external loader", "external", [c_checkpoint], checkpoint_facts, 4, ports=weights_ports),
        _node("wrapper_loaded_set", "storage", "primary", "Loaded Set", "wrapper result", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 5, ports=weights_ports),
        _node("model_loaded_weight", "data", "primary", "Loaded Weight", "model load loop", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 6, ports=weights_ports),
        _node("fp8_scale_remap", "process", "primary", "Optional FP8 Scale Remap", "name remap", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 7, ports=weights_ports),
        _node("mapping_dispatch", "process", "primary", "Mapping Dispatch", "stacked / expert / regular", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 8, ports=weights_ports, size=(240, 110)),
        _node("stacked_mapping", "mapping", "branch", "Stacked Mapping", "q/k/v + gate/up", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 9, ports=weights_ports),
        _node("qkv_proj", "component", "branch", "qkv_proj", "packed target", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 10, ports=weights_ports),
        _node("gate_up_proj", "component", "branch", "gate_up_proj", "stacked target", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 11, ports=weights_ports),
        _node("expert_mapping", "mapping", "branch", "Expert Mapping", "expert_id + shard_id", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 12, ports=weights_ports),
        _node("fused_moe_params", "component", "branch", "FusedMoE Parameters", "expert params", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 13, ports=weights_ports),
        _node("regular_parameter", "mapping", "branch", "Regular Parameter", "router.gate rename", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 14, ports=weights_ports),
        _node("pp_missing_filter", "process", "auxiliary", "PP Missing Filter", "applies to branches", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 15, ports=weights_ports),
        _node("param_weight_loader", "process", "primary", "param.weight_loader", "parameter loader", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 16, ports=weights_ports),
        _node("default_loader", "process", "auxiliary", "default_weight_loader", "fallback", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 17, ports=weights_ports),
        _node("model_loaded_params", "storage", "primary", "loaded_params", "model result", "checkpoint_loading", [c_checkpoint], checkpoint_facts, 18, ports=weights_ports),
    ]
    edges = [
        _edge("wrapper_weights_to_filter", "mapping", "wrapper_weights", "wrapper_filter", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("wrapper_filter_to_speculative", "mapping", "wrapper_filter", "speculative_filter", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("wrapper_speculative_to_tied", "mapping", "speculative_filter", "tied_lm_head_filter", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("wrapper_tied_to_auto_loader", "mapping", "tied_lm_head_filter", "auto_weights_loader", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("wrapper_loader_to_loaded", "mapping", "auto_weights_loader", "wrapper_loaded_set", "external", [c_checkpoint], checkpoint_facts),
        _edge("model_weight_to_fp8", "mapping", "model_loaded_weight", "fp8_scale_remap", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("model_fp8_to_dispatch", "mapping", "fp8_scale_remap", "mapping_dispatch", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("dispatch_to_stacked", "mapping", "mapping_dispatch", "stacked_mapping", "checkpoint_loading", [c_checkpoint], checkpoint_facts, route_hint="branch_top"),
        _edge("stacked_to_qkv", "mapping", "stacked_mapping", "qkv_proj", "checkpoint_loading", [c_checkpoint], checkpoint_facts, label="q/k/v", show_label=True),
        _edge("stacked_to_gate_up", "mapping", "stacked_mapping", "gate_up_proj", "checkpoint_loading", [c_checkpoint], checkpoint_facts, label="gate/up", show_label=True),
        _edge("dispatch_to_expert", "mapping", "mapping_dispatch", "expert_mapping", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("expert_to_params", "mapping", "expert_mapping", "fused_moe_params", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("dispatch_to_regular", "mapping", "mapping_dispatch", "regular_parameter", "checkpoint_loading", [c_checkpoint], checkpoint_facts, route_hint="branch_bottom"),
        _edge("pp_filter_to_stacked", "mapping", "pp_missing_filter", "stacked_mapping", "checkpoint_loading", [c_checkpoint], checkpoint_facts, label="filter", show_label=True, priority="secondary", visible=False),
        _edge("pp_filter_to_expert", "mapping", "pp_missing_filter", "expert_mapping", "checkpoint_loading", [c_checkpoint], checkpoint_facts, label="filter", show_label=True, priority="secondary", visible=False),
        _edge("pp_filter_to_regular", "mapping", "pp_missing_filter", "regular_parameter", "checkpoint_loading", [c_checkpoint], checkpoint_facts, label="filter", show_label=True, priority="secondary", visible=False),
        _edge("regular_to_param_loader", "mapping", "regular_parameter", "param_weight_loader", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("regular_to_default_loader", "mapping", "regular_parameter", "default_loader", "checkpoint_loading", [c_checkpoint], checkpoint_facts, label="fallback", show_label=True, priority="secondary"),
        _edge("qkv_to_loaded_params", "mapping", "qkv_proj", "model_loaded_params", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("gate_up_to_loaded_params", "mapping", "gate_up_proj", "model_loaded_params", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("expert_to_loaded_params", "mapping", "fused_moe_params", "model_loaded_params", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
        _edge("param_loader_to_loaded_params", "mapping", "param_weight_loader", "model_loaded_params", "checkpoint_loading", [c_checkpoint], checkpoint_facts, visible=False),
        _edge("default_loader_to_loaded_params", "mapping", "default_loader", "model_loaded_params", "checkpoint_loading", [c_checkpoint], checkpoint_facts),
    ]
    return _page(
        "checkpoint_weight_loading",
        "Checkpoint and Weight Loading",
        "How do HF checkpoint weights become vLLM parameters?",
        "Separate wrapper AutoWeightsLoader flow from model-level mapping dispatch and parameter loaders.",
        "mapping_dispatch",
        ["HYV3ForCausalLM.load_weights", "HYV3Model.load_weights"],
        ["wrapper_weights", "wrapper_filter", "speculative_filter", "tied_lm_head_filter", "auto_weights_loader", "wrapper_loaded_set"],
        nodes,
        edges,
        groups=[_group("wrapper_loader_lane", "HYV3ForCausalLM.load_weights", "lane", ["wrapper_weights", "wrapper_filter", "speculative_filter", "tied_lm_head_filter", "auto_weights_loader", "wrapper_loaded_set"]), _group("model_loader_lane", "HYV3Model.load_weights", "lane", ["model_loaded_weight", "fp8_scale_remap", "mapping_dispatch", "stacked_mapping", "expert_mapping", "regular_parameter", "model_loaded_params"])],
        annotations=[_annotation("checkpoint_two_entrypoints", "Wrapper and model load_weights are separate entrypoints.", ["wrapper_weights", "model_loaded_weight"], checkpoint_facts)],
        coverage=[_coverage("checkpoint loading", [c_checkpoint], checkpoint_facts)],
    )


def _parallel_page(c_parallel: str, parallel_facts: list[str], *, include_ep: bool) -> dict[str, Any]:
    nodes = [
        _node("tensor_parallel_panel", "strategy", "strategy", "Tensor Parallel", "independent panel", "parallel_partition", [c_parallel], parallel_facts, 0, ports=[], badges=["TP"], size=(310, 220)),
        _node("tp_components", "annotation", "annotation", "World size, head partition\nEmbedding, QKV, row linear, LM head", "", "parallel_partition", [c_parallel], parallel_facts, 1, ports=[], size=(340, 128)),
        _node("pipeline_parallel_panel", "strategy", "strategy", "Pipeline Parallel", "independent panel", "parallel_partition", [c_parallel], parallel_facts, 2, ports=[], badges=["PP"], size=(310, 220)),
        _node("pp_components", "annotation", "annotation", "make_layers, start/end, PPMissingLayer\nrank input/output, missing filter", "", "parallel_partition", [c_parallel], parallel_facts, 3, ports=[], size=(340, 128)),
        _node("expert_parallel_panel", "strategy", "strategy", "Expert Parallel", "independent panel", "parallel_partition", [c_parallel], parallel_facts, 4, ports=[], badges=["EP"], size=(310, 220)),
        _node("ep_components", "annotation", "annotation", "get_ep_group, ep_rank/size, local experts, FusedMoE, EPLB", "", "parallel_partition", [c_parallel], parallel_facts, 5, ports=[], size=(340, 128)),
    ]
    if not include_ep:
        nodes = [node for node in nodes if not node["id"].startswith("expert") and not node["id"].startswith("ep_")]
    return _page(
        "parallel_strategies",
        "Parallel Strategies",
        "How are TP, PP and EP organized without implying a serial flow?",
        "Use independent panels for tensor, pipeline and expert parallel strategy facts.",
        "strategy_matrix",
        ["parallel setup"],
        [],
        nodes,
        [],
        groups=[_group("tp_panel", "Tensor Parallel", "region", ["tensor_parallel_panel", "tp_components"]), _group("pp_panel", "Pipeline Parallel", "region", ["pipeline_parallel_panel", "pp_components"]), _group("ep_panel", "Expert Parallel", "region", ["expert_parallel_panel", "ep_components"])],
        annotations=[_annotation("parallel_no_chain", "No TP -> PP -> EP runtime chain is implied.", ["tensor_parallel_panel", "pipeline_parallel_panel", "expert_parallel_panel"], parallel_facts)],
        coverage=[_coverage("parallel strategies", [c_parallel], parallel_facts)],
    )


def _boundary_page(
    c_adapter: str,
    c_boundary: str,
    c_backend: str,
    c_moe: str,
    c_checkpoint: str,
    c_parallel: str,
    c_caps: str,
    boundary_facts: list[str],
    cap_facts: list[str],
    *,
    include_moe: bool,
) -> dict[str, Any]:
    nodes = [
        _node("hyv3_for_causal_lm", "component", "primary", "HYV3ForCausalLM", "local wrapper", "construction", [c_adapter], boundary_facts, 0),
        _node("hyv3_model", "component", "primary", "HYV3Model", "local model body", "construction", [c_adapter], boundary_facts, 1),
        _node("hyv3_attention", "component", "primary", "HYV3Attention", "local adapter attention", "construction", [c_backend], boundary_facts, 2),
        _node("hyv3_moe", "component", "primary", "HYV3MoEFused", "local MoE adapter", "construction", [c_moe], boundary_facts, 3),
        _node("local_weight_mapping", "process", "primary", "Local Weight Mapping", "load_weights", "checkpoint_loading", [c_checkpoint], boundary_facts, 4),
        _node("local_parallel_logic", "process", "primary", "Local Parallel Orchestration", "TP / PP / EP", "parallel_partition", [c_parallel], boundary_facts, 5),
        _node("adapter_boundary", "boundary", "boundary", "Adapter Boundary", "local evidence stops here", "external", [c_boundary], boundary_facts, 6, ports=[_port("local", "local", "input", "control", "left", 0), _port("external", "external", "output", "control", "right", 0)]),
        _node("external_attention", "external_component", "boundary", "Attention Backend", "vLLM Attention", "external", [c_backend], boundary_facts, 7),
        _node("external_hpc_rope", "external_component", "boundary", "HpcRopeNorm", "external fused path", "external", [c_backend], boundary_facts, 8),
        _node("external_fused_moe", "external_component", "boundary", "FusedMoE Runtime", "external experts", "external", [c_moe], boundary_facts, 9),
        _node("external_auto_loader", "external_component", "boundary", "AutoWeightsLoader", "external wrapper loader", "external", [c_checkpoint], boundary_facts, 10),
        _node("external_param_loaders", "external_component", "boundary", "Parameter Loaders", "weight_loader/default", "external", [c_checkpoint], boundary_facts, 11),
        _node("capabilities", "annotation", "annotation", "Capabilities", "SupportsPP / LoRA / MoE / torch.compile", "construction", [c_caps], cap_facts, 12, ports=[], badges=["PP", "LoRA", "MoE", "compile"], size=(260, 86)),
    ]
    if not include_moe:
        nodes = [node for node in nodes if node["id"] not in {"hyv3_moe", "external_fused_moe"}]
    edges = [
        _edge("boundary_wrapper_contains_model", "dependency", "hyv3_for_causal_lm", "hyv3_model", "construction", [c_adapter], boundary_facts, label="contains", show_label=True),
        _edge("boundary_attention_to_backend", "delegation", "hyv3_attention", "external_attention", "external", [c_backend], boundary_facts, label="delegates", show_label=True),
        _edge("boundary_attention_to_hpc", "delegation", "hyv3_attention", "external_hpc_rope", "external", [c_backend], boundary_facts, label="optional fused", show_label=True),
        _edge("boundary_moe_to_fused", "delegation", "hyv3_moe", "external_fused_moe", "external", [c_moe], boundary_facts, label="delegates", show_label=True),
        _edge("boundary_wrapper_to_auto_loader", "delegation", "local_weight_mapping", "external_auto_loader", "external", [c_checkpoint], boundary_facts, label="wrapper loader", show_label=True),
        _edge("boundary_model_to_param_loaders", "delegation", "local_weight_mapping", "external_param_loaders", "external", [c_checkpoint], boundary_facts, label="param loaders", show_label=True),
        _edge("boundary_parallel_configures_model", "dependency", "local_parallel_logic", "hyv3_model", "parallel_partition", [c_parallel], boundary_facts, label="configures", show_label=True, visible=False),
    ]
    if not include_moe:
        edges = [edge for edge in edges if "moe" not in edge["id"]]
    return _page(
        "adapter_boundary",
        "vLLM Adapter Boundary",
        "What is local adapter behavior and what is external vLLM runtime behavior?",
        "Separate source-proven local classes and methods from imported runtime implementations.",
        "boundary_map",
        ["hy_v3.py single-file boundary"],
        [],
        nodes,
        edges,
        groups=[_group("local_adapter_region", "Local Model Adapter", "region", ["hyv3_for_causal_lm", "hyv3_model", "hyv3_attention", "hyv3_moe", "local_weight_mapping", "local_parallel_logic", "capabilities"]), _group("external_runtime_region", "External vLLM Runtime", "boundary", ["external_attention", "external_hpc_rope", "external_fused_moe", "external_auto_loader", "external_param_loaders"])],
        annotations=[_annotation("boundary_external_note", "External nodes are imported behavior boundaries, not direct local implementation claims.", ["external_attention", "external_fused_moe", "external_auto_loader"], boundary_facts)],
        coverage=[_coverage("adapter boundary", [c_adapter, c_boundary, c_backend, c_checkpoint, c_parallel, c_caps], boundary_facts + cap_facts)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architect_brief", type=Path)
    parser.add_argument("architecture_concept", type=Path)
    parser.add_argument("boundary_report", type=Path)
    parser.add_argument("--author-type", choices=["baseline", "agent"], default="baseline")
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        design = build_baseline_design(
            _load_json(args.architect_brief),
            _load_json(args.architecture_concept),
            _load_json(args.boundary_report),
            author_type=args.author_type,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote baseline Architecture Design template to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
