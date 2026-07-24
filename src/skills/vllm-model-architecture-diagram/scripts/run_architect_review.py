#!/usr/bin/env python3
"""Create an Architecture Design from a Source Fact Graph.

In the interactive v1.0 Skill workflow the active Agent may author this JSON
directly. This script provides a conservative, source-grounded architect pass
that can be validated and reproduced in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DESIGN_VERSION = "1.0"
FACT_GRAPH_VERSION = "1.0"
CONCEPT_TYPES = {
    "model_component",
    "execution_strategy",
    "parallel_strategy",
    "checkpoint_strategy",
    "configuration",
    "capability",
    "boundary",
    "optimization",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    return data


def _facts_matching(graph: dict[str, Any], *needles: str, limit: int = 8) -> list[str]:
    matches: list[str] = []
    lowered = [needle.lower() for needle in needles]
    for fact in graph.get("facts", []):
        if not isinstance(fact, dict) or not isinstance(fact.get("id"), str):
            continue
        haystack = json.dumps(fact, ensure_ascii=False).lower()
        if any(needle in haystack for needle in lowered):
            matches.append(fact["id"])
        if len(matches) >= limit:
            break
    return matches


def _concept(
    concept_id: str,
    title: str,
    concept_type: str,
    purpose: str,
    implementation: list[str],
    evidence: list[str],
    confidence: float = 0.9,
) -> dict[str, Any]:
    if concept_type not in CONCEPT_TYPES:
        raise ValueError(f"invalid concept type: {concept_type}")
    return {
        "id": concept_id,
        "type": concept_type,
        "title": title,
        "purpose": purpose,
        "implementation": implementation,
        "evidence": sorted(set(evidence)),
        "confidence": confidence,
    }


def _relationship(
    rel_id: str,
    source: str,
    target: str,
    relation: str,
    evidence: list[str],
    evidence_type: str = "derived",
) -> dict[str, Any]:
    return {
        "id": rel_id,
        "source": source,
        "target": target,
        "relation": relation,
        "evidence_type": evidence_type,
        "evidence": sorted(set(evidence)),
    }


def build_architecture_design(
    fact_graph: dict[str, Any],
    source_analysis: dict[str, Any] | None = None,
    semantic_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if fact_graph.get("schema_version") != FACT_GRAPH_VERSION:
        raise ValueError(f"Source Fact Graph schema_version must be {FACT_GRAPH_VERSION!r}")
    graph = fact_graph
    source_analysis = source_analysis or {}
    model_name = str(graph.get("model_name") or source_analysis.get("model_name") or "hy-v3")

    evidence = {
        "adapter": _facts_matching(graph, "ForCausalLM", "HYV3Model", "self.model"),
        "execution": _facts_matching(graph, "forward", "input_ids", "logits_processor"),
        "decoder": _facts_matching(graph, "HYV3DecoderLayer", "input_layernorm", "post_attention_layernorm"),
        "attention": _facts_matching(graph, "HYV3Attention", "QKVParallelLinear", "hpc_rope_norm", "Attention"),
        "moe": _facts_matching(graph, "HYV3MoEFused", "FusedMoE", "num_experts", "shared_mlp"),
        "parallel": _facts_matching(graph, "get_pp_group", "get_ep_group", "get_tensor_model_parallel_world_size", "parallel"),
        "checkpoint": _facts_matching(graph, "load_weights", "weight_loader", "stacked_params_mapping", "AutoWeightsLoader"),
        "boundary": _facts_matching(graph, "Attention", "FusedMoE", "AutoWeightsLoader"),
    }

    concepts = [
        _concept(
            "concept:hyv3_vllm_adapter",
            "HY V3 vLLM Adapter",
            "model_component",
            "Explain how the HY V3 model adapter enters vLLM through wrapper, model body and adapter interfaces.",
            ["HYV3ForCausalLM", "HYV3Model"],
            evidence["adapter"],
        ),
        _concept(
            "concept:transformer_execution_flow",
            "Transformer Execution Flow",
            "execution_strategy",
            "Summarize the forward path from inputs through embedding, decoder stack, final normalization and logits.",
            ["forward", "compute_logits"],
            evidence["execution"] + evidence["decoder"],
        ),
        _concept(
            "concept:tensor_parallel_attention_projection",
            "Tensor Parallel QKV Projection",
            "parallel_strategy",
            "Split QKV projection and output projection work across tensor parallel ranks.",
            ["QKVParallelLinear", "RowParallelLinear"],
            evidence["attention"] + _facts_matching(graph, "QKVParallelLinear", "RowParallelLinear"),
        ),
        _concept(
            "concept:qk_positional_processing",
            "Q/K Positional Processing",
            "execution_strategy",
            "Group optional Q/K RMSNorm, RoPE processing and the HPC fused path without claiming external backend internals.",
            ["HpcRopeNorm", "rotary_emb", "q_norm", "k_norm"],
            evidence["attention"],
        ),
        _concept(
            "concept:kv_cache_boundary",
            "KV Cache Boundary",
            "boundary",
            "Show KV cache participation as an integration boundary; local source proves adapter calls, not backend cache internals.",
            ["Attention", "kv_cache"],
            evidence["attention"],
        ),
        _concept(
            "concept:vllm_attention_backend_boundary",
            "vLLM Attention Backend Boundary",
            "boundary",
            "Represent the imported vLLM Attention backend as an external boundary with only local adapter calls as direct evidence.",
            ["Attention"],
            evidence["attention"],
        ),
        _concept(
            "concept:moe_execution_strategy",
            "MoE Execution Strategy",
            "execution_strategy",
            "Explain routing, fused experts, optional shared experts and restore-shape behavior at an architectural level.",
            ["GateLinear", "FusedMoE", "shared_mlp"],
            evidence["moe"],
        ),
        _concept(
            "concept:parallel_execution_strategy",
            "Parallel Execution Strategy",
            "parallel_strategy",
            "Separate tensor, pipeline and expert parallel strategies instead of drawing them as one data-flow chain.",
            ["TP", "PP", "EP"],
            evidence["parallel"],
        ),
        _concept(
            "concept:checkpoint_adaptation",
            "Checkpoint Adaptation",
            "checkpoint_strategy",
            "Describe wrapper and model load_weights flows, packed mappings, expert mappings and loader dispatch.",
            ["load_weights", "packed_modules_mapping", "AutoWeightsLoader"],
            evidence["checkpoint"],
        ),
        _concept(
            "concept:adapter_capabilities",
            "Adapter Capabilities",
            "capability",
            "Show PP, LoRA, MoE and torch.compile support as capabilities instead of fake inheritance chains.",
            ["SupportsPP", "SupportsLoRA", "MixtureOfExperts", "support_torch_compile"],
            _facts_matching(graph, "SupportsPP", "SupportsLoRA", "MixtureOfExperts", "support_torch_compile"),
        ),
        _concept(
            "concept:vllm_integration_boundary",
            "vLLM Integration Boundary",
            "boundary",
            "State which behaviors are proven in the model file and which are delegated to imported vLLM components.",
            ["Attention", "FusedMoE", "AutoWeightsLoader"],
            evidence["boundary"],
        ),
    ]

    relationships = [
        _relationship("concept-rel:adapter:execution", "concept:hyv3_vllm_adapter", "concept:transformer_execution_flow", "organizes", evidence["adapter"] + evidence["execution"]),
        _relationship("concept-rel:execution:attention", "concept:transformer_execution_flow", "concept:tensor_parallel_attention_projection", "uses", evidence["attention"]),
        _relationship("concept-rel:attention:kv-cache", "concept:qk_positional_processing", "concept:kv_cache_boundary", "feeds_boundary", evidence["attention"]),
        _relationship("concept-rel:kv-cache:backend", "concept:kv_cache_boundary", "concept:vllm_attention_backend_boundary", "external_boundary", evidence["attention"], "external"),
        _relationship("concept-rel:execution:moe", "concept:transformer_execution_flow", "concept:moe_execution_strategy", "uses", evidence["moe"]),
        _relationship("concept-rel:parallel:attention", "concept:parallel_execution_strategy", "concept:tensor_parallel_attention_projection", "parallelizes", evidence["parallel"] + evidence["attention"]),
        _relationship("concept-rel:parallel:moe", "concept:parallel_execution_strategy", "concept:moe_execution_strategy", "parallelizes", evidence["parallel"] + evidence["moe"]),
        _relationship("concept-rel:checkpoint:adapter", "concept:checkpoint_adaptation", "concept:hyv3_vllm_adapter", "loads", evidence["checkpoint"]),
        _relationship("concept-rel:adapter:capabilities", "concept:hyv3_vllm_adapter", "concept:adapter_capabilities", "exposes", _facts_matching(graph, "SupportsPP", "SupportsLoRA", "MixtureOfExperts", "support_torch_compile")),
        _relationship("concept-rel:boundary:attention", "concept:vllm_integration_boundary", "concept:vllm_attention_backend_boundary", "documents_external", evidence["boundary"], "external"),
    ]

    views = [
        {"id": "overview", "title": "HY V3 vLLM Adapter Overview", "purpose": "Explain how this model adapter enters vLLM.", "concepts": ["concept:hyv3_vllm_adapter", "concept:adapter_capabilities", "concept:vllm_integration_boundary"]},
        {"id": "decoder_layer_detail", "title": "Transformer Execution Flow", "purpose": "Explain one forward pass without expanding external kernels.", "concepts": ["concept:transformer_execution_flow", "concept:parallel_execution_strategy"]},
        {"id": "attention_detail", "title": "Attention Implementation", "purpose": "Explain how HF-style attention modules are adapted to vLLM attention.", "concepts": ["concept:tensor_parallel_attention_projection", "concept:qk_positional_processing", "concept:kv_cache_boundary", "concept:vllm_attention_backend_boundary"]},
        {"id": "moe_detail", "title": "MoE Execution Strategy", "purpose": "Explain router, experts, optional shared experts and EP adaptation.", "concepts": ["concept:moe_execution_strategy", "concept:parallel_execution_strategy"]},
        {"id": "parallelism", "title": "Parallel Execution Strategy", "purpose": "Explain TP, PP and EP as separate strategies.", "concepts": ["concept:parallel_execution_strategy", "concept:tensor_parallel_attention_projection", "concept:moe_execution_strategy"]},
        {"id": "weight_loading", "title": "Checkpoint Adaptation", "purpose": "Explain how HF checkpoint names and tensors are adapted to vLLM loaders.", "concepts": ["concept:checkpoint_adaptation"]},
        {"id": "integration_boundary", "title": "vLLM Integration Boundary", "purpose": "Explain what is proven locally and what remains inside imported vLLM components.", "concepts": ["concept:vllm_integration_boundary", "concept:vllm_attention_backend_boundary"]},
    ]

    boundaries = [
        {
            "name": "vLLM Attention",
            "concept_id": "concept:vllm_attention_backend_boundary",
            "reason": "Implemented outside hy_v3.py; local evidence proves construction and calls, not backend internals.",
            "allowed_claim": "Provides the attention backend used by the adapter.",
            "evidence": evidence["attention"],
        },
        {
            "name": "FusedMoE",
            "concept_id": "concept:moe_execution_strategy",
            "reason": "Imported fused expert implementation; local evidence proves adapter construction and invocation parameters.",
            "allowed_claim": "Executes fused MoE behind an external implementation boundary.",
            "evidence": evidence["moe"],
        },
        {
            "name": "AutoWeightsLoader",
            "concept_id": "concept:checkpoint_adaptation",
            "reason": "Imported loader implementation; local evidence proves wrapper load_weights delegates to it.",
            "allowed_claim": "Loads compatible weights through vLLM's external loader.",
            "evidence": evidence["checkpoint"],
        },
    ]

    return {
        "schema_version": DESIGN_VERSION,
        "source_fact_graph_version": fact_graph.get("schema_version"),
        "model_name": model_name,
        "concepts": concepts,
        "relationships": relationships,
        "views": views,
        "boundaries": boundaries,
        "assumptions": [
            {
                "id": "assumption:single_file_boundary",
                "text": "Claims are limited to hy_v3.py and imported component boundaries; external internals are not treated as direct facts.",
            }
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_fact_graph", type=Path)
    parser.add_argument("--source-analysis", type=Path)
    parser.add_argument("--semantic-inventory", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        graph = _load_json(args.source_fact_graph)
        analysis = _load_json(args.source_analysis) if args.source_analysis else None
        inventory = _load_json(args.semantic_inventory) if args.semantic_inventory else None
        design = build_architecture_design(graph, analysis, inventory)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Architecture Design to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
