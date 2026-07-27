#!/usr/bin/env python3
"""Build a compact Architect Brief for the active Codex Agent.

The brief is not a design and does not choose final pages. It packages source
facts, concepts, boundaries and known baseline problems into a readable context
for an Agent-authored Architecture Design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

BRIEF_VERSION = "1.0"


CLUSTER_NEEDLES: dict[str, tuple[str, ...]] = {
    "model_execution": ("ForCausalLM", "HYV3Model", "forward", "compute_logits", "logits_processor"),
    "decoder_block": ("DecoderLayer", "input_layernorm", "post_attention_layernorm", "residual"),
    "attention": ("HYV3Attention", "QKVParallelLinear", "hpc_rope_norm", "rotary", "Attention"),
    "moe": ("HYV3MoEFused", "FusedMoE", "GateLinear", "shared_mlp", "num_experts"),
    "tensor_parallel": ("tensor", "QKVParallelLinear", "RowParallelLinear", "VocabParallelEmbedding", "ParallelLMHead"),
    "pipeline_parallel": ("get_pp_group", "make_layers", "PPMissingLayer", "IntermediateTensors"),
    "expert_parallel": ("get_ep_group", "ep_rank", "ep_size", "EPLB", "physical expert"),
    "checkpoint_wrapper_loader": ("HYV3ForCausalLM.load_weights", "AutoWeightsLoader", "_filter_weights", "tie_word_embeddings"),
    "checkpoint_model_loader": ("HYV3Model.load_weights", "weight_loader", "stacked_params_mapping", "router.gate", "default_weight_loader"),
    "external_boundaries": ("Attention", "FusedMoE", "AutoWeightsLoader", "HpcRopeNorm"),
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _facts_matching(facts: list[dict[str, Any]], needles: tuple[str, ...], limit: int = 18) -> list[dict[str, Any]]:
    lowered = tuple(item.lower() for item in needles)
    selected: list[dict[str, Any]] = []
    for fact in facts:
        haystack = json.dumps(fact, ensure_ascii=False).lower()
        if any(needle.lower() in haystack for needle in lowered):
            selected.append(fact)
        if len(selected) >= limit:
            break
    return selected


def _brief_fact(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_id": fact.get("id"),
        "type": fact.get("type"),
        "owner_class": fact.get("owner_class"),
        "method": fact.get("method"),
        "line": fact.get("line"),
        "summary": fact.get("summary"),
    }


def _cluster(
    name: str,
    facts: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    needles: tuple[str, ...],
) -> dict[str, Any]:
    selected = _facts_matching(facts, needles)
    fact_ids = {fact.get("id") for fact in selected}
    candidate_concepts = [
        {
            "id": concept.get("id"),
            "title": concept.get("title"),
            "type": concept.get("type"),
        }
        for concept in concepts
        if any(fact_id in set(concept.get("evidence", [])) for fact_id in fact_ids)
    ]
    return {
        "id": name,
        "fact_ids": [str(fact.get("id")) for fact in selected if isinstance(fact.get("id"), str)],
        "source_snippets": [_brief_fact(fact) for fact in selected[:8]],
        "methods": sorted({str(fact.get("method")) for fact in selected if fact.get("method")}),
        "candidate_concepts": candidate_concepts,
        "unresolved_questions": [],
    }


def _concept_summary(concept: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": concept.get("id"),
        "title": concept.get("title"),
        "type": concept.get("type"),
        "purpose": concept.get("purpose"),
        "implementation": concept.get("implementation", []),
        "evidence": concept.get("evidence", [])[:12],
    }


def _model_profile(source_analysis: dict[str, Any], facts: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = "\n".join(str(fact.get("summary", "")) for fact in facts)
    return {
        "top_level_wrapper": "HYV3ForCausalLM" if "HYV3ForCausalLM" in summaries else None,
        "base_model": "HYV3Model" if "HYV3Model" in summaries else None,
        "decoder_class": "HYV3DecoderLayer" if "HYV3DecoderLayer" in summaries else None,
        "attention_class": "HYV3Attention" if "HYV3Attention" in summaries else None,
        "dense_ffn_class": "HYV3FeedForward" if "HYV3FeedForward" in summaries else None,
        "moe_class": "HYV3MoEFused" if "HYV3MoEFused" in summaries else None,
        "supported_interfaces": [
            item
            for item in ["SupportsPP", "SupportsLoRA", "MixtureOfExperts", "support_torch_compile"]
            if item in summaries
        ],
        "decorators": source_analysis.get("decorators", []),
        "parallelism": {
            "tp": any(item in summaries for item in ["QKVParallelLinear", "RowParallelLinear", "VocabParallelEmbedding", "ParallelLMHead"]),
            "pp": any(item in summaries for item in ["get_pp_group", "make_layers", "PPMissingLayer"]),
            "ep": any(item in summaries for item in ["get_ep_group", "FusedMoE", "EPLB"]),
        },
        "load_weights_entrypoints": [
            item
            for item in ["HYV3ForCausalLM.load_weights", "HYV3Model.load_weights"]
            if item in summaries
        ],
    }


def build_architect_brief(
    source_analysis: dict[str, Any],
    fact_graph: dict[str, Any],
    concept_graph: dict[str, Any],
    boundary_report: dict[str, Any],
    source_file: Path,
    model_name: str | None = None,
) -> dict[str, Any]:
    facts = [fact for fact in fact_graph.get("facts", []) if isinstance(fact, dict)]
    concepts = [concept for concept in concept_graph.get("concepts", []) if isinstance(concept, dict)]
    resolved_model_name = str(model_name or concept_graph.get("model_name") or fact_graph.get("model_name") or source_analysis.get("model_name") or "model")
    return {
        "schema_version": BRIEF_VERSION,
        "model_name": resolved_model_name,
        "source_file": str(source_file),
        "source_sha256": _sha256(source_file),
        "model_profile": _model_profile(source_analysis, facts),
        "candidate_pages": [
            {"id": "model_execution_overview", "title": "Model Execution Overview", "view_kind": "pipeline"},
            {"id": "decoder_block", "title": "Decoder Block", "view_kind": "block_with_residual"},
            {"id": "attention_adaptation", "title": "Attention Adaptation", "view_kind": "branch_merge"},
            {"id": "moe_execution", "title": "MoE Execution", "view_kind": "routed_container"},
            {"id": "checkpoint_weight_loading", "title": "Checkpoint and Weight Loading", "view_kind": "mapping_dispatch"},
            {"id": "parallel_strategies", "title": "Parallel Strategies", "view_kind": "strategy_matrix"},
            {"id": "adapter_boundary", "title": "vLLM Adapter Boundary", "view_kind": "boundary_map"},
        ],
        "concepts": [_concept_summary(concept) for concept in concepts],
        "fact_clusters": [
            _cluster(name, facts, concepts, needles)
            for name, needles in CLUSTER_NEEDLES.items()
        ],
        "external_boundaries": boundary_report.get("external_components", []),
        "required_topics": [
            "model execution",
            "decoder hidden/residual channels",
            "attention QKV branch/merge",
            "MoE routing container",
            "checkpoint wrapper/model loaders",
            "TP/PP/EP independent strategies",
            "local/external vLLM boundary",
        ],
        "diagram_grammar": {
            "node_roles": ["primary", "branch", "auxiliary", "contained", "strategy", "boundary", "annotation"],
            "edge_kinds": ["tensor_flow", "residual_flow", "dependency", "mapping", "parallel_effect", "delegation", "external_boundary", "control"],
            "view_kinds": ["pipeline", "block_with_residual", "branch_merge", "routed_container", "mapping_dispatch", "strategy_matrix", "boundary_map"],
            "rule": "Use implementation names as subtitles; keep the architecture story in labels.",
        },
        "known_baseline_problems": [
            "v1.1 architect mode used a static Python design template.",
            "v1.1 did not include a Decoder Block page.",
            "v1.1 Checkpoint page did not split wrapper and model load_weights lanes.",
            "v1.1 MoE page implied experts after FusedMoE instead of contained by FusedMoE.",
            "v1.1 layout was row/column based and did not enforce view-kind patterns.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_analysis", type=Path)
    parser.add_argument("source_fact_graph", type=Path)
    parser.add_argument("architecture_concept", type=Path)
    parser.add_argument("boundary_report", type=Path)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        brief = build_architect_brief(
            _load_json(args.source_analysis),
            _load_json(args.source_fact_graph),
            _load_json(args.architecture_concept),
            _load_json(args.boundary_report),
            args.source_file,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(brief, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Architect Brief to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
