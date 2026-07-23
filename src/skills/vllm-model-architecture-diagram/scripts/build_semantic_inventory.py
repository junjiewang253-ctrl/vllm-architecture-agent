#!/usr/bin/env python3
"""Build a semantic fact inventory from source-analysis 0.3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

INVENTORY_VERSION = "0.1"
SOURCE_VERSION = "0.3"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def _important(summary: str, category: str) -> bool:
    tokens = (
        "hpc_rope_norm",
        "qkv",
        "self.q_norm",
        "self.k_norm",
        "rotary_emb",
        "self.attn",
        "residual",
        "input_layernorm",
        "post_attention_layernorm",
        "FusedMoE",
        "GateLinear",
        "shared_mlp",
        "expert_bias",
        "view(orig_shape)",
        "vllm_config",
        "make_layers",
        "get_pp_group",
        "IntermediateTensors",
        "PPMissingLayer",
        "get_ep_group",
        "ep_rank",
        "ep_size",
        "maybe_remap_kv_scale_name",
        "is_pp_missing_parameter",
        "router.gate",
        "default_weight_loader",
        "AutoWeightsLoader",
        "tie_word_embeddings",
    )
    return category in {
        "model_structure",
        "runtime_flow",
        "construction",
        "configuration",
        "tensor_parallel",
        "pipeline_parallel",
        "expert_parallel",
        "checkpoint_loading",
        "capability",
        "external_boundary",
    } and any(token in summary for token in tokens)


def build_semantic_inventory(source_analysis: dict[str, Any]) -> dict[str, Any]:
    if source_analysis.get("schema_version") != SOURCE_VERSION:
        raise ValueError(f"source-analysis schema_version must be {SOURCE_VERSION!r}")
    facts: list[dict[str, Any]] = []
    for fact in source_analysis.get("semantic_facts", []):
        if not isinstance(fact, dict) or not isinstance(fact.get("fact_id"), str):
            continue
        category = str(fact.get("category") or "informational")
        summary = str(fact.get("summary") or "")
        relevance = "required" if _important(summary, category) else "informational"
        if category == "external_boundary":
            relevance = "required"
        item = {
            "fact_id": fact["fact_id"],
            "category": category,
            "relevance": relevance,
            "source_location": fact.get("source_location", {}),
            "summary": summary,
        }
        if relevance == "informational":
            item["exclusion_reason"] = "Informational syntax fact not required by the v0.8.1 diagram contract."
        facts.append(item)
    return {
        "schema_version": INVENTORY_VERSION,
        "source_analysis_version": SOURCE_VERSION,
        "model_name": Path(str(source_analysis.get("source_file", "model.py"))).stem.replace("_", "-"),
        "facts": sorted(facts, key=lambda item: (item.get("source_location", {}).get("line") or 0, item["fact_id"])),
        "unresolved": [],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic fact inventory.")
    parser.add_argument("input", type=Path, help="source-analysis JSON file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output semantic inventory JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists() or not args.input.is_file():
        print(f"error: file does not exist: {args.input}", file=sys.stderr)
        return 2
    try:
        inventory = build_semantic_inventory(_load_json(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote semantic inventory to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
