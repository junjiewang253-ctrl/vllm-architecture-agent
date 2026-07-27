#!/usr/bin/env python3
"""Build a Source Fact Graph from source-analysis JSON.

The graph is a normalized fact layer: it records facts and conservative
relations between facts, but it does not create architecture concepts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SOURCE_ANALYSIS_VERSION = "0.3"
FACT_GRAPH_VERSION = "1.0"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return text or "unknown"


def _fact_node(record: dict[str, Any], fallback_type: str) -> dict[str, Any] | None:
    fact_id = record.get("fact_id")
    if not isinstance(fact_id, str):
        return None
    owner = record.get("owner_class")
    method = record.get("method")
    line = record.get("line")
    summary = record.get("summary") or record.get("source") or record.get("expression") or record.get("target") or fact_id
    node: dict[str, Any] = {
        "id": fact_id,
        "type": str(record.get("type") or fallback_type),
        "owner_class": owner if isinstance(owner, str) else None,
        "method": method if isinstance(method, str) else None,
        "line": line if isinstance(line, int) else None,
        "summary": str(summary),
        "evidence": [fact_id],
    }
    if isinstance(record.get("target"), str):
        node["target"] = record["target"]
    if isinstance(record.get("implementation"), str):
        node["implementation"] = record["implementation"]
    return node


def _add_relation(relations: list[dict[str, Any]], source: str, target: str, relation: str, evidence: list[str]) -> None:
    if not source or not target:
        return
    relations.append(
        {
            "id": f"rel:{_slug(source)}:{relation}:{_slug(target)}",
            "source": source,
            "target": target,
            "relation": relation,
            "evidence": sorted(set(evidence)),
        }
    )


def _records(data: dict[str, Any], key: str, fallback_type: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in data.get(key, []):
        if isinstance(item, dict):
            node = _fact_node(item, fallback_type)
            if node:
                result.append(node)
    return result


def build_source_fact_graph(source_analysis: dict[str, Any]) -> dict[str, Any]:
    if source_analysis.get("schema_version") != SOURCE_ANALYSIS_VERSION:
        raise ValueError(f"source-analysis schema_version must be {SOURCE_ANALYSIS_VERSION!r}")

    facts_by_id: dict[str, dict[str, Any]] = {}
    for key, fallback in [
        ("imports", "import"),
        ("classes", "class"),
        ("semantic_facts", "semantic_fact"),
        ("calls", "call"),
        ("assignments", "assignment"),
        ("branches", "branch"),
        ("returns", "return"),
        ("config_accesses", "config_access"),
        ("parallelism_facts", "parallelism_fact"),
    ]:
        for node in _records(source_analysis, key, fallback):
            facts_by_id.setdefault(node["id"], node)

    relations: list[dict[str, Any]] = []
    module_symbols = {
        "QKVParallelLinear": "concept-seed:tensor_parallel_attention_projection",
        "RowParallelLinear": "concept-seed:tensor_parallel_output_projection",
        "VocabParallelEmbedding": "concept-seed:tensor_parallel_embedding",
        "ParallelLMHead": "concept-seed:tensor_parallel_lm_head",
        "FusedMoE": "concept-seed:expert_parallel_moe",
        "Attention": "concept-seed:vllm_attention_backend",
        "AutoWeightsLoader": "concept-seed:auto_weights_loader_boundary",
    }

    for fact in facts_by_id.values():
        summary = fact.get("summary", "")
        fact_id = fact["id"]
        for symbol, target in module_symbols.items():
            if symbol in summary:
                _add_relation(relations, fact_id, target, "implements", [fact_id])
        if "make_layers" in summary or "get_pp_group" in summary or "PPMissingLayer" in summary:
            _add_relation(relations, fact_id, "concept-seed:pipeline_parallel_partitioning", "parallelizes", [fact_id])
        if "get_ep_group" in summary or "ep_rank" in summary or "ep_size" in summary:
            _add_relation(relations, fact_id, "concept-seed:expert_parallel_partitioning", "parallelizes", [fact_id])
        if "load_weights" in str(fact.get("method")) or "weight_loader" in summary or "mapping" in summary:
            _add_relation(relations, fact_id, "concept-seed:checkpoint_adaptation", "maps_to", [fact_id])
        if "forward" == fact.get("method"):
            _add_relation(relations, fact_id, "concept-seed:transformer_execution_flow", "calls", [fact_id])

    for item in source_analysis.get("module_assignments", []):
        if not isinstance(item, dict):
            continue
        fact_id = item.get("fact_id")
        owner = item.get("owner_class")
        target = item.get("target")
        implementation = item.get("call")
        if isinstance(fact_id, str) and isinstance(owner, str) and isinstance(target, str):
            facts_by_id.setdefault(
                fact_id,
                {
                    "id": fact_id,
                    "type": "module_instance",
                    "owner_class": owner,
                    "method": "__init__",
                    "line": item.get("line"),
                    "class": owner,
                    "implementation": implementation,
                    "summary": f"{owner} contains {target}",
                    "evidence": [fact_id],
                },
            )
            _add_relation(relations, f"class:{owner}", fact_id, "contains", [fact_id])
            if isinstance(implementation, str):
                _add_relation(relations, fact_id, f"symbol:{implementation}", "constructs", [fact_id])

    return {
        "schema_version": FACT_GRAPH_VERSION,
        "source_analysis_version": source_analysis.get("schema_version"),
        "model_name": source_analysis.get("model_name", "hy-v3"),
        "source_file": source_analysis.get("source_file"),
        "facts": sorted(facts_by_id.values(), key=lambda item: str(item["id"])),
        "relations": sorted(relations, key=lambda item: str(item["id"])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_analysis", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        graph = build_source_fact_graph(_load_json(args.source_analysis))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Source Fact Graph to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
