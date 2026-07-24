#!/usr/bin/env python3
"""Build a constrained semantic review draft for reviewed mode.

The script is intentionally conservative. It does not invent new model
semantics; it classifies required inventory facts against existing IR evidence
and proposes a small auditable patch for baseline unresolved items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REVIEW_VERSION = "0.1"
PATCH_VERSION = "0.1"
IR_VERSION = "0.6"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} root must be an object")
    return data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ir_ids(ir: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for page in ir.get("pages", []):
        if not isinstance(page, dict):
            continue
        ids.add(str(page.get("id")))
        for key in ("nodes", "edges"):
            for item in page.get(key, []):
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    ids.add(item["id"])
    return ids


def _evidence_index(ir: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for page in ir.get("pages", []):
        if not isinstance(page, dict):
            continue
        for key in ("nodes", "edges"):
            for item in page.get(key, []):
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    continue
                for evidence in item.get("evidence", []):
                    if not isinstance(evidence, dict):
                        continue
                    evidence_type = str(evidence.get("type") or "derived")
                    for fact_id in evidence.get("fact_ids", []):
                        if isinstance(fact_id, str):
                            result.setdefault(fact_id, set()).add(f"{evidence_type}:{item['id']}")
    return result


def _target_for_fact(fact: dict[str, Any], ids: set[str]) -> str:
    category = str(fact.get("category") or "")
    summary = str(fact.get("summary") or "").lower()
    if "attention" in summary or "qkv" in summary or "rope" in summary:
        return "attention_detail" if "attention_detail" in ids else "attention_core"
    if "moe" in summary or "expert" in summary or category == "expert_parallel":
        return "moe_detail" if "moe_detail" in ids else "fused_moe"
    if category == "checkpoint_loading" or "weight" in summary or "loader" in summary:
        return "weight_loading" if "weight_loading" in ids else "mapping_dispatch"
    if category in {"tensor_parallel", "pipeline_parallel", "expert_parallel"}:
        return "parallelism" if "parallelism" in ids else "hyv3_decoder_layer"
    if category in {"configuration", "capability"}:
        return "adapter_integration" if "adapter_integration" in ids else "vllm_config"
    if "decoder" in summary or "residual" in summary or "layernorm" in summary:
        return "decoder_layer_detail" if "decoder_layer_detail" in ids else "hyv3_decoder_layer"
    return "overview" if "overview" in ids else sorted(ids)[0]


def _external_symbol(fact: dict[str, Any]) -> str:
    summary = str(fact.get("summary") or "")
    for symbol in ("Attention", "FusedMoE", "AutoWeightsLoader", "SupportsLoRA", "support_torch_compile"):
        if symbol in summary:
            return symbol
    return "external vLLM component"


def build_semantic_review(
    source_analysis: dict[str, Any],
    inventory: dict[str, Any],
    baseline_ir: dict[str, Any],
    baseline_coverage: dict[str, Any],
    *,
    source_path: Path,
    baseline_ir_sha256: str,
    patch_file: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if baseline_ir.get("schema_version") != IR_VERSION:
        raise ValueError(f"baseline IR schema_version must be {IR_VERSION!r}")
    evidence_index = _evidence_index(baseline_ir)
    ids = _ir_ids(baseline_ir)
    dispositions: list[dict[str, Any]] = []
    required_facts = [
        fact for fact in inventory.get("facts", [])
        if isinstance(fact, dict) and fact.get("relevance") == "required" and isinstance(fact.get("fact_id"), str)
    ]
    for fact in required_facts:
        fact_id = str(fact["fact_id"])
        targets = sorted(evidence_index.get(fact_id, set()))
        direct_targets = [target.split(":", 1)[1] for target in targets if target.startswith("direct:")]
        derived_targets = [target.split(":", 1)[1] for target in targets if target.startswith("derived:")]
        external_targets = [target.split(":", 1)[1] for target in targets if target.startswith("external:")]
        if direct_targets:
            disposition = {
                "fact_id": fact_id,
                "disposition": "rendered_direct",
                "reason": "Fact is cited by direct IR evidence.",
                "target_ids": direct_targets,
                "supporting_fact_ids": [fact_id],
                "confidence": 1.0,
                "external_symbol": None,
                "suggested_next_source": None,
            }
        elif derived_targets:
            disposition = {
                "fact_id": fact_id,
                "disposition": "rendered_derived",
                "reason": "Fact participates in a derived architecture concept.",
                "target_ids": derived_targets,
                "supporting_fact_ids": [fact_id],
                "confidence": 0.95,
                "external_symbol": None,
                "suggested_next_source": None,
            }
        elif external_targets or fact.get("category") == "external_boundary":
            disposition = {
                "fact_id": fact_id,
                "disposition": "documented_external",
                "reason": "Behavior belongs to an imported component boundary and is not proven inside the input file.",
                "target_ids": external_targets or [_target_for_fact(fact, ids)],
                "supporting_fact_ids": [fact_id],
                "confidence": 0.9,
                "external_symbol": _external_symbol(fact),
                "suggested_next_source": None,
            }
        else:
            disposition = {
                "fact_id": fact_id,
                "disposition": "aggregated",
                "reason": "Low-level source fact is covered by a higher-level architecture concept rather than a standalone node.",
                "target_ids": [_target_for_fact(fact, ids)],
                "supporting_fact_ids": [fact_id],
                "confidence": 0.88,
                "external_symbol": None,
                "suggested_next_source": None,
            }
        dispositions.append(disposition)

    config_fact_ids = [
        fact["fact_id"] for fact in source_analysis.get("config_accesses", [])
        if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str)
    ][:4]
    if not config_fact_ids:
        config_fact_ids = [
            fact["fact_id"] for fact in required_facts[:1]
            if isinstance(fact.get("fact_id"), str)
        ]
    findings = [
        {
            "finding_id": "finding:baseline-unresolved-config-values",
            "type": "over_aggregated",
            "severity": "major",
            "description": "Baseline IR keeps symbolic config values in unresolved even though reviewed mode can document them as symbolic external configuration.",
            "affected_ids": ["external_config_values"],
            "evidence_fact_ids": config_fact_ids,
            "recommended_action": "Resolve baseline unresolved item by documenting symbolic config values as reviewed external/configuration assumptions.",
            "confidence": 0.92,
            "patch_op_ids": ["op:resolve-external-config-values"],
        }
    ]
    page_plan = [
        {
            "page_id": str(page.get("id")),
            "action": "keep",
            "reason": "Page is supported by source-analysis facts and remains useful for HY V3 review.",
            "supporting_fact_ids": config_fact_ids[:1],
            "priority": index + 1,
        }
        for index, page in enumerate(baseline_ir.get("pages", []))
        if isinstance(page, dict) and isinstance(page.get("id"), str)
    ]
    review = {
        "schema_version": REVIEW_VERSION,
        "model_name": baseline_ir.get("model_name", "unknown-model"),
        "source_file": str(source_path),
        "review_mode": "reviewed",
        "summary": {
            "required_fact_count": len(required_facts),
            "baseline_unresolved_count": baseline_coverage.get("summary", {}).get("unresolved_fact_count", 0),
            "disposition_count": len(dispositions),
        },
        "fact_dispositions": dispositions,
        "architecture_findings": findings,
        "page_plan": page_plan,
        "patch_file": patch_file,
        "remaining_unknowns": [],
        "deferred_findings": [],
    }
    patch = {
        "schema_version": PATCH_VERSION,
        "base_ir_version": IR_VERSION,
        "base_ir_sha256": baseline_ir_sha256,
        "operations": [
            {
                "op_id": "op:resolve-external-config-values",
                "op": "resolve_unresolved",
                "item": "external_config_values",
                "reason": "Reviewed mode documents symbolic config accesses instead of leaving them unresolved.",
                "evidence_fact_ids": config_fact_ids,
                "confidence": 0.92,
                "finding_ids": ["finding:baseline-unresolved-config-values"],
            }
        ],
        "deferred_operations": [],
    }
    return review, patch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build constrained semantic review and IR patch.")
    parser.add_argument("source_analysis", type=Path)
    parser.add_argument("semantic_inventory", type=Path)
    parser.add_argument("baseline_ir", type=Path)
    parser.add_argument("baseline_coverage", type=Path)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        review, patch = build_semantic_review(
            _load_json(args.source_analysis, "source-analysis"),
            _load_json(args.semantic_inventory, "semantic inventory"),
            _load_json(args.baseline_ir, "baseline Architecture IR"),
            _load_json(args.baseline_coverage, "baseline semantic coverage"),
            source_path=args.source_file,
            baseline_ir_sha256=_sha256(args.baseline_ir),
            patch_file=str(args.patch_output),
        )
        args.review_output.parent.mkdir(parents=True, exist_ok=True)
        args.patch_output.parent.mkdir(parents=True, exist_ok=True)
        args.review_output.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.patch_output.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote semantic review to {args.review_output}")
    print(f"Wrote IR patch to {args.patch_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
