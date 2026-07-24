#!/usr/bin/env python3
"""Validate semantic coverage from source-analysis through Architecture IR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

COVERAGE_VERSION = "0.2"
STATUSES = (
    "rendered_direct",
    "rendered_derived",
    "aggregated",
    "documented_external",
    "excluded",
    "unresolved",
    "orphaned",
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} root must be an object")
    return data


def _evidence_fact_ids_by_type(ir: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for page in ir.get("pages", []):
        if not isinstance(page, dict):
            continue
        for collection in ("nodes", "edges"):
            for item in page.get(collection, []):
                if not isinstance(item, dict):
                    continue
                for evidence in item.get("evidence", []):
                    if not isinstance(evidence, dict):
                        continue
                    evidence_type = str(evidence.get("type") or "derived")
                    for fact_id in evidence.get("fact_ids", []):
                        if isinstance(fact_id, str):
                            result.setdefault(fact_id, set()).add(evidence_type)
    return result


def _fact_by_id(source_analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in (
        "semantic_facts",
        "imports",
        "classes",
        "class_attributes",
        "module_assignments",
        "layer_factories",
        "methods",
        "method_control_flows",
        "calls",
        "assignments",
        "branches",
        "returns",
        "config_accesses",
        "parallelism_hints",
        "parallelism_facts",
        "weight_mappings",
        "weight_loading_hints",
    ):
        for fact in source_analysis.get(key, []):
            if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str):
                result[fact["fact_id"]] = fact
    for flow in source_analysis.get("weight_loading_flows", []):
        if not isinstance(flow, dict):
            continue
        for stage in flow.get("stages", []):
            if isinstance(stage, dict) and isinstance(stage.get("fact_id"), str):
                result[stage["fact_id"]] = stage
    return result


def _status_for_fact(fact: dict[str, Any], consumed: dict[str, set[str]]) -> tuple[str, str | None]:
    fact_id = str(fact["fact_id"])
    if fact_id in consumed:
        types = consumed[fact_id]
        if "direct" in types:
            return "rendered_direct", None
        if "external" in types:
            return "documented_external", "External component boundary is documented by IR evidence."
        return "rendered_derived", None
    relevance = fact.get("relevance")
    category = fact.get("category")
    if relevance == "excluded":
        return "excluded", str(fact.get("exclusion_reason") or "Excluded by inventory.")
    if relevance == "informational":
        return "excluded", str(fact.get("exclusion_reason") or "Informational fact outside required diagram coverage.")
    if category == "external_boundary":
        return "unresolved", "External component behavior is outside single-file proof boundary."
    return "unresolved", "Required semantic fact is tracked but not rendered as a standalone IR node or edge."


def _review_dispositions(review: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not review:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for disposition in review.get("fact_dispositions", []):
        if isinstance(disposition, dict) and isinstance(disposition.get("fact_id"), str):
            result[disposition["fact_id"]] = disposition
    return result


def _matrix() -> dict[str, int]:
    return {status: 0 for status in STATUSES}


def _entry_from_review(fact: dict[str, Any], disposition: dict[str, Any]) -> dict[str, Any]:
    status = str(disposition.get("disposition"))
    entry = {
        "fact_id": fact["fact_id"],
        "status": status,
        "category": fact.get("category"),
        "relevance": fact.get("relevance"),
        "reason": disposition.get("reason"),
        "target_ids": disposition.get("target_ids", []),
        "supporting_fact_ids": disposition.get("supporting_fact_ids", []),
        "confidence": disposition.get("confidence"),
    }
    if status == "aggregated":
        entry["aggregate_target_id"] = (disposition.get("target_ids") or [None])[0]
    elif status == "documented_external":
        entry["external_symbol"] = disposition.get("external_symbol")
        entry["boundary_reason"] = disposition.get("reason")
        entry["local_evidence_fact_ids"] = disposition.get("supporting_fact_ids", [])
        entry["allowed_claim"] = "Local file proves the adapter boundary, not the imported component internals."
    elif status == "excluded":
        entry["exclusion_reason"] = disposition.get("reason")
        entry["excluded_from_pages"] = disposition.get("excluded_from_pages", [])
    elif status == "unresolved":
        entry["unresolved_reason"] = disposition.get("reason")
        entry["attempted_evidence_fact_ids"] = disposition.get("supporting_fact_ids", [])
        entry["suggested_next_source"] = disposition.get("suggested_next_source")
    return entry


def validate_semantic_coverage(
    source_analysis: dict[str, Any],
    inventory: dict[str, Any],
    ir: dict[str, Any],
    semantic_review: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if source_analysis.get("schema_version") != "0.3":
        errors.append("source-analysis schema_version must be '0.3'")
    if ir.get("schema_version") != "0.6":
        errors.append("Architecture IR schema_version must be '0.6'")

    source_facts = _fact_by_id(source_analysis)
    consumed_ids = _evidence_fact_ids_by_type(ir)
    dispositions = _review_dispositions(semantic_review)
    facts = [fact for fact in inventory.get("facts", []) if isinstance(fact, dict)]
    statuses: list[dict[str, Any]] = []
    relevance_matrix: dict[str, dict[str, int]] = {
        "required": _matrix(),
        "optional": _matrix(),
        "informational": _matrix(),
        "excluded": _matrix(),
    }

    for fact in facts:
        fact_id = fact.get("fact_id")
        if not isinstance(fact_id, str):
            continue
        if fact_id not in source_facts and not fact_id.startswith("stage:"):
            errors.append(f"inventory fact is not present in source-analysis: {fact_id}")
        if semantic_review and fact.get("relevance") == "required":
            disposition = dispositions.get(fact_id)
            if disposition is None:
                status = "orphaned"
                entry = {
                    "fact_id": fact_id,
                    "status": status,
                    "category": fact.get("category"),
                    "relevance": fact.get("relevance"),
                    "orphaned_reason": "Required fact has no semantic review disposition.",
                }
            else:
                entry = _entry_from_review(fact, disposition)
                status = str(entry["status"])
        else:
            status, reason = _status_for_fact(fact, consumed_ids)
            entry = {"fact_id": fact_id, "status": status, "category": fact.get("category"), "relevance": fact.get("relevance")}
            if reason:
                if status == "excluded":
                    entry["exclusion_reason"] = reason
                elif status == "documented_external":
                    entry["boundary_reason"] = reason
                else:
                    entry["unresolved_reason"] = reason
        relevance = str(fact.get("relevance") or "informational")
        relevance_matrix.setdefault(relevance, _matrix())
        relevance_matrix[relevance][status] = relevance_matrix[relevance].get(status, 0) + 1
        statuses.append(entry)

    for fact_id in consumed_ids:
        if fact_id not in source_facts and not fact_id.startswith("stage:"):
            errors.append(f"IR evidence references unknown fact_id: {fact_id}")

    totals = _matrix()
    for counts in relevance_matrix.values():
        for status in STATUSES:
            totals[status] += counts.get(status, 0)
    required_count = sum(relevance_matrix.get("required", {}).values())
    consumed_count = totals["rendered_direct"] + totals["rendered_derived"]
    excluded_count = totals["excluded"]
    unresolved_count = totals["unresolved"]
    orphaned_count = totals["orphaned"]
    summary = {
        "required": relevance_matrix.get("required", _matrix()),
        "optional": relevance_matrix.get("optional", _matrix()),
        "informational": relevance_matrix.get("informational", _matrix()),
        "excluded": relevance_matrix.get("excluded", _matrix()),
        "totals": totals,
        "rendered_direct_count": totals["rendered_direct"],
        "rendered_derived_count": totals["rendered_derived"],
        "aggregated_count": totals["aggregated"],
        "documented_external_count": totals["documented_external"],
        "required_fact_count": required_count,
        "consumed_fact_count": consumed_count,
        "excluded_fact_count": excluded_count,
        "unresolved_fact_count": unresolved_count,
        "orphaned_fact_count": orphaned_count,
    }
    if semantic_review:
        if relevance_matrix.get("required", {}).get("unresolved", 0):
            errors.append(f"required unresolved facts found: {relevance_matrix['required']['unresolved']}")
        if relevance_matrix.get("required", {}).get("orphaned", 0):
            errors.append(f"orphaned required facts found: {relevance_matrix['required']['orphaned']}")
    return {
        "schema_version": COVERAGE_VERSION,
        "model_name": ir.get("model_name", inventory.get("model_name", "unknown-model")),
        "summary": summary,
        "facts": statuses,
    }, errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate semantic coverage.")
    parser.add_argument("source_analysis", type=Path, help="source-analysis JSON")
    parser.add_argument("semantic_inventory", type=Path, help="semantic inventory JSON")
    parser.add_argument("architecture_ir", type=Path, help="Architecture IR JSON")
    parser.add_argument("--semantic-review", type=Path, help="Semantic review JSON for coverage 0.2 dispositions")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output semantic coverage JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        coverage, errors = validate_semantic_coverage(
            _load_json(args.source_analysis, "source-analysis"),
            _load_json(args.semantic_inventory, "semantic inventory"),
            _load_json(args.architecture_ir, "Architecture IR"),
            _load_json(args.semantic_review, "semantic review") if args.semantic_review else None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Semantic coverage validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
