#!/usr/bin/env python3
"""Validate agent-guided semantic review artifacts before applying IR patches."""

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
DISPOSITIONS = {
    "rendered_direct",
    "rendered_derived",
    "aggregated",
    "documented_external",
    "excluded",
    "unresolved",
}
ALLOWED_PATCH_OPS = {
    "add_node",
    "remove_node",
    "update_node",
    "add_edge",
    "remove_edge",
    "update_edge",
    "add_evidence",
    "replace_evidence",
    "set_evidence_type",
    "add_port",
    "update_port",
    "remove_port",
    "add_annotation",
    "update_annotation",
    "set_unresolved",
    "resolve_unresolved",
    "add_page",
    "remove_page",
    "rename_page",
    "move_item_to_page",
    "merge_nodes",
    "split_node",
}


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


def _source_facts(source_analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    for key, value in source_analysis.items():
        if key == "weight_loading_flows" or not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("fact_id"), str):
                facts[item["fact_id"]] = item
    for flow in source_analysis.get("weight_loading_flows", []):
        if not isinstance(flow, dict):
            continue
        for stage in flow.get("stages", []):
            if isinstance(stage, dict) and isinstance(stage.get("fact_id"), str):
                facts[stage["fact_id"]] = stage
    return facts


def _required_fact_ids(inventory: dict[str, Any]) -> set[str]:
    return {
        fact["fact_id"]
        for fact in inventory.get("facts", [])
        if isinstance(fact, dict)
        and fact.get("relevance") == "required"
        and isinstance(fact.get("fact_id"), str)
    }


def _ir_ids(ir: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for page in ir.get("pages", []):
        if not isinstance(page, dict):
            continue
        if isinstance(page.get("id"), str):
            ids.add(page["id"])
        for key in ("nodes", "edges"):
            for item in page.get(key, []):
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    ids.add(item["id"])
    return ids


def _patch_added_ids(patch: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for operation in patch.get("operations", []):
        if not isinstance(operation, dict):
            continue
        op = operation.get("op")
        if op == "add_page" and isinstance(operation.get("page"), dict):
            page_id = operation["page"].get("id")
            if isinstance(page_id, str):
                ids.add(page_id)
        if op == "add_node" and isinstance(operation.get("node"), dict):
            node_id = operation["node"].get("id")
            if isinstance(node_id, str):
                ids.add(node_id)
        if op == "add_edge" and isinstance(operation.get("edge"), dict):
            edge_id = operation["edge"].get("id")
            if isinstance(edge_id, str):
                ids.add(edge_id)
    return ids


def _is_import_only(fact_ids: list[str], source_facts: dict[str, dict[str, Any]]) -> bool:
    if not fact_ids:
        return False
    kinds = {source_facts.get(fact_id, {}).get("type") for fact_id in fact_ids}
    return bool(kinds) and kinds <= {"import"}


def _finding_deferred(finding_id: str, review: dict[str, Any], patch: dict[str, Any]) -> bool:
    deferred = review.get("deferred_findings", [])
    for item in deferred:
        if item == finding_id:
            return True
        if isinstance(item, dict) and item.get("finding_id") == finding_id and item.get("reason"):
            return True
    for operation in patch.get("deferred_operations", []):
        if isinstance(operation, dict) and finding_id in operation.get("finding_ids", []) and operation.get("reason"):
            return True
    return False


def validate_semantic_review(
    source_analysis: dict[str, Any],
    inventory: dict[str, Any],
    baseline_ir: dict[str, Any],
    baseline_ir_path: Path,
    review: dict[str, Any],
    patch: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if review.get("schema_version") != REVIEW_VERSION:
        errors.append("semantic review schema_version must be '0.1'")
    if patch.get("schema_version") != PATCH_VERSION:
        errors.append("IR patch schema_version must be '0.1'")
    if patch.get("base_ir_version") != IR_VERSION or baseline_ir.get("schema_version") != IR_VERSION:
        errors.append("IR patch and baseline IR must use Architecture IR 0.6")
    expected_hash = _sha256(baseline_ir_path)
    if patch.get("base_ir_sha256") != expected_hash:
        errors.append("IR patch base_ir_sha256 does not match baseline IR")

    source_facts = _source_facts(source_analysis)
    required_ids = _required_fact_ids(inventory)
    known_fact_ids = set(source_facts) | required_ids
    ir_ids = _ir_ids(baseline_ir) | _patch_added_ids(patch)

    seen_dispositions: set[str] = set()
    excluded_count = 0
    for disposition in review.get("fact_dispositions", []):
        if not isinstance(disposition, dict):
            errors.append("fact disposition must be an object")
            continue
        fact_id = disposition.get("fact_id")
        if not isinstance(fact_id, str):
            errors.append("fact disposition missing fact_id")
            continue
        if fact_id in seen_dispositions:
            errors.append(f"duplicate disposition for required fact: {fact_id}")
        seen_dispositions.add(fact_id)
        if fact_id not in required_ids:
            errors.append(f"disposition references non-required or unknown fact: {fact_id}")
        kind = disposition.get("disposition") or disposition.get("status")
        if kind not in DISPOSITIONS:
            errors.append(f"invalid disposition for {fact_id}: {kind}")
        confidence = disposition.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            errors.append(f"invalid disposition confidence for {fact_id}")
        for support_id in disposition.get("supporting_fact_ids", []):
            if support_id not in known_fact_ids:
                errors.append(f"unknown supporting_fact_id for {fact_id}: {support_id}")
        if kind == "aggregated":
            if not disposition.get("target_ids"):
                errors.append(f"aggregated disposition must name target_ids: {fact_id}")
            for target_id in disposition.get("target_ids", []):
                if target_id not in ir_ids:
                    errors.append(f"aggregated target does not exist for {fact_id}: {target_id}")
        if kind == "documented_external" and not disposition.get("external_symbol"):
            errors.append(f"documented_external disposition must name external_symbol: {fact_id}")
        if kind == "excluded":
            excluded_count += 1
            if not str(disposition.get("reason") or "").strip():
                errors.append(f"excluded disposition must explain reason: {fact_id}")
        if kind == "unresolved" and not (
            str(disposition.get("reason") or "").strip() or str(disposition.get("suggested_next_source") or "").strip()
        ):
            errors.append(f"unresolved disposition must explain missing evidence: {fact_id}")

    missing = required_ids - seen_dispositions
    for fact_id in sorted(missing):
        errors.append(f"missing disposition for required fact: {fact_id}")
    if required_ids and excluded_count / len(required_ids) > 0.55:
        errors.append("too many required facts were excluded; review appears to be bypassing coverage")

    patched_findings: set[str] = set()
    for operation in patch.get("operations", []):
        if not isinstance(operation, dict):
            errors.append("patch operation must be an object")
            continue
        op_id = operation.get("op_id")
        op = operation.get("op")
        if not isinstance(op_id, str) or not op_id:
            errors.append("patch operation missing op_id")
        if op not in ALLOWED_PATCH_OPS:
            errors.append(f"invalid patch operation: {op}")
        confidence = operation.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            errors.append(f"invalid patch confidence for {op_id}")
        elif float(confidence) < 0.70:
            errors.append(f"low-confidence patch operation must be deferred, not auto-applied: {op_id}")
        fact_ids = [fact_id for fact_id in operation.get("evidence_fact_ids", []) if isinstance(fact_id, str)]
        if op not in {"resolve_unresolved", "remove_page", "remove_node", "remove_edge"} and not fact_ids:
            errors.append(f"semantic patch operation requires evidence_fact_ids: {op_id}")
        for fact_id in fact_ids:
            if fact_id not in known_fact_ids:
                errors.append(f"patch operation references unknown fact_id: {op_id}: {fact_id}")
        if op in {"add_edge", "update_edge", "set_evidence_type", "replace_evidence"} and _is_import_only(fact_ids, source_facts):
            errors.append(f"behavior patch cannot rely only on import evidence: {op_id}")
        if op == "set_evidence_type" and operation.get("evidence_type") == "direct":
            for fact_id in fact_ids:
                if source_facts.get(fact_id, {}).get("category") == "external_boundary":
                    errors.append(f"external behavior cannot be promoted to direct evidence: {op_id}")
        for finding_id in operation.get("finding_ids", []):
            if isinstance(finding_id, str):
                patched_findings.add(finding_id)

    baseline_node_count = sum(
        len(page.get("nodes", [])) for page in baseline_ir.get("pages", []) if isinstance(page, dict)
    )
    removed_node_count = sum(
        1 for operation in patch.get("operations", [])
        if isinstance(operation, dict) and operation.get("op") == "remove_node"
    )
    if baseline_node_count and removed_node_count / baseline_node_count > 0.35:
        errors.append("patch deletes more than 35% of baseline semantic nodes")

    review_findings = review.get("architecture_findings")
    if not isinstance(review_findings, list):
        review_findings = review.get("findings", [])
    for finding in review_findings:
        if not isinstance(finding, dict):
            continue
        finding_id = finding.get("finding_id") or finding.get("id")
        severity = finding.get("severity")
        if severity in {"critical", "major"} and isinstance(finding_id, str):
            if finding_id not in patched_findings and not _finding_deferred(finding_id, review, patch):
                errors.append(f"critical/major finding is neither patched nor deferred: {finding_id}")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate semantic review and Architecture IR patch.")
    parser.add_argument("source_analysis", type=Path)
    parser.add_argument("semantic_inventory", type=Path)
    parser.add_argument("baseline_ir", type=Path)
    parser.add_argument("semantic_review", type=Path)
    parser.add_argument("architecture_ir_patch", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        errors = validate_semantic_review(
            _load_json(args.source_analysis, "source-analysis"),
            _load_json(args.semantic_inventory, "semantic inventory"),
            _load_json(args.baseline_ir, "baseline Architecture IR"),
            args.baseline_ir,
            _load_json(args.semantic_review, "semantic review"),
            _load_json(args.architecture_ir_patch, "Architecture IR patch"),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Semantic review validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
