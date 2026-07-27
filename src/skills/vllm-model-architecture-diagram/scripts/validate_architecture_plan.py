"""Validate an Agent-authored lightweight Architecture Plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VALID_PATTERNS = {
    "pipeline",
    "block",
    "branch_merge",
    "routed_container",
    "mapping_flow",
    "strategy_matrix",
    "boundary_map",
    "multimodal_pipeline",
    "state_machine",
    "component_map",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_plan(
    plan: dict[str, Any],
    evidence: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if plan.get("schema_version") != "2.0":
        errors.append("plan.schema_version must be 2.0")
    for key in ("model_name", "target_file"):
        if not plan.get(key):
            errors.append(f"plan.{key} must be non-empty")
    files_read = plan.get("files_read")
    if not isinstance(files_read, list) or not files_read:
        errors.append("plan.files_read must list the files Codex read")
    pages = plan.get("pages")
    if not isinstance(pages, list):
        errors.append("plan.pages must be a list")
        pages = []
    if not (1 <= len(pages) <= 12):
        errors.append("plan must contain between 1 and 12 pages")

    claim_ids = {claim.get("id") for claim in evidence.get("claims", [])} if evidence else set()
    seen_pages: set[str] = set()
    category_candidates = set(context.get("classification", {}).get("category_candidates", [])) if context else set()
    capability_signals = context.get("capability_signals", {}) if context else {}
    for index, page in enumerate(pages):
        page_id = page.get("id")
        if not page_id:
            errors.append(f"page[{index}] is missing id")
            continue
        if page_id in seen_pages:
            errors.append(f"duplicate page id: {page_id}")
        seen_pages.add(page_id)
        for field in ("title", "question", "purpose"):
            if not str(page.get(field, "")).strip():
                errors.append(f"{page_id}: {field} must be non-empty")
        pattern = page.get("view_pattern")
        if pattern not in VALID_PATTERNS:
            errors.append(f"{page_id}: invalid view_pattern {pattern!r}")
        page_claims = page.get("claim_ids")
        if not isinstance(page_claims, list) or not page_claims:
            errors.append(f"{page_id}: claim_ids must be non-empty")
        elif evidence:
            for claim_id in page_claims:
                if claim_id not in claim_ids:
                    errors.append(f"{page_id}: claim_id does not exist in evidence: {claim_id}")
        budget = page.get("detail_budget", {})
        if budget:
            for budget_key in ("max_primary_nodes", "max_total_nodes", "max_visible_edges"):
                value = budget.get(budget_key)
                if not isinstance(value, int) or value < 1:
                    errors.append(f"{page_id}: detail_budget.{budget_key} must be a positive integer")
            if budget.get("max_total_nodes", 1) > 24:
                warnings.append(f"{page_id}: max_total_nodes is high for a readable page")
        title_blob = f"{page_id} {page.get('title', '')} {page.get('purpose', '')}".lower()
        if "moe" in title_blob and capability_signals.get("moe", {}).get("detected") is False:
            errors.append(f"{page_id}: MoE page conflicts with source context capabilities")
        if "multimodal" in title_blob and "multimodal" not in category_candidates and capability_signals.get("multimodal", {}).get("detected") is False:
            errors.append(f"{page_id}: multimodal page conflicts with source context capabilities")
        if not page.get("main_story") and pattern in {"pipeline", "block", "branch_merge", "mapping_flow"}:
            warnings.append(f"{page_id}: flow-oriented page should describe main_story")
    if len({page.get("purpose") for page in pages}) < len(pages):
        warnings.append("some page purposes are duplicated")
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--context", type=Path)
    args = parser.parse_args(argv)

    plan = _load_json(args.plan)
    evidence = _load_json(args.evidence) if args.evidence else None
    context = _load_json(args.context) if args.context else None
    errors, warnings = validate_plan(plan, evidence=evidence, context=context)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Architecture plan validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
