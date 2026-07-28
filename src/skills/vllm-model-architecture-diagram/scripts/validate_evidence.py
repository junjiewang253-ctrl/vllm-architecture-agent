"""Validate v2.1 evidence claims for complete Architecture Plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vllm_architecture_agent.paths import path_is_within, resolve_repo_path  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_slice(path: Path, start: int, end: int) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if start < 1 or end < start or end > len(lines):
        raise ValueError(f"invalid line range {start}-{end} for {path}")
    return lines[start - 1 : end]


def _direct_is_import_only(entries: list[dict[str, Any]], repo_root: Path) -> bool:
    non_import_seen = False
    for entry in entries:
        path = resolve_repo_path(repo_root, entry["file"])
        for line in _line_slice(path, int(entry["start_line"]), int(entry["end_line"])):
            stripped = line.strip()
            if stripped and not stripped.startswith(("import ", "from ")):
                non_import_seen = True
    return not non_import_seen


def _source_item_ids(context: dict[str, Any]) -> set[str]:
    known: set[str] = set()
    for key, id_key in (
        ("classes", "class_id"),
        ("methods", "method_id"),
        ("module_functions", "function_id"),
        ("branches", "branch_id"),
        ("loops", "loop_id"),
        ("weight_mappings", "mapping_id"),
    ):
        known.update(item[id_key] for item in context.get(key, []) if item.get(id_key))
    for name, signal in context.get("capability_signals", {}).items():
        known.add(signal.get("capability_id") or f"capability:{name}")
    return known


def _review_claim_ids(plan: dict[str, Any]) -> set[str]:
    claim_ids: set[str] = set()
    for collection in ("class_review", "method_review", "function_review"):
        for item in plan.get(collection, []):
            claim_ids.update(item.get("claim_ids", []))
    manifest = plan.get("coverage_manifest", {})
    for collection in ("branches", "weight_mappings", "capabilities", "external_boundaries"):
        for item in manifest.get(collection, []):
            claim_ids.update(item.get("claim_ids", []))
    for page in plan.get("pages", []):
        claim_ids.update(page.get("claim_ids", []))
        for region in page.get("detail_regions", []):
            claim_ids.update(region.get("claim_ids", []))
    return claim_ids


def validate_evidence(
    evidence: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    warnings: list[str] = []
    repo_root_value = context.get("target", {}).get("repo_root") if context else None
    if repo_root is None:
        repo_root = Path(repo_root_value).resolve() if repo_root_value and repo_root_value != "." else Path.cwd().resolve()
    else:
        repo_root = repo_root.resolve()

    if evidence.get("schema_version") != "2.1":
        errors.append("evidence.schema_version must be 2.1")
    if not evidence.get("source_sha256"):
        errors.append("evidence.source_sha256 is required")
    if context and evidence.get("source_sha256") != context.get("source_sha256"):
        errors.append("evidence.source_sha256 must match context.source_sha256")
    if context and evidence.get("target_file") != context.get("target", {}).get("target_file"):
        errors.append("evidence.target_file must match context target file")
    if plan:
        if evidence.get("source_sha256") != plan.get("source_sha256"):
            errors.append("evidence.source_sha256 must match plan.source_sha256")
        if evidence.get("target_file") != plan.get("target_file"):
            errors.append("evidence.target_file must match plan.target_file")

    claims = evidence.get("claims")
    if not isinstance(claims, list):
        errors.append("evidence.claims must be a list")
        claims = []

    plan_page_ids = {page.get("id") for page in plan.get("pages", [])} if plan else set()
    files_read = {str(resolve_repo_path(repo_root, path)) for path in plan.get("files_read", [])} if plan else set()
    known_source_items = _source_item_ids(context or {})
    claim_ids: set[str] = set()
    used_by_page_or_review = _review_claim_ids(plan or {})
    summary = {"direct": 0, "derived": 0, "external": 0}

    for index, claim in enumerate(claims):
        claim_id = claim.get("id")
        if not claim_id:
            errors.append(f"claim[{index}] is missing id")
            continue
        if claim_id in claim_ids:
            errors.append(f"duplicate claim id: {claim_id}")
        claim_ids.add(claim_id)
        confidence = claim.get("confidence")
        if confidence not in {"direct", "derived", "external"}:
            errors.append(f"{claim_id}: confidence must be direct, derived, or external")
            continue
        summary[confidence] += 1
        if claim.get("phase") not in {"construction", "runtime", "loading", "parallel", "capability", "external"}:
            errors.append(f"{claim_id}: invalid or missing phase")
        source_item_ids = claim.get("source_item_ids", [])
        if not isinstance(source_item_ids, list) or not source_item_ids:
            errors.append(f"{claim_id}: source_item_ids must be non-empty")
        elif context:
            for item_id in source_item_ids:
                if item_id not in known_source_items:
                    errors.append(f"{claim_id}: unknown source_item_id {item_id}")
        entries = claim.get("evidence")
        if not isinstance(entries, list) or not entries:
            errors.append(f"{claim_id}: evidence must be a non-empty list")
            continue
        if confidence == "derived":
            if len(entries) < 2:
                errors.append(f"{claim_id}: derived claims require at least two evidence entries")
            if not str(claim.get("derivation", "")).strip():
                errors.append(f"{claim_id}: derived claims require derivation")
        if confidence == "external" and not str(claim.get("external_boundary", "")).strip():
            errors.append(f"{claim_id}: external claims require external_boundary")
        for page_id in claim.get("used_in_pages", []):
            if plan and page_id not in plan_page_ids:
                errors.append(f"{claim_id}: used_in_pages references unknown page {page_id}")
        for entry_index, entry in enumerate(entries):
            file_value = entry.get("file")
            if not file_value:
                errors.append(f"{claim_id}.evidence[{entry_index}] is missing file")
                continue
            path = resolve_repo_path(repo_root, file_value)
            if not path.exists():
                errors.append(f"{claim_id}.evidence[{entry_index}] file does not exist: {path}")
                continue
            if repo_root and not path_is_within(path, repo_root):
                errors.append(f"{claim_id}.evidence[{entry_index}] file is outside repo root: {path}")
            if plan and str(path.resolve()) not in files_read:
                errors.append(f"{claim_id}.evidence[{entry_index}] file is not listed in plan.files_read: {file_value}")
            try:
                start = int(entry.get("start_line"))
                end = int(entry.get("end_line"))
                source_lines = _line_slice(path, start, end)
            except (TypeError, ValueError) as exc:
                errors.append(f"{claim_id}.evidence[{entry_index}] {exc}")
                continue
            symbol = str(entry.get("symbol", "")).strip()
            if symbol and not any(symbol in line for line in source_lines):
                warnings.append(f"{claim_id}.evidence[{entry_index}] symbol not found in cited lines: {symbol}")
        if confidence == "direct":
            try:
                if _direct_is_import_only(entries, repo_root):
                    errors.append(f"{claim_id}: direct claim is supported only by import lines")
            except (OSError, ValueError) as exc:
                errors.append(f"{claim_id}: unable to check direct import-only evidence: {exc}")
        if plan and claim_id not in used_by_page_or_review:
            warnings.append(f"{claim_id}: claim is not referenced by any page, region, or review")

    if plan:
        missing = sorted(used_by_page_or_review - claim_ids)
        for claim_id in missing:
            errors.append(f"plan references missing claim: {claim_id}")
        external_only_pages: list[str] = []
        for page in plan.get("pages", []):
            page_claims = set(page.get("claim_ids", []))
            confidences = {
                claim.get("confidence")
                for claim in claims
                if claim.get("id") in page_claims
            }
            if confidences == {"external"}:
                external_only_pages.append(page.get("id", "<unknown>"))
        for page_id in external_only_pages:
            warnings.append(f"{page_id}: page has only external claims")
        claim_by_id = {claim.get("id"): claim for claim in claims}
        for page in plan.get("pages", []):
            for region in page.get("detail_regions", []):
                region_claims = [claim_by_id.get(claim_id) for claim_id in region.get("claim_ids", [])]
                if region_claims and not any(claim and claim.get("confidence") in {"direct", "derived"} for claim in region_claims):
                    warnings.append(f"{page.get('id')}.{region.get('id')}: region has no direct/derived claim")
    return errors, warnings, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)

    evidence = _load_json(args.evidence)
    context = _load_json(args.context) if args.context else None
    plan = _load_json(args.plan) if args.plan else None
    errors, warnings, summary = validate_evidence(evidence, context=context, plan=plan, repo_root=args.repo_root)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Evidence validation passed")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
