"""Validate source evidence claims for Agent-authored architecture plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _line_slice(path: Path, start: int, end: int) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if start < 1 or end < start or end > len(lines):
        raise ValueError(f"invalid line range {start}-{end} for {path}")
    return lines[start - 1 : end]


def _direct_is_import_only(entries: list[dict[str, Any]]) -> bool:
    non_import_seen = False
    for entry in entries:
        path = Path(entry["file"])
        start = int(entry["start_line"])
        end = int(entry["end_line"])
        for line in _line_slice(path, start, end):
            stripped = line.strip()
            if stripped and not stripped.startswith(("import ", "from ")):
                non_import_seen = True
    return not non_import_seen


def validate_evidence(
    evidence: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    warnings: list[str] = []
    repo_root_value = None
    if context:
        repo_root_value = context.get("target", {}).get("repo_root")
    repo_root = Path(repo_root_value).resolve() if repo_root_value else None

    if evidence.get("schema_version") != "2.0":
        errors.append("evidence.schema_version must be 2.0")
    claims = evidence.get("claims")
    if not isinstance(claims, list):
        errors.append("evidence.claims must be a list")
        claims = []

    claim_ids: set[str] = set()
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
        entries = claim.get("evidence")
        if not isinstance(entries, list) or not entries:
            errors.append(f"{claim_id}: evidence must be a non-empty list")
            continue
        if confidence == "derived" and len(entries) < 2:
            errors.append(f"{claim_id}: derived claims require at least two evidence entries")
        if confidence == "external":
            boundary = claim.get("external_boundary") or " ".join(
                str(entry.get("reason", "")) for entry in entries
            )
            if "external" not in boundary.lower() and "boundary" not in boundary.lower():
                errors.append(f"{claim_id}: external claims must explain the external boundary")
        for entry_index, entry in enumerate(entries):
            file_value = entry.get("file")
            if not file_value:
                errors.append(f"{claim_id}.evidence[{entry_index}] is missing file")
                continue
            path = Path(file_value)
            if not path.exists():
                errors.append(f"{claim_id}.evidence[{entry_index}] file does not exist: {path}")
                continue
            if repo_root and not _is_within(path, repo_root):
                errors.append(f"{claim_id}.evidence[{entry_index}] file is outside repo root: {path}")
            try:
                source_lines = _line_slice(path, int(entry.get("start_line")), int(entry.get("end_line")))
            except (TypeError, ValueError) as exc:
                errors.append(f"{claim_id}.evidence[{entry_index}] {exc}")
                continue
            symbol = str(entry.get("symbol", "")).strip()
            if symbol and not any(symbol in line for line in source_lines):
                warnings.append(f"{claim_id}.evidence[{entry_index}] symbol not found in cited lines: {symbol}")
        if confidence == "direct":
            try:
                if _direct_is_import_only(entries):
                    errors.append(f"{claim_id}: direct claim is supported only by import lines")
            except (OSError, ValueError) as exc:
                errors.append(f"{claim_id}: unable to check direct import-only evidence: {exc}")

    if plan:
        plan_claims = {
            claim_id
            for page in plan.get("pages", [])
            for claim_id in page.get("claim_ids", [])
        }
        missing = sorted(plan_claims - claim_ids)
        for claim_id in missing:
            errors.append(f"plan references missing claim: {claim_id}")
        target_file = evidence.get("target_file")
        if target_file and plan.get("target_file") and str(target_file) != str(plan.get("target_file")):
            errors.append("plan.target_file and evidence.target_file differ")
    return errors, warnings, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args(argv)

    evidence = _load_json(args.evidence)
    context = _load_json(args.context) if args.context else None
    plan = _load_json(args.plan) if args.plan else None
    errors, warnings, summary = validate_evidence(evidence, context=context, plan=plan)
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
