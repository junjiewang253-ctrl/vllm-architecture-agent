#!/usr/bin/env python3
"""Run Codex Visual Review and persist review plus Diagram View patch."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from review_runner_common import (
    extract_json_object,
    load_json,
    read_codex_or_mock,
    stable_write_json,
    write_failure,
)

REVIEW_VERSION = "0.1"
PATCH_VERSION = "0.1"
REVIEW_ROOT = Path(__file__).resolve().parents[1] / "review"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_prompt(template_path: Path, replacements: dict[str, str]) -> str:
    prompt = template_path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


def _validate_raw_visual_review(review: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(review.get("model_name"), str) or not review["model_name"].strip():
        errors.append("visual_review.model_name must be a non-empty string")
    if not isinstance(review.get("findings"), list):
        errors.append("visual_review.findings must be a list")
        return errors
    for index, finding in enumerate(review["findings"]):
        if not isinstance(finding, dict):
            errors.append(f"findings[{index}] must be an object")
            continue
        if not isinstance(finding.get("id"), str) or not finding["id"]:
            errors.append(f"findings[{index}].id is required")
        if not isinstance(finding.get("page_id"), str) or not finding["page_id"]:
            errors.append(f"findings[{index}].page_id is required")
        if finding.get("severity") not in {"critical", "major", "minor", "informational"}:
            errors.append(f"findings[{index}].severity is invalid")
        if finding.get("semantic_change") is not False:
            errors.append(f"findings[{index}] must not propose semantic_change")
    return errors


def _validate_patch(patch: dict[str, Any], base_hash: str) -> list[str]:
    errors: list[str] = []
    if patch.get("schema_version") != PATCH_VERSION:
        errors.append("diagram_view_patch.schema_version must be '0.1'")
    if patch.get("base_view_sha256") != base_hash:
        errors.append("diagram_view_patch.base_view_sha256 does not match baseline Diagram View")
    if not isinstance(patch.get("operations"), list):
        errors.append("diagram_view_patch.operations must be a list")
    return errors


def _normalize_review(raw: dict[str, Any], *, patch_output: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for item in raw.get("findings", []):
        finding = dict(item)
        finding["finding_id"] = finding.get("id")
        findings.append(finding)
    return {
        "schema_version": REVIEW_VERSION,
        "model_name": raw.get("model_name", "unknown-model"),
        "review_mode": "reviewed",
        "summary": {
            "finding_count": len(findings),
            "codex_review": True,
        },
        "findings": findings,
        "patch_file": str(patch_output),
    }


def run_visual_review(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    reviewed_ir = load_json(args.reviewed_ir, "reviewed Architecture IR")
    diagram_view = load_json(args.diagram_view, "Diagram View")
    layout_metrics = load_json(args.layout_metrics, "layout metrics")
    base_hash = _sha256(args.diagram_view)
    screenshot_text = ", ".join(str(path) for path in args.screenshots) if args.screenshots else "not provided"
    prompt = _render_prompt(
        args.prompt_template,
        {
            "reviewed_ir": str(args.reviewed_ir),
            "diagram_view": str(args.diagram_view),
            "layout_metrics": str(args.layout_metrics),
            "screenshots": screenshot_text,
            "base_view_sha256": base_hash,
        },
    )
    if args.prompt_output:
        args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_output.write_text(prompt, encoding="utf-8")
    text = read_codex_or_mock(
        prompt,
        mock_response=args.mock_response,
        codex_bin=args.codex_bin,
        cwd=args.cwd,
        timeout_seconds=args.timeout_seconds,
    )
    data = extract_json_object(text)
    raw_review = data.get("visual_review", data)
    patch = data.get("diagram_view_patch")
    if not isinstance(raw_review, dict) or not isinstance(patch, dict):
        raise ValueError("Codex response must contain visual_review and diagram_view_patch objects")
    if args.mock_response is not None and patch.get("base_view_sha256") != base_hash:
        patch["base_view_sha256"] = base_hash
    errors = _validate_raw_visual_review(raw_review)
    errors.extend(_validate_patch(patch, base_hash))
    if reviewed_ir.get("schema_version") != "0.6":
        errors.append("reviewed Architecture IR schema_version must be '0.6'")
    if diagram_view.get("schema_version") != "0.1":
        errors.append("Diagram View schema_version must be '0.1'")
    if "pages" not in layout_metrics:
        errors.append("layout metrics must contain pages")
    if errors:
        raise ValueError("; ".join(errors))
    return _normalize_review(raw_review, patch_output=args.patch_output), patch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Codex Visual Review.")
    parser.add_argument("reviewed_ir", type=Path)
    parser.add_argument("diagram_view", type=Path)
    parser.add_argument("layout_metrics", type=Path)
    parser.add_argument("--screenshots", type=Path, nargs="*", default=[])
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, default=REVIEW_ROOT / "prompts" / "visual_review.md")
    parser.add_argument("--prompt-output", type=Path)
    parser.add_argument("--mock-response", type=Path)
    parser.add_argument("--review-failed-output", type=Path)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    failure_path = args.review_failed_output or args.review_output.with_name("review_failed.json")
    try:
        review, patch = run_visual_review(args)
        stable_write_json(args.review_output, review)
        stable_write_json(args.patch_output, patch)
    except (OSError, ValueError) as exc:
        write_failure(failure_path, str(exc))
        print(f"error: visual review failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Codex visual review to {args.review_output}")
    print(f"Wrote Diagram View patch to {args.patch_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
