#!/usr/bin/env python3
"""Build a constrained visual review and Diagram View patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REVIEW_VERSION = "0.1"
PATCH_VERSION = "0.1"


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


def _page_ids(view: dict[str, Any]) -> set[str]:
    return {page["id"] for page in view.get("pages", []) if isinstance(page, dict) and isinstance(page.get("id"), str)}


def build_visual_review(
    ir: dict[str, Any],
    view: dict[str, Any],
    layout_plan: dict[str, Any],
    metrics: dict[str, Any],
    *,
    base_view_sha256: str,
    patch_file: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pages = _page_ids(view)
    findings: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    if "attention_detail" in pages:
        findings.append(
            {
                "finding_id": "visual:attention-external-boundary-note",
                "page_id": "attention_detail",
                "type": "missing_boundary_note",
                "severity": "major",
                "affected_ids": ["kv_cache", "kv_cache_to_attention_core"],
                "description": "KV cache read behavior is an imported vLLM Attention boundary and should be visually labelled as such.",
                "recommended_action": "Add a compact boundary note without changing semantic edges.",
                "semantic_change": False,
                "patch_op_ids": ["view-op:add-attention-boundary-note"],
            }
        )
        operations.append(
            {
                "op_id": "view-op:add-attention-boundary-note",
                "op": "add_boundary_note",
                "page_id": "attention_detail",
                "annotation": {
                    "id": "decorative_note_attention_external_boundary",
                    "type": "boundary_note",
                    "text": "External boundary: KV cache read is inside vLLM Attention.",
                    "x": 1020,
                    "y": 74,
                    "width": 330,
                    "height": 58,
                },
                "reason": "Clarify external boundary discovered by semantic review.",
                "confidence": 0.93,
                "finding_ids": ["visual:attention-external-boundary-note"],
            }
        )
    if "adapter_integration" in pages:
        member_ids = [
            "model_to_input_components",
            "model_to_attention_components",
            "model_to_ffn_moe_components",
            "causal_lm_to_output_components",
        ]
        findings.append(
            {
                "finding_id": "visual:adapter-execution-bundle",
                "page_id": "adapter_integration",
                "type": "long_edge",
                "severity": "major",
                "affected_ids": member_ids,
                "description": "Several adapter-to-component relations are visually related and can be bundled as a presentation hint.",
                "recommended_action": "Add a Diagram View bundle; semantic edge endpoints stay unchanged.",
                "semantic_change": False,
                "patch_op_ids": ["view-op:add-adapter-execution-bundle"],
            }
        )
        operations.append(
            {
                "op_id": "view-op:add-adapter-execution-bundle",
                "op": "add_bundle",
                "page_id": "adapter_integration",
                "bundle": {
                    "id": "adapter_execution_bundle",
                    "member_ids": member_ids,
                    "label": "adapter execution components",
                },
                "reason": "Bundle related adaptation edges without changing semantic relationships.",
                "confidence": 0.9,
                "finding_ids": ["visual:adapter-execution-bundle"],
            }
        )
    summary = {
        "finding_count": len(findings),
        "operation_count": len(operations),
        "page_count": len(pages),
        "baseline_layout_pages": len(layout_plan.get("pages", [])),
        "metrics_pages": len(metrics.get("pages", [])),
    }
    review = {
        "schema_version": REVIEW_VERSION,
        "model_name": ir.get("model_name", view.get("model_name", "unknown-model")),
        "review_mode": "reviewed",
        "summary": summary,
        "findings": findings,
        "patch_file": patch_file,
    }
    patch = {
        "schema_version": PATCH_VERSION,
        "base_view_sha256": base_view_sha256,
        "operations": operations,
        "deferred_operations": [],
    }
    return review, patch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual review and Diagram View patch.")
    parser.add_argument("reviewed_ir", type=Path)
    parser.add_argument("baseline_view", type=Path)
    parser.add_argument("baseline_layout_plan", type=Path)
    parser.add_argument("baseline_layout_metrics", type=Path)
    parser.add_argument("baseline_drawio", type=Path)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        review, patch = build_visual_review(
            _load_json(args.reviewed_ir, "reviewed Architecture IR"),
            _load_json(args.baseline_view, "baseline Diagram View"),
            _load_json(args.baseline_layout_plan, "baseline Layout Plan"),
            _load_json(args.baseline_layout_metrics, "baseline layout metrics"),
            base_view_sha256=_sha256(args.baseline_view),
            patch_file=str(args.patch_output),
        )
        if not args.baseline_drawio.exists():
            raise ValueError(f"baseline Draw.io does not exist: {args.baseline_drawio}")
        args.review_output.parent.mkdir(parents=True, exist_ok=True)
        args.patch_output.parent.mkdir(parents=True, exist_ok=True)
        args.review_output.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.patch_output.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote visual review to {args.review_output}")
    print(f"Wrote Diagram View patch to {args.patch_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
