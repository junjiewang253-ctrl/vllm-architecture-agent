#!/usr/bin/env python3
"""Build deterministic baseline review templates for CI/deterministic mode.

Reviewed mode should use run_semantic_review.py and run_visual_review.py. This
script keeps the old conservative template behavior available without presenting
it as Codex review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from build_semantic_review import build_semantic_review
from build_visual_review import build_visual_review
from review_runner_common import load_json, stable_write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic baseline review template.")
    subparsers = parser.add_subparsers(dest="kind", required=True)

    semantic = subparsers.add_parser("semantic")
    semantic.add_argument("source_analysis", type=Path)
    semantic.add_argument("semantic_inventory", type=Path)
    semantic.add_argument("baseline_ir", type=Path)
    semantic.add_argument("baseline_coverage", type=Path)
    semantic.add_argument("--source-file", type=Path, required=True)
    semantic.add_argument("--review-output", type=Path, required=True)
    semantic.add_argument("--patch-output", type=Path, required=True)

    visual = subparsers.add_parser("visual")
    visual.add_argument("reviewed_ir", type=Path)
    visual.add_argument("baseline_view", type=Path)
    visual.add_argument("baseline_layout_plan", type=Path)
    visual.add_argument("baseline_layout_metrics", type=Path)
    visual.add_argument("baseline_drawio", type=Path)
    visual.add_argument("--review-output", type=Path, required=True)
    visual.add_argument("--patch-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.kind == "semantic":
            review, patch = build_semantic_review(
                load_json(args.source_analysis, "source-analysis"),
                load_json(args.semantic_inventory, "semantic inventory"),
                load_json(args.baseline_ir, "baseline Architecture IR"),
                load_json(args.baseline_coverage, "baseline semantic coverage"),
                source_path=args.source_file,
                baseline_ir_sha256=__import__("hashlib").sha256(args.baseline_ir.read_bytes()).hexdigest(),
                patch_file=str(args.patch_output),
            )
        else:
            review, patch = build_visual_review(
                load_json(args.reviewed_ir, "reviewed Architecture IR"),
                load_json(args.baseline_view, "baseline Diagram View"),
                load_json(args.baseline_layout_plan, "baseline Layout Plan"),
                load_json(args.baseline_layout_metrics, "baseline layout metrics"),
                base_view_sha256=__import__("hashlib").sha256(args.baseline_view.read_bytes()).hexdigest(),
                patch_file=str(args.patch_output),
            )
            if not args.baseline_drawio.exists():
                raise ValueError(f"baseline Draw.io does not exist: {args.baseline_drawio}")
        stable_write_json(args.review_output, review)
        stable_write_json(args.patch_output, patch)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote baseline {args.kind} review template to {args.review_output}")
    print(f"Wrote baseline {args.kind} patch template to {args.patch_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
