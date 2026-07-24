#!/usr/bin/env python3
"""Create a reproducibility lock for reviewed-mode artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LOCK_VERSION = "0.1"


def _sha256(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"required artifact does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_lock(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "schema_version": LOCK_VERSION,
        "source_sha256": _sha256(paths["source"]),
        "source_analysis_sha256": _sha256(paths["source_analysis"]),
        "semantic_inventory_sha256": _sha256(paths["semantic_inventory"]),
        "baseline_ir_sha256": _sha256(paths["baseline_ir"]),
        "semantic_review_sha256": _sha256(paths["semantic_review"]),
        "ir_patch_sha256": _sha256(paths["ir_patch"]),
        "reviewed_ir_sha256": _sha256(paths["reviewed_ir"]),
        "visual_review_sha256": _sha256(paths["visual_review"]),
        "view_patch_sha256": _sha256(paths["view_patch"]),
        "reviewed_view_sha256": _sha256(paths["reviewed_view"]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reviewed-mode reproducibility lock.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-analysis", type=Path, required=True)
    parser.add_argument("--semantic-inventory", type=Path, required=True)
    parser.add_argument("--baseline-ir", type=Path, required=True)
    parser.add_argument("--semantic-review", type=Path, required=True)
    parser.add_argument("--ir-patch", type=Path, required=True)
    parser.add_argument("--reviewed-ir", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--view-patch", type=Path, required=True)
    parser.add_argument("--reviewed-view", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "source": args.source,
        "source_analysis": args.source_analysis,
        "semantic_inventory": args.semantic_inventory,
        "baseline_ir": args.baseline_ir,
        "semantic_review": args.semantic_review,
        "ir_patch": args.ir_patch,
        "reviewed_ir": args.reviewed_ir,
        "visual_review": args.visual_review,
        "view_patch": args.view_patch,
        "reviewed_view": args.reviewed_view,
    }
    try:
        lock = build_lock(paths)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote review lock to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
