#!/usr/bin/env python3
"""Validate reviewed-mode reproducibility locks and detect stale review inputs."""

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


def _load_lock(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read review lock: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("review lock root must be an object")
    return {str(key): str(value) for key, value in data.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate reviewed-mode reproducibility lock.")
    parser.add_argument("review_lock", type=Path)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "source_sha256": args.source,
        "source_analysis_sha256": args.source_analysis,
        "semantic_inventory_sha256": args.semantic_inventory,
        "baseline_ir_sha256": args.baseline_ir,
        "semantic_review_sha256": args.semantic_review,
        "ir_patch_sha256": args.ir_patch,
        "reviewed_ir_sha256": args.reviewed_ir,
        "visual_review_sha256": args.visual_review,
        "view_patch_sha256": args.view_patch,
        "reviewed_view_sha256": args.reviewed_view,
    }
    try:
        lock = _load_lock(args.review_lock)
        errors: list[str] = []
        if lock.get("schema_version") != LOCK_VERSION:
            errors.append("review lock schema_version must be '0.1'")
        for key, path in paths.items():
            current = _sha256(path)
            if lock.get(key) != current:
                errors.append(f"stale review lock entry {key}: expected {lock.get(key)}, got {current}")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Review lock validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
