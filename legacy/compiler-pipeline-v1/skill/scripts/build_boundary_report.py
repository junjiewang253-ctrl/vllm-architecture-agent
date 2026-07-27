#!/usr/bin/env python3
"""Build a semantic boundary report from Architecture Design."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def build_boundary_report(design: dict[str, Any]) -> dict[str, Any]:
    boundaries = []
    for item in design.get("boundaries", []):
        if not isinstance(item, dict):
            continue
        boundaries.append(
            {
                "name": item.get("name"),
                "concept_id": item.get("concept_id"),
                "reason": item.get("reason"),
                "allowed_claim": item.get("allowed_claim"),
                "evidence": item.get("evidence", []),
            }
        )
    return {
        "schema_version": "1.0",
        "source_design_version": design.get("schema_version"),
        "model_name": design.get("model_name", "unknown-model"),
        "external_components": boundaries,
        "local_truth_boundary": "Only behavior evidenced in the input model adapter file may be direct. Imported component internals are boundaries.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_design", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_boundary_report(_load_json(args.architecture_design))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Boundary Report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
