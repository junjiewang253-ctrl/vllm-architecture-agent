#!/usr/bin/env python3
"""Build a mentor-facing Architecture report from v1.0 artifacts."""

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


def build_mentor_report(design: dict[str, Any], view: dict[str, Any], boundary: dict[str, Any]) -> str:
    lines: list[str] = [
        "# HY V3 vLLM Adapter Architecture Report",
        "",
        "This report summarizes a source-driven, Agent-architected model adapter analysis.",
        "The scripts define what is evidenced; the Architect layer decides how those facts are grouped into human architecture concepts.",
        "",
        "## Architecture Concepts",
        "",
    ]
    for concept in design.get("concepts", []):
        if not isinstance(concept, dict):
            continue
        lines.extend(
            [
                f"### {concept.get('title')}",
                "",
                f"- Type: `{concept.get('type')}`",
                f"- Purpose: {concept.get('purpose')}",
                f"- Implementation symbols: {', '.join(concept.get('implementation', []))}",
                f"- Evidence facts: {len(concept.get('evidence', []))}",
                "",
            ]
        )
    lines.extend(["## Architecture Views", ""])
    for page in view.get("pages", []):
        if not isinstance(page, dict):
            continue
        lines.extend(
            [
                f"### {page.get('title')}",
                "",
                f"Question answered: {page.get('purpose')}",
                f"View nodes: {len(page.get('nodes', page.get('visible_nodes', [])))}",
                f"View edges: {len(page.get('edges', page.get('visible_edges', [])))}",
                "",
            ]
        )
    lines.extend(["## Semantic Boundaries", ""])
    for component in boundary.get("external_components", []):
        if not isinstance(component, dict):
            continue
        lines.extend(
            [
                f"### {component.get('name')}",
                "",
                f"- Boundary reason: {component.get('reason')}",
                f"- Allowed claim: {component.get('allowed_claim')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Limits",
            "",
            "- The report does not claim to analyze external vLLM internals.",
            "- The architecture concepts are evidence-backed abstractions, not a raw code relationship graph.",
            "- Multi-model validation remains future work.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_design", type=Path)
    parser.add_argument("architecture_view", type=Path)
    parser.add_argument("boundary_report", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_mentor_report(_load_json(args.architecture_design), _load_json(args.architecture_view), _load_json(args.boundary_report))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote mentor report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
