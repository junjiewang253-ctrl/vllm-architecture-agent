#!/usr/bin/env python3
"""Validate Architecture Concept Graph and v1.0.1 Architecture View quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CONCEPT_TYPES = {
    "model_component",
    "execution_strategy",
    "parallel_strategy",
    "checkpoint_strategy",
    "configuration",
    "capability",
    "boundary",
    "optimization",
}
REQUIRED_VIEW_TITLES = {
    "Model Overview",
    "Attention Implementation",
    "MoE Execution Strategy",
    "Parallel Strategy",
    "Checkpoint Adaptation",
    "vLLM Integration Boundary",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def validate_architecture_quality(
    fact_graph: dict[str, Any],
    design: dict[str, Any],
    view: dict[str, Any],
    boundary_report: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if fact_graph.get("schema_version") != "1.0":
        errors.append("Source Fact Graph schema_version must be '1.0'")
    if design.get("schema_version") != "1.0":
        errors.append("Architecture Design schema_version must be '1.0'")
    if view.get("schema_version") != "0.1" or view.get("view_graph_type") != "architecture_view_graph":
        errors.append("Architecture View Graph schema_version must be '0.1'")

    fact_ids = {fact.get("id") for fact in fact_graph.get("facts", []) if isinstance(fact, dict)}
    concepts = [item for item in design.get("concepts", []) if isinstance(item, dict)]
    concept_ids = {concept.get("id") for concept in concepts}
    if not concepts:
        errors.append("Architecture Design must contain concepts")
    for concept in concepts:
        concept_id = concept.get("id")
        if concept.get("type") not in CONCEPT_TYPES:
            errors.append(f"concept {concept_id} has invalid type {concept.get('type')!r}")
        evidence = concept.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"concept {concept_id} must have fact evidence")
        else:
            missing = [fact_id for fact_id in evidence if fact_id not in fact_ids]
            if missing:
                errors.append(f"concept {concept_id} references missing facts: {', '.join(map(str, missing[:3]))}")
        confidence = concept.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            errors.append(f"concept {concept_id} confidence must be between 0 and 1")

    related_ids: set[Any] = set()
    for relationship in design.get("relationships", []):
        if not isinstance(relationship, dict):
            errors.append("relationship must be an object")
            continue
        source = relationship.get("source")
        target = relationship.get("target")
        if source not in concept_ids or target not in concept_ids:
            errors.append(f"relationship {relationship.get('id')} references unknown concept")
        related_ids.update([source, target])
        evidence = relationship.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"relationship {relationship.get('id')} must have evidence")
    isolated = sorted(str(cid) for cid in concept_ids - related_ids if cid)
    if isolated:
        errors.append(f"isolated concepts: {', '.join(isolated)}")

    pages = [page for page in view.get("pages", []) if isinstance(page, dict)]
    titles = {page.get("title") for page in pages}
    missing_titles = sorted(REQUIRED_VIEW_TITLES - {title for title in titles if isinstance(title, str)})
    if missing_titles:
        errors.append(f"Architecture View missing required pages: {', '.join(missing_titles)}")
    for page in pages:
        if not str(page.get("purpose") or "").strip():
            errors.append(f"page {page.get('id')} must have purpose")
        if not page.get("nodes"):
            errors.append(f"page {page.get('id')} must contain view nodes")

    report = boundary_report or {}
    external_components = report.get("external_components")
    if not isinstance(external_components, list) or not external_components:
        errors.append("Boundary Report must contain external_components")
    else:
        for component in external_components:
            if not isinstance(component, dict):
                errors.append("external component must be an object")
                continue
            if not component.get("reason") or not component.get("allowed_claim"):
                errors.append(f"external component {component.get('name')} must include reason and allowed_claim")
            if "direct" in str(component.get("allowed_claim", "")).lower():
                errors.append(f"external component {component.get('name')} allowed_claim must not claim direct internals")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_fact_graph", type=Path)
    parser.add_argument("architecture_design", type=Path)
    parser.add_argument("architecture_view", type=Path)
    parser.add_argument("--boundary-report", type=Path)
    args = parser.parse_args(argv)
    try:
        errors = validate_architecture_quality(
            _load_json(args.source_fact_graph),
            _load_json(args.architecture_design),
            _load_json(args.architecture_view),
            _load_json(args.boundary_report) if args.boundary_report else None,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Architecture quality validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
