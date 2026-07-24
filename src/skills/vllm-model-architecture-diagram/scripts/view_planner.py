#!/usr/bin/env python3
"""Plan concept-driven Architecture Views from Architecture Design JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ARCHITECTURE_VIEW_VERSION = "1.0"
DESIGN_VERSION = "1.0"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _node_id(concept_id: str) -> str:
    return concept_id.replace("concept:", "concept_").replace(":", "_")


def _ports() -> list[dict[str, str]]:
    return [
        {"id": "in", "label": "in", "direction": "input", "data_kind": "control"},
        {"id": "out", "label": "out", "direction": "output", "data_kind": "control"},
    ]


def _visible_node(concept: dict[str, Any], region_id: str) -> dict[str, Any]:
    title = str(concept.get("title") or concept.get("id"))
    purpose = str(concept.get("purpose") or "")
    kind = "note" if concept.get("type") in {"boundary", "capability"} else "container"
    badges = []
    if concept.get("type") == "parallel_strategy":
        badges = ["TP", "PP", "EP"]
    elif concept.get("type") == "boundary":
        badges = ["external"]
    return {
        "semantic_id": _node_id(str(concept["id"])),
        "concept_id": concept["id"],
        "display_label": title,
        "display_subtitle": purpose[:96],
        "region_id": region_id,
        "lane_id": None,
        "kind": kind,
        "preferred_size": {"width": 260, "height": 96 if kind == "note" else 120},
        "ports": _ports(),
        "badges": badges,
    }


def _page_type(page_id: str) -> str:
    mapping = {
        "overview": "architecture_overview",
        "decoder_layer_detail": "execution_flow",
        "attention_detail": "attention_architecture",
        "moe_detail": "moe_architecture",
        "parallelism": "parallel_strategy",
        "weight_loading": "checkpoint_strategy",
        "integration_boundary": "integration_boundary",
    }
    return mapping.get(page_id, "architecture_view")


def build_architecture_view(design: dict[str, Any]) -> dict[str, Any]:
    if design.get("schema_version") != DESIGN_VERSION:
        raise ValueError(f"Architecture Design schema_version must be {DESIGN_VERSION!r}")
    concept_by_id = {
        concept["id"]: concept
        for concept in design.get("concepts", [])
        if isinstance(concept, dict) and isinstance(concept.get("id"), str)
    }
    relationships = [
        rel for rel in design.get("relationships", [])
        if isinstance(rel, dict) and rel.get("source") in concept_by_id and rel.get("target") in concept_by_id
    ]
    pages: list[dict[str, Any]] = []
    for view in design.get("views", []):
        if not isinstance(view, dict) or not isinstance(view.get("id"), str):
            continue
        page_id = str(view["id"])
        concept_ids = [cid for cid in view.get("concepts", []) if isinstance(cid, str) and cid in concept_by_id]
        region_id = f"region_{page_id}_concepts"
        visible_nodes = [_visible_node(concept_by_id[cid], region_id) for cid in concept_ids]
        concept_node_ids = {node["concept_id"]: node["semantic_id"] for node in visible_nodes}
        visible_edges: list[dict[str, Any]] = []
        for rel in relationships:
            source = concept_node_ids.get(rel["source"])
            target = concept_node_ids.get(rel["target"])
            if not source or not target:
                continue
            relation = str(rel.get("relation") or "relates")
            visible_edges.append(
                {
                    "semantic_id": rel["id"].replace("concept-rel:", "concept_edge_").replace(":", "_"),
                    "concept_relationship_id": rel["id"],
                    "source": source,
                    "target": target,
                    "source_port": "out",
                    "target_port": "in",
                    "style_kind": "dependency" if relation in {"documents_external", "organizes"} else "summary",
                    "label": relation.replace("_", " "),
                    "label_visible": relation in {"documents_external", "parallelizes", "loads"},
                    "route_class": "horizontal_lane",
                    "bundle_id": None,
                }
            )
        purpose = str(view.get("purpose") or "")
        page = {
            "id": page_id,
            "title": str(view.get("title") or page_id),
            "page_type": _page_type(page_id),
            "purpose": purpose,
            "regions": [
                {
                    "id": region_id,
                    "label": "Architecture concepts",
                    "orientation": "grid",
                    "member_ids": [node["semantic_id"] for node in visible_nodes],
                }
            ],
            "lanes": [],
            "visible_nodes": visible_nodes,
            "visible_edges": visible_edges,
            "bundles": [],
            "annotations": [
                {
                    "id": f"decorative_purpose_{page_id}",
                    "text": purpose,
                    "x": 760,
                    "y": 16,
                    "width": 420,
                    "height": 48,
                }
            ],
            "layout_constraints": {"page_size": [1280, 720], "answer": purpose},
            "concepts": concept_ids,
        }
        pages.append(page)

    return {
        "schema_version": ARCHITECTURE_VIEW_VERSION,
        "source_design_version": design.get("schema_version"),
        "model_name": design.get("model_name", "unknown-model"),
        "style_profile": "architect_mode",
        "pages": pages,
        "unresolved": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_design", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        view = build_architecture_view(_load_json(args.architecture_design))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Architecture View to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
