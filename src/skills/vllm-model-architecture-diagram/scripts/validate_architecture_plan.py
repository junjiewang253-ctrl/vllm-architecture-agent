"""Validate v2.1 legacy and v2.2 integrated Architecture Plans."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vllm_architecture_agent.paths import resolve_repo_path  # noqa: E402

VALID_PATTERNS = {
    "pipeline",
    "block",
    "branch_merge",
    "routed_container",
    "mapping_flow",
    "strategy_matrix",
    "boundary_map",
    "multimodal_pipeline",
    "state_machine",
    "component_map",
}

VALID_DIAGRAM_STRATEGIES = {
    "integrated_single_canvas",
    "composite_pages",
    "compact_boundary",
}

VALID_TOPICS = {
    "model_composition",
    "model_execution",
    "decoder",
    "attention",
    "ffn",
    "moe",
    "multimodal",
    "pooling",
    "classification",
    "recurrent",
    "state_space",
    "tensor_parallel",
    "pipeline_parallel",
    "expert_parallel",
    "weight_loading",
    "quantization",
    "lora",
    "external_boundary",
    "other",
}

VALID_INTEGRATION_MODES = {
    "inline_expansion",
    "nested_container",
    "branch_merge",
    "side_dependency",
    "loading_injection",
    "boundary_crossing",
}

VALID_VISUAL_REPRESENTATIONS = {
    "node",
    "container",
    "storage",
    "badge",
    "merge",
    "annotation",
}

VALID_VISUAL_ROLES = {
    "input",
    "runtime_stage",
    "branch",
    "merge",
    "detail",
    "output",
    "loading_stage",
    "strategy",
    "capability",
    "external_boundary",
    "annotation",
}

VALID_VISUAL_PHASES = {
    "construction",
    "runtime",
    "loading",
    "parallel",
    "capability",
    "external",
}

VALID_STYLE_ROLES = {
    "runtime",
    "residual",
    "loading",
    "construction",
    "metadata",
    "external",
    "containment",
    "capability",
}

VALID_VISUAL_RELATIONSHIP_KINDS = {
    "runtime_flow",
    "residual_flow",
    "loading_mapping",
    "construction_dependency",
    "metadata_dependency",
    "external_delegation",
}

VALID_REGION_TYPES = {
    "embedded_subgraph",
    "component_tree",
    "nested_container",
    "parameter_panel",
    "capability_panel",
    "strategy_panel",
    "variant_table",
    "mapping_tree",
    "boundary_panel",
    "callout",
}

VALID_REVIEW_STATUS = {
    "rendered_detail",
    "rendered_aggregate",
    "documented_external",
    "excluded_with_reason",
    "unresolved",
}

MIN_MAIN_STORY_STEPS = {
    "pipeline": 5,
    "block": 5,
    "branch_merge": 5,
    "routed_container": 5,
    "mapping_flow": 4,
    "multimodal_pipeline": 5,
    "state_machine": 4,
}

VALID_CLASS_ROLES = {
    "primary_component",
    "supporting_component",
    "utility_or_helper",
    "configuration_or_metadata",
    "loading_only",
    "external_boundary",
    "not_architecturally_relevant",
    "unresolved",
}

VALID_IMPORTANCE = {"core", "supporting", "trivial"}

CAPABILITY_TOPIC_MAP = {
    "moe": "moe",
    "multimodal": "multimodal",
    "expert_parallel": "expert_parallel",
    "pooling": "pooling",
    "recurrent_or_ssm": "recurrent",
}

STRUCTURAL_REGION_TYPES = {
    "embedded_subgraph",
    "component_tree",
    "nested_container",
    "mapping_tree",
}

SUBSYSTEM_CATEGORY_GROUPS = {
    "main compute": {"attention", "ffn"},
    "mixture-of-experts": {"moe", "router"},
    "multimodal": {"multimodal"},
    "recurrent/state-space": {"recurrent", "state_space"},
}

MAX_REGION_SOURCE_ITEMS_PER_ANCHOR = 8
MIN_STRUCTURAL_REGION_ANCHORS = 4
MIN_STRUCTURAL_REGION_RELATIONSHIPS = 3


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counter_duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _ids(items: list[dict[str, Any]], key: str) -> set[str]:
    return {item[key] for item in items if item.get(key)}


def _page_maps(plan: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[tuple[str, str]]]:
    pages = {page["id"]: page for page in plan.get("pages", []) if page.get("id")}
    regions: set[tuple[str, str]] = set()
    for page in pages.values():
        for region in page.get("detail_regions", []):
            if region.get("id"):
                regions.add((page["id"], region["id"]))
    return pages, regions


def _known_source_item_ids(context: dict[str, Any]) -> set[str]:
    known: set[str] = set()
    for key, id_key in (
        ("classes", "class_id"),
        ("methods", "method_id"),
        ("module_functions", "function_id"),
        ("branches", "branch_id"),
        ("loops", "loop_id"),
        ("weight_mappings", "mapping_id"),
    ):
        known.update(_ids(context.get(key, []), id_key))
    for name, signal in context.get("capability_signals", {}).items():
        known.add(signal.get("capability_id") or f"capability:{name}")
    return known


def _has_directed_path(
    adjacency: dict[str, set[str]],
    source: str,
    target: str,
) -> bool:
    pending = [source]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(adjacency.get(current, set()) - seen)
    return False


def _semantic_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if token not in {"a", "an", "and", "of", "or", "the", "to"}
    }


def _validate_visual_contract(
    *,
    page_id: str,
    page: dict[str, Any],
    known_source_ids: set[str],
    claim_ids: set[str],
    evidence_claims: dict[str, dict[str, Any]],
    context_available: bool,
    evidence_available: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    contract = page.get("visual_contract")
    if not isinstance(contract, dict):
        errors.append(f"{page_id}: Plan 2.2 pages require visual_contract")
        return

    canvas = contract.get("canvas", {})
    for field, minimum in (
        ("min_width", 3200),
        ("min_height", 1600),
        ("min_export_width", 2400),
        ("min_export_height", 1200),
    ):
        value = canvas.get(field)
        if not isinstance(value, int) or value < minimum:
            errors.append(f"{page_id}: visual_contract.canvas.{field} must be at least {minimum}")
    if canvas.get("orientation") != "landscape":
        errors.append(f"{page_id}: integrated canvas orientation must be landscape")
    min_font_size = canvas.get("min_font_size")
    if not isinstance(min_font_size, int) or not (10 <= min_font_size <= 16):
        errors.append(f"{page_id}: visual_contract.canvas.min_font_size must be from 10 to 16")
    min_fill = canvas.get("min_content_fill_ratio")
    if not isinstance(min_fill, (int, float)) or not (0.30 <= float(min_fill) <= 0.80):
        errors.append(
            f"{page_id}: visual_contract.canvas.min_content_fill_ratio must be from 0.30 to 0.80"
        )

    page_claim_ids = set(page.get("claim_ids", []))
    anchors = contract.get("required_anchors")
    if not isinstance(anchors, list) or not anchors:
        errors.append(f"{page_id}: visual_contract.required_anchors must be non-empty")
        anchors = []
    elif len(anchors) < 18:
        errors.append(f"{page_id}: integrated visual contract requires at least 18 anchors")
    anchor_ids = [anchor.get("id") for anchor in anchors if isinstance(anchor, dict)]
    for duplicate in _counter_duplicates([value for value in anchor_ids if value]):
        errors.append(f"{page_id}: duplicate visual anchor {duplicate}")
    anchor_map = {
        anchor["id"]: anchor
        for anchor in anchors
        if isinstance(anchor, dict) and anchor.get("id")
    }
    for anchor in anchors:
        if not isinstance(anchor, dict):
            errors.append(f"{page_id}: visual anchors must be objects")
            continue
        anchor_id = anchor.get("id", "<missing>")
        if not str(anchor_id).startswith("visual:"):
            errors.append(f"{page_id}: visual anchor IDs must start with visual: ({anchor_id})")
        if not str(anchor.get("label", "")).strip():
            errors.append(f"{page_id}.{anchor_id}: label must be non-empty")
        if anchor.get("representation") not in VALID_VISUAL_REPRESENTATIONS:
            errors.append(f"{page_id}.{anchor_id}: invalid representation")
        if anchor.get("role") not in VALID_VISUAL_ROLES:
            errors.append(f"{page_id}.{anchor_id}: invalid role")
        if anchor.get("phase") not in VALID_VISUAL_PHASES:
            errors.append(f"{page_id}.{anchor_id}: invalid phase")
        if anchor.get("style_role") not in VALID_STYLE_ROLES:
            errors.append(f"{page_id}.{anchor_id}: invalid style_role")
        anchor_claims = anchor.get("claim_ids")
        if not isinstance(anchor_claims, list) or not anchor_claims:
            errors.append(f"{page_id}.{anchor_id}: claim_ids must be non-empty")
        elif evidence_available:
            for claim_id in anchor_claims:
                if claim_id not in claim_ids:
                    errors.append(f"{page_id}.{anchor_id}: unknown claim_id {claim_id}")
                if claim_id not in page_claim_ids:
                    errors.append(
                        f"{page_id}.{anchor_id}: anchor claim_id is not declared by its page: {claim_id}"
                    )
        source_ids = anchor.get("source_item_ids")
        if not isinstance(source_ids, list) or not source_ids:
            errors.append(f"{page_id}.{anchor_id}: source_item_ids must be non-empty")
        elif context_available:
            for source_id in source_ids:
                if source_id not in known_source_ids:
                    errors.append(f"{page_id}.{anchor_id}: unknown source_item_id {source_id}")
        if anchor.get("role") == "external_boundary":
            if anchor.get("phase") != "external" or anchor.get("style_role") != "external":
                errors.append(
                    f"{page_id}.{anchor_id}: external boundaries require external phase and style_role"
                )
            if evidence_available and not any(
                evidence_claims.get(claim_id, {}).get("confidence") == "external"
                for claim_id in anchor_claims
            ):
                errors.append(
                    f"{page_id}.{anchor_id}: external boundary requires an external Evidence claim"
                )
        if anchor.get("role") == "capability" and anchor.get("style_role") != "capability":
            errors.append(f"{page_id}.{anchor_id}: capability anchors require capability style_role")

    for anchor_id, anchor in anchor_map.items():
        parent_id = anchor.get("parent_anchor_id")
        if parent_id is None:
            continue
        parent = anchor_map.get(parent_id)
        if parent is None:
            errors.append(f"{page_id}.{anchor_id}: unknown parent_anchor_id {parent_id}")
        elif parent.get("representation") != "container":
            errors.append(f"{page_id}.{anchor_id}: parent_anchor_id must reference a container")

    relationships = contract.get("required_relationships")
    if not isinstance(relationships, list) or not relationships:
        errors.append(f"{page_id}: visual_contract.required_relationships must be non-empty")
        relationships = []
    elif len(relationships) < 14:
        errors.append(f"{page_id}: integrated visual contract requires at least 14 relationships")
    relationship_ids = [
        relationship.get("id")
        for relationship in relationships
        if isinstance(relationship, dict)
    ]
    for duplicate in _counter_duplicates([value for value in relationship_ids if value]):
        errors.append(f"{page_id}: duplicate visual relationship {duplicate}")

    adjacency: dict[str, set[str]] = {}
    connectivity: dict[str, set[str]] = {}
    degree: Counter[str] = Counter()
    incoming_degree: Counter[str] = Counter()
    outgoing_degree: Counter[str] = Counter()
    endpoint_pairs: set[tuple[str, str]] = set()
    for relationship in relationships:
        if not isinstance(relationship, dict):
            errors.append(f"{page_id}: visual relationships must be objects")
            continue
        relationship_id = relationship.get("id", "<missing>")
        if not str(relationship_id).startswith("visual-rel:"):
            errors.append(
                f"{page_id}: visual relationship IDs must start with visual-rel: ({relationship_id})"
            )
        source = relationship.get("source_anchor_id")
        target = relationship.get("target_anchor_id")
        if source not in anchor_map:
            errors.append(f"{page_id}.{relationship_id}: unknown source anchor {source}")
        if target not in anchor_map:
            errors.append(f"{page_id}.{relationship_id}: unknown target anchor {target}")
        if relationship.get("kind") not in VALID_VISUAL_RELATIONSHIP_KINDS:
            errors.append(f"{page_id}.{relationship_id}: invalid relationship kind")
        relationship_claims = relationship.get("claim_ids")
        if not isinstance(relationship_claims, list) or not relationship_claims:
            errors.append(f"{page_id}.{relationship_id}: claim_ids must be non-empty")
        elif evidence_available:
            for claim_id in relationship_claims:
                if claim_id not in claim_ids:
                    errors.append(f"{page_id}.{relationship_id}: unknown claim_id {claim_id}")
                if claim_id not in page_claim_ids:
                    errors.append(
                        f"{page_id}.{relationship_id}: relationship claim_id is not declared by its page: {claim_id}"
                    )
        expected_phases = {
            "runtime_flow": {"runtime"},
            "residual_flow": {"runtime"},
            "loading_mapping": {"loading"},
            "construction_dependency": {"construction"},
            "metadata_dependency": {"construction", "parallel", "capability"},
            "external_delegation": {"external"},
        }.get(relationship.get("kind"), set())
        if evidence_available and expected_phases and relationship_claims:
            actual_phases = {
                evidence_claims.get(claim_id, {}).get("phase")
                for claim_id in relationship_claims
            }
            if not actual_phases & expected_phases:
                errors.append(
                    f"{page_id}.{relationship_id}: {relationship.get('kind')} requires "
                    f"Evidence phase in {sorted(expected_phases)}"
                )
        if source in anchor_map and target in anchor_map:
            endpoint_pair = (source, target)
            if endpoint_pair in endpoint_pairs:
                errors.append(
                    f"{page_id}.{relationship_id}: duplicate visual relationship "
                    f"endpoints {source} -> {target}; combine the semantics or use "
                    "distinct architecture endpoints"
                )
            endpoint_pairs.add(endpoint_pair)
            adjacency.setdefault(source, set()).add(target)
            connectivity.setdefault(source, set()).add(target)
            connectivity.setdefault(target, set()).add(source)
            degree[source] += 1
            degree[target] += 1
            outgoing_degree[source] += 1
            incoming_degree[target] += 1

    for anchor_id, anchor in anchor_map.items():
        parent_id = anchor.get("parent_anchor_id")
        if parent_id in anchor_map:
            connectivity.setdefault(parent_id, set()).add(anchor_id)
            connectivity.setdefault(anchor_id, set()).add(parent_id)

    main_story_anchors = contract.get("main_story_anchor_ids")
    if not isinstance(main_story_anchors, list) or len(main_story_anchors) < 7:
        errors.append(f"{page_id}: integrated main_story_anchor_ids must contain at least 7 anchors")
        main_story_anchors = []
    for anchor_id in main_story_anchors:
        if anchor_id not in anchor_map:
            errors.append(f"{page_id}: main_story_anchor_ids references unknown anchor {anchor_id}")
    if len(main_story_anchors) != len(set(main_story_anchors)):
        errors.append(f"{page_id}: main_story_anchor_ids must be unique")
    if len(main_story_anchors) != len(page.get("main_story", [])):
        errors.append(
            f"{page_id}: main_story and main_story_anchor_ids must describe the same number of stages"
        )
    for stage, anchor_id in zip(page.get("main_story", []), main_story_anchors, strict=False):
        anchor = anchor_map.get(anchor_id, {})
        stage_tokens = _semantic_tokens(str(stage))
        anchor_tokens = _semantic_tokens(str(anchor.get("label", "")))
        if stage_tokens and anchor_tokens and not stage_tokens & anchor_tokens:
            errors.append(
                f"{page_id}: main story stage {stage!r} does not match anchor label "
                f"{anchor.get('label')!r}"
            )
    for source, target in zip(main_story_anchors, main_story_anchors[1:], strict=False):
        if source in anchor_map and target in anchor_map and not _has_directed_path(adjacency, source, target):
            errors.append(
                f"{page_id}: no required visual path connects main story anchors {source} -> {target}"
            )

    child_counts: Counter[str] = Counter(
        anchor.get("parent_anchor_id")
        for anchor in anchors
        if isinstance(anchor, dict) and anchor.get("parent_anchor_id")
    )
    disconnected_exempt_roles = {"strategy", "capability", "annotation"}
    for anchor_id, anchor in anchor_map.items():
        if (
            degree[anchor_id] == 0
            and not anchor.get("parent_anchor_id")
            and child_counts[anchor_id] == 0
            and anchor.get("role") not in disconnected_exempt_roles
        ):
            errors.append(f"{page_id}.{anchor_id}: required architecture anchor is isolated")
        if anchor.get("role") == "branch" and outgoing_degree[anchor_id] < 2:
            errors.append(
                f"{page_id}.{anchor_id}: branch anchors must expose at least two "
                "outgoing alternatives; do not collapse mutually exclusive paths "
                "into one serial box"
            )
        if anchor.get("role") == "merge" and incoming_degree[anchor_id] < 2:
            errors.append(
                f"{page_id}.{anchor_id}: merge anchors must receive at least two "
                "incoming alternatives"
            )

    for region in page.get("detail_regions", []):
        region_id = region.get("id", "<missing>")
        integration = region.get("integration")
        if not isinstance(integration, dict):
            errors.append(f"{page_id}.{region_id}: Plan 2.2 detail regions require integration")
            continue
        if integration.get("mode") not in VALID_INTEGRATION_MODES:
            errors.append(f"{page_id}.{region_id}: invalid integration mode")
        region_anchor_ids = integration.get("anchor_ids")
        if not isinstance(region_anchor_ids, list) or not region_anchor_ids:
            errors.append(f"{page_id}.{region_id}: integration.anchor_ids must be non-empty")
            region_anchor_ids = []
        for anchor_id in region_anchor_ids:
            if anchor_id not in anchor_map:
                errors.append(f"{page_id}.{region_id}: unknown integration anchor {anchor_id}")
        attach_to = integration.get("attach_to_anchor_id")
        if attach_to not in anchor_map:
            errors.append(f"{page_id}.{region_id}: unknown attach_to_anchor_id {attach_to}")
        elif attach_to not in main_story_anchors:
            warnings.append(
                f"{page_id}.{region_id}: detail region attaches outside the declared main story"
            )
        for anchor_id in region_anchor_ids:
            if (
                anchor_id in anchor_map
                and attach_to in anchor_map
                and anchor_id != attach_to
                and not _has_directed_path(connectivity, attach_to, anchor_id)
                and anchor_map[anchor_id].get("parent_anchor_id") != attach_to
            ):
                errors.append(
                    f"{page_id}.{region_id}: anchor {anchor_id} is not connected to attachment {attach_to}"
                )
        if region.get("type") in STRUCTURAL_REGION_TYPES:
            semantic_region_anchors = {
                anchor_id
                for anchor_id in region_anchor_ids
                if anchor_map.get(anchor_id, {}).get("representation")
                not in {"annotation", "badge"}
            }
            internal_relationships = [
                relationship
                for relationship in relationships
                if relationship.get("source_anchor_id") in semantic_region_anchors
                and relationship.get("target_anchor_id") in semantic_region_anchors
            ]
            if len(semantic_region_anchors) < MIN_STRUCTURAL_REGION_ANCHORS:
                errors.append(
                    f"{page_id}.{region_id}: structural detail region requires at least "
                    f"{MIN_STRUCTURAL_REGION_ANCHORS} semantic anchors"
                )
            if len(internal_relationships) < MIN_STRUCTURAL_REGION_RELATIONSHIPS:
                errors.append(
                    f"{page_id}.{region_id}: structural detail region requires at least "
                    f"{MIN_STRUCTURAL_REGION_RELATIONSHIPS} internal relationships"
                )
            source_item_count = len(set(region.get("source_item_ids", [])))
            source_capacity = max(
                1,
                len(semantic_region_anchors),
            ) * MAX_REGION_SOURCE_ITEMS_PER_ANCHOR
            if source_item_count > source_capacity:
                errors.append(
                    f"{page_id}.{region_id}: {source_item_count} source items are compressed "
                    f"into {len(semantic_region_anchors)} semantic anchors; split the region "
                    "or make more architecture detail visible"
                )

    gates = contract.get("quality_gates", {})
    for field, minimum, maximum in (
        ("min_semantic_nodes", 18, 100),
        ("min_visible_edges", 14, 140),
        ("min_visual_styles", 4, 8),
    ):
        value = gates.get(field)
        if not isinstance(value, int) or not (minimum <= value <= maximum):
            errors.append(
                f"{page_id}: visual_contract.quality_gates.{field} must be from {minimum} to {maximum}"
            )
    for field in (
        "require_nested_container",
        "require_external_boundary",
        "require_loading_plane",
    ):
        if not isinstance(gates.get(field), bool):
            errors.append(f"{page_id}: visual_contract.quality_gates.{field} must be boolean")
    if page.get("external_boundaries") and not gates.get("require_external_boundary"):
        errors.append(f"{page_id}: declared external boundaries require the external-boundary quality gate")
    if "weight_loading" in page.get("topics", []) and not gates.get("require_loading_plane"):
        errors.append(f"{page_id}: weight_loading topic requires the loading-plane quality gate")
    if gates.get("require_nested_container") and not any(
        anchor.get("parent_anchor_id") for anchor in anchors if isinstance(anchor, dict)
    ):
        errors.append(f"{page_id}: quality gate requires at least one nested visual anchor")
    if gates.get("require_external_boundary") and not any(
        anchor.get("role") == "external_boundary" for anchor in anchors if isinstance(anchor, dict)
    ):
        errors.append(f"{page_id}: quality gate requires an external boundary anchor")
    if gates.get("require_loading_plane"):
        if not any(
            anchor.get("phase") == "loading" for anchor in anchors if isinstance(anchor, dict)
        ):
            errors.append(f"{page_id}: quality gate requires loading-phase anchors")
        if not any(
            relationship.get("kind") == "loading_mapping"
            for relationship in relationships
            if isinstance(relationship, dict)
        ):
            errors.append(f"{page_id}: quality gate requires loading_mapping relationships")


def _validate_page_refs(
    owner: str,
    page_refs: list[dict[str, str]],
    regions: set[tuple[str, str]],
    errors: list[str],
) -> None:
    if not isinstance(page_refs, list):
        errors.append(f"{owner}: page_refs must be a list")
        return
    for ref in page_refs:
        page_id = ref.get("page_id")
        region_id = ref.get("region_id")
        if (page_id, region_id) not in regions:
            errors.append(f"{owner}: page_ref points to unknown page/region {page_id}/{region_id}")


def _validate_review_collection(
    *,
    label: str,
    review_items: list[dict[str, Any]],
    context_ids: set[str],
    id_key: str,
    regions: set[tuple[str, str]],
    claim_ids: set[str],
    errors: list[str],
) -> None:
    seen = [item.get(id_key) for item in review_items]
    missing = sorted(context_ids - set(seen))
    extra = sorted(set(seen) - context_ids)
    duplicates = _counter_duplicates([value for value in seen if value])
    for item_id in missing:
        errors.append(f"missing {label} review: {item_id}")
    for item_id in extra:
        errors.append(f"{label} review references unknown item: {item_id}")
    for item_id in duplicates:
        errors.append(f"duplicate {label} review: {item_id}")
    for item in review_items:
        item_id = item.get(id_key, "<missing>")
        status = item.get("status")
        if status not in VALID_REVIEW_STATUS:
            errors.append(f"{label} {item_id}: invalid or missing status")
        if status in {"rendered_detail", "rendered_aggregate"}:
            if not item.get("page_refs"):
                errors.append(f"{label} {item_id}: rendered items require page_refs")
            _validate_page_refs(f"{label} {item_id}", item.get("page_refs", []), regions, errors)
        if status in {"excluded_with_reason", "unresolved"} and not str(item.get("reason", "")).strip():
            errors.append(f"{label} {item_id}: {status} requires reason")
        for claim_id in item.get("claim_ids", []):
            if claim_id not in claim_ids:
                errors.append(f"{label} {item_id}: unknown claim_id {claim_id}")


def summarize_coverage(plan: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    class_reviews = plan.get("class_review", [])
    method_reviews = plan.get("method_review", [])
    function_reviews = plan.get("function_review", [])
    manifest = plan.get("coverage_manifest", {})
    core_methods = [item for item in context.get("methods", []) if item.get("importance_candidate") == "core"]
    detected_capabilities = [
        value
        for value in context.get("capability_signals", {}).values()
        if value.get("detected")
    ]
    required_branches = [
        item
        for item in context.get("branches", [])
        if item.get("architecture_relevance_candidate") in {"high", "medium"}
    ]
    return {
        "classes": {
            "total": len(context.get("classes", [])),
            "unreviewed": sum(1 for item in class_reviews if not item.get("status")),
            "rendered_detail": sum(1 for item in class_reviews if item.get("status") == "rendered_detail"),
            "rendered_aggregate": sum(1 for item in class_reviews if item.get("status") == "rendered_aggregate"),
            "documented_external": sum(1 for item in class_reviews if item.get("status") == "documented_external"),
            "excluded": sum(1 for item in class_reviews if item.get("status") == "excluded_with_reason"),
            "unresolved": sum(1 for item in class_reviews if item.get("status") == "unresolved"),
        },
        "methods": {
            "total": len(context.get("methods", [])),
            "core": len(core_methods),
            "unreviewed": sum(1 for item in method_reviews if not item.get("status")),
        },
        "functions": {
            "total": len(context.get("module_functions", [])),
            "unreviewed": sum(1 for item in function_reviews if not item.get("status")),
        },
        "branches": {
            "required": len(required_branches),
            "covered": len(manifest.get("branches", [])),
        },
        "weight_mappings": {
            "total": len(context.get("weight_mappings", [])),
            "covered": len(manifest.get("weight_mappings", [])),
        },
        "capabilities": {
            "detected": len(detected_capabilities),
            "covered": len(manifest.get("capabilities", [])),
        },
        "pages": len(plan.get("pages", [])),
    }


def validate_plan(
    plan: dict[str, Any],
    evidence: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    repo_root = repo_root.resolve() if repo_root else Path.cwd().resolve()
    schema_version = plan.get("schema_version")
    if schema_version not in {"2.1", "2.2"}:
        errors.append("plan.schema_version must be 2.1 or 2.2")
    integrated_contract = schema_version == "2.2"
    diagram_strategy = plan.get("diagram_strategy")
    if integrated_contract:
        if diagram_strategy not in VALID_DIAGRAM_STRATEGIES:
            errors.append("plan.diagram_strategy is invalid or missing")
        if not str(plan.get("integration_rationale", "")).strip():
            errors.append("plan.integration_rationale must explain the selected diagram strategy")
    if plan.get("detail_level") not in {"overview", "standard", "complete"}:
        errors.append("plan.detail_level must be overview, standard, or complete")
    if not plan.get("model_name"):
        errors.append("plan.model_name must be non-empty")
    if not plan.get("target_file"):
        errors.append("plan.target_file must be non-empty")

    if context:
        if plan.get("source_sha256") != context.get("source_sha256"):
            errors.append("plan.source_sha256 must match context.source_sha256")
        if plan.get("target_file") != context.get("target", {}).get("target_file"):
            errors.append("plan.target_file must match context target file")

    files_read = plan.get("files_read")
    if not isinstance(files_read, list) or not files_read:
        errors.append("plan.files_read must list the files Codex read")
    elif plan.get("target_file") not in files_read:
        errors.append("plan.files_read must include target_file")

    pages = plan.get("pages")
    if not isinstance(pages, list):
        errors.append("plan.pages must be a list")
        pages = []
    page_budget = plan.get("page_budget", {})
    max_pages = page_budget.get("max_pages")
    if not isinstance(max_pages, int) or max_pages > 5 or max_pages < 1:
        errors.append("plan.page_budget.max_pages must be an integer from 1 to 5")
        max_pages = 5
    if len(pages) > max_pages:
        errors.append("plan pages exceed page_budget.max_pages")
    if integrated_contract and diagram_strategy == "integrated_single_canvas":
        if page_budget.get("target_pages") != 1 or max_pages != 1:
            errors.append(
                "integrated_single_canvas requires page_budget.target_pages=1 and max_pages=1"
            )
        if len(pages) != 1:
            errors.append("integrated_single_canvas plans must contain exactly one page")
    elif integrated_contract and diagram_strategy == "compact_boundary":
        if not (1 <= len(pages) <= 2):
            errors.append("compact_boundary plans must contain 1 to 2 pages")
    elif context:
        status = context.get("classification", {}).get("status")
        if status == "full" and not (2 <= len(pages) <= 5):
            errors.append("composite full-model plans must contain 2 to 5 pages")
        elif status == "partial" and not (1 <= len(pages) <= 5):
            errors.append("partial model plans must contain 1 to 5 pages")
        elif status in {"helper", "boundary-only"} and not (1 <= len(pages) <= 2):
            errors.append("helper or boundary-only plans must contain 1 to 2 pages")
    elif not (1 <= len(pages) <= 5):
        errors.append("plans must contain 1 to 5 pages")

    evidence_claims = {
        claim.get("id"): claim
        for claim in evidence.get("claims", [])
        if claim.get("id")
    } if evidence else {}
    claim_ids = set(evidence_claims)
    if evidence and plan.get("global_boundaries"):
        external_claims = [
            claim
            for claim in evidence.get("claims", [])
            if claim.get("confidence") == "external"
        ]
        if not external_claims:
            errors.append("plan declares global_boundaries but evidence contains no external claim")
    seen_pages: set[str] = set()
    known_source_ids = _known_source_item_ids(context or {})
    for index, page in enumerate(pages):
        page_id = page.get("id")
        if not page_id:
            errors.append(f"page[{index}] is missing id")
            continue
        if page_id in seen_pages:
            errors.append(f"duplicate page id: {page_id}")
        seen_pages.add(page_id)
        for field in ("title", "export_name", "question", "purpose"):
            if not str(page.get(field, "")).strip():
                errors.append(f"{page_id}: {field} must be non-empty")
        topics = page.get("topics", [])
        if not isinstance(topics, list) or not topics:
            errors.append(f"{page_id}: topics must be a non-empty list")
        for topic in topics:
            if topic not in VALID_TOPICS:
                errors.append(f"{page_id}: invalid topic {topic!r}")
        pattern = page.get("view_pattern")
        if pattern not in VALID_PATTERNS:
            errors.append(f"{page_id}: invalid view_pattern {pattern!r}")
        main_story = page.get("main_story")
        if not isinstance(main_story, list):
            errors.append(f"{page_id}: main_story must be a list")
        elif pattern in MIN_MAIN_STORY_STEPS and len(main_story) < MIN_MAIN_STORY_STEPS[pattern]:
            errors.append(
                f"{page_id}: {pattern} main_story must contain at least "
                f"{MIN_MAIN_STORY_STEPS[pattern]} concrete ordered stages"
            )
        page_claims = page.get("claim_ids")
        if not isinstance(page_claims, list) or not page_claims:
            errors.append(f"{page_id}: claim_ids must be non-empty")
        elif evidence:
            for claim_id in page_claims:
                if claim_id not in claim_ids:
                    errors.append(f"{page_id}: claim_id does not exist in evidence: {claim_id}")
        regions = page.get("detail_regions")
        if not isinstance(regions, list) or not regions:
            errors.append(f"{page_id}: detail_regions must contain at least one region")
            regions = []
        elif integrated_contract and len(regions) < 3:
            errors.append(f"{page_id}: integrated pages require at least three detail regions")
        region_ids: set[str] = set()
        for region in regions:
            region_id = region.get("id")
            if not region_id:
                errors.append(f"{page_id}: detail region missing id")
                continue
            if region_id in region_ids:
                errors.append(f"{page_id}: duplicate detail region id {region_id}")
            region_ids.add(region_id)
            if region.get("type") not in VALID_REGION_TYPES:
                errors.append(f"{page_id}.{region_id}: invalid detail region type")
            for field in ("title", "question", "visual_form"):
                if not str(region.get(field, "")).strip():
                    errors.append(f"{page_id}.{region_id}: {field} must be non-empty")
            for source_id in region.get("source_item_ids", []):
                if context and source_id not in known_source_ids:
                    errors.append(f"{page_id}.{region_id}: unknown source_item_id {source_id}")
            for claim_id in region.get("claim_ids", []):
                if evidence and claim_id not in claim_ids:
                    errors.append(f"{page_id}.{region_id}: unknown claim_id {claim_id}")
        if integrated_contract:
            _validate_visual_contract(
                page_id=page_id,
                page=page,
                known_source_ids=known_source_ids,
                claim_ids=claim_ids,
                evidence_claims=evidence_claims,
                context_available=context is not None,
                evidence_available=evidence is not None,
                errors=errors,
                warnings=warnings,
            )
        budget = page.get("detail_budget", {})
        for budget_key in ("max_primary_nodes", "max_total_nodes", "max_visible_edges"):
            value = budget.get(budget_key)
            if not isinstance(value, int) or value < 1:
                errors.append(f"{page_id}: detail_budget.{budget_key} must be a positive integer")
        if integrated_contract and diagram_strategy == "integrated_single_canvas":
            if budget.get("max_total_nodes", 0) > 100:
                errors.append(f"{page_id}: integrated max_total_nodes above 100 is too dense")
            elif budget.get("max_total_nodes", 0) > 84:
                warnings.append(f"{page_id}: integrated max_total_nodes above 84 requires careful review")
            if budget.get("max_visible_edges", 0) > 140:
                errors.append(f"{page_id}: integrated max_visible_edges above 140 is too dense")
            elif budget.get("max_visible_edges", 0) > 110:
                warnings.append(f"{page_id}: integrated max_visible_edges above 110 requires careful review")
        else:
            if budget.get("max_total_nodes", 0) > 32:
                errors.append(f"{page_id}: max_total_nodes above 32 is too dense")
            elif budget.get("max_total_nodes", 0) > 28:
                warnings.append(f"{page_id}: max_total_nodes above 28 may be hard to read")
            if budget.get("max_visible_edges", 0) > 32:
                errors.append(f"{page_id}: max_visible_edges above 32 is too dense")
            elif budget.get("max_visible_edges", 0) > 26:
                warnings.append(f"{page_id}: max_visible_edges above 26 may be hard to read")

    if context:
        capability_signals = context.get("capability_signals", {})
        for page in pages:
            for capability, topic in CAPABILITY_TOPIC_MAP.items():
                if topic in page.get("topics", []) and capability_signals.get(capability, {}).get("detected") is False:
                    errors.append(f"{page.get('id')}: topic {topic} conflicts with Source Context capability {capability}=false")

    pages_by_id, regions = _page_maps(plan)
    class_ids = _ids((context or {}).get("classes", []), "class_id")
    method_ids = _ids((context or {}).get("methods", []), "method_id")
    function_ids = _ids((context or {}).get("module_functions", []), "function_id")

    _validate_review_collection(
        label="class",
        review_items=plan.get("class_review", []),
        context_ids=class_ids,
        id_key="class_id",
        regions=regions,
        claim_ids=claim_ids,
        errors=errors,
    )
    for item in plan.get("class_review", []):
        item_id = item.get("class_id", "<missing>")
        if item.get("role") not in VALID_CLASS_ROLES:
            errors.append(f"class {item_id}: invalid or missing role")
        if item.get("role") == "primary_component" and item.get("status") == "excluded_with_reason":
            errors.append(f"class {item_id}: primary_component cannot be excluded")

    _validate_review_collection(
        label="method",
        review_items=plan.get("method_review", []),
        context_ids=method_ids,
        id_key="method_id",
        regions=regions,
        claim_ids=claim_ids,
        errors=errors,
    )
    context_methods = {item["method_id"]: item for item in (context or {}).get("methods", [])}
    for item in plan.get("method_review", []):
        item_id = item.get("method_id", "<missing>")
        if item.get("importance") not in VALID_IMPORTANCE:
            errors.append(f"method {item_id}: invalid or missing importance")
        method = context_methods.get(item_id, {})
        is_core = method.get("importance_candidate") == "core" or item.get("importance") == "core"
        if is_core:
            if item.get("status") not in {"rendered_detail", "rendered_aggregate"}:
                errors.append(f"method {item_id}: core methods must be rendered_detail or rendered_aggregate")
            if not item.get("claim_ids"):
                errors.append(f"method {item_id}: core methods require at least one claim_id")
            if not item.get("page_refs"):
                errors.append(f"method {item_id}: core methods require page_refs")

    _validate_review_collection(
        label="function",
        review_items=plan.get("function_review", []),
        context_ids=function_ids,
        id_key="function_id",
        regions=regions,
        claim_ids=claim_ids,
        errors=errors,
    )
    for item in plan.get("function_review", []):
        item_id = item.get("function_id", "<missing>")
        if item.get("importance") not in VALID_IMPORTANCE:
            errors.append(f"function {item_id}: invalid or missing importance")

    manifest = plan.get("coverage_manifest", {})
    for manifest_key, context_key, id_key, required_filter in (
        (
            "branches",
            "branches",
            "branch_id",
            lambda item: item.get("architecture_relevance_candidate") in {"high", "medium"},
        ),
        ("weight_mappings", "weight_mappings", "mapping_id", lambda item: True),
    ):
        required_ids = {
            item[id_key]
            for item in (context or {}).get(context_key, [])
            if required_filter(item)
        }
        manifest_items = manifest.get(manifest_key, [])
        seen = [item.get("item_id") for item in manifest_items]
        for item_id in sorted(required_ids - set(seen)):
            errors.append(f"coverage_manifest.{manifest_key} missing required item {item_id}")
        for item_id in _counter_duplicates([value for value in seen if value]):
            errors.append(f"coverage_manifest.{manifest_key} duplicate item {item_id}")
        for item in manifest_items:
            item_id = item.get("item_id", "<missing>")
            if item.get("status") not in VALID_REVIEW_STATUS:
                errors.append(f"coverage_manifest.{manifest_key} {item_id}: invalid or missing status")
            if item.get("status") in {"rendered_detail", "rendered_aggregate"}:
                if not item.get("page_refs"):
                    errors.append(f"coverage_manifest.{manifest_key} {item_id}: rendered items require page_refs")
                _validate_page_refs(f"coverage_manifest.{manifest_key} {item_id}", item.get("page_refs", []), regions, errors)
            if item.get("status") in {"excluded_with_reason", "unresolved"} and not str(item.get("reason", "")).strip():
                errors.append(f"coverage_manifest.{manifest_key} {item_id}: reason required")
            for claim_id in item.get("claim_ids", []):
                if claim_id not in claim_ids:
                    errors.append(f"coverage_manifest.{manifest_key} {item_id}: unknown claim_id {claim_id}")

    capability_required = {
        (signal.get("capability_id") or f"capability:{name}")
        for name, signal in (context or {}).get("capability_signals", {}).items()
        if signal.get("detected")
    }
    capability_seen = [item.get("item_id") for item in manifest.get("capabilities", [])]
    for item_id in sorted(capability_required - set(capability_seen)):
        errors.append(f"coverage_manifest.capabilities missing detected capability {item_id}")
    for item in manifest.get("capabilities", []):
        item_id = item.get("item_id", "<missing>")
        if item.get("status") not in VALID_REVIEW_STATUS:
            errors.append(f"coverage_manifest.capabilities {item_id}: invalid or missing status")
        if item.get("status") in {"excluded_with_reason", "unresolved"} and not str(item.get("reason", "")).strip():
            errors.append(f"coverage_manifest.capabilities {item_id}: reason required")

    if integrated_contract:
        anchored_source_ids: set[str] = set()
        region_source_ids: set[str] = set()
        for page in pages:
            for anchor in page.get("visual_contract", {}).get("required_anchors", []):
                anchored_source_ids.update(anchor.get("source_item_ids", []))
            for region in page.get("detail_regions", []):
                region_source_ids.update(region.get("source_item_ids", []))

        for label, review_items, id_key in (
            ("class", plan.get("class_review", []), "class_id"),
            ("method", plan.get("method_review", []), "method_id"),
            ("function", plan.get("function_review", []), "function_id"),
        ):
            for item in review_items:
                item_id = item.get(id_key)
                if item.get("status") == "rendered_detail" and item_id not in anchored_source_ids:
                    errors.append(
                        f"{label} {item_id}: rendered_detail must be represented by a required visual anchor"
                    )
                if (
                    item.get("status") == "rendered_aggregate"
                    and item_id not in anchored_source_ids
                    and item_id not in region_source_ids
                ):
                    errors.append(
                        f"{label} {item_id}: rendered_aggregate must appear in a detail region or visual anchor"
                    )

        for manifest_key in ("branches", "weight_mappings", "capabilities"):
            for item in manifest.get(manifest_key, []):
                item_id = item.get("item_id")
                if item.get("status") == "rendered_detail" and item_id not in anchored_source_ids:
                    errors.append(
                        f"coverage_manifest.{manifest_key} {item_id}: "
                        "rendered_detail must be represented by a required visual anchor"
                    )
                if (
                    item.get("status") == "rendered_aggregate"
                    and item_id not in anchored_source_ids
                    and item_id not in region_source_ids
                ):
                    errors.append(
                        f"coverage_manifest.{manifest_key} {item_id}: "
                        "rendered_aggregate must appear in a detail region or visual anchor"
                    )

        complete_full_model = bool(
            context
            and plan.get("detail_level") == "complete"
            and diagram_strategy == "integrated_single_canvas"
            and context.get("classification", {}).get("status") == "full"
        )
        if complete_full_model:
            class_reviews = plan.get("class_review", [])
            class_detail_count = sum(
                item.get("status") == "rendered_detail"
                for item in class_reviews
            )
            required_class_details = min(
                len(class_reviews),
                3,
                max(1, math.ceil(len(class_reviews) * 0.15)),
            )
            if class_detail_count < required_class_details:
                errors.append(
                    "complete integrated plan renders too few classes in detail: "
                    f"{class_detail_count} < {required_class_details}"
                )

            core_review_items = [
                item
                for item in plan.get("method_review", [])
                if item.get("importance") == "core"
            ]
            detailed_core_methods = [
                item
                for item in core_review_items
                if item.get("status") == "rendered_detail"
            ]
            required_core_details = min(
                len(core_review_items),
                8,
                max(2, math.ceil(len(core_review_items) * 0.20)),
            )
            if len(detailed_core_methods) < required_core_details:
                errors.append(
                    "complete integrated plan renders too few core methods in detail: "
                    f"{len(detailed_core_methods)} < {required_core_details}"
                )

            core_by_region: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for item in core_review_items:
                for page_ref in item.get("page_refs", []):
                    key = (page_ref.get("page_id", ""), page_ref.get("region_id", ""))
                    core_by_region.setdefault(key, []).append(item)
            for (ref_page, ref_region), items in core_by_region.items():
                if len(items) < 4:
                    continue
                required_region_details = min(
                    4,
                    max(1, math.ceil(len(items) * 0.15)),
                )
                actual_region_details = sum(
                    item.get("status") == "rendered_detail" for item in items
                )
                if actual_region_details < required_region_details:
                    errors.append(
                        f"{ref_page}.{ref_region}: aggregates {len(items)} core methods "
                        f"but renders only {actual_region_details} in detail; "
                        f"minimum is {required_region_details}"
                    )

            high_branch_ids = {
                item.get("branch_id")
                for item in context.get("branches", [])
                if item.get("architecture_relevance_candidate") == "high"
            }
            high_branch_reviews = [
                item
                for item in manifest.get("branches", [])
                if item.get("item_id") in high_branch_ids
            ]
            required_branch_details = min(
                len(high_branch_reviews),
                6,
                max(1, math.ceil(len(high_branch_reviews) * 0.10)),
            )
            actual_branch_details = sum(
                item.get("status") == "rendered_detail"
                for item in high_branch_reviews
            )
            if actual_branch_details < required_branch_details:
                errors.append(
                    "complete integrated plan renders too few high-relevance branches "
                    f"in detail: {actual_branch_details} < {required_branch_details}"
                )

            mapping_reviews = manifest.get("weight_mappings", [])
            if mapping_reviews:
                required_mapping_details = min(3, len(mapping_reviews))
                actual_mapping_details = sum(
                    item.get("status") == "rendered_detail"
                    for item in mapping_reviews
                )
                if actual_mapping_details < required_mapping_details:
                    errors.append(
                        "complete integrated plan renders too few weight-mapping groups "
                        f"in detail: {actual_mapping_details} < {required_mapping_details}"
                    )

            anchor_map_by_page = {
                page.get("id"): {
                    anchor.get("id"): anchor
                    for anchor in page.get("visual_contract", {}).get(
                        "required_anchors", []
                    )
                }
                for page in pages
            }
            relationship_map_by_page = {
                page.get("id"): page.get("visual_contract", {}).get(
                    "required_relationships", []
                )
                for page in pages
            }
            class_sources_by_group: dict[str, set[str]] = {}
            for group_name, categories in SUBSYSTEM_CATEGORY_GROUPS.items():
                class_sources_by_group[group_name] = {
                    item.get("class_id")
                    for item in context.get("classes", [])
                    if categories
                    & set(item.get("category_candidates", []))
                }
            for group_name, source_ids in class_sources_by_group.items():
                if not source_ids:
                    continue
                matching_regions: list[tuple[dict[str, Any], dict[str, Any]]] = []
                for page in pages:
                    for region in page.get("detail_regions", []):
                        if (
                            region.get("type") in STRUCTURAL_REGION_TYPES
                            and source_ids & set(region.get("source_item_ids", []))
                        ):
                            matching_regions.append((page, region))
                if not matching_regions:
                    errors.append(
                        f"complete integrated plan has no structural detail region for "
                        f"the detected {group_name} subsystem"
                    )
                    continue
                sufficiently_expanded = False
                for page, region in matching_regions:
                    region_anchor_ids = set(
                        region.get("integration", {}).get("anchor_ids", [])
                    )
                    page_anchors = anchor_map_by_page.get(page.get("id"), {})
                    semantic_anchors = {
                        anchor_id
                        for anchor_id in region_anchor_ids
                        if page_anchors.get(anchor_id, {}).get("representation")
                        not in {"annotation", "badge"}
                    }
                    internal_relationships = [
                        relationship
                        for relationship in relationship_map_by_page.get(
                            page.get("id"), []
                        )
                        if relationship.get("source_anchor_id") in semantic_anchors
                        and relationship.get("target_anchor_id") in semantic_anchors
                    ]
                    if len(semantic_anchors) >= 4 and len(internal_relationships) >= 3:
                        sufficiently_expanded = True
                        break
                if not sufficiently_expanded:
                    errors.append(
                        f"detected {group_name} subsystem must be expanded with at least "
                        "four semantic anchors and three internal relationships"
                    )

    if evidence and files_read:
        files_read_resolved = {str(resolve_repo_path(repo_root, path).resolve()) for path in files_read}
        for claim in evidence.get("claims", []):
            for entry in claim.get("evidence", []):
                if str(resolve_repo_path(repo_root, entry.get("file", "")).resolve()) not in files_read_resolved:
                    errors.append(f"claim {claim.get('id')} evidence file is not listed in plan.files_read: {entry.get('file')}")
    if len({page.get("purpose") for page in pages}) < len(pages):
        warnings.append("some page purposes are duplicated")
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)

    plan = _load_json(args.plan)
    evidence = _load_json(args.evidence) if args.evidence else None
    context = _load_json(args.context) if args.context else None
    errors, warnings = validate_plan(plan, evidence=evidence, context=context, repo_root=args.repo_root)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Architecture plan validation passed")
    if context:
        print(json.dumps(summarize_coverage(plan, context), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
