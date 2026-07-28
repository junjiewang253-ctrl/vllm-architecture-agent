"""Validate v2.1 high-density Architecture Plans for source completeness."""

from __future__ import annotations

import argparse
import json
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
    if plan.get("schema_version") != "2.1":
        errors.append("plan.schema_version must be 2.1")
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
    if context:
        status = context.get("classification", {}).get("status")
        if status == "full" and not (3 <= len(pages) <= 5):
            errors.append("full model plans must contain 3 to 5 pages")
        elif status == "partial" and not (1 <= len(pages) <= 5):
            errors.append("partial model plans must contain 1 to 5 pages")
        elif status in {"helper", "boundary-only"} and not (1 <= len(pages) <= 2):
            errors.append("helper or boundary-only plans must contain 1 to 2 pages")
    elif not (1 <= len(pages) <= 5):
        errors.append("plans must contain 1 to 5 pages")

    claim_ids = {claim.get("id") for claim in evidence.get("claims", [])} if evidence else set()
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
        budget = page.get("detail_budget", {})
        for budget_key in ("max_primary_nodes", "max_total_nodes", "max_visible_edges"):
            value = budget.get(budget_key)
            if not isinstance(value, int) or value < 1:
                errors.append(f"{page_id}: detail_budget.{budget_key} must be a positive integer")
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
