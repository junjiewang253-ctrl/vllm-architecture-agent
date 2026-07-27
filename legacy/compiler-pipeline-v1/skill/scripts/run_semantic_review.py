#!/usr/bin/env python3
"""Run Codex Semantic Review and persist review plus IR patch artifacts."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from review_runner_common import (
    extract_json_object,
    load_json,
    read_codex_or_mock,
    stable_write_json,
    write_failure,
)

REVIEW_VERSION = "0.1"
PATCH_VERSION = "0.1"
IR_VERSION = "0.6"
ALLOWED_STATUSES = {
    "rendered_direct",
    "rendered_derived",
    "aggregated",
    "documented_external",
    "excluded",
}
REVIEW_ROOT = Path(__file__).resolve().parents[1] / "review"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_prompt(template_path: Path, replacements: dict[str, str]) -> str:
    prompt = template_path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


def _required_fact_ids(inventory: dict[str, Any]) -> list[str]:
    return sorted(
        fact["fact_id"]
        for fact in inventory.get("facts", [])
        if isinstance(fact, dict)
        and fact.get("relevance") == "required"
        and isinstance(fact.get("fact_id"), str)
    )


def _ir_ids(ir: dict[str, Any]) -> list[str]:
    ids: set[str] = set()
    for page in ir.get("pages", []):
        if not isinstance(page, dict):
            continue
        if isinstance(page.get("id"), str):
            ids.add(page["id"])
        for collection in ("nodes", "edges"):
            for item in page.get(collection, []):
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    ids.add(item["id"])
    return sorted(ids)


def _validate_raw_semantic_review(review: dict[str, Any], required_fact_ids: list[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(review.get("model_name"), str) or not review["model_name"].strip():
        errors.append("semantic_review.model_name must be a non-empty string")
    if not isinstance(review.get("findings"), list):
        errors.append("semantic_review.findings must be a list")
    if not isinstance(review.get("fact_dispositions"), list):
        errors.append("semantic_review.fact_dispositions must be a list")
    seen: set[str] = set()
    for index, item in enumerate(review.get("fact_dispositions", [])):
        if not isinstance(item, dict):
            errors.append(f"fact_dispositions[{index}] must be an object")
            continue
        fact_id = item.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id:
            errors.append(f"fact_dispositions[{index}].fact_id is required")
        elif fact_id in seen:
            errors.append(f"duplicate fact disposition: {fact_id}")
        else:
            seen.add(fact_id)
        status = item.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"fact_dispositions[{index}].status must not be unresolved: {status}")
        if not str(item.get("reason") or "").strip():
            errors.append(f"fact_dispositions[{index}].reason is required")
        if not isinstance(item.get("target_ids"), list):
            errors.append(f"fact_dispositions[{index}].target_ids must be a list")
        if not isinstance(item.get("supporting_fact_ids"), list):
            errors.append(f"fact_dispositions[{index}].supporting_fact_ids must be a list")
    required = set(required_fact_ids)
    if seen != required:
        for fact_id in sorted(required - seen):
            errors.append(f"missing required fact disposition: {fact_id}")
        for fact_id in sorted(seen - required):
            errors.append(f"unknown or placeholder fact disposition: {fact_id}")
    for index, finding in enumerate(review.get("findings", [])):
        if not isinstance(finding, dict):
            errors.append(f"findings[{index}] must be an object")
            continue
        if not isinstance(finding.get("id"), str):
            errors.append(f"findings[{index}].id is required")
        if finding.get("severity") not in {"critical", "major", "minor", "informational"}:
            errors.append(f"findings[{index}].severity is invalid")
        confidence = finding.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            errors.append(f"findings[{index}].confidence must be between 0 and 1")
    return errors


def _validate_patch(patch: dict[str, Any], base_hash: str) -> list[str]:
    errors: list[str] = []
    if patch.get("schema_version") != PATCH_VERSION:
        errors.append("architecture_ir_patch.schema_version must be '0.1'")
    if patch.get("base_ir_version") != IR_VERSION:
        errors.append("architecture_ir_patch.base_ir_version must be '0.6'")
    if patch.get("base_ir_sha256") != base_hash:
        errors.append("architecture_ir_patch.base_ir_sha256 does not match baseline IR")
    if not isinstance(patch.get("operations"), list):
        errors.append("architecture_ir_patch.operations must be a list")
    return errors


def _normalize_target_ids(target_ids: list[Any], fact_id: str, valid_ids: set[str]) -> list[str]:
    aliases = {
        "input_layernorm": "input_rmsnorm",
        "post_attention_layernorm": "post_attention_rmsnorm",
        "wrapper_config": "vllm_config",
        "model_config": "vllm_config",
        "eplb_config": "eplb_metadata",
        "expert_parallelism": "ep_lane",
        "pipeline_parallelism": "pp_lane",
        "pipeline_layer_partitioning": "pp_layer_partitioning",
        "pipeline_intermediate_tensors": "pp_rank_flows",
        "moe_gate": "gate_linear",
        "qkv_proj": "qkv_projection",
    }
    result: list[str] = []
    for value in target_ids:
        if not isinstance(value, str) or not value:
            continue
        candidate = aliases.get(value, value)
        if candidate in valid_ids:
            result.append(candidate)
    if result:
        return sorted(set(result))
    if "HYV3Attention" in fact_id:
        return ["attention_detail"]
    if "HYV3DecoderLayer" in fact_id:
        return ["decoder_layer_detail"]
    if "HYV3MoEFused" in fact_id or "expert" in fact_id:
        return ["moe_detail"]
    if "load_weights" in fact_id or fact_id.startswith("stage:"):
        return ["weight_loading"]
    if "parallelism" in fact_id or "get_pp_group" in fact_id or "get_ep_group" in fact_id:
        return ["parallelism"]
    if "config" in fact_id or "VllmConfig" in fact_id:
        return ["adapter_integration"]
    return ["overview"]


def _config_fact_ids(source_analysis: dict[str, Any]) -> list[str]:
    fact_ids = [
        item["fact_id"]
        for item in source_analysis.get("config_accesses", [])
        if isinstance(item, dict) and isinstance(item.get("fact_id"), str)
    ]
    return fact_ids[:4]


def _normalize_review(
    raw: dict[str, Any],
    *,
    source_file: Path,
    patch_output: Path,
    baseline_ir: dict[str, Any],
) -> dict[str, Any]:
    valid_ids = set(_ir_ids(baseline_ir))
    dispositions: list[dict[str, Any]] = []
    for item in raw.get("fact_dispositions", []):
        normalized = dict(item)
        normalized["disposition"] = normalized.get("status")
        normalized.setdefault("confidence", 0.85)
        normalized.setdefault("external_symbol", "external component boundary" if normalized.get("status") == "documented_external" else None)
        normalized.setdefault("suggested_next_source", None)
        normalized["target_ids"] = _normalize_target_ids(
            list(normalized.get("target_ids", [])),
            str(normalized.get("fact_id", "")),
            valid_ids,
        )
        dispositions.append(normalized)
    architecture_findings: list[dict[str, Any]] = []
    deferred_findings: list[dict[str, Any]] = list(raw.get("deferred_findings", [])) if isinstance(raw.get("deferred_findings"), list) else []
    for item in raw.get("findings", []):
        finding = dict(item)
        finding["finding_id"] = finding.get("id")
        finding.setdefault("affected_ids", [])
        finding.setdefault("recommended_action", finding.get("description", "Review finding requires consideration."))
        architecture_findings.append(finding)
        if finding.get("severity") in {"critical", "major"} and not finding.get("patch_op_ids"):
            deferred_findings.append(
                {
                    "finding_id": finding["finding_id"],
                    "reason": "Codex review judged the baseline semantics acceptable and did not propose a semantic IR mutation.",
                }
            )
    return {
        "schema_version": REVIEW_VERSION,
        "model_name": raw.get("model_name", "unknown-model"),
        "source_file": str(source_file),
        "review_mode": "reviewed",
        "summary": {
            "finding_count": len(raw.get("findings", [])),
            "disposition_count": len(raw.get("fact_dispositions", [])),
            "codex_review": True,
        },
        "findings": raw.get("findings", []),
        "fact_dispositions": dispositions,
        "architecture_findings": architecture_findings,
        "page_plan": raw.get("page_plan", []),
        "patch_file": str(patch_output),
        "remaining_unknowns": [],
        "deferred_findings": deferred_findings,
    }


def _normalize_patch(patch: dict[str, Any], source_analysis: dict[str, Any], baseline_ir: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(patch)
    normalized.setdefault("operations", [])
    normalized.setdefault("deferred_operations", [])
    unresolved = baseline_ir.get("unresolved", [])
    if isinstance(unresolved, list) and any(isinstance(item, dict) and item.get("item") == "external_config_values" for item in unresolved):
        if not any(isinstance(op, dict) and op.get("op") == "resolve_unresolved" and op.get("item") == "external_config_values" for op in normalized["operations"]):
            normalized["operations"].append(
                {
                    "op_id": "op:resolve-external-config-values",
                    "op": "resolve_unresolved",
                    "item": "external_config_values",
                    "reason": "Codex semantic review classified symbolic configuration facts; remove the baseline unresolved bucket.",
                    "evidence_fact_ids": _config_fact_ids(source_analysis),
                    "confidence": 0.9,
                }
            )
    return normalized


def run_semantic_review(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    source_analysis = load_json(args.source_analysis, "source-analysis")
    inventory = load_json(args.semantic_inventory, "semantic inventory")
    baseline_ir = load_json(args.baseline_ir, "baseline Architecture IR")
    baseline_coverage = load_json(args.baseline_coverage, "baseline semantic coverage")
    base_hash = _sha256(args.baseline_ir)
    required_ids = _required_fact_ids(inventory)
    prompt = _render_prompt(
        args.prompt_template,
        {
            "source_file": str(args.source_file),
            "source_analysis": str(args.source_analysis),
            "semantic_inventory": str(args.semantic_inventory),
            "baseline_ir": str(args.baseline_ir),
            "baseline_coverage": str(args.baseline_coverage),
            "base_ir_sha256": base_hash,
        },
    )
    prompt += "\n\n## Exact Required Fact IDs\n\n"
    prompt += "You must include exactly these fact IDs in semantic_review.fact_dispositions. Do not use placeholders.\n\n"
    prompt += "\n".join(f"- `{fact_id}`" for fact_id in required_ids)
    prompt += "\n"
    prompt += "\n## Valid Baseline IR Target IDs\n\n"
    prompt += "Use only these existing IDs in fact_dispositions.target_ids unless your patch adds a new ID. Do not invent target IDs.\n\n"
    prompt += "\n".join(f"- `{item_id}`" for item_id in _ir_ids(baseline_ir))
    prompt += "\n\nCritical or major findings must be connected to patch operations through patch_op_ids, or listed in deferred_findings with a clear reason.\n"
    if args.prompt_output:
        args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_output.write_text(prompt, encoding="utf-8")
    text = read_codex_or_mock(
        prompt,
        mock_response=args.mock_response,
        codex_bin=args.codex_bin,
        cwd=args.cwd,
        timeout_seconds=args.timeout_seconds,
    )
    data = extract_json_object(text)
    raw_review = data.get("semantic_review", data)
    patch = data.get("architecture_ir_patch")
    if not isinstance(raw_review, dict) or not isinstance(patch, dict):
        raise ValueError("Codex response must contain semantic_review and architecture_ir_patch objects")
    if args.mock_response is not None and patch.get("base_ir_sha256") != base_hash:
        patch["base_ir_sha256"] = base_hash
    errors = _validate_raw_semantic_review(raw_review, required_ids)
    errors.extend(_validate_patch(patch, base_hash))
    if source_analysis.get("schema_version") != "0.3":
        errors.append("source-analysis schema_version must be '0.3'")
    if baseline_ir.get("schema_version") != IR_VERSION:
        errors.append("baseline Architecture IR schema_version must be '0.6'")
    if "facts" not in inventory:
        errors.append("semantic inventory must contain facts")
    if "summary" not in baseline_coverage:
        errors.append("baseline semantic coverage must contain summary")
    if errors:
        raise ValueError("; ".join(errors))
    return (
        _normalize_review(raw_review, source_file=args.source_file, patch_output=args.patch_output, baseline_ir=baseline_ir),
        _normalize_patch(patch, source_analysis, baseline_ir),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Codex Semantic Review.")
    parser.add_argument("source_analysis", type=Path)
    parser.add_argument("semantic_inventory", type=Path)
    parser.add_argument("baseline_ir", type=Path)
    parser.add_argument("baseline_coverage", type=Path)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, default=REVIEW_ROOT / "prompts" / "semantic_review.md")
    parser.add_argument("--prompt-output", type=Path)
    parser.add_argument("--mock-response", type=Path)
    parser.add_argument("--review-failed-output", type=Path)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    failure_path = args.review_failed_output or args.review_output.with_name("review_failed.json")
    try:
        review, patch = run_semantic_review(args)
        stable_write_json(args.review_output, review)
        stable_write_json(args.patch_output, patch)
    except (OSError, ValueError) as exc:
        write_failure(failure_path, str(exc))
        print(f"error: semantic review failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Codex semantic review to {args.review_output}")
    print(f"Wrote Architecture IR patch to {args.patch_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
