from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "src" / "skills" / "vllm-model-architecture-diagram"
SCRIPTS = SKILL_ROOT / "scripts"
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"


def load_module(filename: str, name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pipeline_artifacts(tmp_path: Path) -> dict[str, Any]:
    extractor = load_module("extract_architecture.py", "extract_architecture_v09_tests")
    inventory_builder = load_module("build_semantic_inventory.py", "build_inventory_v09_tests")
    ir_builder = load_module("build_architecture_ir.py", "build_ir_v09_tests")
    coverage_validator = load_module("validate_semantic_coverage.py", "coverage_v09_tests")
    semantic_review_builder = load_module("build_semantic_review.py", "semantic_review_builder_v09_tests")
    semantic_review_validator = load_module("validate_semantic_review.py", "semantic_review_validator_v09_tests")
    ir_patch = load_module("apply_ir_patch.py", "apply_ir_patch_v09_tests")
    view_builder = load_module("build_diagram_view.py", "view_builder_v09_tests")
    layout_builder = load_module("layout_diagram.py", "layout_builder_v09_tests")
    renderer = load_module("render_drawio.py", "renderer_v09_tests")
    visual_review_builder = load_module("build_visual_review.py", "visual_review_builder_v09_tests")
    visual_review_validator = load_module("validate_visual_review.py", "visual_review_validator_v09_tests")
    view_patch = load_module("apply_view_patch.py", "apply_view_patch_v09_tests")

    analysis = extractor.extract_architecture(HY_V3_PATH)
    inventory = inventory_builder.build_semantic_inventory(analysis)
    baseline_ir = ir_builder.build_architecture_ir(analysis)
    baseline_ir_path = tmp_path / "baseline-ir.json"
    baseline_ir_path.write_text(json.dumps(baseline_ir, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    baseline_coverage, baseline_errors = coverage_validator.validate_semantic_coverage(analysis, inventory, baseline_ir)
    assert baseline_errors == []
    review, patch = semantic_review_builder.build_semantic_review(
        analysis,
        inventory,
        baseline_ir,
        baseline_coverage,
        source_path=HY_V3_PATH,
        baseline_ir_sha256=semantic_review_builder._sha256(baseline_ir_path),
        patch_file=str(tmp_path / "ir.patch.json"),
    )
    patch_path = tmp_path / "ir.patch.json"
    review_path = tmp_path / "semantic-review.json"
    patch_path.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert semantic_review_validator.validate_semantic_review(
        analysis,
        inventory,
        baseline_ir,
        baseline_ir_path,
        review,
        patch,
    ) == []
    reviewed_ir = ir_patch.apply_ir_patch(baseline_ir_path, patch_path)
    reviewed_coverage, reviewed_errors = coverage_validator.validate_semantic_coverage(
        analysis,
        inventory,
        reviewed_ir,
        review,
    )
    assert reviewed_errors == []
    baseline_view = view_builder.build_diagram_view(reviewed_ir)
    baseline_view_path = tmp_path / "baseline-view.json"
    baseline_view_path.write_text(json.dumps(baseline_view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    baseline_layout = layout_builder.build_layout_plan(baseline_view)
    metrics = {"pages": [{"id": page["id"]} for page in baseline_view["pages"]]}
    baseline_drawio = renderer.render_drawio(baseline_view, baseline_layout)
    drawio_path = tmp_path / "baseline.drawio"
    drawio_path.write_text(baseline_drawio, encoding="utf-8")
    visual_review, view_patch_doc = visual_review_builder.build_visual_review(
        reviewed_ir,
        baseline_view,
        baseline_layout,
        metrics,
        base_view_sha256=visual_review_builder._sha256(baseline_view_path),
        patch_file=str(tmp_path / "view.patch.json"),
    )
    view_patch_path = tmp_path / "view.patch.json"
    view_patch_path.write_text(json.dumps(view_patch_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert visual_review_validator.validate_visual_review(
        reviewed_ir,
        baseline_view,
        baseline_view_path,
        visual_review,
        view_patch_doc,
    ) == []
    reviewed_view = view_patch.apply_view_patch(baseline_view_path, view_patch_path)
    reviewed_layout = layout_builder.build_layout_plan(reviewed_view)
    reviewed_drawio = renderer.render_drawio(reviewed_view, reviewed_layout)
    return {
        "analysis": analysis,
        "inventory": inventory,
        "baseline_ir": baseline_ir,
        "baseline_coverage": baseline_coverage,
        "review": review,
        "patch": patch,
        "reviewed_ir": reviewed_ir,
        "reviewed_coverage": reviewed_coverage,
        "baseline_view": baseline_view,
        "visual_review": visual_review,
        "view_patch": view_patch_doc,
        "reviewed_view": reviewed_view,
        "reviewed_layout": reviewed_layout,
        "reviewed_drawio": reviewed_drawio,
        "paths": {
            "baseline_ir": baseline_ir_path,
            "patch": patch_path,
            "baseline_view": baseline_view_path,
            "view_patch": view_patch_path,
        },
    }


def test_semantic_review_disposes_every_required_fact(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    required = {
        fact["fact_id"]
        for fact in artifacts["inventory"]["facts"]
        if fact.get("relevance") == "required"
    }
    dispositions = [item["fact_id"] for item in artifacts["review"]["fact_dispositions"]]
    assert set(dispositions) == required
    assert len(dispositions) == len(set(dispositions))
    assert all(0 <= item["confidence"] <= 1 for item in artifacts["review"]["fact_dispositions"])
    assert all(item["target_ids"] for item in artifacts["review"]["fact_dispositions"] if item["disposition"] == "aggregated")


def test_reviewed_coverage_has_zero_required_unresolved_and_orphaned(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    summary = artifacts["reviewed_coverage"]["summary"]
    assert artifacts["reviewed_coverage"]["schema_version"] == "0.2"
    assert summary["required"]["unresolved"] == 0
    assert summary["required"]["orphaned"] == 0
    assert summary["aggregated_count"] > 0
    assert summary["rendered_direct_count"] > 0


def test_semantic_review_validator_rejects_duplicate_and_mass_excluded(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    validator = load_module("validate_semantic_review.py", "semantic_review_validator_negative_v09_tests")
    review = json.loads(json.dumps(artifacts["review"]))
    review["fact_dispositions"].append(dict(review["fact_dispositions"][0]))
    errors = validator.validate_semantic_review(
        artifacts["analysis"],
        artifacts["inventory"],
        artifacts["baseline_ir"],
        artifacts["paths"]["baseline_ir"],
        review,
        artifacts["patch"],
    )
    assert any("duplicate disposition" in error for error in errors)
    review = json.loads(json.dumps(artifacts["review"]))
    for item in review["fact_dispositions"]:
        item["disposition"] = "excluded"
        item["reason"] = "no diagram"
    errors = validator.validate_semantic_review(
        artifacts["analysis"],
        artifacts["inventory"],
        artifacts["baseline_ir"],
        artifacts["paths"]["baseline_ir"],
        review,
        artifacts["patch"],
    )
    assert any("too many required facts" in error for error in errors)


def test_ir_patch_apply_resolves_unresolved_and_is_deterministic(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    applier = load_module("apply_ir_patch.py", "apply_ir_patch_determinism_v09_tests")
    first = applier.apply_ir_patch(artifacts["paths"]["baseline_ir"], artifacts["paths"]["patch"])
    second = applier.apply_ir_patch(artifacts["paths"]["baseline_ir"], artifacts["paths"]["patch"])
    assert first == second
    assert first["review"]["applied_operations"] == ["op:resolve-external-config-values"]
    assert all(item.get("item") != "external_config_values" for item in first["unresolved"])


def test_ir_patch_rejects_hash_mismatch_and_external_to_direct(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    applier = load_module("apply_ir_patch.py", "apply_ir_patch_negative_v09_tests")
    bad_patch = dict(artifacts["patch"])
    bad_patch["base_ir_sha256"] = "0" * 64
    path = tmp_path / "bad.patch.json"
    path.write_text(json.dumps(bad_patch), encoding="utf-8")
    try:
        applier.apply_ir_patch(artifacts["paths"]["baseline_ir"], path)
    except ValueError as exc:
        assert "base_ir_sha256" in str(exc)
    else:
        raise AssertionError("hash mismatch should fail")

    patch = {
        "schema_version": "0.1",
        "base_ir_version": "0.6",
        "base_ir_sha256": artifacts["patch"]["base_ir_sha256"],
        "operations": [
            {
                "op_id": "op:external-direct",
                "op": "set_evidence_type",
                "page_id": "attention_detail",
                "item_type": "edge",
                "item_id": "kv_cache_read_to_attention",
                "evidence_type": "direct",
                "reason": "bad",
                "evidence_fact_ids": ["fact:HYV3Attention.forward:external:kv_cache_read"],
                "confidence": 0.9,
            }
        ],
    }
    path.write_text(json.dumps(patch), encoding="utf-8")
    try:
        applier.apply_ir_patch(artifacts["paths"]["baseline_ir"], path)
    except ValueError as exc:
        assert "external evidence" in str(exc)
    else:
        raise AssertionError("external evidence should not become direct")


def test_visual_review_patch_is_presentation_only_and_deterministic(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    assert artifacts["visual_review"]["summary"]["finding_count"] >= 1
    assert artifacts["view_patch"]["operations"]
    assert artifacts["reviewed_view"] == load_module("apply_view_patch.py", "apply_view_patch_determinism_v09_tests").apply_view_patch(
        artifacts["paths"]["baseline_view"],
        artifacts["paths"]["view_patch"],
    )
    attention = next(page for page in artifacts["reviewed_view"]["pages"] if page["id"] == "attention_detail")
    assert any(annotation["id"] == "decorative_note_attention_external_boundary" for annotation in attention["annotations"])
    assert "decorative_note_attention_external_boundary" in artifacts["reviewed_drawio"]


def test_visual_review_validator_rejects_semantic_and_hash_changes(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    validator = load_module("validate_visual_review.py", "visual_review_validator_negative_v09_tests")
    bad_patch = dict(artifacts["view_patch"])
    bad_patch["base_view_sha256"] = "0" * 64
    errors = validator.validate_visual_review(
        artifacts["reviewed_ir"],
        artifacts["baseline_view"],
        artifacts["paths"]["baseline_view"],
        artifacts["visual_review"],
        bad_patch,
    )
    assert any("base_view_sha256" in error for error in errors)
    bad_patch = {"schema_version": "0.1", "base_view_sha256": artifacts["view_patch"]["base_view_sha256"], "operations": [{"op_id": "bad", "op": "add_edge", "reason": "bad", "confidence": 0.9}]}
    errors = validator.validate_visual_review(
        artifacts["reviewed_ir"],
        artifacts["baseline_view"],
        artifacts["paths"]["baseline_view"],
        artifacts["visual_review"],
        bad_patch,
    )
    assert any("forbidden semantic operation" in error for error in errors)


def test_review_lock_detects_stale_source(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    lock_builder = load_module("build_review_lock.py", "build_review_lock_v09_tests")
    source_analysis = tmp_path / "source-analysis.json"
    inventory = tmp_path / "inventory.json"
    reviewed_ir = tmp_path / "reviewed-ir.json"
    visual_review = tmp_path / "visual-review.json"
    reviewed_view = tmp_path / "reviewed-view.json"
    for path, value in [
        (source_analysis, artifacts["analysis"]),
        (inventory, artifacts["inventory"]),
        (reviewed_ir, artifacts["reviewed_ir"]),
        (visual_review, artifacts["visual_review"]),
        (reviewed_view, artifacts["reviewed_view"]),
    ]:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lock = lock_builder.build_lock(
        {
            "source": HY_V3_PATH,
            "source_analysis": source_analysis,
            "semantic_inventory": inventory,
            "baseline_ir": artifacts["paths"]["baseline_ir"],
            "semantic_review": tmp_path / "semantic-review.json",
            "ir_patch": artifacts["paths"]["patch"],
            "reviewed_ir": reviewed_ir,
            "visual_review": visual_review,
            "view_patch": artifacts["paths"]["view_patch"],
            "reviewed_view": reviewed_view,
        }
    )
    assert lock["source_sha256"]
    mutated_source = tmp_path / "hy_v3.py"
    mutated_source.write_text(HY_V3_PATH.read_text(encoding="utf-8") + "\n# mutation\n", encoding="utf-8")
    assert lock["source_sha256"] != lock_builder._sha256(mutated_source)
