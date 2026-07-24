from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "src" / "skills" / "vllm-model-architecture-diagram"
SCRIPTS = SKILL_ROOT / "scripts"
CLI_PATH = ROOT / "src" / "vllm_architecture_agent" / "cli.py"
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


def _codex_semantic_mock(review: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_review": {
            "model_name": review["model_name"],
            "findings": [
                {
                    "id": finding["finding_id"],
                    "type": finding["type"],
                    "severity": finding["severity"],
                    "description": finding["description"],
                    "evidence_fact_ids": finding["evidence_fact_ids"],
                    "confidence": finding["confidence"],
                    "patch_op_ids": finding.get("patch_op_ids", []),
                }
                for finding in review["architecture_findings"]
            ],
            "fact_dispositions": [
                {
                    "fact_id": item["fact_id"],
                    "status": item["disposition"],
                    "reason": item["reason"],
                    "target_ids": item["target_ids"],
                    "supporting_fact_ids": item["supporting_fact_ids"],
                    "confidence": item["confidence"],
                    "external_symbol": item.get("external_symbol"),
                }
                for item in review["fact_dispositions"]
            ],
        },
        "architecture_ir_patch": patch,
    }


def _codex_visual_mock(review: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    return {
        "visual_review": {
            "model_name": review["model_name"],
            "findings": [
                {
                    "id": finding["finding_id"],
                    "page_id": finding["page_id"],
                    "type": finding["type"],
                    "severity": finding["severity"],
                    "affected_ids": finding["affected_ids"],
                    "description": finding["description"],
                    "recommended_action": finding["recommended_action"],
                    "semantic_change": finding["semantic_change"],
                    "patch_op_ids": finding.get("patch_op_ids", []),
                }
                for finding in review["findings"]
            ],
        },
        "diagram_view_patch": patch,
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


def test_semantic_review_runner_uses_mock_codex_and_saves_patch(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    runner = load_module("run_semantic_review.py", "run_semantic_review_v091_tests")
    (tmp_path / "source-analysis.json").write_text(json.dumps(artifacts["analysis"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (tmp_path / "inventory.json").write_text(json.dumps(artifacts["inventory"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (tmp_path / "baseline-coverage.json").write_text(json.dumps(artifacts["baseline_coverage"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mock = tmp_path / "mock-semantic.md"
    mock.write_text(
        "Codex review result\n```json\n"
        + json.dumps(_codex_semantic_mock(artifacts["review"], artifacts["patch"]))
        + "\n```\n",
        encoding="utf-8",
    )
    review_out = tmp_path / "runner-semantic-review.json"
    patch_out = tmp_path / "runner-ir.patch.json"
    rc = runner.main(
        [
            str(tmp_path / "source-analysis.json"),
            str(tmp_path / "inventory.json"),
            str(artifacts["paths"]["baseline_ir"]),
            str(tmp_path / "baseline-coverage.json"),
            "--source-file",
            str(HY_V3_PATH),
            "--review-output",
            str(review_out),
            "--patch-output",
            str(patch_out),
            "--mock-response",
            str(mock),
        ]
    )
    assert rc == 0
    saved_review = json.loads(review_out.read_text(encoding="utf-8"))
    saved_patch = json.loads(patch_out.read_text(encoding="utf-8"))
    assert saved_review["summary"]["codex_review"] is True
    assert saved_patch["base_ir_sha256"] == artifacts["patch"]["base_ir_sha256"]


def test_semantic_review_runner_invalid_json_writes_failure(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    runner = load_module("run_semantic_review.py", "run_semantic_review_failure_v091_tests")
    (tmp_path / "source-analysis.json").write_text(json.dumps(artifacts["analysis"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (tmp_path / "inventory.json").write_text(json.dumps(artifacts["inventory"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (tmp_path / "baseline-coverage.json").write_text(json.dumps(artifacts["baseline_coverage"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mock = tmp_path / "bad-response.txt"
    mock.write_text("not json", encoding="utf-8")
    failure = tmp_path / "review_failed.json"
    rc = runner.main(
        [
            str(tmp_path / "source-analysis.json"),
            str(tmp_path / "inventory.json"),
            str(artifacts["paths"]["baseline_ir"]),
            str(tmp_path / "baseline-coverage.json"),
            "--source-file",
            str(HY_V3_PATH),
            "--review-output",
            str(tmp_path / "unused-review.json"),
            "--patch-output",
            str(tmp_path / "unused.patch.json"),
            "--mock-response",
            str(mock),
            "--review-failed-output",
            str(failure),
        ]
    )
    assert rc == 1
    assert json.loads(failure.read_text(encoding="utf-8"))["status"] == "failed"
    assert not (tmp_path / "unused.patch.json").exists()


def test_visual_review_runner_uses_mock_codex_and_saves_view_patch(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    runner = load_module("run_visual_review.py", "run_visual_review_v091_tests")
    (tmp_path / "reviewed-ir.json").write_text(json.dumps(artifacts["reviewed_ir"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"pages": [{"id": "overview"}]}), encoding="utf-8")
    mock = tmp_path / "mock-visual.md"
    mock.write_text(
        "```json\n" + json.dumps(_codex_visual_mock(artifacts["visual_review"], artifacts["view_patch"])) + "\n```\n",
        encoding="utf-8",
    )
    review_out = tmp_path / "runner-visual-review.json"
    patch_out = tmp_path / "runner-view.patch.json"
    rc = runner.main(
        [
            str(tmp_path / "reviewed-ir.json"),
            str(artifacts["paths"]["baseline_view"]),
            str(metrics),
            "--review-output",
            str(review_out),
            "--patch-output",
            str(patch_out),
            "--mock-response",
            str(mock),
        ]
    )
    assert rc == 0
    saved_review = json.loads(review_out.read_text(encoding="utf-8"))
    saved_patch = json.loads(patch_out.read_text(encoding="utf-8"))
    assert saved_review["summary"]["codex_review"] is True
    assert saved_patch["base_view_sha256"] == artifacts["view_patch"]["base_view_sha256"]


def test_vllm_arch_reviewed_mode_uses_existing_semantic_artifacts(tmp_path: Path):
    artifacts = pipeline_artifacts(tmp_path)
    semantic_review = tmp_path / "semantic-review.lock.json"
    ir_patch = tmp_path / "architecture-ir.patch.lock.json"
    visual_mock = tmp_path / "visual-mock.json"
    semantic_review.write_text(json.dumps(artifacts["review"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ir_patch.write_text(json.dumps(artifacts["patch"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visual_mock.write_text(json.dumps(_codex_visual_mock(artifacts["visual_review"], artifacts["view_patch"])), encoding="utf-8")
    outputs = tmp_path / "cli-outputs"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "run",
            "--input",
            str(HY_V3_PATH),
            "--outputs-dir",
            str(outputs),
            "--model-name",
            "hy-v3-v091-cli",
            "--mode",
            "reviewed",
            "--semantic-review",
            str(semantic_review),
            "--ir-patch",
            str(ir_patch),
            "--mock-visual-review",
            str(visual_mock),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert (outputs / "hy-v3-v091-cli-architecture.drawio").exists()
    coverage = json.loads((outputs / "hy-v3-v091-cli-semantic-coverage.json").read_text(encoding="utf-8"))
    assert coverage["summary"]["required"]["unresolved"] == 0
    assert coverage["summary"]["required"]["orphaned"] == 0
    assert (outputs / "hy-v3-v091-cli-review-lock.json").exists()


def test_vllm_arch_reviewed_mode_requires_review_artifacts(tmp_path: Path):
    outputs = tmp_path / "cli-missing-review"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "run",
            "--input",
            str(HY_V3_PATH),
            "--outputs-dir",
            str(outputs),
            "--model-name",
            "hy-v3-v091-missing",
            "--mode",
            "reviewed",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "requires VSCode Codex to write" in completed.stderr
    assert not (outputs / "hy-v3-v091-missing-architecture.drawio").exists()
