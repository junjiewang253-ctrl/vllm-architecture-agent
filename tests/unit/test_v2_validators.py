from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from test_v2_registry_and_context import make_repo


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "src" / "skills" / "vllm-model-architecture-diagram" / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_context(tmp_path: Path):
    repo = make_repo(tmp_path)
    resolver = load_script("resolve_model_target")
    collector = load_script("collect_source_context")
    context = collector.collect_source_context(repo, resolver.resolve_model_target(repo, architecture="DenseForCausalLM"))
    return repo, context


def valid_evidence(context: dict) -> dict:
    target = context["target"]["target_file"]
    return {
        "schema_version": "2.0",
        "target_file": target,
        "source_sha256": context["source_sha256"],
        "claims": [
            {
                "id": "claim:model-flow",
                "statement": "The wrapper delegates forward execution to the base model.",
                "category": "runtime",
                "confidence": "direct",
                "evidence": [
                    {
                        "file": target,
                        "start_line": 33,
                        "end_line": 34,
                        "symbol": "self.model",
                        "reason": "forward calls the local base model",
                    }
                ],
                "used_in_pages": ["overview"],
            },
            {
                "id": "claim:parallel-output",
                "statement": "Output logits use a parallel head and logits processor.",
                "category": "runtime",
                "confidence": "derived",
                "evidence": [
                    {
                        "file": target,
                        "start_line": 29,
                        "end_line": 30,
                        "symbol": "lm_head",
                        "reason": "constructor creates output head",
                    },
                    {
                        "file": target,
                        "start_line": 36,
                        "end_line": 37,
                        "symbol": "logits_processor",
                        "reason": "compute_logits passes hidden states to logits processor",
                    },
                ],
                "used_in_pages": ["overview"],
            },
            {
                "id": "claim:external-logits",
                "statement": "The logits processor implementation is external.",
                "category": "boundary",
                "confidence": "external",
                "external_boundary": "External imported logits processor implementation is not analyzed here.",
                "evidence": [
                    {
                        "file": target,
                        "start_line": 1,
                        "end_line": 1,
                        "symbol": "LogitsProcessor",
                        "reason": "local file imports an external component",
                    }
                ],
                "used_in_pages": ["boundary"],
            },
        ],
    }


def valid_plan(context: dict) -> dict:
    return {
        "schema_version": "2.0",
        "model_name": "dense",
        "target_file": context["target"]["target_file"],
        "model_category": context["classification"]["category_candidates"],
        "files_read": [context["target"]["target_file"]],
        "pages": [
            {
                "id": "overview",
                "title": "Model Overview",
                "question": "How does the model run?",
                "purpose": "Show the primary runtime path.",
                "view_pattern": "pipeline",
                "scope": ["DenseForCausalLM.forward"],
                "claim_ids": ["claim:model-flow", "claim:parallel-output"],
                "main_story": ["input", "model", "logits"],
                "secondary_topics": [],
                "external_boundaries": [],
                "visual_decisions": [],
                "detail_budget": {
                    "max_primary_nodes": 8,
                    "max_total_nodes": 12,
                    "max_visible_edges": 12,
                },
            },
            {
                "id": "boundary",
                "title": "Adapter Boundary",
                "question": "Which imported components are external?",
                "purpose": "Separate local claims from imported behavior.",
                "view_pattern": "boundary_map",
                "scope": ["DenseForCausalLM"],
                "claim_ids": ["claim:external-logits"],
                "main_story": [],
                "secondary_topics": ["external logits processor"],
                "external_boundaries": ["LogitsProcessor"],
                "visual_decisions": [],
                "detail_budget": {
                    "max_primary_nodes": 4,
                    "max_total_nodes": 8,
                    "max_visible_edges": 8,
                },
            },
        ],
        "global_boundaries": ["LogitsProcessor"],
        "omissions": [],
        "unresolved": [],
    }


def test_plan_and_evidence_validators_pass(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    plan = valid_plan(context)

    evidence_mod = load_script("validate_evidence")
    plan_mod = load_script("validate_architecture_plan")
    errors, warnings, summary = evidence_mod.validate_evidence(evidence, context=context, plan=plan)
    assert errors == []
    assert summary == {"direct": 1, "derived": 1, "external": 1}
    plan_errors, _ = plan_mod.validate_plan(plan, evidence=evidence, context=context)
    assert plan_errors == []


def test_direct_import_only_evidence_fails(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    evidence["claims"][0]["evidence"] = [
        {
            "file": context["target"]["target_file"],
            "start_line": 1,
            "end_line": 1,
            "symbol": "LogitsProcessor",
            "reason": "import line only",
        }
    ]

    evidence_mod = load_script("validate_evidence")
    errors, _, _ = evidence_mod.validate_evidence(evidence, context=context)
    assert any("import lines" in error for error in errors)


def test_plan_rejects_unsupported_moe_page(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    plan = valid_plan(context)
    plan["pages"][0]["id"] = "moe"
    plan["pages"][0]["title"] = "MoE Detail"

    plan_mod = load_script("validate_architecture_plan")
    errors, _ = plan_mod.validate_plan(plan, evidence=evidence, context=context)
    assert any("MoE page conflicts" in error for error in errors)


def write_drawio(path: Path, page_names: list[str], *, html: bool = False, background: str = "#ffffff") -> None:
    diagrams = []
    for index, name in enumerate(page_names, start=1):
        html_style = "html=1;" if html else "html=0;"
        diagrams.append(
            f'''<diagram id="page-{index}" name="{name}"><mxGraphModel background="{background}"><root><mxCell id="0"/><mxCell id="1" parent="0"/><mxCell id="{name.lower().replace(' ', '_')}_a" value="A" style="{html_style}rounded=1;" vertex="1" parent="1"><mxGeometry x="40" y="40" width="120" height="50" as="geometry"/></mxCell><mxCell id="{name.lower().replace(' ', '_')}_b" value="B" style="html=0;rounded=1;" vertex="1" parent="1"><mxGeometry x="200" y="40" width="120" height="50" as="geometry"/></mxCell><mxCell id="{name.lower().replace(' ', '_')}_c" value="C" style="html=0;rounded=1;" vertex="1" parent="1"><mxGeometry x="360" y="40" width="120" height="50" as="geometry"/></mxCell></root></mxGraphModel></diagram>'''
        )
    path.write_text(f'<mxfile host="app.diagrams.net">{"".join(diagrams)}</mxfile>', encoding="utf-8")


def test_drawio_validator_checks_pages_and_html(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = valid_plan(context)
    drawio = tmp_path / "architecture.drawio"
    write_drawio(drawio, ["Model Overview", "Adapter Boundary"])
    validator = load_script("validate_drawio")
    assert validator.validate_drawio(drawio, plan=plan) == []

    bad = tmp_path / "bad.drawio"
    write_drawio(bad, ["Model Overview", "Adapter Boundary"], html=True)
    assert any("html=1" in error for error in validator.validate_drawio(bad, plan=plan))


def test_drawio_validator_fails_missing_plan_page(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = valid_plan(context)
    drawio = tmp_path / "architecture.drawio"
    write_drawio(drawio, ["Model Overview"])
    validator = load_script("validate_drawio")
    errors = validator.validate_drawio(drawio, plan=plan)
    assert any("missing from drawio" in error for error in errors)
