from __future__ import annotations

import importlib.util
import struct
import xml.etree.ElementTree as ET
import zlib
from copy import deepcopy
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


def make_context(tmp_path: Path, architecture: str = "DenseForCausalLM"):
    repo = make_repo(tmp_path)
    resolver = load_script("resolve_model_target")
    collector = load_script("collect_source_context")
    context = collector.collect_source_context(repo, resolver.resolve_model_target(repo, architecture=architecture))
    return repo, context


def _ref(page_id: str, region_id: str) -> list[dict[str, str]]:
    return [{"page_id": page_id, "region_id": region_id}]


def valid_evidence(context: dict) -> dict:
    target = context["target"]["target_file"]
    first_method = context["methods"][0]["method_id"]
    second_method = context["methods"][1]["method_id"]
    return {
        "schema_version": "2.1",
        "target_file": target,
        "source_sha256": context["source_sha256"],
        "claims": [
            {
                "id": "claim:model-runtime",
                "statement": "The model defines a runtime path through local model methods.",
                "category": "runtime",
                "phase": "runtime",
                "confidence": "direct",
                "derivation": "",
                "external_boundary": "",
                "source_item_ids": [first_method],
                "evidence": [
                    {
                        "file": target,
                        "start_line": 22,
                        "end_line": 26,
                        "symbol": "forward",
                        "reason": "forward embeds input IDs and returns normalized hidden states",
                    }
                ],
                "used_in_pages": ["overview"],
            },
            {
                "id": "claim:construction",
                "statement": "The model constructs embedding, layer and output components.",
                "category": "construction",
                "phase": "construction",
                "confidence": "derived",
                "derivation": "Constructor assignments and wrapper construction combine into the model component structure.",
                "source_item_ids": [first_method, second_method],
                "external_boundary": "",
                "evidence": [
                    {
                        "file": target,
                        "start_line": 17,
                        "end_line": 20,
                        "symbol": "make_layers",
                        "reason": "DenseModel constructs the repeated layer stack",
                    },
                    {
                        "file": target,
                        "start_line": 30,
                        "end_line": 33,
                        "symbol": "self.model",
                        "reason": "Wrapper constructs the base model and output components",
                    },
                ],
                "used_in_pages": ["composition"],
            },
            {
                "id": "claim:external-logits",
                "statement": "LogitsProcessor internals are imported and external to this file.",
                "category": "boundary",
                "phase": "external",
                "confidence": "external",
                "derivation": "",
                "external_boundary": "External boundary: local source imports LogitsProcessor but does not define its internals.",
                "source_item_ids": [context["classes"][-1]["class_id"]],
                "evidence": [
                    {
                        "file": target,
                        "start_line": 1,
                        "end_line": 1,
                        "symbol": "LogitsProcessor",
                        "reason": "imported component boundary",
                    }
                ],
                "used_in_pages": ["boundary"],
            },
        ],
    }


def valid_plan(context: dict) -> dict:
    pages = [
        {
            "id": "overview",
            "title": "Model Overview",
            "export_name": "model-overview",
            "question": "How does the model run?",
            "purpose": "Show the primary runtime path.",
            "topics": ["model_execution"],
            "view_pattern": "pipeline",
            "scope": ["forward"],
            "claim_ids": ["claim:model-runtime"],
            "main_story": ["input", "embedding", "model", "normalization", "output"],
            "detail_regions": [
                {
                    "id": "runtime",
                    "title": "Runtime Path",
                    "type": "embedded_subgraph",
                    "question": "How does forward move hidden states?",
                    "source_item_ids": [context["methods"][0]["method_id"]],
                    "claim_ids": ["claim:model-runtime"],
                    "visual_form": "left-to-right flow",
                }
            ],
            "secondary_topics": [],
            "external_boundaries": [],
            "visual_decisions": [],
            "detail_budget": {"max_primary_nodes": 8, "max_total_nodes": 18, "max_visible_edges": 16},
        },
        {
            "id": "composition",
            "title": "Composition",
            "export_name": "composition",
            "question": "Which components are constructed?",
            "purpose": "Show local component structure.",
            "topics": ["model_composition"],
            "view_pattern": "component_map",
            "scope": ["__init__"],
            "claim_ids": ["claim:construction"],
            "main_story": [],
            "detail_regions": [
                {
                    "id": "components",
                    "title": "Component Tree",
                    "type": "component_tree",
                    "question": "How are local modules contained?",
                    "source_item_ids": [context["classes"][0]["class_id"]],
                    "claim_ids": ["claim:construction"],
                    "visual_form": "nested component tree",
                }
            ],
            "secondary_topics": [],
            "external_boundaries": [],
            "visual_decisions": [],
            "detail_budget": {"max_primary_nodes": 8, "max_total_nodes": 18, "max_visible_edges": 16},
        },
        {
            "id": "boundary",
            "title": "Adapter Boundary",
            "export_name": "adapter-boundary",
            "question": "Which parts are external?",
            "purpose": "Show imported runtime boundaries.",
            "topics": ["external_boundary"],
            "view_pattern": "boundary_map",
            "scope": ["imports"],
            "claim_ids": ["claim:external-logits"],
            "main_story": [],
            "detail_regions": [
                {
                    "id": "external",
                    "title": "External Boundary",
                    "type": "boundary_panel",
                    "question": "Which imported behavior is not local?",
                    "source_item_ids": [context["classes"][-1]["class_id"]],
                    "claim_ids": ["claim:external-logits"],
                    "visual_form": "local/external boundary",
                }
            ],
            "secondary_topics": [],
            "external_boundaries": ["LogitsProcessor"],
            "visual_decisions": [],
            "detail_budget": {"max_primary_nodes": 5, "max_total_nodes": 10, "max_visible_edges": 8},
        },
    ]
    plan = {
        "schema_version": "2.1",
        "model_name": "dense",
        "target_file": context["target"]["target_file"],
        "source_sha256": context["source_sha256"],
        "detail_level": "complete",
        "model_category": context["classification"]["category_candidates"],
        "files_read": [context["target"]["target_file"]],
        "traversal_notes": [{"file": context["target"]["target_file"], "reason": "target file reviewed"}],
        "page_budget": {"target_pages": 4, "max_pages": 5},
        "pages": pages,
        "class_review": [],
        "method_review": [],
        "function_review": [],
        "coverage_manifest": {"branches": [], "weight_mappings": [], "capabilities": [], "external_boundaries": []},
        "global_boundaries": ["LogitsProcessor"],
        "omissions": [],
        "unresolved": [],
    }
    for cls in context["classes"]:
        plan["class_review"].append(
            {
                "class_id": cls["class_id"],
                "role": "primary_component" if cls["qualified_name"].endswith("ForCausalLM") else "supporting_component",
                "status": "rendered_aggregate",
                "page_refs": _ref("composition", "components"),
                "claim_ids": ["claim:construction"],
                "reason": "Reviewed and aggregated into composition page.",
            }
        )
    for method in context["methods"]:
        plan["method_review"].append(
            {
                "method_id": method["method_id"],
                "importance": method["importance_candidate"],
                "status": "rendered_aggregate",
                "page_refs": _ref("overview", "runtime"),
                "claim_ids": ["claim:model-runtime"],
                "reason": "Reviewed and aggregated into runtime/composition regions.",
            }
        )
    for function in context["module_functions"]:
        plan["function_review"].append(
            {
                "function_id": function["function_id"],
                "importance": function["importance_candidate"],
                "status": "excluded_with_reason",
                "page_refs": [],
                "claim_ids": [],
                "reason": "No architecture effect for this plan.",
            }
        )
    for name, signal in context["capability_signals"].items():
        if signal.get("detected"):
            plan["coverage_manifest"]["capabilities"].append(
                {
                    "item_id": signal.get("capability_id") or f"capability:{name}",
                    "status": "rendered_aggregate",
                    "page_refs": _ref("composition", "components"),
                    "claim_ids": ["claim:construction"],
                    "reason": "Capability shown as badge or side annotation.",
                }
            )
    return plan


def test_plan_and_evidence_validators_pass(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    plan = valid_plan(context)

    evidence_mod = load_script("validate_evidence")
    plan_mod = load_script("validate_architecture_plan")
    errors, _, summary = evidence_mod.validate_evidence(evidence, context=context, plan=plan, repo_root=repo)
    assert errors == []
    assert summary == {"direct": 1, "derived": 1, "external": 1}
    plan_errors, _ = plan_mod.validate_plan(plan, evidence=evidence, context=context, repo_root=repo)
    assert plan_errors == []
    coverage = plan_mod.summarize_coverage(plan, context)
    assert coverage["classes"]["unreviewed"] == 0
    assert coverage["methods"]["unreviewed"] == 0


def test_direct_import_only_evidence_fails(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
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
    errors, _, _ = evidence_mod.validate_evidence(evidence, context=context, plan=valid_plan(context), repo_root=repo)
    assert any("import lines" in error for error in errors)


def test_completeness_validator_catches_missing_method_review(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    plan = valid_plan(context)
    plan["method_review"] = plan["method_review"][:-1]

    plan_mod = load_script("validate_architecture_plan")
    errors, _ = plan_mod.validate_plan(plan, evidence=evidence, context=context, repo_root=repo)
    assert any("missing method review" in error for error in errors)


def test_core_method_cannot_be_excluded(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    plan = valid_plan(context)
    plan["method_review"][0]["status"] = "excluded_with_reason"
    plan["method_review"][0]["reason"] = "bad idea"

    plan_mod = load_script("validate_architecture_plan")
    errors, _ = plan_mod.validate_plan(plan, evidence=evidence, context=context, repo_root=repo)
    assert any("core methods must be rendered" in error for error in errors)


def test_bad_page_ref_fails(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    plan = valid_plan(context)
    plan["class_review"][0]["page_refs"] = _ref("overview", "missing")

    plan_mod = load_script("validate_architecture_plan")
    errors, _ = plan_mod.validate_plan(plan, evidence=evidence, context=context, repo_root=repo)
    assert any("unknown page/region" in error for error in errors)


def test_source_hash_mismatch_fails(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    plan = valid_plan(context)
    plan["source_sha256"] = "not-the-source"

    plan_mod = load_script("validate_architecture_plan")
    evidence_mod = load_script("validate_evidence")
    errors, _ = plan_mod.validate_plan(plan, evidence=evidence, context=context, repo_root=repo)
    assert any("source_sha256" in error for error in errors)
    evidence_errors, _, _ = evidence_mod.validate_evidence(evidence, context=context, plan=plan, repo_root=repo)
    assert any("source_sha256" in error for error in evidence_errors)


def test_branch_mapping_capability_manifest_required(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    resolver = load_script("resolve_model_target")
    collector = load_script("collect_source_context")
    target = repo / "vllm" / "model_executor" / "models" / "complex_branch_adapter.py"
    context = collector.collect_source_context(repo, resolver.resolve_model_target(repo, input_path=target))
    evidence = {
        "schema_version": "2.1",
        "target_file": context["target"]["target_file"],
        "source_sha256": context["source_sha256"],
        "claims": [
            {
                "id": "claim:branch",
                "statement": "Branchy fixture has architecture branches.",
                "category": "runtime",
                "phase": "runtime",
                "confidence": "direct",
                "derivation": "",
                "external_boundary": "",
                "source_item_ids": [context["branches"][0]["branch_id"]],
                "evidence": [
                    {
                        "file": context["target"]["target_file"],
                        "start_line": context["branches"][0]["start_line"],
                        "end_line": context["branches"][0]["start_line"],
                        "symbol": "if",
                        "reason": "branch exists",
                    }
                ],
                "used_in_pages": ["overview"],
            }
        ],
    }
    plan = valid_plan(context)
    plan["pages"] = plan["pages"][:3]
    plan["pages"][0]["detail_regions"][0]["source_item_ids"] = [context["branches"][0]["branch_id"]]
    plan["pages"][0]["detail_regions"][0]["claim_ids"] = ["claim:branch"]
    plan["pages"][0]["claim_ids"] = ["claim:branch"]
    plan["method_review"] = [
        {
            "method_id": method["method_id"],
            "importance": method["importance_candidate"],
            "status": "rendered_aggregate",
            "page_refs": _ref("overview", "runtime"),
            "claim_ids": ["claim:branch"],
            "reason": "reviewed",
        }
        for method in context["methods"]
    ]
    plan["class_review"] = [
        {
            "class_id": cls["class_id"],
            "role": "primary_component",
            "status": "rendered_aggregate",
            "page_refs": _ref("overview", "runtime"),
            "claim_ids": ["claim:branch"],
            "reason": "reviewed",
        }
        for cls in context["classes"]
    ]
    plan["coverage_manifest"] = {"branches": [], "weight_mappings": [], "capabilities": [], "external_boundaries": []}
    plan_mod = load_script("validate_architecture_plan")
    errors, _ = plan_mod.validate_plan(plan, evidence=evidence, context=context, repo_root=repo)
    assert any("coverage_manifest.branches missing" in error for error in errors)
    assert any("coverage_manifest.weight_mappings missing" in error for error in errors)


def test_flow_page_requires_ordered_main_story(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    plan = valid_plan(context)
    plan["pages"][0]["main_story"] = ["runtime summary"]

    validator = load_script("validate_architecture_plan")
    errors, _ = validator.validate_plan(
        plan,
        evidence=evidence,
        context=context,
        repo_root=repo,
    )

    assert any("main_story must contain at least 5 concrete ordered stages" in error for error in errors)


def test_declared_external_boundary_requires_external_evidence(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    evidence["claims"] = [
        claim
        for claim in evidence["claims"]
        if claim["confidence"] != "external"
    ]
    plan = valid_plan(context)

    validator = load_script("validate_architecture_plan")
    errors, _ = validator.validate_plan(
        plan,
        evidence=evidence,
        context=context,
        repo_root=repo,
    )

    assert any("global_boundaries but evidence contains no external claim" in error for error in errors)


def write_drawio(
    path: Path,
    page_names: list[str],
    *,
    html: bool = False,
    background: str = "#ffffff",
    include_edges: bool = True,
    questions: dict[str, str] | None = None,
) -> None:
    diagrams = []
    for index, name in enumerate(page_names, start=1):
        html_style = "html=1;" if html else "html=0;"
        values = "Model Overview Runtime Path Composition Component Tree Adapter Boundary External Boundary"
        prefix = name.lower().replace(" ", "_")
        question = (questions or {}).get(name, "What engineering question does this page answer?")
        edges = ""
        if include_edges:
            edges = (
                f'<mxCell id="{prefix}_edge_1" edge="1" source="{prefix}_a" target="{prefix}_b" '
                'parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>'
                f'<mxCell id="{prefix}_edge_2" edge="1" source="{prefix}_b" target="{prefix}_c" '
                'parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>'
                f'<mxCell id="{prefix}_edge_3" edge="1" source="{prefix}_c" target="{prefix}_d" '
                'parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>'
                f'<mxCell id="{prefix}_edge_4" edge="1" source="{prefix}_d" target="{prefix}_e" '
                'parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>'
                f'<mxCell id="{prefix}_edge_5" edge="1" source="{prefix}_e" target="{prefix}_f" '
                'parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>'
                f'<mxCell id="{prefix}_edge_6" edge="1" source="{prefix}_f" target="{prefix}_g" '
                'parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>'
                f'<mxCell id="{prefix}_edge_7" edge="1" source="{prefix}_g" target="{prefix}_h" '
                'parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>'
            )
        diagrams.append(
            f'''<diagram id="page-{index}" name="{name}"><mxGraphModel background="{background}"><root><mxCell id="0"/><mxCell id="1" parent="0"/><mxCell id="{prefix}_a" value="{name}" style="{html_style}rounded=1;fillColor=#ffffff;strokeColor=#ffffff;" vertex="1" parent="1"><mxGeometry x="40" y="20" width="180" height="40" as="geometry"/></mxCell><mxCell id="{prefix}_q" value="{question}" style="html=0;rounded=0;fillColor=#ffffff;strokeColor=#ffffff;" vertex="1" parent="1"><mxGeometry x="40" y="65" width="420" height="30" as="geometry"/></mxCell><mxCell id="{prefix}_b" value="{values}" style="html=0;rounded=1;fillColor=#e7f5ff;strokeColor=#2b7dbd;" vertex="1" parent="1"><mxGeometry x="40" y="120" width="200" height="70" as="geometry"/></mxCell><mxCell id="{prefix}_c" value="C" style="html=0;rounded=1;fillColor=#eaf7ea;strokeColor=#4b9d58;" vertex="1" parent="1"><mxGeometry x="270" y="120" width="120" height="50" as="geometry"/></mxCell><mxCell id="{prefix}_d" value="D" style="html=0;rounded=1;fillColor=#e7f5ff;strokeColor=#2b7dbd;" vertex="1" parent="1"><mxGeometry x="420" y="120" width="120" height="50" as="geometry"/></mxCell><mxCell id="{prefix}_e" value="E" style="html=0;rounded=1;fillColor=#eaf7ea;strokeColor=#4b9d58;" vertex="1" parent="1"><mxGeometry x="570" y="120" width="120" height="50" as="geometry"/></mxCell><mxCell id="{prefix}_f" value="F" style="html=0;rounded=1;fillColor=#e7f5ff;strokeColor=#2b7dbd;" vertex="1" parent="1"><mxGeometry x="720" y="120" width="120" height="50" as="geometry"/></mxCell><mxCell id="{prefix}_g" value="G" style="html=0;rounded=1;fillColor=#eaf7ea;strokeColor=#4b9d58;" vertex="1" parent="1"><mxGeometry x="870" y="120" width="120" height="50" as="geometry"/></mxCell><mxCell id="{prefix}_h" value="H" style="html=0;rounded=1;fillColor=#e7f5ff;strokeColor=#2b7dbd;" vertex="1" parent="1"><mxGeometry x="1020" y="120" width="120" height="50" as="geometry"/></mxCell><mxCell id="{prefix}_i" value="I" style="html=0;rounded=1;fillColor=#eaf7ea;strokeColor=#4b9d58;" vertex="1" parent="1"><mxGeometry x="1170" y="120" width="120" height="50" as="geometry"/></mxCell>{edges}</root></mxGraphModel></diagram>'''
        )
    path.write_text(f'<mxfile host="app.diagrams.net">{"".join(diagrams)}</mxfile>', encoding="utf-8")


def write_png(path: Path, width: int = 640, height: int = 360) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    scanline = b"\x00" + (b"\xff\xff\xff" * width)
    image_data = zlib.compress(scanline * height)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", image_data)
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_visual_review(path: Path, page_names: list[str]) -> None:
    pages = "\n".join(
        f"- {name}: Draft issue identified, revised through MCP, and accepted in final review."
        for name in page_names
    )
    path.write_text(
        "# Visual Review\n\n"
        "## Draft 1 Findings\n"
        f"{pages}\n\n"
        "## Revision Round 1\n"
        "Geometry changes: moved and resized groups, rerouted long edges, and "
        "adjusted spacing after opening every PNG export. Labels and semantic "
        "styles were then refined without changing evidence-backed semantics.\n\n"
        "## Final Review\n"
        "All pages were re-exported and inspected. Architecture story, detail coverage, "
        "semantic distinction, topology, readability and information density meet the rubric.\n",
        encoding="utf-8",
    )


def test_drawio_validator_checks_pages_regions_and_html(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = valid_plan(context)
    drawio = tmp_path / "architecture.drawio"
    page_names = [page["title"] for page in plan["pages"]]
    questions = {page["title"]: page["question"] for page in plan["pages"]}
    write_drawio(drawio, page_names, questions=questions)
    images = tmp_path / "images"
    images.mkdir()
    for page in plan["pages"]:
        write_png(images / f"{page['export_name']}.png")
    write_visual_review(tmp_path / "visual-review.md", page_names)
    validator = load_script("validate_drawio")
    assert validator.validate_drawio(drawio, plan=plan, images_dir=images) == []

    bad = tmp_path / "bad.drawio"
    write_drawio(bad, page_names, html=True, questions=questions)
    assert any("html=1" in error for error in validator.validate_drawio(bad, plan=plan))


def test_drawio_validator_fails_missing_region_title_and_png(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = valid_plan(context)
    drawio = tmp_path / "architecture.drawio"
    page_names = [page["title"] for page in plan["pages"]]
    questions = {page["title"]: page["question"] for page in plan["pages"]}
    write_drawio(drawio, page_names, questions=questions)
    images = tmp_path / "images"
    images.mkdir()
    write_visual_review(tmp_path / "visual-review.md", page_names)
    validator = load_script("validate_drawio")
    errors = validator.validate_drawio(drawio, plan=plan, images_dir=images)
    assert any("expected PNG export is missing" in error for error in errors)


def test_drawio_validator_fails_missing_detail_region_title(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = valid_plan(context)
    plan["pages"][0]["detail_regions"][0]["title"] = "Not Present In Diagram"
    drawio = tmp_path / "architecture.drawio"
    write_drawio(
        drawio,
        [page["title"] for page in plan["pages"]],
        questions={page["title"]: page["question"] for page in plan["pages"]},
    )
    validator = load_script("validate_drawio")
    errors = validator.validate_drawio(drawio, plan=plan)
    assert any("detail region title is missing" in error for error in errors)


def test_drawio_validator_rejects_placeholder_png_and_card_only_flow(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = valid_plan(context)
    drawio = tmp_path / "architecture.drawio"
    page_names = [page["title"] for page in plan["pages"]]
    write_drawio(
        drawio,
        page_names,
        include_edges=False,
        questions={page["title"]: page["question"] for page in plan["pages"]},
    )
    images = tmp_path / "images"
    images.mkdir()
    for page in plan["pages"]:
        (images / f"{page['export_name']}.png").write_bytes(b"")
    write_visual_review(tmp_path / "visual-review.md", page_names)

    validator = load_script("validate_drawio")
    errors = validator.validate_drawio(drawio, plan=plan, images_dir=images)

    assert any("flow-oriented page has" in error for error in errors)
    assert any("PNG export is empty" in error for error in errors)


def test_drawio_validator_rejects_sparse_single_style_pages(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = valid_plan(context)
    drawio = tmp_path / "architecture.drawio"
    page_names = [page["title"] for page in plan["pages"]]
    write_drawio(
        drawio,
        page_names,
        questions={page["title"]: page["question"] for page in plan["pages"]},
    )

    tree = ET.parse(drawio)
    for diagram in tree.getroot().findall("diagram"):
        graph_root = diagram.find("./mxGraphModel/root")
        assert graph_root is not None
        for cell in list(graph_root):
            cell_id = cell.get("id", "")
            if cell_id.endswith(("_f", "_g", "_h", "_i")):
                graph_root.remove(cell)
            elif cell.get("vertex") == "1":
                cell.set("style", "html=0;rounded=1;")
    tree.write(drawio, encoding="utf-8")

    validator = load_script("validate_drawio")
    errors = validator.validate_drawio(drawio, plan=plan)

    assert any("minimum is 8" in error for error in errors)
    assert any("at least two visual treatments" in error for error in errors)


def test_drawio_validator_requires_visual_review_with_exports(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = valid_plan(context)
    drawio = tmp_path / "architecture.drawio"
    page_names = [page["title"] for page in plan["pages"]]
    write_drawio(
        drawio,
        page_names,
        questions={page["title"]: page["question"] for page in plan["pages"]},
    )
    images = tmp_path / "images"
    images.mkdir()
    for page in plan["pages"]:
        write_png(images / f"{page['export_name']}.png")

    validator = load_script("validate_drawio")
    errors = validator.validate_drawio(drawio, plan=plan, images_dir=images)

    assert any("visual review does not exist" in error for error in errors)
