from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import pytest

from test_v2_validators import (
    ROOT,
    load_script,
    make_context,
    valid_evidence,
    valid_plan,
)


def _integrated_plan(context: dict) -> dict:
    plan = valid_plan(context)
    all_source_ids = [
        *(item["class_id"] for item in context["classes"]),
        *(item["method_id"] for item in context["methods"]),
        *(item["function_id"] for item in context["module_functions"]),
        *(
            signal.get("capability_id") or f"capability:{name}"
            for name, signal in context["capability_signals"].items()
            if signal.get("detected")
        ),
    ]
    source_id = all_source_ids[0]
    anchors = [
        {
            "id": "visual:input",
            "label": "Model Input",
            "representation": "node",
            "role": "input",
            "phase": "runtime",
            "style_role": "runtime",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:model",
        },
        {
            "id": "visual:embedding",
            "label": "Embedding",
            "representation": "node",
            "role": "runtime_stage",
            "phase": "runtime",
            "style_role": "runtime",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:model",
        },
        {
            "id": "visual:decoder",
            "label": "Decoder Stack",
            "representation": "container",
            "role": "runtime_stage",
            "phase": "runtime",
            "style_role": "containment",
            "claim_ids": ["claim:construction"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:model",
        },
        {
            "id": "visual:block",
            "label": "Expanded Repeated Block",
            "representation": "container",
            "role": "runtime_stage",
            "phase": "runtime",
            "style_role": "containment",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:decoder",
        },
        {
            "id": "visual:norm",
            "label": "Final Norm",
            "representation": "node",
            "role": "runtime_stage",
            "phase": "runtime",
            "style_role": "runtime",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:model",
        },
        {
            "id": "visual:external-logits",
            "label": "External Logits Processor",
            "representation": "node",
            "role": "external_boundary",
            "phase": "external",
            "style_role": "external",
            "claim_ids": ["claim:external-logits"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:model",
        },
        {
            "id": "visual:output",
            "label": "Logits",
            "representation": "node",
            "role": "output",
            "phase": "runtime",
            "style_role": "runtime",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:model",
        },
        {
            "id": "visual:model",
            "label": "Local Model Adapter",
            "representation": "container",
            "role": "detail",
            "phase": "construction",
            "style_role": "containment",
            "claim_ids": ["claim:construction"],
            "source_item_ids": [source_id],
            "parent_anchor_id": None,
        },
        {
            "id": "visual:attention",
            "label": "Attention Adapter",
            "representation": "node",
            "role": "detail",
            "phase": "runtime",
            "style_role": "runtime",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:block",
        },
        {
            "id": "visual:ffn",
            "label": "Feed Forward",
            "representation": "node",
            "role": "detail",
            "phase": "runtime",
            "style_role": "runtime",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:block",
        },
        {
            "id": "visual:residual",
            "label": "Residual Bus",
            "representation": "node",
            "role": "detail",
            "phase": "runtime",
            "style_role": "residual",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:block",
        },
        {
            "id": "visual:config",
            "label": "Construction Configuration",
            "representation": "node",
            "role": "detail",
            "phase": "construction",
            "style_role": "construction",
            "claim_ids": ["claim:construction"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:decoder",
        },
        {
            "id": "visual:capability",
            "label": "Capabilities",
            "representation": "badge",
            "role": "capability",
            "phase": "capability",
            "style_role": "capability",
            "claim_ids": ["claim:construction"],
            "source_item_ids": [source_id],
            "parent_anchor_id": None,
        },
        {
            "id": "visual:branch",
            "label": "Input Branch",
            "representation": "node",
            "role": "branch",
            "phase": "runtime",
            "style_role": "runtime",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:model",
        },
        {
            "id": "visual:merge",
            "label": "Merge",
            "representation": "merge",
            "role": "merge",
            "phase": "runtime",
            "style_role": "runtime",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:model",
        },
        {
            "id": "visual:parallel",
            "label": "Parallel Metadata",
            "representation": "node",
            "role": "strategy",
            "phase": "parallel",
            "style_role": "metadata",
            "claim_ids": ["claim:construction"],
            "source_item_ids": [source_id],
            "parent_anchor_id": None,
        },
        {
            "id": "visual:boundary-note",
            "label": "External Boundary Note",
            "representation": "annotation",
            "role": "annotation",
            "phase": "external",
            "style_role": "external",
            "claim_ids": ["claim:external-logits"],
            "source_item_ids": [source_id],
            "parent_anchor_id": None,
        },
        {
            "id": "visual:runtime-detail",
            "label": "Runtime Detail",
            "representation": "node",
            "role": "detail",
            "phase": "runtime",
            "style_role": "runtime",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:block",
        },
        {
            "id": "visual:output-projection",
            "label": "Output Projection",
            "representation": "node",
            "role": "detail",
            "phase": "runtime",
            "style_role": "runtime",
            "claim_ids": ["claim:model-runtime"],
            "source_item_ids": [source_id],
            "parent_anchor_id": "visual:block",
        },
    ]
    relationships = [
        ("visual-rel:input-embedding", "visual:input", "visual:embedding", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:embedding-decoder", "visual:embedding", "visual:decoder", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:decoder-block", "visual:decoder", "visual:block", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:block-norm", "visual:block", "visual:norm", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:norm-external", "visual:norm", "visual:external-logits", "external_delegation", "claim:external-logits"),
        ("visual-rel:external-output", "visual:external-logits", "visual:output", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:block-attention", "visual:block", "visual:attention", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:attention-ffn", "visual:attention", "visual:ffn", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:residual-norm", "visual:residual", "visual:norm", "residual_flow", "claim:model-runtime"),
        ("visual-rel:config-attention", "visual:config", "visual:attention", "construction_dependency", "claim:construction"),
        ("visual-rel:config-ffn", "visual:config", "visual:ffn", "construction_dependency", "claim:construction"),
        ("visual-rel:parallel-decoder", "visual:parallel", "visual:decoder", "metadata_dependency", "claim:construction"),
        ("visual-rel:branch-merge", "visual:branch", "visual:merge", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:branch-attention", "visual:branch", "visual:attention", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:input-merge", "visual:input", "visual:merge", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:merge-block", "visual:merge", "visual:block", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:note-external", "visual:boundary-note", "visual:external-logits", "external_delegation", "claim:external-logits"),
        ("visual-rel:runtime-detail-attention", "visual:runtime-detail", "visual:attention", "runtime_flow", "claim:model-runtime"),
        ("visual-rel:capability-model", "visual:capability", "visual:model", "metadata_dependency", "claim:construction"),
        ("visual-rel:ffn-output-projection", "visual:ffn", "visual:output-projection", "runtime_flow", "claim:model-runtime"),
    ]

    page = {
        "id": "integrated",
        "title": "Integrated Architecture Flow",
        "export_name": "integrated-architecture-flow",
        "question": "How do model execution, expanded internals and external boundaries form one system?",
        "purpose": "Show one connected architecture story with details expanded at their execution points.",
        "topics": ["model_composition", "model_execution", "external_boundary"],
        "view_pattern": "pipeline",
        "scope": ["complete target file"],
        "claim_ids": ["claim:model-runtime", "claim:construction", "claim:external-logits"],
        "main_story": [
            "Model Input",
            "Embedding",
            "Decoder Stack",
            "Expanded Repeated Block",
            "Final Norm",
            "External Logits Processor",
            "Logits",
        ],
        "detail_regions": [
            {
                "id": "runtime",
                "title": "End-to-End Runtime",
                "type": "embedded_subgraph",
                "question": "How does input become output?",
                "source_item_ids": all_source_ids,
                "claim_ids": ["claim:model-runtime"],
                "visual_form": "continuous runtime spine",
                "integration": {
                    "mode": "inline_expansion",
                    "attach_to_anchor_id": "visual:input",
                    "anchor_ids": [
                        "visual:input",
                        "visual:embedding",
                        "visual:decoder",
                        "visual:block",
                        "visual:norm",
                        "visual:output",
                    ],
                },
            },
            {
                "id": "expanded-block",
                "title": "Expanded Repeated Block",
                "type": "nested_container",
                "question": "What happens inside the repeated block?",
                "source_item_ids": all_source_ids,
                "claim_ids": ["claim:model-runtime", "claim:construction"],
                "visual_form": "nested runtime and residual lanes",
                "integration": {
                    "mode": "nested_container",
                    "attach_to_anchor_id": "visual:decoder",
                    "anchor_ids": [
                        "visual:block",
                        "visual:attention",
                        "visual:ffn",
                        "visual:residual",
                        "visual:config",
                    ],
                },
            },
            {
                "id": "boundary",
                "title": "External Runtime Boundary",
                "type": "boundary_panel",
                "question": "Where does local source delegate to imported runtime behavior?",
                "source_item_ids": all_source_ids,
                "claim_ids": ["claim:external-logits"],
                "visual_form": "boundary at the real call site",
                "integration": {
                    "mode": "boundary_crossing",
                    "attach_to_anchor_id": "visual:external-logits",
                    "anchor_ids": [
                        "visual:external-logits",
                        "visual:boundary-note",
                    ],
                },
            },
        ],
        "secondary_topics": ["parallel metadata", "capabilities"],
        "external_boundaries": ["LogitsProcessor"],
        "visual_decisions": [
            "Use a single landscape canvas.",
            "Expand the repeated block inside the runtime spine.",
        ],
        "detail_budget": {
            "max_primary_nodes": 18,
            "max_total_nodes": 72,
            "max_visible_edges": 90,
        },
        "visual_contract": {
            "canvas": {
                "min_width": 4200,
                "min_height": 2000,
                "orientation": "landscape",
                "min_export_width": 2400,
                "min_export_height": 1200,
                "min_font_size": 12,
                "min_content_fill_ratio": 0.45,
            },
            "main_story_anchor_ids": [
                "visual:input",
                "visual:embedding",
                "visual:decoder",
                "visual:block",
                "visual:norm",
                "visual:external-logits",
                "visual:output",
            ],
            "required_anchors": anchors,
            "required_relationships": [
                {
                    "id": relationship_id,
                    "source_anchor_id": source,
                    "target_anchor_id": target,
                    "kind": kind,
                    "claim_ids": [claim_id],
                }
                for relationship_id, source, target, kind, claim_id in relationships
            ],
            "quality_gates": {
                "min_semantic_nodes": 18,
                "min_visible_edges": 14,
                "min_visual_styles": 4,
                "require_nested_container": True,
                "require_external_boundary": True,
                "require_loading_plane": False,
            },
        },
    }

    plan["schema_version"] = "2.2"
    plan["diagram_strategy"] = "integrated_single_canvas"
    plan["integration_rationale"] = "The runtime, expanded block and boundary are one connected engineering story."
    plan["page_budget"] = {"target_pages": 1, "max_pages": 1}
    plan["pages"] = [page]
    for collection in ("class_review", "method_review"):
        for item in plan[collection]:
            item["page_refs"] = [{"page_id": "integrated", "region_id": "runtime"}]
    for item in plan["function_review"]:
        if item["status"] in {"rendered_detail", "rendered_aggregate"}:
            item["page_refs"] = [{"page_id": "integrated", "region_id": "runtime"}]
    for collection in plan["coverage_manifest"].values():
        for item in collection:
            if item["status"] in {"rendered_detail", "rendered_aggregate"}:
                item["page_refs"] = [{"page_id": "integrated", "region_id": "runtime"}]
    plan["class_review"][0]["status"] = "rendered_detail"
    class_id = plan["class_review"][0]["class_id"]
    model_anchor = next(
        anchor for anchor in anchors if anchor["id"] == "visual:model"
    )
    if class_id not in model_anchor["source_item_ids"]:
        model_anchor["source_item_ids"].append(class_id)
    core_method_ids = [
        method["method_id"]
        for method in context["methods"]
        if method["importance_candidate"] == "core"
    ][:2]
    for method_id, anchor_id in zip(
        core_method_ids,
        ("visual:attention", "visual:ffn"),
        strict=False,
    ):
        review = next(
            item
            for item in plan["method_review"]
            if item["method_id"] == method_id
        )
        review["status"] = "rendered_detail"
        next(
            anchor for anchor in anchors if anchor["id"] == anchor_id
        )["source_item_ids"].append(method_id)
    return plan


def _node_style(anchor: dict) -> str:
    representation = anchor["representation"]
    style_role = anchor["style_role"]
    base = "html=0;whiteSpace=wrap;fontSize=12;"
    if representation == "container":
        return base + "swimlane;rounded=1;fillColor=#FFFFFF;strokeColor=#94A3B8;"
    if representation == "storage":
        return base + "shape=cylinder3;fillColor=#FFF7ED;strokeColor=#EA580C;dashed=1;"
    if representation == "merge":
        return base + "shape=ellipse;fillColor=#DBEAFE;strokeColor=#2563EB;"
    if representation == "badge":
        return base + "rounded=1;arcSize=50;fillColor=#E0F2FE;strokeColor=#0284C7;"
    if style_role == "external":
        return base + "rounded=1;fillColor=#FFF7ED;strokeColor=#EA580C;dashed=1;"
    if style_role == "construction":
        return base + "rounded=1;fillColor=#FFFBEB;strokeColor=#D97706;dashed=1;"
    if style_role == "metadata":
        return base + "rounded=1;fillColor=#F8FAFC;strokeColor=#64748B;dashed=1;"
    if style_role == "residual":
        return base + "rounded=1;fillColor=#ECFEFF;strokeColor=#0EA5E9;"
    return base + "rounded=1;fillColor=#EFF6FF;strokeColor=#2563EB;"


def _edge_style(kind: str) -> str:
    base = "edgeStyle=orthogonalEdgeStyle;html=0;endArrow=block;endFill=1;"
    if kind == "runtime_flow":
        return base + "strokeColor=#2563EB;strokeWidth=3;"
    if kind == "residual_flow":
        return base + "strokeColor=#0EA5E9;strokeWidth=2;"
    if kind == "loading_mapping":
        return base + "strokeColor=#7C3AED;strokeWidth=2.5;"
    if kind == "construction_dependency":
        return base + "strokeColor=#D97706;strokeWidth=2;dashed=1;"
    if kind == "metadata_dependency":
        return base + "strokeColor=#64748B;strokeWidth=1.5;dashed=1;"
    return base + "strokeColor=#EA580C;strokeWidth=2;dashed=1;"


def _write_integrated_drawio(path: Path, plan: dict) -> None:
    page = plan["pages"][0]
    contract = page["visual_contract"]
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net"})
    diagram = ET.SubElement(mxfile, "diagram", {"id": "integrated", "name": page["title"]})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "background": "#ffffff",
            "page": "1",
            "pageWidth": "5200",
            "pageHeight": "2500",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    for cell_id, value, x, y, width, height, font_size in (
        ("title", page["title"], 60, 20, 1200, 50, 24),
        ("question", page["question"], 60, 75, 2500, 40, 16),
        ("region-runtime", "End-to-End Runtime", 80, 150, 500, 35, 16),
        ("region-block", "Expanded Repeated Block", 1200, 350, 500, 35, 16),
        ("region-boundary", "External Runtime Boundary", 3900, 350, 600, 35, 16),
    ):
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": (
                    "html=0;rounded=0;fillColor=#ffffff;strokeColor=#ffffff;"
                    f"fontSize={font_size};"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "as": "geometry",
            },
        )

    positions = [
        (100, 300, 220, 90),
        (420, 300, 220, 90),
        (760, 220, 2600, 1450),
        (280, 180, 1900, 900),
        (3500, 300, 220, 90),
        (3890, 300, 270, 100),
        (4500, 300, 220, 90),
        (50, 170, 5000, 2050),
        (420, 330, 260, 90),
        (820, 330, 260, 90),
        (300, 700, 320, 70),
        (120, 1120, 420, 80),
        (3800, 90, 500, 60),
        (420, 520, 220, 80),
        (690, 520, 100, 100),
        (3500, 850, 300, 80),
        (3900, 500, 400, 90),
        (1200, 520, 260, 90),
        (1500, 760, 240, 90),
    ]
    positions[15] = (3500, 1900, 300, 80)
    anchor_cells: dict[str, str] = {}
    for index, (anchor, position) in enumerate(
        zip(contract["required_anchors"], positions, strict=True),
        start=1,
    ):
        cell_id = f"anchor-{index}"
        anchor_cells[anchor["id"]] = cell_id
        parent_id = "1"
        parent_anchor_id = anchor.get("parent_anchor_id")
        if parent_anchor_id and parent_anchor_id in anchor_cells:
            parent_id = anchor_cells[parent_anchor_id]
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": cell_id,
                "value": anchor["label"],
                "style": _node_style(anchor),
                "vertex": "1",
                "parent": parent_id,
                "dataAnchor": anchor["id"],
            },
        )
        x, y, width, height = position
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "as": "geometry",
            },
        )

    for index, relationship in enumerate(contract["required_relationships"], start=1):
        edge = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"edge-{index}",
                "value": "",
                "style": _edge_style(relationship["kind"]),
                "edge": "1",
                "parent": "1",
                "source": anchor_cells[relationship["source_anchor_id"]],
                "target": anchor_cells[relationship["target_anchor_id"]],
                "dataAnchor": relationship["id"],
            },
        )
        ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})
    ET.ElementTree(mxfile).write(path, encoding="utf-8", xml_declaration=False)


def test_plan_22_integrated_contract_passes(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    for claim in evidence["claims"]:
        claim["used_in_pages"] = ["integrated"]
    plan = _integrated_plan(context)
    validator = load_script("validate_architecture_plan")

    errors, warnings = validator.validate_plan(
        plan,
        evidence=evidence,
        context=context,
        repo_root=repo,
    )

    assert errors == []
    assert warnings == []


def test_plan_22_rejects_detached_detail_region(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    for claim in evidence["claims"]:
        claim["used_in_pages"] = ["integrated"]
    plan = _integrated_plan(context)
    plan["pages"][0]["visual_contract"]["required_relationships"] = [
        relationship
        for relationship in plan["pages"][0]["visual_contract"]["required_relationships"]
        if relationship["id"] != "visual-rel:note-external"
    ]
    plan["pages"][0]["visual_contract"]["required_relationships"].append(
        {
            "id": "visual-rel:replacement",
            "source_anchor_id": "visual:input",
            "target_anchor_id": "visual:embedding",
            "kind": "runtime_flow",
            "claim_ids": ["claim:model-runtime"],
        }
    )
    validator = load_script("validate_architecture_plan")

    errors, _ = validator.validate_plan(
        plan,
        evidence=evidence,
        context=context,
        repo_root=repo,
    )

    assert any("not connected to attachment" in error for error in errors)


def test_plan_22_rejects_external_relation_without_external_evidence(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    for claim in evidence["claims"]:
        claim["used_in_pages"] = ["integrated"]
    plan = _integrated_plan(context)
    relationship = next(
        item
        for item in plan["pages"][0]["visual_contract"]["required_relationships"]
        if item["kind"] == "external_delegation"
    )
    relationship["claim_ids"] = ["claim:model-runtime"]
    validator = load_script("validate_architecture_plan")

    errors, _ = validator.validate_plan(
        plan,
        evidence=evidence,
        context=context,
        repo_root=repo,
    )

    assert any("external_delegation requires Evidence phase" in error for error in errors)


def test_plan_22_rejects_serial_branch_and_duplicate_endpoints(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    for claim in evidence["claims"]:
        claim["used_in_pages"] = ["integrated"]
    plan = _integrated_plan(context)
    relationships = plan["pages"][0]["visual_contract"]["required_relationships"]
    relationships[:] = [
        relationship
        for relationship in relationships
        if relationship["id"] != "visual-rel:branch-attention"
    ]
    relationships.append(
        {
            "id": "visual-rel:duplicate-input-embedding",
            "source_anchor_id": "visual:input",
            "target_anchor_id": "visual:embedding",
            "kind": "runtime_flow",
            "claim_ids": ["claim:model-runtime"],
        }
    )
    validator = load_script("validate_architecture_plan")

    errors, _ = validator.validate_plan(
        plan,
        evidence=evidence,
        context=context,
        repo_root=repo,
    )

    assert any("branch anchors must expose at least two" in error for error in errors)
    assert any("duplicate visual relationship endpoints" in error for error in errors)


def test_plan_22_rejects_all_aggregate_complete_model(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    for claim in evidence["claims"]:
        claim["used_in_pages"] = ["integrated"]
    plan = _integrated_plan(context)
    for item in plan["class_review"]:
        if item["status"] == "rendered_detail":
            item["status"] = "rendered_aggregate"
    for item in plan["method_review"]:
        if item["status"] == "rendered_detail":
            item["status"] = "rendered_aggregate"
    validator = load_script("validate_architecture_plan")

    errors, _ = validator.validate_plan(
        plan,
        evidence=evidence,
        context=context,
        repo_root=repo,
    )

    assert any("too few classes in detail" in error for error in errors)
    assert any("too few core methods in detail" in error for error in errors)
    assert any("but renders only 0 in detail" in error for error in errors)


def test_plan_22_rejects_shallow_structural_region(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    for claim in evidence["claims"]:
        claim["used_in_pages"] = ["integrated"]
    plan = _integrated_plan(context)
    region = next(
        item
        for item in plan["pages"][0]["detail_regions"]
        if item["id"] == "expanded-block"
    )
    region["integration"]["anchor_ids"] = [
        "visual:block",
        "visual:attention",
    ]
    validator = load_script("validate_architecture_plan")

    errors, _ = validator.validate_plan(
        plan,
        evidence=evidence,
        context=context,
        repo_root=repo,
    )

    assert any(
        "structural detail region requires at least 4 semantic anchors"
        in error
        for error in errors
    )
    assert any(
        "structural detail region requires at least 3 internal relationships"
        in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("category", "subsystem_label"),
    (
        ("moe", "mixture-of-experts"),
        ("multimodal", "multimodal"),
    ),
)
def test_plan_22_requires_detected_subsystem_subgraph(
    tmp_path: Path,
    category: str,
    subsystem_label: str,
) -> None:
    repo, context = make_context(tmp_path)
    subsystem_class_id = context["classes"][0]["class_id"]
    context["classes"][0]["category_candidates"] = [category]
    evidence = valid_evidence(context)
    for claim in evidence["claims"]:
        claim["used_in_pages"] = ["integrated"]
    plan = _integrated_plan(context)
    for region in plan["pages"][0]["detail_regions"]:
        if region["type"] in {
            "embedded_subgraph",
            "component_tree",
            "nested_container",
            "mapping_tree",
        }:
            region["source_item_ids"] = [
                item_id
                for item_id in region["source_item_ids"]
                if item_id != subsystem_class_id
            ]
    validator = load_script("validate_architecture_plan")

    errors, _ = validator.validate_plan(
        plan,
        evidence=evidence,
        context=context,
        repo_root=repo,
    )

    assert any(
        f"no structural detail region for the detected {subsystem_label} subsystem"
        in error
        for error in errors
    )


def test_drawio_22_integrated_contract_passes(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = _integrated_plan(context)
    drawio = tmp_path / "architecture.drawio"
    _write_integrated_drawio(drawio, plan)
    validator = load_script("validate_drawio")

    assert validator.validate_drawio(drawio, plan=plan) == []


def test_drawio_22_rejects_missing_anchor_and_wrong_edge_style(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = _integrated_plan(context)
    drawio = tmp_path / "architecture.drawio"
    _write_integrated_drawio(drawio, plan)
    tree = ET.parse(drawio)
    root = tree.getroot()
    missing = root.find(".//mxCell[@dataAnchor='visual:attention']")
    assert missing is not None
    missing.attrib.pop("dataAnchor")
    wrong_edge = root.find(".//mxCell[@dataAnchor='visual-rel:input-embedding']")
    assert wrong_edge is not None
    wrong_edge.set(
        "style",
        "edgeStyle=orthogonalEdgeStyle;html=0;strokeColor=#7C3AED;"
        "strokeWidth=2.5;endArrow=block;endFill=1;",
    )
    tree.write(drawio, encoding="utf-8")
    validator = load_script("validate_drawio")

    errors = validator.validate_drawio(drawio, plan=plan)

    assert any("required visual anchor is missing: visual:attention" in error for error in errors)
    assert any("edge style does not match runtime_flow" in error for error in errors)


def test_drawio_22_rejects_small_canvas_and_wrong_endpoint(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = _integrated_plan(context)
    drawio = tmp_path / "architecture.drawio"
    _write_integrated_drawio(drawio, plan)
    tree = ET.parse(drawio)
    model = tree.getroot().find(".//mxGraphModel")
    assert model is not None
    model.set("pageWidth", "1200")
    model.set("pageHeight", "900")
    edge = tree.getroot().find(".//mxCell[@dataAnchor='visual-rel:input-embedding']")
    output = tree.getroot().find(".//mxCell[@dataAnchor='visual:output']")
    assert edge is not None and output is not None
    edge.set("target", output.get("id", ""))
    tree.write(drawio, encoding="utf-8")
    validator = load_script("validate_drawio")

    errors = validator.validate_drawio(drawio, plan=plan)

    assert any("integrated canvas is" in error for error in errors)
    assert any("edge target does not match Plan anchor" in error for error in errors)


def test_drawio_22_rejects_hidden_mislabeled_or_off_canvas_anchor(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = _integrated_plan(context)
    drawio = tmp_path / "architecture.drawio"
    _write_integrated_drawio(drawio, plan)
    tree = ET.parse(drawio)
    anchor = tree.getroot().find(".//mxCell[@dataAnchor='visual:input']")
    assert anchor is not None
    anchor.set("value", "Unrelated Placeholder")
    anchor.set("visible", "0")
    geometry = anchor.find("mxGeometry")
    assert geometry is not None
    geometry.set("x", "6000")
    tree.write(drawio, encoding="utf-8")
    validator = load_script("validate_drawio")

    errors = validator.validate_drawio(drawio, plan=plan)

    assert any("required visual anchor must be visible" in error for error in errors)
    assert any("visible label does not match" in error for error in errors)
    assert any("required anchor lies outside the canvas" in error for error in errors)


def test_drawio_22_rejects_sparse_required_anchor_distribution(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = _integrated_plan(context)
    drawio = tmp_path / "architecture.drawio"
    _write_integrated_drawio(drawio, plan)
    tree = ET.parse(drawio)
    for cell in tree.getroot().findall(".//mxCell[@dataAnchor]"):
        if not cell.get("dataAnchor", "").startswith("visual:"):
            continue
        geometry = cell.find("mxGeometry")
        if geometry is not None:
            geometry.set("y", "260")
    tree.write(drawio, encoding="utf-8")
    validator = load_script("validate_drawio")

    errors = validator.validate_drawio(drawio, plan=plan)

    assert any("occupied content height" in error for error in errors)
    assert any("occupied-content grid" in error for error in errors)


def test_drawio_22_rejects_combined_heading_and_detached_region_title(
    tmp_path: Path,
) -> None:
    _, context = make_context(tmp_path)
    plan = _integrated_plan(context)
    drawio = tmp_path / "architecture.drawio"
    _write_integrated_drawio(drawio, plan)
    tree = ET.parse(drawio)
    root = tree.getroot()
    title = root.find(".//mxCell[@id='title']")
    question = root.find(".//mxCell[@id='question']")
    region = root.find(".//mxCell[@id='region-runtime']")
    assert title is not None and question is not None and region is not None
    title.set("value", f"{plan['pages'][0]['title']} {plan['pages'][0]['question']}")
    question.set("value", " ")
    region_geometry = region.find("mxGeometry")
    assert region_geometry is not None
    region_geometry.set("y", "2200")
    tree.write(drawio, encoding="utf-8")
    validator = load_script("validate_drawio")

    errors = validator.validate_drawio(drawio, plan=plan)

    assert any("title must appear once as a separate" in error for error in errors)
    assert any("question must appear once as a separate" in error for error in errors)
    assert any("detail region title is detached" in error for error in errors)


def test_visual_review_requires_geometry_or_routing_revision(tmp_path: Path) -> None:
    review = tmp_path / "visual-review.md"
    review.write_text(
        "# Visual Review\n\n"
        "## Draft 1 Findings\n"
        "Integrated Architecture Flow was opened and inspected. The canvas had "
        "large empty areas and weak visual hierarchy around the local block.\n\n"
        "## Revision Round 1\n"
        "Added compact explanatory notes and clarified several labels without "
        "changing the architecture semantics.\n\n"
        "## Final Review\n"
        "Integrated Architecture Flow was exported and inspected again. The "
        "remaining risks are documented for the final handoff.\n",
        encoding="utf-8",
    )
    validator = load_script("validate_drawio")

    errors = validator._validate_visual_review(
        review,
        ["Integrated Architecture Flow"],
    )

    assert any("no substantive geometry or routing revision" in error for error in errors)


def test_visual_revision_requires_region_level_geometry_change(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    plan = _integrated_plan(context)
    draft = tmp_path / "architecture-draft1.drawio"
    final = tmp_path / "architecture.drawio"
    _write_integrated_drawio(draft, plan)
    _write_integrated_drawio(final, plan)
    tree = ET.parse(final)
    anchor = tree.getroot().find(".//mxCell[@dataAnchor='visual:input']")
    assert anchor is not None
    geometry = anchor.find("mxGeometry")
    assert geometry is not None
    geometry.set("x", "140")
    edge = tree.getroot().find(
        ".//mxCell[@dataAnchor='visual-rel:input-embedding']"
    )
    assert edge is not None
    edge.set("style", edge.get("style", "") + "rounded=1;")
    tree.write(final, encoding="utf-8")
    validator = load_script("validate_drawio")

    errors = validator._validate_visual_revision(
        draft_path=draft,
        final_path=final,
    )

    assert any("changed only 1 required anchor" in error for error in errors)
    assert any("rerouted only 1 required relationship" in error for error in errors)


def test_complete_plan_rejects_imprecise_evidence_symbol(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    for claim in evidence["claims"]:
        claim["used_in_pages"] = ["integrated"]
    evidence["claims"][0]["evidence"][0]["symbol"] = "human summary not in source"
    plan = _integrated_plan(context)
    validator = load_script("validate_evidence")

    errors, warnings, _ = validator.validate_evidence(
        evidence,
        context=context,
        plan=plan,
        repo_root=repo,
    )

    assert any("symbol not found in cited lines" in error for error in errors)
    assert not any("symbol not found in cited lines" in warning for warning in warnings)


def test_integrated_baselines_remain_large_connected_references() -> None:
    for model_name, min_anchor_count, min_edge_count in (
        ("hy_v3", 70, 80),
        ("qwen3_moe", 65, 75),
    ):
        drawio = ROOT / "examples" / "integrated-flow" / model_name / "architecture.drawio"
        tree = ET.parse(drawio)
        model = tree.getroot().find(".//mxGraphModel")
        assert model is not None
        assert float(model.get("pageWidth", "0")) >= 5000
        assert float(model.get("pageHeight", "0")) >= 2400
        anchors = model.findall(".//mxCell[@dataAnchor]")
        edges = model.findall(".//mxCell[@edge='1']")
        assert len(anchors) >= min_anchor_count
        assert len(edges) >= min_edge_count
        assert model.get("background") == "#ffffff"


def test_prepare_template_defaults_to_integrated_single_canvas(tmp_path: Path) -> None:
    _, context = make_context(tmp_path)
    collector = load_script("collect_source_context")

    template = collector.plan_template(context, "dense")

    assert template["schema_version"] == "2.2"
    assert template["diagram_strategy"] == "integrated_single_canvas"
    assert template["page_budget"] == {"target_pages": 1, "max_pages": 1}
    assert template["pages"] == []


def test_legacy_plan_21_remains_valid(tmp_path: Path) -> None:
    repo, context = make_context(tmp_path)
    evidence = valid_evidence(context)
    plan = deepcopy(valid_plan(context))
    validator = load_script("validate_architecture_plan")

    errors, _ = validator.validate_plan(
        plan,
        evidence=evidence,
        context=context,
        repo_root=repo,
    )

    assert errors == []
