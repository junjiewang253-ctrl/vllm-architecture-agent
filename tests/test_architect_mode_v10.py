from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "src" / "skills" / "vllm-model-architecture-diagram" / "scripts"
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


def architect_artifacts() -> dict[str, Any]:
    extractor = load_module("extract_architecture.py", "extract_architecture_v10_tests")
    inventory_builder = load_module("build_semantic_inventory.py", "inventory_v10_tests")
    graph_builder = load_module("build_source_fact_graph.py", "fact_graph_v10_tests")
    architect = load_module("run_architect_review.py", "architect_review_v10_tests")
    view_architect = load_module("run_view_architect.py", "view_architect_v101_tests")
    boundary_builder = load_module("build_boundary_report.py", "boundary_report_v10_tests")
    analysis = extractor.extract_architecture(HY_V3_PATH)
    inventory = inventory_builder.build_semantic_inventory(analysis)
    graph = graph_builder.build_source_fact_graph(analysis)
    design = architect.build_architecture_design(graph, analysis, inventory)
    view = view_architect.build_architecture_view_graph(design, graph)
    boundary = boundary_builder.build_boundary_report(design)
    return {"analysis": analysis, "inventory": inventory, "graph": graph, "design": design, "view": view, "boundary": boundary}


def test_source_fact_graph_tracks_qkv_and_tp_relations():
    artifacts = architect_artifacts()
    graph = artifacts["graph"]
    assert graph["schema_version"] == "1.0"
    summaries = "\n".join(str(fact.get("summary")) for fact in graph["facts"])
    assert "QKVParallelLinear" in summaries
    relations = graph["relations"]
    assert any(rel["relation"] == "implements" and "tensor_parallel_attention_projection" in rel["target"] for rel in relations)


def test_architecture_design_concepts_have_evidence_and_attention_projection():
    design = architect_artifacts()["design"]
    concepts = {concept["id"]: concept for concept in design["concepts"]}
    projection = concepts["concept:tensor_parallel_attention_projection"]
    assert projection["title"] == "Tensor Parallel QKV Projection"
    assert "QKVParallelLinear" in projection["implementation"]
    assert projection["evidence"]
    assert all(concept["evidence"] for concept in concepts.values())


def test_attention_view_contains_qkv_kv_cache_and_backend_boundary():
    view = architect_artifacts()["view"]
    attention = next(page for page in view["pages"] if page["title"] == "Attention Implementation")
    labels = {node["label"] for node in attention["nodes"]}
    assert "QKV Projection" in labels
    assert "Q/K/V Split" in labels
    assert "HPC Fused Processing" in labels
    assert "Optional QK Norm" in labels
    assert "KV Cache Boundary" in labels
    assert "vLLM Attention Backend" in labels
    edges = {edge["id"]: edge for edge in attention["edges"]}
    assert edges["attention_split_to_q"]["source_port"] == "q"
    assert edges["attention_split_to_k"]["source_port"] == "k"
    assert edges["attention_split_to_v"]["source_port"] == "v"
    assert attention["purpose"]


def test_view_graph_contains_runtime_flows_not_concept_cards():
    view = architect_artifacts()["view"]
    assert view["schema_version"] == "0.1"
    for page in view["pages"]:
        assert len(page["nodes"]) >= 3
        assert any(edge["type"] != "annotation" for edge in page["edges"])
        assert not all(str(node.get("visual_role", "")).startswith("concept") for node in page["nodes"])
    for page_id in {"model_overview", "attention", "moe"}:
        page = next(item for item in view["pages"] if item["id"] == page_id)
        assert any(edge["type"] == "runtime_flow" for edge in page["edges"])


def test_moe_and_checkpoint_views_have_architecture_nodes():
    view = architect_artifacts()["view"]
    moe = next(page for page in view["pages"] if page["id"] == "moe")
    moe_labels = {node["label"] for node in moe["nodes"]}
    assert {"Router", "Top-K Routing", "FusedMoE", "Routed Experts", "Shared Experts", "Expert Parallel"}.issubset(moe_labels)
    checkpoint = next(page for page in view["pages"] if page["id"] == "checkpoint")
    checkpoint_labels = {node["label"] for node in checkpoint["nodes"]}
    assert {"HF Checkpoint", "Weight Name Processing", "Packed Mapping", "q_proj / k_proj / v_proj", "qkv_proj", "Expert Params", "Loader Dispatch", "vLLM Parameters"}.issubset(checkpoint_labels)


def test_external_boundary_does_not_claim_direct_internals():
    boundary = architect_artifacts()["boundary"]
    names = {item["name"] for item in boundary["external_components"]}
    assert {"vLLM Attention", "FusedMoE", "AutoWeightsLoader"}.issubset(names)
    assert all("direct" not in item["allowed_claim"].lower() for item in boundary["external_components"])


def test_architecture_quality_validator_accepts_architect_artifacts():
    artifacts = architect_artifacts()
    validator = load_module("validate_architecture_quality.py", "quality_v10_tests")
    assert validator.validate_architecture_quality(
        artifacts["graph"],
        artifacts["design"],
        artifacts["view"],
        artifacts["boundary"],
    ) == []
    view_validator = load_module("validate_architecture_view.py", "view_quality_v101_tests")
    assert view_validator.validate_architecture_view(
        artifacts["view"],
        artifacts["design"],
        artifacts["graph"],
    ) == []


def test_vllm_arch_architect_mode_end_to_end(tmp_path: Path):
    outputs = tmp_path / "architect"
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
            "hy-v3-v10",
            "--mode",
            "architect",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert (outputs / "hy-v3-v10-source-fact-graph.json").exists()
    assert (outputs / "hy-v3-v10-architecture-concept.json").exists()
    assert (outputs / "hy-v3-v10-architecture-view.json").exists()
    assert (outputs / "hy-v3-v10-boundary-report.json").exists()
    assert (outputs / "hy-v3-v10-architecture.drawio").exists()
    assert (outputs / "hy-v3-v10-architecture-report.md").exists()
