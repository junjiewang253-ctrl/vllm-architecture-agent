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
    planner = load_module("view_planner.py", "view_planner_v10_tests")
    boundary_builder = load_module("build_boundary_report.py", "boundary_report_v10_tests")
    analysis = extractor.extract_architecture(HY_V3_PATH)
    inventory = inventory_builder.build_semantic_inventory(analysis)
    graph = graph_builder.build_source_fact_graph(analysis)
    design = architect.build_architecture_design(graph, analysis, inventory)
    view = planner.build_architecture_view(design)
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
    labels = {node["display_label"] for node in attention["visible_nodes"]}
    assert "Tensor Parallel QKV Projection" in labels
    assert "KV Cache Boundary" in labels
    assert "vLLM Attention Backend Boundary" in labels
    assert attention["purpose"]


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
    assert (outputs / "hy-v3-v10-architecture-design.json").exists()
    assert (outputs / "hy-v3-v10-architecture-view.json").exists()
    assert (outputs / "hy-v3-v10-boundary-report.json").exists()
    assert (outputs / "hy-v3-v10-architecture.drawio").exists()
    assert (outputs / "hy-v3-v10-mentor-report.md").exists()
