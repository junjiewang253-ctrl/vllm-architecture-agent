from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "src" / "skills" / "vllm-model-architecture-diagram"
SCRIPTS = SKILL / "scripts"
CLI_PATH = ROOT / "src" / "vllm_architecture_agent" / "cli.py"
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"
SIMPLE_PATH = ROOT / "samples" / "simple_model.py"


def load_module(filename: str, name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def architect_context(input_path: Path = HY_V3_PATH, model: str = "hy-v3-v12-tests") -> dict[str, Any]:
    extractor = load_module("extract_architecture.py", f"extract_architecture_{model}")
    inventory_builder = load_module("build_semantic_inventory.py", f"inventory_{model}")
    graph_builder = load_module("build_source_fact_graph.py", f"fact_graph_{model}")
    concept_builder = load_module("build_baseline_concept_graph.py", f"concept_{model}")
    boundary_builder = load_module("build_boundary_report.py", f"boundary_{model}")
    brief_builder = load_module("build_architect_brief.py", f"brief_{model}")
    design_builder = load_module("build_baseline_design.py", f"design_{model}")
    view_compiler = load_module("compile_architecture_view.py", f"view_{model}")
    design_validator = load_module("validate_architecture_design.py", f"design_validator_{model}")
    view_validator = load_module("validate_architecture_view.py", f"view_validator_{model}")

    analysis = extractor.extract_architecture(input_path)
    inventory = inventory_builder.build_semantic_inventory(analysis)
    graph = graph_builder.build_source_fact_graph(analysis)
    concept = concept_builder.build_architecture_design(graph, analysis, inventory)
    boundary = boundary_builder.build_boundary_report(concept)
    brief = brief_builder.build_architect_brief(analysis, graph, concept, boundary, source_file=input_path, model_name=model)
    design = design_builder.build_baseline_design(brief, concept, boundary, author_type="agent")
    view = view_compiler.compile_architecture_view(design)
    return {
        "analysis": analysis,
        "inventory": inventory,
        "graph": graph,
        "concept": concept,
        "boundary": boundary,
        "brief": brief,
        "design": design,
        "view": view,
        "design_errors": design_validator.validate_architecture_design(design, graph, concept, source_file=input_path),
        "view_errors": view_validator.validate_architecture_view(view, concept, graph),
    }


def page(root: dict[str, Any], page_id: str) -> dict[str, Any]:
    return next(item for item in root["pages"] if item["id"] == page_id)


def test_architect_brief_clusters_and_model_profile_are_compact():
    artifacts = architect_context()
    brief = artifacts["brief"]
    assert brief["schema_version"] == "1.0"
    profile = brief["model_profile"]
    assert profile["top_level_wrapper"]
    assert profile["attention_class"]
    assert profile["moe_class"]
    cluster_ids = {cluster["id"] for cluster in brief["fact_clusters"]}
    assert {"attention", "moe", "checkpoint_wrapper_loader", "checkpoint_model_loader"} <= cluster_ids
    assert all(len(cluster["fact_ids"]) <= 24 for cluster in brief["fact_clusters"])


def test_strict_agent_design_has_seven_story_pages_and_validates():
    artifacts = architect_context()
    design = artifacts["design"]
    assert artifacts["design_errors"] == []
    assert design["schema_version"] == "1.0"
    assert design["author"]["type"] == "agent"
    assert [item["id"] for item in design["pages"]] == [
        "model_execution_overview",
        "decoder_block",
        "attention_adaptation",
        "moe_execution",
        "checkpoint_weight_loading",
        "parallel_strategies",
        "adapter_boundary",
    ]
    assert {item["view_kind"] for item in design["pages"]} >= {
        "pipeline",
        "block_with_residual",
        "branch_merge",
        "routed_container",
        "mapping_dispatch",
        "strategy_matrix",
        "boundary_map",
    }


def test_attention_moe_checkpoint_and_parallel_semantics_are_source_grounded():
    design = architect_context()["design"]
    attention = page(design, "attention_adaptation")
    attention_edges = {edge["id"]: edge for edge in attention["edges"]}
    assert attention_edges["attention_qkv_to_hpc"]["source"] == "qkv_projection"
    assert attention_edges["attention_qkv_to_hpc"]["source_port"] == "qkv_out"
    assert attention_edges["attention_v_to_backend"]["source"] == "v_stream"
    assert attention_edges["attention_v_to_backend"]["target"] == "attention_backend"
    assert all(edge["target"] not in {"q_norm", "k_norm", "rope"} for edge in attention["edges"] if edge["source"] == "v_stream")

    moe = page(design, "moe_execution")
    moe_edges = {(edge["source"], edge["target"]) for edge in moe["edges"]}
    assert ("flatten_tokens", "fused_moe") in moe_edges
    assert ("router_logits", "fused_moe") in moe_edges
    assert page(design, "moe_execution")["view_kind"] == "routed_container"
    assert {node["id"]: node for node in moe["nodes"]}["routed_experts"]["parent_id"] == "fused_moe"

    checkpoint = page(design, "checkpoint_weight_loading")
    lanes = {group["id"] for group in checkpoint["groups"]}
    assert {"wrapper_loader_lane", "model_loader_lane"} <= lanes
    parallel = page(design, "parallel_strategies")
    assert not parallel["edges"]


def test_architecture_view_compiles_from_design_not_concept_cards():
    artifacts = architect_context()
    view = artifacts["view"]
    assert artifacts["view_errors"] == []
    assert view["schema_version"] == "1.0"
    for item in view["pages"]:
        assert item["visible_nodes"]
        assert not all(node.get("role") == "annotation" for node in item["visible_nodes"])
    attention = page(view, "attention_adaptation")
    labels = {node["label"] for node in attention["visible_nodes"]}
    assert {"QKV Projection", "Split Q / K / V", "Q", "K", "V", "KV Cache", "vLLM Attention Backend"} <= labels


def test_vllm_arch_prepare_does_not_generate_final_design(tmp_path: Path):
    outputs = tmp_path / "prepare"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "prepare",
            "--input",
            str(HY_V3_PATH),
            "--outputs-dir",
            str(outputs),
            "--model-name",
            "hy-v3-v12",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert (outputs / "hy-v3-v12-architect-brief.json").exists()
    assert (outputs / "hy-v3-v12-architecture-design.template.json").exists()
    assert not (outputs / "hy-v3-v12-architecture-design.json").exists()


def test_vllm_arch_architect_mode_requires_design(tmp_path: Path):
    outputs = tmp_path / "architect-missing"
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
            "hy-v3-v12",
            "--mode",
            "architect",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "requires an Agent-authored design file" in completed.stderr


def test_vllm_arch_finalize_end_to_end_with_agent_design(tmp_path: Path):
    outputs = tmp_path / "finalize"
    subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "prepare",
            "--input",
            str(HY_V3_PATH),
            "--outputs-dir",
            str(outputs),
            "--model-name",
            "hy-v3-v12",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    design_builder = load_module("build_baseline_design.py", "design_for_cli_v12")
    brief = json.loads((outputs / "hy-v3-v12-architect-brief.json").read_text(encoding="utf-8"))
    concept = json.loads((outputs / "hy-v3-v12-architecture-concept.json").read_text(encoding="utf-8"))
    boundary = json.loads((outputs / "hy-v3-v12-boundary-report.json").read_text(encoding="utf-8"))
    design = design_builder.build_baseline_design(brief, concept, boundary, author_type="agent")
    design_path = outputs / "hy-v3-v12-architecture-design.json"
    design_path.write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "finalize",
            "--design",
            str(design_path),
            "--outputs-dir",
            str(outputs),
            "--model-name",
            "hy-v3-v12",
            "--source-file",
            str(HY_V3_PATH),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert (outputs / "hy-v3-v12-architecture-view.json").exists()
    assert (outputs / "hy-v3-v12-layout-plan.json").exists()
    assert (outputs / "hy-v3-v12-layout-metrics.json").exists()
    assert (outputs / "hy-v3-v12-architecture.drawio").exists()
    assert (outputs / "hy-v3-v12-architecture-report.md").exists()


def test_simple_model_uses_smaller_page_set_without_moe_or_expert_parallel():
    artifacts = architect_context(SIMPLE_PATH, "simple-v12")
    page_ids = [item["id"] for item in artifacts["design"]["pages"]]
    assert "moe_execution" not in page_ids
    assert "attention_adaptation" not in page_ids
    assert "checkpoint_weight_loading" not in page_ids
    parallel = page(artifacts["design"], "parallel_strategies")
    labels = {node["label"] for node in parallel["nodes"]}
    assert "Expert Parallel" not in labels
