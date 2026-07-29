from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

from vllm_architecture_agent.patterns import detect_capabilities


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "src" / "skills" / "vllm-model-architecture-diagram" / "scripts"
FIXTURES = ROOT / "tests" / "fixtures"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    models = repo / "vllm" / "model_executor" / "models"
    models.mkdir(parents=True)
    mapping = {
        "dense_decoder_adapter.py": "dense_decoder_adapter.py",
        "moe_adapter.py": "moe_adapter.py",
        "multimodal_adapter.py": "multimodal_adapter.py",
        "embedding_adapter.py": "embedding_adapter.py",
        "hybrid_adapter.py": "hybrid_adapter.py",
        "custom_weight_loader_adapter.py": "custom_weight_loader_adapter.py",
        "helper_module.py": "helper_module.py",
        "nested_class_adapter.py": "nested_class_adapter.py",
        "complex_branch_adapter.py": "complex_branch_adapter.py",
    }
    for source, destination in mapping.items():
        shutil.copyfile(FIXTURES / source, models / destination)
    (models / "registry.py").write_text(
        """
_TEXT_GENERATION_MODELS = {
    "DenseForCausalLM": ("dense_decoder_adapter", "DenseForCausalLM"),
    "CustomLoaderForCausalLM": ("custom_weight_loader_adapter", "CustomLoaderForCausalLM"),
}
_MOE_MODELS = {
    "MoeForCausalLM": ("moe_adapter", "MoeForCausalLM"),
}
_MULTIMODAL_MODELS = {
    "MultiModalForConditionalGeneration": ("multimodal_adapter", "MultiModalForConditionalGeneration"),
}
_EMBEDDING_MODELS = {
    "TextEmbeddingModel": ("embedding_adapter", "TextEmbeddingModel"),
}
_HYBRID_MODELS = {
    "HybridStateModel": ("hybrid_adapter", "HybridStateModel"),
}
_VLLM_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_MOE_MODELS,
    **_MULTIMODAL_MODELS,
    **_EMBEDDING_MODELS,
    **_HYBRID_MODELS,
}
""",
        encoding="utf-8",
    )
    return repo


def test_registry_static_parse_and_resolve(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    resolver = load_script("resolve_model_target")

    entries, warnings = resolver.list_registry_models(repo)
    assert not [warning for warning in warnings if "not found" in warning]
    names = {entry["architecture"] for entry in entries}
    assert "DenseForCausalLM" in names
    assert "MoeForCausalLM" in names

    resolved = resolver.resolve_model_target(repo, architecture="DenseForCausalLM")
    assert resolved["status"] == "resolved"
    assert resolved["registered_classes"] == ["DenseForCausalLM"]
    assert resolved["target_file"].endswith("dense_decoder_adapter.py")


def test_context_collector_detects_distinct_model_categories(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    resolver = load_script("resolve_model_target")
    collector = load_script("collect_source_context")

    dense = collector.collect_source_context(
        repo,
        resolver.resolve_model_target(repo, architecture="DenseForCausalLM"),
    )
    assert "decoder_text_generation" in dense["classification"]["category_candidates"]
    assert dense["capability_signals"]["moe"]["detected"] is False
    assert dense["module_assignments"]
    assert dense["schema_version"] == "2.1"
    assert dense["source_coverage"]["all_classes_indexed"] is True
    assert dense["source_coverage"]["all_methods_indexed"] is True
    assert dense["source_coverage"]["unindexed_nodes"] == []
    assert dense["capability_signals"]["pooling"]["detected"] is False
    assert "pages" not in dense

    moe = collector.collect_source_context(repo, resolver.resolve_model_target(repo, architecture="MoeForCausalLM"))
    assert moe["capability_signals"]["moe"]["detected"] is True
    assert moe["capability_signals"]["expert_parallel"]["detected"] is True

    multimodal = collector.collect_source_context(
        repo,
        resolver.resolve_model_target(repo, architecture="MultiModalForConditionalGeneration"),
    )
    assert "multimodal" in multimodal["classification"]["category_candidates"]

    embedding = collector.collect_source_context(
        repo,
        resolver.resolve_model_target(repo, architecture="TextEmbeddingModel"),
    )
    assert "decoder_text_generation" not in embedding["classification"]["category_candidates"]
    assert "embedding_or_pooling" in embedding["classification"]["category_candidates"]

    hybrid = collector.collect_source_context(repo, resolver.resolve_model_target(repo, architecture="HybridStateModel"))
    assert "hybrid_or_recurrent" in hybrid["classification"]["category_candidates"]


def test_helper_file_is_not_full_model(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    resolver = load_script("resolve_model_target")
    collector = load_script("collect_source_context")
    helper_path = repo / "vllm" / "model_executor" / "models" / "helper_module.py"
    context = collector.collect_source_context(repo, resolver.resolve_model_target(repo, input_path=helper_path))

    assert context["classification"]["status"] == "helper"
    assert "helper_or_shared_module" in context["classification"]["category_candidates"]


def test_prepare_templates_are_empty_skeletons(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    output = tmp_path / "out"
    collector = load_script("collect_source_context")
    exit_code = collector.main(
        [
            "--repo-root",
            str(repo),
            "--architecture",
            "DenseForCausalLM",
            "--output",
            str(output / "source-context.json"),
            "--plan-template",
            str(output / "architecture-plan.template.json"),
            "--evidence-template",
            str(output / "evidence.template.json"),
        ]
    )
    assert exit_code == 0
    plan = json.loads((output / "architecture-plan.template.json").read_text(encoding="utf-8"))
    evidence = json.loads((output / "evidence.template.json").read_text(encoding="utf-8"))
    assert plan["pages"] == []
    assert len(plan["class_review"]) > 0
    assert len(plan["method_review"]) > 0
    assert "coverage_manifest" in plan
    assert plan["schema_version"] == "2.2"
    assert plan["diagram_strategy"] == "integrated_single_canvas"
    assert plan["page_budget"] == {"target_pages": 1, "max_pages": 1}
    assert evidence["claims"] == []
    assert evidence["schema_version"] == "2.1"


def test_recursive_catalog_indexes_nested_classes_and_module_functions(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    resolver = load_script("resolve_model_target")
    collector = load_script("collect_source_context")
    target = repo / "vllm" / "model_executor" / "models" / "nested_class_adapter.py"
    context = collector.collect_source_context(repo, resolver.resolve_model_target(repo, input_path=target))

    class_names = {item["qualified_name"]: item for item in context["classes"]}
    assert "OuterModel" in class_names
    assert "OuterModel.InnerBlock" in class_names
    assert class_names["OuterModel.InnerBlock"]["parent_class_id"] == class_names["OuterModel"]["class_id"]
    methods = {item["qualified_name"]: item for item in context["methods"]}
    assert "OuterModel.InnerBlock.forward" in methods
    assert "OuterModel.forward" in methods
    assert "OuterModel.hidden_size" in methods
    assert {item["qualified_name"] for item in context["module_functions"]} == {"module_helper"}
    outer_calls = {
        call["target"]
        for call in context["calls"]
        if call["owner_id"] == methods["OuterModel.forward"]["method_id"]
    }
    assert "should_not_be_counted" not in " ".join(outer_calls)
    assert context["source_coverage"]["ast_class_count"] == context["source_coverage"]["catalogued_class_count"]
    assert context["source_coverage"]["ast_method_count"] == context["source_coverage"]["catalogued_method_count"]


def test_branch_and_weight_mapping_catalog(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    resolver = load_script("resolve_model_target")
    collector = load_script("collect_source_context")
    target = repo / "vllm" / "model_executor" / "models" / "complex_branch_adapter.py"
    context = collector.collect_source_context(repo, resolver.resolve_model_target(repo, input_path=target))

    branch_kinds = {item["kind"] for item in context["branches"]}
    assert {"if", "match"}.issubset(branch_kinds)
    assert any(item["architecture_relevance_candidate"] == "high" for item in context["branches"])
    mapping_kinds = {item["kind"] for item in context["weight_mappings"]}
    assert "rename" in mapping_kinds
    assert "default_loader" in mapping_kinds or "loader_dispatch" in mapping_kinds
    assert len(context["weight_mappings"]) <= 6
    assert all(item["event_count"] >= 1 for item in context["weight_mappings"])
    assert all(item["evidence_ranges"] for item in context["weight_mappings"])


def test_method_importance_distinguishes_delegates_and_output_boundary(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    resolver = load_script("resolve_model_target")
    collector = load_script("collect_source_context")
    context = collector.collect_source_context(repo, resolver.resolve_model_target(repo, architecture="DenseForCausalLM"))

    methods = {item["qualified_name"]: item["importance_candidate"] for item in context["methods"]}
    assert methods["DenseModel.forward"] == "core"
    assert methods["DenseForCausalLM.forward"] == "supporting"
    assert methods["DenseForCausalLM.compute_logits"] == "core"


def test_capability_detection_precision() -> None:
    embedding_only = detect_capabilities("def embed_input_ids(self, input_ids): return self.embed_tokens(input_ids)")
    assert embedding_only["pooling"]["detected"] is False

    explicit_pooling = detect_capabilities("self.pooler = PoolingHead(config.hidden_size)\ndef pooling(self, hidden_states): pass")
    assert explicit_pooling["pooling"]["detected"] is True

    fused_moe_only = detect_capabilities("self.experts = FusedMoE(num_experts=config.num_experts)")
    assert fused_moe_only["moe"]["detected"] is True
    assert fused_moe_only["expert_parallel"]["detected"] is False

    ep_group = detect_capabilities("self.ep_group = get_ep_group(); self.ep_rank = self.ep_group.rank")
    assert ep_group["expert_parallel"]["detected"] is True

    variable_name_only = detect_capabilities("attn_mask = mask")
    assert variable_name_only["custom_attention"]["detected"] is False
