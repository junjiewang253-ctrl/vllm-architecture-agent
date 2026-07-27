from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path


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
    assert evidence["claims"] == []
