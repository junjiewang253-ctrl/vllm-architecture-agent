from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_PATH = (
    ROOT
    / "src"
    / "skills"
    / "vllm-model-architecture-diagram"
    / "scripts"
    / "extract_architecture.py"
)
SAMPLE_PATH = ROOT / "samples" / "simple_model.py"


def load_extractor_module():
    spec = importlib.util.spec_from_file_location("extract_architecture", EXTRACTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analysis():
    module = load_extractor_module()
    return module.extract_architecture(SAMPLE_PATH)


def test_extracts_expected_classes():
    result = analysis()
    class_names = {item["name"] for item in result["classes"]}
    assert {
        "SimpleAttention",
        "SimpleMLP",
        "SimpleMoE",
        "SimpleDecoderLayer",
        "SimpleForCausalLM",
    }.issubset(class_names)


def test_extracts_module_assignments():
    result = analysis()
    assignments = {
        (item["owner_class"], item["attribute"], item["constructor"])
        for item in result["module_assignments"]
    }
    assert ("SimpleDecoderLayer", "self_attn", "SimpleAttention") in assignments
    assert ("SimpleForCausalLM", "embed_tokens", "Embedding") in assignments
    assert ("SimpleMoE", "router", "Linear") in assignments


def test_preserves_forward_call_order():
    result = analysis()
    flow = next(
        item
        for item in result["forward_flows"]
        if item["class"] == "SimpleForCausalLM"
    )
    targets = [call["target"] for call in flow["calls"]]
    assert targets == [
        "self.embed_tokens",
        "self.layer",
        "self.norm",
        "self.lm_head",
    ]
    assert [call["order"] for call in flow["calls"]] == [1, 2, 3, 4]


def test_extracts_conditional_modules():
    result = analysis()
    condition = next(
        item
        for item in result["conditions"]
        if item["owner_class"] == "SimpleDecoderLayer"
    )
    assert condition["condition"] == "use_moe"
    assert condition["true_assignments"][0]["constructor"] == "SimpleMoE"
    assert condition["false_assignments"][0]["constructor"] == "SimpleMLP"


def test_every_major_record_has_line_numbers():
    result = analysis()
    assert all(item["line"] > 0 for item in result["classes"])
    assert all(item["line"] > 0 for item in result["module_assignments"])
    assert all(item["line"] > 0 for item in result["forward_flows"])
    assert all(item["line"] > 0 for item in result["conditions"])


def test_result_is_json_serializable():
    result = analysis()
    rendered = json.dumps(result, ensure_ascii=False)
    assert '"schema_version": "0.1"' in rendered
