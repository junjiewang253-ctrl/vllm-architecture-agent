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
SIMPLE_PATH = ROOT / "samples" / "simple_model.py"
HY_V3_PATH = ROOT / "samples" / "hy_v3.py"


def load_extractor_module():
    spec = importlib.util.spec_from_file_location("extract_architecture", EXTRACTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract(path: Path):
    module = load_extractor_module()
    return module.extract_architecture(path)


def walk_control_flow(items):
    for item in items:
        yield item
        if item.get("type") == "if":
            yield from walk_control_flow(item.get("then", []))
            yield from walk_control_flow(item.get("else", []))
        elif item.get("type") == "for":
            yield from walk_control_flow(item.get("body", []))
            yield from walk_control_flow(item.get("else", []))


def test_extracts_expected_classes():
    result = extract(SIMPLE_PATH)
    class_names = {item["name"] for item in result["classes"]}
    assert {
        "SimpleAttention",
        "SimpleMLP",
        "SimpleMoE",
        "SimpleDecoderLayer",
        "SimpleForCausalLM",
    }.issubset(class_names)


def test_classifies_submodules():
    result = extract(SIMPLE_PATH)
    assignments = {
        (item["owner_class"], item["attribute"], item["constructor"], item["assignment_kind"])
        for item in result["module_assignments"]
    }
    assert ("SimpleDecoderLayer", "self_attn", "SimpleAttention", "submodule") in assignments
    assert ("SimpleForCausalLM", "embed_tokens", "Embedding", "submodule") in assignments


def test_state_calls_are_not_submodules():
    result = extract(HY_V3_PATH)
    by_source = {item["source"]: item for item in result["module_assignments"]}
    assert by_source["self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)"]["assignment_kind"] == "state"
    assert by_source["self.use_qk_norm = getattr(config, 'qk_norm', False)"]["assignment_kind"] == "state"
    assert by_source["self.num_moe_layers = len(self.moe_layers)"]["assignment_kind"] == "state"


def test_preserves_flat_forward_call_order_for_compatibility():
    result = extract(SIMPLE_PATH)
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


def test_extracts_conditional_modules():
    result = extract(SIMPLE_PATH)
    condition = next(
        item
        for item in result["conditions"]
        if item["owner_class"] == "SimpleDecoderLayer"
    )
    assert condition["condition"] == "use_moe"
    assert condition["true_assignments"][0]["constructor"] == "SimpleMoE"
    assert condition["false_assignments"][0]["constructor"] == "SimpleMLP"


def test_attention_control_flow_keeps_hpc_and_fallback_separate():
    result = extract(HY_V3_PATH)
    flow = next(
        item
        for item in result["forward_control_flows"]
        if item["class"] == "HYV3Attention"
    )
    top_if = next(item for item in flow["body"] if item.get("type") == "if")
    assert top_if["condition"] == "self.hpc_rope_norm is not None"
    then_calls = json.dumps(top_if["then"])
    else_calls = json.dumps(top_if["else"])
    assert "self.hpc_rope_norm" in then_calls
    assert "self.rotary_emb" not in then_calls
    assert "self.rotary_emb" in else_calls
    assert "self.hpc_rope_norm" not in else_calls


def test_model_forward_resolves_layer_loop():
    result = extract(HY_V3_PATH)
    flow = next(
        item
        for item in result["forward_control_flows"]
        if item["class"] == "HYV3Model"
    )
    loop = next(item for item in flow["body"] if item.get("type") == "for")
    body = list(walk_control_flow(loop["body"]))
    assignment = next(item for item in body if item.get("type") == "assignment")
    assert assignment["value"]["target"] == "layer"
    assert assignment["value"]["resolved_collection"] == "self.layers"


def test_make_layers_factory_is_extracted():
    result = extract(HY_V3_PATH)
    factory = result["layer_factories"][0]
    assert factory["targets"] == ["self.start_layer", "self.end_layer", "self.layers"]
    assert factory["repeat_expression"] == "config.num_hidden_layers"
    assert factory["layer_constructor"] == "HYV3DecoderLayer"


def test_residual_add_is_structured():
    result = extract(HY_V3_PATH)
    flow = next(
        item
        for item in result["forward_control_flows"]
        if item["class"] == "HYV3Model"
    )
    add_assignment = next(
        item
        for item in flow["body"]
        if item.get("type") == "assignment"
        and isinstance(item.get("value"), dict)
        and item["value"].get("type") == "add"
    )
    assert add_assignment["targets"] == ["hidden_states"]
    assert add_assignment["value"]["right"] == "residual"


def test_weight_mappings_are_normalized_and_unique():
    result = extract(HY_V3_PATH)
    mappings = result["weight_mappings"]
    keys = {(item["source"], item["target"], json.dumps(item["shard"])) for item in mappings}
    assert len(keys) == len(mappings)
    assert (".q_proj", ".qkv_proj", '"q"') in keys
    assert (".gate_proj", ".gate_up_proj", "0") in keys


def test_every_major_record_has_line_numbers():
    result = extract(SIMPLE_PATH)
    assert all(item["line"] > 0 for item in result["classes"])
    assert all(item["line"] > 0 for item in result["module_assignments"])
    assert all(item["line"] > 0 for item in result["forward_flows"])
    assert all(item["line"] > 0 for item in result["conditions"])


def test_result_is_json_serializable():
    result = extract(SIMPLE_PATH)
    rendered = json.dumps(result, ensure_ascii=False)
    assert '"schema_version": "0.3"' in rendered
