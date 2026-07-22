from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    ROOT
    / "src"
    / "skills"
    / "vllm-model-architecture-diagram"
    / "scripts"
    / "validate_architecture_ir.py"
)


def load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_architecture_ir", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_ir():
    return {
        "schema_version": "0.2",
        "model_name": "Example",
        "detail_level": "overview",
        "pages": [
            {
                "id": "overview",
                "title": "Overview",
                "nodes": [
                    {
                        "id": "input",
                        "label": "Input",
                        "kind": "input",
                        "phase": "runtime",
                        "scope": "ExampleModel.forward",
                        "parent_id": None,
                        "badges": [],
                        "evidence": [{"type": "direct", "line": 1}],
                    },
                    {
                        "id": "decoder",
                        "label": "Decoder",
                        "kind": "repeated_block",
                        "phase": "runtime",
                        "scope": "ExampleModel.forward",
                        "parent_id": None,
                        "badges": ["PP"],
                        "repetition": {"count_expression": "config.num_hidden_layers"},
                        "variants": [
                            {
                                "condition": "layer_idx < config.first_k_dense_replace",
                                "component": "ExampleFeedForward",
                                "phase": "construction",
                            },
                            {
                                "condition": "layer_idx >= config.first_k_dense_replace",
                                "component": "ExampleMoE",
                                "phase": "construction",
                            },
                        ],
                        "evidence": [{"type": "direct", "line": 2}],
                    },
                    {
                        "id": "add",
                        "label": "Add",
                        "kind": "add",
                        "phase": "runtime",
                        "scope": "ExampleModel.forward",
                        "parent_id": None,
                        "badges": [],
                        "evidence": [{"type": "direct", "line": 3}],
                    },
                    {
                        "id": "logits",
                        "label": "LogitsProcessor",
                        "kind": "logits_processor",
                        "phase": "runtime",
                        "scope": "ExampleForCausalLM.compute_logits",
                        "parent_id": None,
                        "badges": [],
                        "evidence": [{"type": "direct", "line": 4}],
                    },
                ],
                "edges": [
                    {
                        "id": "e1",
                        "source": "input",
                        "target": "decoder",
                        "kind": "runtime",
                        "phase": "runtime",
                        "scope": "ExampleModel.forward",
                        "evidence": [{"type": "direct", "line": 2}],
                    },
                    {
                        "id": "e2",
                        "source": "decoder",
                        "target": "add",
                        "kind": "runtime",
                        "phase": "runtime",
                        "scope": "ExampleModel.forward",
                        "evidence": [{"type": "direct", "line": 3}],
                    },
                    {
                        "id": "e3",
                        "source": "decoder",
                        "target": "add",
                        "kind": "residual",
                        "phase": "runtime",
                        "scope": "ExampleModel.forward",
                        "evidence": [{"type": "direct", "line": 3}],
                    },
                    {
                        "id": "e4",
                        "source": "add",
                        "target": "logits",
                        "kind": "summary",
                        "phase": "runtime",
                        "scope": "ExampleForCausalLM.compute_logits",
                        "target_port": "hidden_states",
                        "evidence": [{"type": "derived", "lines": [3, 4]}],
                    },
                ],
            }
        ],
        "unresolved": [],
    }


def test_valid_ir_passes():
    module = load_validator_module()
    assert module.validate_architecture_ir(valid_ir()) == []


def test_unknown_edge_endpoint_fails():
    module = load_validator_module()
    data = valid_ir()
    data["pages"][0]["edges"][0]["target"] = "missing"
    errors = module.validate_architecture_ir(data)
    assert any("unknown node" in error for error in errors)


def test_invalid_parent_id_fails():
    module = load_validator_module()
    data = valid_ir()
    data["pages"][0]["nodes"][1]["parent_id"] = "missing"
    errors = module.validate_architecture_ir(data)
    assert any("parent_id references unknown node" in error for error in errors)


def test_runtime_edge_cross_scope_fails():
    module = load_validator_module()
    data = valid_ir()
    data["pages"][0]["edges"][3]["kind"] = "runtime"
    errors = module.validate_architecture_ir(data)
    assert any("runtime edge cannot cross scopes" in error for error in errors)
    assert any("cross-scope edge must use invocation or summary" in error for error in errors)


def test_invocation_edge_cross_scope_passes():
    module = load_validator_module()
    data = valid_ir()
    data["pages"][0]["edges"][3]["kind"] = "invocation"
    errors = module.validate_architecture_ir(data)
    assert errors == []


def test_construction_node_with_runtime_edge_fails():
    module = load_validator_module()
    data = valid_ir()
    data["pages"][0]["nodes"][1]["phase"] = "construction"
    errors = module.validate_architecture_ir(data)
    assert any("construction phase nodes cannot use runtime edges" in error for error in errors)


def test_weight_mapping_wrong_phase_fails():
    module = load_validator_module()
    data = valid_ir()
    data["pages"][0]["edges"][0]["kind"] = "weight_mapping"
    data["pages"][0]["edges"][0]["phase"] = "runtime"
    errors = module.validate_architecture_ir(data)
    assert any("weight_mapping edge must use checkpoint_loading phase" in error for error in errors)


def test_repetition_missing_count_expression_fails():
    module = load_validator_module()
    data = valid_ir()
    data["pages"][0]["nodes"][1]["repetition"] = {"count_expression": ""}
    errors = module.validate_architecture_ir(data)
    assert any("repetition.count_expression must be non-empty" in error for error in errors)


def test_invalid_badge_fails():
    module = load_validator_module()
    data = valid_ir()
    data["pages"][0]["nodes"][1]["badges"] = ["TP", "DP"]
    errors = module.validate_architecture_ir(data)
    assert any("invalid badge" in error for error in errors)


def test_residual_must_target_add_or_merge():
    module = load_validator_module()
    data = valid_ir()
    data["pages"][0]["edges"][2]["target"] = "decoder"
    errors = module.validate_architecture_ir(data)
    assert any("residual edge must target" in error for error in errors)


def test_overview_has_at_most_twelve_major_nodes():
    module = load_validator_module()
    data = valid_ir()
    base = deepcopy(data["pages"][0]["nodes"][0])
    for index in range(9):
        node = deepcopy(base)
        node["id"] = f"extra_{index}"
        node["label"] = f"Extra {index}"
        data["pages"][0]["nodes"].append(node)
    errors = module.validate_architecture_ir(data)
    assert any("maximum is 12" in error for error in errors)
