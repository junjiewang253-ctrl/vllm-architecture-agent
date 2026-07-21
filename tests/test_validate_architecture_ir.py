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
        "schema_version": "0.1",
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
                        "badges": [],
                        "evidence": [{"type": "direct", "line": 1}],
                    },
                    {
                        "id": "decision",
                        "label": "Use MoE?",
                        "kind": "decision",
                        "badges": [],
                        "evidence": [{"type": "direct", "line": 2}],
                    },
                    {
                        "id": "ffn",
                        "label": "FFN",
                        "kind": "ffn",
                        "badges": [],
                        "evidence": [{"type": "direct", "line": 3}],
                    },
                    {
                        "id": "moe",
                        "label": "MoE",
                        "kind": "moe",
                        "badges": ["EP"],
                        "evidence": [{"type": "direct", "line": 4}],
                    },
                    {
                        "id": "merge",
                        "label": "Merge",
                        "kind": "merge",
                        "badges": [],
                        "evidence": [{"type": "derived", "lines": [3, 4]}],
                    },
                ],
                "edges": [
                    {
                        "id": "e1",
                        "source": "input",
                        "target": "decision",
                        "kind": "runtime",
                        "evidence": [{"type": "direct", "line": 2}],
                    },
                    {
                        "id": "e2",
                        "source": "decision",
                        "target": "ffn",
                        "kind": "conditional_false",
                        "evidence": [{"type": "direct", "line": 3}],
                    },
                    {
                        "id": "e3",
                        "source": "decision",
                        "target": "moe",
                        "kind": "conditional_true",
                        "evidence": [{"type": "direct", "line": 4}],
                    },
                    {
                        "id": "e4",
                        "source": "ffn",
                        "target": "merge",
                        "kind": "runtime",
                        "evidence": [{"type": "derived", "lines": [3, 4]}],
                    },
                    {
                        "id": "e5",
                        "source": "moe",
                        "target": "merge",
                        "kind": "runtime",
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


def test_residual_must_target_add_or_merge():
    module = load_validator_module()
    data = valid_ir()
    data["pages"][0]["edges"].append(
        {
            "id": "residual",
            "source": "input",
            "target": "ffn",
            "kind": "residual",
            "evidence": [{"type": "direct", "line": 8}],
        }
    )
    errors = module.validate_architecture_ir(data)
    assert any("residual edge must target" in error for error in errors)


def test_decision_requires_both_branches():
    module = load_validator_module()
    data = valid_ir()
    data["pages"][0]["edges"] = [
        edge for edge in data["pages"][0]["edges"] if edge["kind"] != "conditional_false"
    ]
    errors = module.validate_architecture_ir(data)
    assert any("requires true and false branches" in error for error in errors)


def test_overview_has_at_most_twelve_major_nodes():
    module = load_validator_module()
    data = valid_ir()
    base = deepcopy(data["pages"][0]["nodes"][0])
    for index in range(8):
        node = deepcopy(base)
        node["id"] = f"extra_{index}"
        node["label"] = f"Extra {index}"
        data["pages"][0]["nodes"].append(node)
    errors = module.validate_architecture_ir(data)
    assert any("maximum is 12" in error for error in errors)
