#!/usr/bin/env python3
"""Compatibility wrapper for deterministic baseline Architecture Design.

The v1.2 architect default no longer uses this script. It remains only for
legacy tests/integrations that call `run_design_architect.py` directly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_script(filename: str, module_name: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to import {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _fallback_brief(concept_graph: dict[str, Any], fact_graph: dict[str, Any]) -> dict[str, Any]:
    source_file = fact_graph.get("source_file")
    source_hash = "0" * 64
    if isinstance(source_file, str) and Path(source_file).exists():
        source_hash = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    summaries = "\n".join(str(fact.get("summary", "")) for fact in fact_graph.get("facts", []) if isinstance(fact, dict))
    return {
        "schema_version": "1.0",
        "model_name": concept_graph.get("model_name") or fact_graph.get("model_name") or "model",
        "source_file": source_file or "",
        "source_sha256": source_hash,
        "model_profile": {
            "decoder_class": "HYV3DecoderLayer" if "DecoderLayer" in summaries else None,
            "attention_class": "HYV3Attention" if "Attention" in summaries else None,
            "moe_class": "HYV3MoEFused" if "FusedMoE" in summaries or "MoE" in summaries else None,
            "load_weights_entrypoints": ["load_weights"] if "load_weights" in summaries else [],
        },
    }


def build_architecture_design_graph(concept_graph: dict[str, Any], fact_graph: dict[str, Any]) -> dict[str, Any]:
    builder = _load_script("build_baseline_design.py", "build_baseline_design_compat")
    boundary_report = {"external_components": concept_graph.get("boundaries", [])}
    return builder.build_baseline_design(_fallback_brief(concept_graph, fact_graph), concept_graph, boundary_report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture_concept", type=Path)
    parser.add_argument("source_fact_graph", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        design = build_architecture_design_graph(_load_json(args.architecture_concept), _load_json(args.source_fact_graph))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote baseline Architecture Design template to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
