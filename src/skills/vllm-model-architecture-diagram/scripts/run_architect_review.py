#!/usr/bin/env python3
"""Compatibility wrapper for the deterministic baseline Concept Graph builder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_baseline() -> Any:
    path = Path(__file__).with_name("build_baseline_concept_graph.py")
    spec = importlib.util.spec_from_file_location("build_baseline_concept_graph_compat", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import build_baseline_concept_graph.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_architecture_design(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible name for existing tests and integrations."""

    return _load_baseline().build_architecture_design(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return int(_load_baseline().main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
