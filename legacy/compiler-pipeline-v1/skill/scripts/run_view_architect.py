#!/usr/bin/env python3
"""Compatibility wrapper for the legacy concept-to-view architect.

v1.2 architect mode compiles Architecture View from an Agent-authored
Architecture Design via compile_architecture_view.py. This wrapper remains only
for older tests and deterministic experiments that still call run_view_architect.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _legacy_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "legacy" / "run_view_architect.py"
    spec = importlib.util.spec_from_file_location("legacy_run_view_architect", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load legacy view architect: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    return int(_legacy_module().main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
