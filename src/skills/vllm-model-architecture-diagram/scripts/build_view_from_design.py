#!/usr/bin/env python3
"""Compatibility wrapper for `compile_architecture_view.py`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_compiler() -> Any:
    path = Path(__file__).with_name("compile_architecture_view.py")
    spec = importlib.util.spec_from_file_location("compile_architecture_view_compat", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import compile_architecture_view.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_view_from_design(design: dict[str, Any]) -> dict[str, Any]:
    return _load_compiler().compile_architecture_view(design)


def main(argv: list[str] | None = None) -> int:
    return int(_load_compiler().main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
