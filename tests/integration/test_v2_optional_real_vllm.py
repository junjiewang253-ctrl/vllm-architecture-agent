from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "src" / "skills" / "vllm-model-architecture-diagram" / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif("VLLM_REPO_ROOT" not in os.environ, reason="requires a local vLLM checkout")
def test_scan_real_vllm_repo_does_not_raise(tmp_path: Path) -> None:
    repo = Path(os.environ["VLLM_REPO_ROOT"])
    scanner = load_script("scan_models_directory")
    report = scanner.scan_models_directory(repo)
    assert report["summary"]["failures"] == 0
    assert report["summary"]["total_python_files"] > 0
