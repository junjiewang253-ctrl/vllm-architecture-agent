from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from vllm_architecture_agent import cli
from test_v2_registry_and_context import make_repo


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "src" / "skills" / "vllm-model-architecture-diagram" / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_cli_only_exposes_v2_commands() -> None:
    parser = cli.build_parser()
    subcommands = parser._subparsers._group_actions[0].choices.keys()
    assert set(subcommands) == {"list-models", "prepare", "validate", "scan"}


def test_cli_prepare_creates_default_outputs(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    outputs = tmp_path / "prepared"
    exit_code = cli.main(
        [
            "prepare",
            "--repo-root",
            str(repo),
            "--architecture",
            "DenseForCausalLM",
            "--outputs-dir",
            str(outputs),
        ]
    )
    assert exit_code == 0
    assert (outputs / "source-context.json").exists()
    assert (outputs / "architecture-plan.template.json").exists()
    assert (outputs / "evidence.template.json").exists()
    plan = json.loads((outputs / "architecture-plan.template.json").read_text(encoding="utf-8"))
    assert plan["pages"] == []


def test_scan_models_directory_continues_on_invalid_file(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    bad_file = repo / "vllm" / "model_executor" / "models" / "broken.py"
    bad_file.write_text("class Broken(:\n", encoding="utf-8")
    scanner = load_script("scan_models_directory")

    report = scanner.scan_models_directory(repo)
    assert report["summary"]["total_python_files"] >= 8
    assert report["summary"]["failures"] == 0
    broken = next(item for item in report["entries"] if item["file"].endswith("broken.py"))
    assert broken["status"] == "unsupported"


def test_scan_report_has_distinct_capabilities(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    scanner = load_script("scan_models_directory")
    report = scanner.scan_models_directory(repo)
    by_name = {Path(item["file"]).name: item for item in report["entries"]}
    assert "moe" in by_name["moe_adapter.py"]["capabilities"]
    assert "multimodal" in by_name["multimodal_adapter.py"]["capabilities"]
    assert "recurrent_or_ssm" in by_name["hybrid_adapter.py"]["capabilities"]
    assert "custom_weight_loading" in by_name["custom_weight_loader_adapter.py"]["capabilities"]
    assert by_name["helper_module.py"]["status"] == "helper"
