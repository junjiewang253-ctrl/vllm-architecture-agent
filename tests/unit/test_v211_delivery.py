from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "src" / "skills" / "vllm-model-architecture-diagram" / "scripts"
EXAMPLE = ROOT / "examples" / "hy_v3"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_example(repo_root: Path, example_dir: Path) -> None:
    context = json.loads((example_dir / "source-context.json").read_text(encoding="utf-8"))
    plan = json.loads((example_dir / "architecture-plan.json").read_text(encoding="utf-8"))
    evidence = json.loads((example_dir / "evidence.json").read_text(encoding="utf-8"))

    evidence_mod = load_script("validate_evidence")
    plan_mod = load_script("validate_architecture_plan")
    drawio_mod = load_script("validate_drawio")

    evidence_errors, _, evidence_summary = evidence_mod.validate_evidence(
        evidence,
        context=context,
        plan=plan,
        repo_root=repo_root,
    )
    assert evidence_errors == []
    assert evidence_summary == {"direct": 14, "derived": 4, "external": 2}

    plan_errors, _ = plan_mod.validate_plan(plan, evidence=evidence, context=context, repo_root=repo_root)
    assert plan_errors == []
    coverage = plan_mod.summarize_coverage(plan, context)
    assert coverage["classes"]["total"] == 6
    assert coverage["classes"]["unreviewed"] == 0
    assert coverage["methods"]["total"] == 20
    assert coverage["methods"]["unreviewed"] == 0
    assert coverage["weight_mappings"]["total"] == 12
    assert coverage["weight_mappings"]["covered"] == 12

    drawio_errors = drawio_mod.validate_drawio(
        example_dir / "architecture.drawio",
        plan=plan,
        images_dir=example_dir / "images",
    )
    assert drawio_errors == []


def test_hy_v3_golden_example_validates() -> None:
    _validate_example(ROOT, EXAMPLE)


def test_hy_v3_example_is_relocatable(tmp_path: Path) -> None:
    relocated = tmp_path / "repo"
    (relocated / "samples").mkdir(parents=True)
    shutil.copyfile(ROOT / "samples" / "hy_v3.py", relocated / "samples" / "hy_v3.py")
    shutil.copytree(EXAMPLE, relocated / "examples" / "hy_v3")

    _validate_example(relocated, relocated / "examples" / "hy_v3")


def test_hy_v3_example_has_no_user_absolute_paths() -> None:
    windows_drive = r"D:" + "\\"
    windows_profile = r"C:" + "\\" + "Users"
    posix_home = "/" + "home/"
    pattern = re.compile("|".join(re.escape(part) for part in [windows_drive, windows_profile, posix_home]))
    offenders: list[str] = []
    for path in EXAMPLE.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".md", ".drawio"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_report_truthfulness_matches_real_validation() -> None:
    _validate_example(ROOT, EXAMPLE)
    report = (EXAMPLE / "report.md").read_text(encoding="utf-8")
    assert "Evidence validation: passed" in report
    assert "Architecture plan validation: passed" in report
    assert "Draw.io validation: passed" in report
