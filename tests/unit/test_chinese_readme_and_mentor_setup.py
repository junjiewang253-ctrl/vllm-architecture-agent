from __future__ import annotations

import re
from pathlib import Path

import vllm_architecture_agent


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_readme_starts_with_chinese_mentor_workflow() -> None:
    readme = read("README.md")
    first_50_lines = "\n".join(readme.splitlines()[:50])

    assert readme.startswith("# vLLM Architecture Agent")
    assert "## 导师最快使用" in first_50_lines
    assert ".\\tools\\setup-mentor.ps1 -ConfigureDrawioMcp" in first_50_lines
    assert "使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，生成默认架构图。" in first_50_lines
    assert "outputs/hy-v3/" in first_50_lines
    assert "不依赖 HY V3 固定模板" in readme
    assert "分析其他模型" in readme
    assert "常见故障排查" in readme


def test_user_facing_docs_have_no_mojibake() -> None:
    suspicious = ["浣跨敤", "鍒嗘瀽", "鐢熸垚", "榛樿", "鏋舵瀯"]
    paths = [
        "README.md",
        "integrations/README.md",
        "tools/setup-codex-dev.ps1",
        "tools/setup-mentor.ps1",
        "docs/mentor/00-executive-summary.md",
        "docs/mentor/01-architecture-walkthrough.md",
        "docs/mentor/02-skill-workflow.md",
        "docs/mentor/03-reproduction-guide.md",
        "docs/mentor/04-validation-and-coverage.md",
        "docs/mentor/05-limitations-and-future-work.md",
    ]
    offenders: dict[str, list[str]] = {}
    for path in paths:
        text = read(path)
        hits = [item for item in suspicious if item in text]
        if hits:
            offenders[path] = hits
    assert offenders == {}


def test_setup_mentor_configures_skill_and_drawio_mcp() -> None:
    script = read("tools/setup-mentor.ps1")
    assert "param(" in script
    assert "[switch]$ConfigureDrawioMcp" in script
    assert 'python -m pip install -e ".[dev]"' in script
    assert ".agents\\skills\\vllm-model-architecture-diagram" in script
    assert "[mcp_servers.drawio]" in script
    assert "@next-ai-drawio/mcp-server@latest" in script
    assert "codex mcp list" in script
    assert "使用 `$vllm-model-architecture-diagram 分析 samples/hy_v3.py，生成默认架构图。" in script


def test_mentor_package_builder_targets_v212() -> None:
    builder = read("tools/build_mentor_package.py")
    assert "v2.1.2" in builder
    assert "vllm-architecture-agent-v2.1.2-mentor.zip" in builder
    assert "tools/setup-mentor.ps1" in builder
    assert "docs/development/v2.1.2-chinese-mentor-usability-report.md" in builder
    assert "legacy" not in re.findall(r'"([^"]+)"', builder)


def test_version_is_212() -> None:
    pyproject = read("pyproject.toml")
    assert 'version = "2.1.2"' in pyproject
    assert vllm_architecture_agent.__version__ == "2.1.2"
