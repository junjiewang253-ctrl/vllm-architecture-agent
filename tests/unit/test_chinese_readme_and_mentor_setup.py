from __future__ import annotations

import re
from pathlib import Path

import vllm_architecture_agent


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_readme_starts_with_general_quick_start() -> None:
    readme = read("README.md")
    first_50_lines = "\n".join(readme.splitlines()[:50])

    assert readme.startswith("# vLLM Architecture Agent")
    assert "## 快速开始" in first_50_lines
    assert ".\\tools\\setup.ps1 -ConfigureDrawioMcp" in first_50_lines
    assert "使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，生成默认架构图。" in first_50_lines
    assert "outputs/hy-v3/" in first_50_lines
    assert "固定模板" in readme
    assert "分析其他模型" in readme
    assert "## 常见问题" in readme
    assert "导师" not in readme


def test_readme_embeds_all_hy_v3_example_images() -> None:
    readme = read("README.md")
    image_names = [
        "model-architecture-and-execution.png",
        "decoder-and-attention.png",
        "moe-architecture-and-routing.png",
        "parallelism-configuration-and-weight-loading.png",
    ]
    for image_name in image_names:
        relative = f"examples/hy_v3/images/{image_name}"
        assert relative in readme
        assert (ROOT / relative).is_file()
    integrated_images = [
        "examples/integrated-flow/hy_v3/architecture.png",
        "examples/integrated-flow/qwen3_moe/architecture.png",
    ]
    for relative in integrated_images:
        assert relative in readme
        assert (ROOT / relative).is_file()


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


def test_general_setup_entrypoint_wraps_installer() -> None:
    script = read("tools/setup.ps1")
    assert "[switch]$ConfigureDrawioMcp" in script
    assert '"setup-mentor.ps1"' in script
    assert "& $installer -RepoRoot $RepoRoot -ConfigureDrawioMcp" in script


def test_package_builder_targets_v220() -> None:
    builder = read("tools/build_mentor_package.py")
    assert "v2.2.0" in builder
    assert "vllm-architecture-agent-v2.2.0.zip" in builder
    assert "tools/setup.ps1" in builder
    assert "tools/setup-mentor.ps1" in builder
    assert "docs/development/v2.2-integrated-visual-contract-report.md" in builder
    assert "examples/integrated-flow" in builder
    assert "legacy" not in re.findall(r'"([^"]+)"', builder)


def test_version_is_220() -> None:
    pyproject = read("pyproject.toml")
    assert 'version = "2.2.0"' in pyproject
    assert vllm_architecture_agent.__version__ == "2.2.0"
