# Reproduction Guide

## Install

From the repository root:

```powershell
python --version
python -m pip install -e ".[dev]"
.\tools\setup-codex-dev.ps1
```

Check Draw.io MCP:

```powershell
codex mcp list
```

If Draw.io is missing, use `integrations/codex/config.toml.example` as a reference.

## Run The Skill

Restart VS Code Codex if needed, then enter:

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，
生成默认架构图。
```

Expected output:

```text
outputs/hy-v3/
  source-context.json
  architecture-plan.json
  evidence.json
  architecture.drawio
  report.md
  visual-review.md
  images/
```

## Validate Manually

```powershell
pytest

vllm-arch validate `
  --repo-root . `
  --context examples\hy_v3\source-context.json `
  --plan examples\hy_v3\architecture-plan.json `
  --evidence examples\hy_v3\evidence.json `
  --drawio examples\hy_v3\architecture.drawio `
  --images-dir examples\hy_v3\images
```

Open `examples/hy_v3/architecture.drawio` in Draw.io to inspect or edit the diagrams.

