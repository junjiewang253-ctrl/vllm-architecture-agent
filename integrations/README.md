# Agent integrations

The canonical cross-Agent Skill lives under:

`src/skills/vllm-model-architecture-diagram/`

Do not duplicate its source here. This directory is reserved for Agent-host
specific manifests, MCP configuration examples and packaging scripts.

Recommended Codex development setup:

```powershell
.\tools\setup-codex-dev.ps1
```

The script:

- checks Python;
- runs `pip install -e ".[dev]"`;
- creates `.agents/skills/vllm-model-architecture-diagram` as a Windows
  junction to `src/skills/vllm-model-architecture-diagram`;
- prints `codex mcp list` so the user can verify Draw.io MCP separately.

It does not edit global Codex MCP configuration. Use
`integrations/codex/config.toml.example` as a reference if Draw.io MCP is not
configured yet.

Claude Code plugin packaging is a later milestone.
