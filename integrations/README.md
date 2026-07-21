# Agent integrations

The canonical cross-Agent Skill lives under:

`src/skills/vllm-model-architecture-diagram/`

Do not duplicate its source here. This directory is reserved for Agent-host
specific manifests, MCP configuration examples and packaging scripts.

Current development uses Codex discovery through a local junction created by
`tools/setup-codex-dev.ps1`. Claude Code plugin packaging is a later milestone.
