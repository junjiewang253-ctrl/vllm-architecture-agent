# vLLM Architecture Agent

An Agent Skill that analyzes a vLLM model adapter Python file and uses the
Next AI Draw.io MCP server to generate an editable architecture diagram.

## Current status

Version 0.2 contains:

- a cross-Agent core Skill under `src/skills/`;
- a Codex development link under `.agents/skills/`;
- a source-grounded Python AST extractor;
- vLLM interpretation and diagram-style references;
- a synthetic dense/MoE sample;
- extractor unit tests;
- a verified Draw.io MCP smoke test.

The current extractor reports syntax-level facts. The Agent is still
responsible for semantic interpretation, layout planning and Draw.io XML
generation.

## Repository layout

```text
src/skills/vllm-model-architecture-diagram/
  SKILL.md
  scripts/extract_architecture.py
  references/
  schemas/

.agents/skills/                 Codex development discovery path
samples/                        Manual end-to-end inputs
tests/                          Deterministic extractor tests
outputs/                        Generated analysis and diagrams
integrations/                   Future Agent-specific packaging
```

## Prerequisites

- Python 3.10+
- Node.js and `npx`
- VS Code Codex or another Agent supporting Agent Skills
- Next AI Draw.io MCP configured in the Agent host

## Codex development setup

From PowerShell at the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup-codex-dev.ps1
```

Restart the Codex session and verify that
`vllm-model-architecture-diagram` is discoverable.

## Run tests

```powershell
python -m pip install -e ".[dev]"
pytest
```

## Run the extractor manually

```powershell
python `
  src\skills\vllm-model-architecture-diagram\scripts\extract_architecture.py `
  samples\simple_model.py `
  --output outputs\simple-model-source-analysis.json
```

## End-to-end Codex prompt

```text
Use $vllm-model-architecture-diagram to analyze samples/simple_model.py.
Run the bundled extractor first and use its JSON as the factual basis.
Generate one overview page through the drawio MCP and export:
outputs/simple-model-overview.drawio
outputs/simple-model-overview.svg
```

## Next milestones

- v0.3: define an explicit architecture IR.
- v0.4: expand vLLM patterns and test real adapters such as `hy_v3.py`.
- v0.5: add deterministic Draw.io rendering and XML validation.
- v0.6: package Codex and Claude Code integrations from the shared core Skill.
