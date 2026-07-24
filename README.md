# vLLM Architecture Agent

An Agent Skill for source-grounded vLLM model adapter architecture analysis.

## Current Status

Version 1.0 introduces **Agent Architect Mode**. The project is no longer just
an AST-to-diagram visualizer. The default research direction is:

```text
Python source
-> Source Fact Graph
-> Agent Architecture Design
-> Architecture Concept Graph
-> Architecture Views
-> Deterministic Renderer
-> Draw.io
```

Scripts still define the trusted boundary: what facts exist, what evidence is
valid and what external behavior cannot be claimed. The Agent decides how those
facts should be understood, grouped and explained as architecture concepts.

## v1.0 Artifacts

For `samples/hy_v3.py`, Architect Mode generates:

- `outputs/hy-v3-v1.0-source-analysis.json`
- `outputs/hy-v3-v1.0-semantic-inventory.json`
- `outputs/hy-v3-v1.0-source-fact-graph.json`
- `outputs/hy-v3-v1.0-architecture-design.json`
- `outputs/hy-v3-v1.0-architecture-view.json`
- `outputs/hy-v3-v1.0-boundary-report.json`
- `outputs/hy-v3-v1.0-layout-plan.json`
- `outputs/hy-v3-v1.0-architecture.drawio`
- `outputs/hy-v3-v1.0-mentor-report.md`

## Architecture Pages

The v1.0 HY V3 Architecture View contains seven concept-driven pages:

- HY V3 vLLM Adapter Overview
- Transformer Execution Flow
- Attention Implementation
- MoE Execution Strategy
- Parallel Execution Strategy
- Checkpoint Adaptation
- vLLM Integration Boundary

## Run

```powershell
python -m pip install -e ".[dev]"
pytest

vllm-arch run `
  --mode architect `
  --input samples\hy_v3.py `
  --model-name hy-v3-v1.0 `
  --outputs-dir outputs
```

Older deterministic and reviewed v0.9.1 modes remain available for regression
and patch-audited workflows.

## Repository Layout

```text
src/skills/vllm-model-architecture-diagram/
  facts/       Source Fact Graph layer
  concepts/    Architecture Concept Layer
  reviews/     Agent Architect and review responsibilities
  views/       Architecture View planning
  schemas/     JSON contracts
  scripts/     deterministic builders, validators, layout and renderer
  references/  review and architecture rules
```

## Boundaries

v1.0 does not recursively analyze external vLLM components. Imported component
internals such as `Attention`, `FusedMoE` and `AutoWeightsLoader` are represented
as explicit boundaries unless the input file itself proves the behavior.
