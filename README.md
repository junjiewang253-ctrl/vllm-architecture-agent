# vLLM Architecture Agent

An Agent Skill for source-grounded vLLM model adapter architecture analysis.

## Current Status

Version 1.0.1 adds **Agent Architect View Generation**. v1.0 introduced
architecture concepts, but drawing concepts directly produced concept cards.
v1.0.1 adds a separate Architecture View Graph so concepts become readable
architecture expressions: components, runtime data flow, module relationships,
checkpoint mappings, parallel strategies and external boundaries.

```text
Python source
-> Source Fact Graph
-> Architecture Concept Graph
-> Architecture View Graph
-> Deterministic Renderer
-> Draw.io
```

## v1.0.1 Artifacts

For `samples/hy_v3.py`, Architect Mode generates:

- `outputs/hy-v3-v1.0.1-source-fact-graph.json`
- `outputs/hy-v3-v1.0.1-architecture-concept.json`
- `outputs/hy-v3-v1.0.1-architecture-view.json`
- `outputs/hy-v3-v1.0.1-boundary-report.json`
- `outputs/hy-v3-v1.0.1-layout-plan.json`
- `outputs/hy-v3-v1.0.1-architecture.drawio`
- `outputs/hy-v3-v1.0.1-architecture-report.md`

## Default Pages

- Model Overview
- Attention Implementation
- MoE Execution Strategy
- Parallel Strategy
- Checkpoint Adaptation
- vLLM Integration Boundary

These pages are not concept cards. They contain View nodes such as `QKV
Projection`, `Q/K/V Split`, `KV Cache Boundary`, `Router`, `FusedMoE`, `HF
Checkpoint`, and `vLLM Attention Backend`.

## Run

```powershell
python -m pip install -e ".[dev]"
pytest

vllm-arch run `
  --mode architect `
  --input samples\hy_v3.py `
  --model-name hy-v3-v1.0.1 `
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
  views/       Architecture View Graph planning
  schemas/     JSON contracts
  scripts/     builders, validators, layout and renderer
  references/  review and architecture rules
```

## Boundaries

v1.0.1 does not recursively analyze external vLLM components. Imported component
internals such as `Attention`, `FusedMoE` and `AutoWeightsLoader` are represented
as explicit boundaries unless the input file itself proves the behavior.
