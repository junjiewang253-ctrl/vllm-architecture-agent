# vLLM Architecture Agent

An Agent Skill for source-grounded vLLM model adapter architecture analysis.

## Current Status

Version 1.2 is an **Evidence-Grounded Agent Diagram Designer**. It keeps
deterministic extraction, fact graphs, concept graphs, validation and rendering,
but requires the current Codex Agent to author the Architecture Design Graph.

```text
Python source
-> Source Analysis
-> Source Fact Graph
-> Architecture Concept Graph
-> Agent-authored Architecture Design Graph
-> Architecture View Graph
-> Pattern-based Layout
-> Draw.io
```

Concepts are not rendered directly. Design states the engineering question,
main story, branches, boundaries, nodes, ports and edge semantics. View is
compiled from Design and is the only renderer input.

## v1.2 Workflow

Prepare deterministic context:

```powershell
vllm-arch prepare `
  --input samples\hy_v3.py `
  --model-name hy-v3 `
  --outputs-dir outputs
```

Codex then writes:

```text
outputs/<model>-architecture-design.json
```

Finalize:

```powershell
vllm-arch finalize `
  --design outputs\hy-v3-architecture-design.json `
  --model-name hy-v3 `
  --outputs-dir outputs `
  --source-file samples\hy_v3.py
```

`vllm-arch run --mode architect` is only a convenience wrapper and requires
`--architecture-design`; it does not silently fall back to the deterministic
template.

## Outputs

For `samples/hy_v3.py`, v1.2 produces:

- `outputs/<model>-source-analysis.json`
- `outputs/<model>-semantic-inventory.json`
- `outputs/<model>-source-fact-graph.json`
- `outputs/<model>-architecture-concept.json`
- `outputs/<model>-boundary-report.json`
- `outputs/<model>-architect-brief.json`
- `outputs/<model>-architecture-design.json`
- `outputs/<model>-architecture-view.json`
- `outputs/<model>-layout-plan.json`
- `outputs/<model>-layout-metrics.json`
- `outputs/<model>-architecture.drawio`
- `outputs/<model>-architecture-report.md`

Expected HY V3 pages:

- Model Execution Overview
- Decoder Block
- Attention Adaptation
- MoE Execution
- Checkpoint and Weight Loading
- Parallel Strategies
- vLLM Adapter Boundary

## Validation

Finalize runs:

- Architecture Design validation
- Architecture View validation
- Draw.io semantic validation
- Visual layout validation

Any validator failure stops the workflow.

## Boundaries

The project does not recursively analyze external vLLM components. Imported
component internals such as `Attention`, `HpcRopeNorm`, `FusedMoE` and
`AutoWeightsLoader` are represented as explicit boundaries unless the input file
itself proves the behavior.
