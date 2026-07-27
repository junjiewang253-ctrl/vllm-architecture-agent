---
name: vllm-model-architecture-diagram
description: Analyze a vLLM model adapter Python file and generate a source-grounded, Agent-designed, View-Graph-rendered Draw.io architecture diagram and architecture report.
compatibility: Requires Python 3, access to the input source file, and a connected Draw.io MCP server for optional opening/export.
---

# Goal

Analyze a vLLM model adapter Python file and create a human-readable,
evidence-grounded architecture diagram.

v1.2 uses six layers:

```text
Python source
-> Source Analysis
-> Source Fact Graph
-> Architecture Concept Graph
-> Agent-authored Architecture Design Graph
-> Architecture View Graph
-> Deterministic Draw.io Renderer
```

Scripts define what is true and evidenced. The current Codex Agent designs the
Architecture Design Graph: page questions, primary stories, branches, merges,
boundaries, nodes, ports and edge semantics. The renderer reads Architecture
View Graph only; it must not read Concept Graph directly or invent architecture.

# Modes

- `architect`: v1.2 Agent Architect mode. Requires an Agent-authored Design.
- `deterministic`: CI/regression fallback.
- `reviewed`: v0.9.1 patch-audited workflow.

# Default Architect Workflow

1. Prepare deterministic context:

```text
vllm-arch prepare --input samples/hy_v3.py --model-name hy-v3 --outputs-dir outputs
```

This writes:

- `outputs/<model>-source-analysis.json`
- `outputs/<model>-semantic-inventory.json`
- `outputs/<model>-source-fact-graph.json`
- `outputs/<model>-architecture-concept.json`
- `outputs/<model>-boundary-report.json`
- `outputs/<model>-architect-brief.json`
- `outputs/<model>-architecture-design.template.json`

2. Codex Agent reads the input source, Architect Brief, Concept Graph, Boundary
   Report, `references/architect-design-prompt.md`,
   `references/diagram-grammar.md`, `references/page-patterns.md`, and
   `references/visual-quality-rubric.md`.

3. Codex Agent writes:

```text
outputs/<model>-architecture-design.json
```

The Design must use schema `1.0`, `author.type = "agent"`, and source-backed
`concept_refs` and `fact_refs`.

4. Finalize:

```text
vllm-arch finalize --design outputs/<model>-architecture-design.json --model-name <model> --outputs-dir outputs --source-file samples/hy_v3.py
```

Finalize runs:

- `validate_architecture_design.py`
- `compile_architecture_view.py`
- `validate_architecture_view.py`
- `apply_view_layout.py`
- `render_drawio.py`
- `validate_drawio.py`
- `validate_visual_layout.py`
- `build_mentor_report.py`

Any validator failure stops the workflow. Do not use Draw.io MCP or manual XML
edits to bypass validation.

# Architect Rules

Codex may:

- decide pages and page questions;
- choose primary stories, branches and merge points;
- decide whether a concept becomes a component, data node, process, container,
  badge or annotation;
- separate runtime flow from dependency, mapping, parallelism and boundaries;
- perform one visual-only review by patching display, size, route hints,
  regions or lanes.

Codex must not:

- modify Source Fact Graph or Concept Graph to hide missing evidence;
- create nodes or edges without source-backed facts;
- directly write Draw.io XML;
- mark external imported behavior as direct local behavior;
- change edge source/target, phase, evidence or concept/fact references during
  visual review.

# Expected HY V3 Pages

For `samples/hy_v3.py`, an Agent-authored Design should produce:

- Model Execution Overview
- Decoder Block
- Attention Adaptation
- MoE Execution
- Checkpoint and Weight Loading
- Parallel Strategies
- vLLM Adapter Boundary

`simple_model.py` should produce a smaller page set; do not hardcode HY V3 page
lists into production architect mode.

# Draw.io MCP

After scripts generate `.drawio`, Draw.io MCP may open, inspect and export.
MCP must not add or remove semantic View nodes or edges. If visual-only MCP
changes are made, rerun Draw.io and visual validators.

# Completion Report

Report generated files, page list, View node/edge counts, validators, test
result, Draw.io path, architecture report path, legacy code used or not used,
and remaining external boundaries.
