---
name: vllm-model-architecture-diagram
description: Analyze a vLLM model adapter Python file and generate a source-grounded, concept-driven Draw.io architecture diagram and mentor report.
compatibility: Requires Python 3, access to the input source file, and a connected Draw.io MCP server for optional export.
---

# Goal

Analyze a vLLM model adapter Python file and create a human-readable,
source-grounded architecture analysis package.

v1.0 introduces **Agent Architect Mode**:

```text
Python source
-> Source Fact Graph
-> Agent Architecture Design
-> Architecture Concept Graph
-> Architecture Views
-> Deterministic Renderer
-> Draw.io
```

Scripts decide what is true, evidenced and forbidden. The Agent decides how to
understand, group and explain those facts. Renderer/layout scripts draw the
chosen Architecture View deterministically.

# Modes

- `architect`: v1.0 default for architecture analysis. Builds Source Fact Graph,
  Architecture Design, Architecture View, Boundary Report, Draw.io and mentor
  report.
- `reviewed`: v0.9.1 patch-audited workflow. Uses VSCode Codex-authored
  Review/Patch files and deterministic validators/apply scripts.
- `deterministic`: CI/regression baseline without Agent-authored review.

# Default Architect Workflow

The short instruction should work:

```text
Use $vllm-model-architecture-diagram to analyze samples/hy_v3.py and generate the default architecture diagram.
```

Run:

```text
vllm-arch run --mode architect --input samples/hy_v3.py --model-name hy-v3-v1.0 --outputs-dir outputs
```

Equivalent explicit steps:

1. Extract source facts:

```text
python <skill-directory>/scripts/extract_architecture.py <input.py> --output outputs/<model>-source-analysis.json
```

2. Build semantic inventory:

```text
python <skill-directory>/scripts/build_semantic_inventory.py outputs/<model>-source-analysis.json --output outputs/<model>-semantic-inventory.json
```

3. Build Source Fact Graph:

```text
python <skill-directory>/scripts/build_source_fact_graph.py outputs/<model>-source-analysis.json --output outputs/<model>-source-fact-graph.json
```

4. Run Agent Architect Review:

```text
python <skill-directory>/scripts/run_architect_review.py outputs/<model>-source-fact-graph.json --source-analysis outputs/<model>-source-analysis.json --semantic-inventory outputs/<model>-semantic-inventory.json --output outputs/<model>-architecture-design.json
```

5. Plan Architecture View and boundary report:

```text
python <skill-directory>/scripts/view_planner.py outputs/<model>-architecture-design.json --output outputs/<model>-architecture-view.json
python <skill-directory>/scripts/build_boundary_report.py outputs/<model>-architecture-design.json --output outputs/<model>-boundary-report.json
```

6. Validate architecture quality:

```text
python <skill-directory>/scripts/validate_architecture_quality.py outputs/<model>-source-fact-graph.json outputs/<model>-architecture-design.json outputs/<model>-architecture-view.json --boundary-report outputs/<model>-boundary-report.json
```

7. Layout, render and report:

```text
python <skill-directory>/scripts/layout_diagram.py outputs/<model>-architecture-view.json --output outputs/<model>-layout-plan.json
python <skill-directory>/scripts/render_drawio.py outputs/<model>-architecture-view.json --layout-plan outputs/<model>-layout-plan.json --output outputs/<model>-architecture.drawio
python <skill-directory>/scripts/build_mentor_report.py outputs/<model>-architecture-design.json outputs/<model>-architecture-view.json outputs/<model>-boundary-report.json --output outputs/<model>-mentor-report.md
```

# Architect Rules

- Every concept must cite Source Fact Graph evidence.
- Agent may aggregate low-level facts into architecture concepts.
- Agent may choose page purpose, abstraction level and display grouping.
- Agent must not create concepts without evidence.
- Agent must not mark imported component internals as direct local behavior.
- Agent must not modify source facts or source code.
- Agent must not directly write Draw.io XML.
- Renderer must read Architecture View and layout plan, not invent semantics.

# Default Pages

For HY V3, v1.0 currently emits:

- HY V3 vLLM Adapter Overview
- Transformer Execution Flow
- Attention Implementation
- MoE Execution Strategy
- Parallel Execution Strategy
- Checkpoint Adaptation
- vLLM Integration Boundary

The pages are concept-driven and answer human architecture questions. They are
not intended to be exhaustive code relationship diagrams.

# Draw.io MCP

Use Draw.io MCP only after script validation succeeds. MCP may open, inspect and
export. It must not add/delete semantic concept nodes or reinterpret evidence.

# Completion Report

Report Source Fact Graph count, concept count, page list, boundary count,
quality validator result, Draw.io path, mentor report path, tests, and remaining
limits.
