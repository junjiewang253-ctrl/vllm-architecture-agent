---
name: vllm-model-architecture-diagram
description: Analyze a vLLM model adapter Python file and generate a source-grounded, View-Graph-driven Draw.io architecture diagram and architecture report.
compatibility: Requires Python 3, access to the input source file, and a connected Draw.io MCP server for optional opening/export.
---

# Goal

Analyze a vLLM model adapter Python file and create a human-readable
architecture diagram.

v1.0.1 uses four layers:

```text
Python source
-> Source Fact Graph
-> Architecture Concept Graph
-> Architecture View Graph
-> Deterministic Draw.io Renderer
```

Concepts explain what source facts mean. View Graph explains how a human should
see those concepts: components, data nodes, runtime flow, mapping flow,
parallel strategy and external boundaries. Renderer must read View Graph, not
Concept Graph.

# Modes

- `architect`: default v1.0.1 workflow.
- `reviewed`: v0.9.1 patch-audited workflow.
- `deterministic`: CI/regression baseline.

# Default Command

```text
vllm-arch run --mode architect --input samples/hy_v3.py --model-name hy-v3-v1.0.1 --outputs-dir outputs
```

# Default Workflow

1. Extract source analysis:

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

4. Build Architecture Concept Graph:

```text
python <skill-directory>/scripts/run_architect_review.py outputs/<model>-source-fact-graph.json --source-analysis outputs/<model>-source-analysis.json --semantic-inventory outputs/<model>-semantic-inventory.json --output outputs/<model>-architecture-concept.json
```

5. Build Architecture View Graph:

```text
python <skill-directory>/scripts/run_view_architect.py outputs/<model>-architecture-concept.json outputs/<model>-source-fact-graph.json --output outputs/<model>-architecture-view.json
```

6. Build boundary report and validate:

```text
python <skill-directory>/scripts/build_boundary_report.py outputs/<model>-architecture-concept.json --output outputs/<model>-boundary-report.json
python <skill-directory>/scripts/validate_architecture_quality.py outputs/<model>-source-fact-graph.json outputs/<model>-architecture-concept.json outputs/<model>-architecture-view.json --boundary-report outputs/<model>-boundary-report.json
python <skill-directory>/scripts/validate_architecture_view.py outputs/<model>-architecture-view.json --architecture-concept outputs/<model>-architecture-concept.json --source-fact-graph outputs/<model>-source-fact-graph.json
```

7. Layout and render:

```text
python <skill-directory>/scripts/apply_view_layout.py outputs/<model>-architecture-view.json --output outputs/<model>-layout-plan.json
python <skill-directory>/scripts/render_drawio.py outputs/<model>-architecture-view.json --layout-plan outputs/<model>-layout-plan.json --output outputs/<model>-architecture.drawio
python <skill-directory>/scripts/build_mentor_report.py outputs/<model>-architecture-concept.json outputs/<model>-architecture-view.json outputs/<model>-boundary-report.json --output outputs/<model>-architecture-report.md
```

# Required Pages

For HY V3, default View Architect output should include:

- Model Overview
- Attention Implementation
- MoE Execution Strategy
- Parallel Strategy
- Checkpoint Adaptation
- vLLM Integration Boundary

# View Requirements

- Concept Graph must not be rendered directly.
- Every core View node must have `concept_refs` and `fact_refs`.
- Each page must have a purpose and at least three nodes.
- Pages must not be only concept cards.
- Attention page must show QKV Projection, Q/K/V Split, KV Cache Boundary,
  vLLM Attention Backend and Output Projection.
- MoE page must show Router, FusedMoE, Routed Experts, Shared Experts and EP.
- Checkpoint page must show HF weights, packed/stacked mappings and expert
  mapping.
- Boundary page must distinguish local adapter behavior from imported vLLM
  component behavior.

# Agent Rules

The Agent may design architecture concepts and view nodes, but may not:

- create nodes without source-backed facts;
- modify Source Fact Graph or Concept Graph to hide missing evidence;
- invent runtime flow;
- treat external component internals as direct local behavior;
- write Draw.io XML directly.

# Draw.io MCP

After scripts generate `.drawio`, use Draw.io MCP to open, inspect and export.
MCP must not add/delete semantic View nodes or reinterpret evidence.

# Completion Report

Report generated files, page list, View node/edge counts, validators, test
result, Draw.io path, architecture report path and remaining limitations.
