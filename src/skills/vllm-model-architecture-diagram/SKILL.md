---
name: vllm-model-architecture-diagram
description: Analyze a vLLM model adapter Python file and generate an editable Draw.io architecture diagram. Use when the user asks to inspect or visualize files under vllm/model_executor/models, explain model structure, show forward data flow, or draw Attention, MLP, MoE, TP, PP, EP, or checkpoint weight mappings.
compatibility: Requires Python 3, access to the input source file, and a connected Draw.io MCP server.
---

# Goal

Analyze a vLLM model adapter Python file and create a source-grounded,
deterministic Draw.io architecture diagram.

Do not invent modules, configuration values, tensor flows, dimensions or
parallelism features that are not supported by the supplied source code.

# Inputs

The user supplies one Python source path. The typical input is a file under:

`vllm/model_executor/models/*.py`

Default output is a six-page architecture diagram:

- Model Overview
- Decoder Layer Detail
- Attention Detail
- MoE Detail
- vLLM Adapter Integration
- Parallelism & Weight Loading

# Default Workflow

The short instruction should work:

```text
Use $vllm-model-architecture-diagram to analyze samples/hy_v3.py and generate the default architecture diagram.
```

Run the pipeline in this order. Stop immediately when any validator fails.

## 1. Validate Input

- Confirm that the path exists.
- Confirm that it is a `.py` file.
- Derive a filesystem-safe model name for output files.

## 2. Extract Source Facts

```text
python <skill-directory>/scripts/extract_architecture.py \
  <input-python-file> \
  --output outputs/<model-name>-source-analysis.json
```

Do not rewrite or replace the extractor before trying it.

## 3. Build Architecture IR

```text
python <skill-directory>/scripts/build_architecture_ir.py \
  outputs/<model-name>-source-analysis.json \
  --output outputs/<model-name>-architecture-ir.json
```

The builder emits Architecture IR 0.5 with semantic ports. Architecture IR is
the only semantic source for rendering.

Read `unresolved`. Only supplement or correct IR when source-analysis evidence
supports the change.

## 4. Validate Architecture IR

```text
python <skill-directory>/scripts/validate_architecture_ir.py \
  outputs/<model-name>-architecture-ir.json
```

## 5. Build Diagram View

```text
python <skill-directory>/scripts/build_diagram_view.py \
  outputs/<model-name>-architecture-ir.json \
  --output outputs/<model-name>-diagram-view.json
```

Diagram View may decide visibility, display labels, regions, lanes, bundles,
ports and route classes. It must not change model semantics.

## 6. Validate Diagram View

```text
python <skill-directory>/scripts/validate_diagram_view.py \
  outputs/<model-name>-architecture-ir.json \
  outputs/<model-name>-diagram-view.json
```

## 7. Build Layout Plan

```text
python <skill-directory>/scripts/layout_diagram.py \
  outputs/<model-name>-diagram-view.json \
  --output outputs/<model-name>-layout-plan.json
```

The layout plan contains deterministic node boxes, port anchors and routed
waypoints.

## 8. Render Draw.io

```text
python <skill-directory>/scripts/render_drawio.py \
  outputs/<model-name>-diagram-view.json \
  --layout-plan outputs/<model-name>-layout-plan.json \
  --output outputs/<model-name>-architecture.drawio
```

Agent must not directly hand-write complete Draw.io XML. The renderer is the
only component that converts the view and layout into Draw.io cells.

Rendering rules:

- IR node IDs map one-to-one to Draw.io semantic node IDs.
- IR edge IDs map one-to-one to Draw.io semantic edge IDs.
- `parent_id` maps to Draw.io `parent`.
- Edge `source_port` and `target_port` are preserved.
- Hidden semantic edges remain Draw.io cells with transparent style.
- Decorative cells use the `decorative_` prefix.
- Dense/MoE variants are construction notes, not runtime decision nodes.
- Evidence remains in IR and is not shown in node titles.

## 9. Validate Draw.io and Visual Layout

```text
python <skill-directory>/scripts/validate_drawio.py \
  outputs/<model-name>-architecture-ir.json \
  outputs/<model-name>-architecture.drawio \
  --view outputs/<model-name>-diagram-view.json \
  --layout-plan outputs/<model-name>-layout-plan.json
```

```text
python <skill-directory>/scripts/validate_visual_layout.py \
  outputs/<model-name>-architecture-ir.json \
  outputs/<model-name>-architecture.drawio \
  --metrics-output outputs/<model-name>-layout-metrics.json
```

Validation failure blocks MCP export. Fix layout-only issues in Diagram View,
layout planning, renderer, or with a non-semantic MCP edit, then rerun Draw.io
and visual validators.

## 10. Use Draw.io MCP

Use the Draw.io MCP only after all validators pass.

Allowed MCP work:

- open the generated diagram;
- inspect visual layout;
- fix pure layout issues without changing semantic IDs, ports or endpoints;
- export.

Forbidden MCP work:

- adding or deleting semantic nodes;
- adding or deleting semantic edges;
- changing edge source, target, source port or target port;
- reinterpreting architecture.

If MCP edits the `.drawio` file, rerun `validate_drawio.py` and
`validate_visual_layout.py` before export.

Default outputs:

- `outputs/<model-name>-source-analysis.json`;
- `outputs/<model-name>-architecture-ir.json`;
- `outputs/<model-name>-diagram-view.json`;
- `outputs/<model-name>-layout-plan.json`;
- `outputs/<model-name>-layout-metrics.json`;
- `outputs/<model-name>-architecture.drawio`;
- page-level `.png` and `.svg` exports for Overview, Decoder Layer, Attention,
  MoE, Adapter Integration, and Parallelism & Weight Loading.

# Diagram Rules

- Overview answers only the top-level inference flow.
- Decoder Detail separates hidden states and residual state into distinct lanes.
- Attention Detail explicitly separates Q, K, V and KV Cache.
- MoE Detail shows Gate, Router, FusedMoE, Routed Experts, Shared Experts and
  EPLB/EP hints without inventing All-to-All internals.
- Adapter Integration does not include checkpoint mapping.
- Parallelism & Weight Loading keeps TP, PP, EP and checkpoint mapping in
  distinct regions.
- Do not draw Scheduler, Worker, EngineCore, Request Batching or other vLLM
  system components unless they are present in the input source.
- Do not treat checkpoint loading as runtime tensor flow.

# Completion Report

Report:

- detected top-level model class;
- detected decoder, Attention and MoE classes;
- detected TP, PP and EP hints;
- unresolved external configuration or parser warnings;
- source-analysis, Architecture IR, Diagram View, Layout Plan, metrics and
  Draw.io paths;
- validator results;
- exported Draw.io, SVG and PNG paths.
