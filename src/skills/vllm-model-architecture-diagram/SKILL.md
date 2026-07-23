---
name: vllm-model-architecture-diagram
description: Analyze a vLLM model adapter Python file and generate an editable Draw.io architecture diagram. Use when the user asks to inspect or visualize files under vllm/model_executor/models, explain model structure, show forward data flow, or draw Attention, MLP, MoE, TP, PP, EP, or checkpoint weight mappings.
compatibility: Requires Python 3, access to the input source file, and a connected Draw.io MCP server.
---

# Goal

Analyze a vLLM model adapter Python file and create an editable,
source-grounded, deterministic Draw.io architecture diagram.

Do not invent modules, configuration values, tensor flows, dimensions or
parallelism features that are not supported by the supplied source code.

# Inputs

The user supplies one Python source path. The typical input is a file under:

`vllm/model_executor/models/*.py`

Default output is a four-page architecture diagram:

- Model Overview
- Decoder Layer Detail
- Attention Detail
- vLLM Adaptation Map

# Workflow

## 1. Validate the input

- Confirm that the path exists.
- Confirm that it is a `.py` file.
- Derive a filesystem-safe model name for output files.

## 2. Extract source facts

Run:

```text
python <skill-directory>/scripts/extract_architecture.py \
  <input-python-file> \
  --output outputs/<model-name>-source-analysis.json
```

Do not rewrite or replace the extractor before trying it.

## 3. Build Architecture IR

Run:

```text
python <skill-directory>/scripts/build_architecture_ir.py \
  outputs/<model-name>-source-analysis.json \
  --output outputs/<model-name>-architecture-ir.json
```

The builder emits Architecture IR 0.4 with four pages: `overview`,
`decoder_layer_detail`, `attention_detail`, and `vllm_adaptation_map`. The IR is
the only semantic source for rendering.

Read `unresolved`. Only supplement or correct IR when source-analysis evidence
supports the change.

## 4. Validate Architecture IR

Run:

```text
python <skill-directory>/scripts/validate_architecture_ir.py \
  outputs/<model-name>-architecture-ir.json
```

Do not render Draw.io until validation succeeds.

## 5. Render Draw.io deterministically

Run:

```text
python <skill-directory>/scripts/render_drawio.py \
  outputs/<model-name>-architecture-ir.json \
  --output outputs/<model-name>-architecture.drawio
```

Agent must not directly hand-write complete Draw.io XML. The renderer is the
only component that converts IR semantics into Draw.io cells.

Rendering rules:

- IR node IDs map one-to-one to Draw.io semantic node IDs.
- IR edge IDs map one-to-one to Draw.io semantic edge IDs.
- `parent_id` maps to Draw.io `parent`.
- `display` controls human-readable text only; it never changes semantics.
- `display.visible=false` edges remain semantic cells but render transparent.
- Decorative cells use the `decorative_` prefix.
- Dense/MoE variants are construction notes, not runtime decision nodes.
- Evidence remains in IR and is not shown in node titles.

## 6. Validate Draw.io XML

Run:

```text
python <skill-directory>/scripts/validate_drawio.py \
  outputs/<model-name>-architecture-ir.json \
  outputs/<model-name>-architecture.drawio
```

Draw.io validation failure blocks export. Fix layout-only issues through the
renderer or a non-semantic MCP edit, then run `validate_drawio.py` again.

## 7. Call the Draw.io MCP

## 7. Validate visual layout

Run:

```text
python <skill-directory>/scripts/validate_visual_layout.py \
  outputs/<model-name>-architecture-ir.json \
  outputs/<model-name>-architecture.drawio
```

Visual validation failure blocks export. Fix layout-only issues in
`render_drawio.py` or with a non-semantic MCP edit, then rerun both Draw.io
validators.

## 8. Call the Draw.io MCP

Use the Draw.io MCP only after all validators pass.

Allowed MCP work:

- open the generated diagram;
- inspect visual layout;
- fix pure layout issues;
- export.

Forbidden MCP work:

- adding or deleting semantic nodes;
- adding or deleting semantic edges;
- changing edge source or target;
- reinterpreting architecture.

If MCP edits the `.drawio` file, run `validate_drawio.py` and
`validate_visual_layout.py` again before export.

Use the MCP tools in this order:

1. `start_session`;
2. `load_diagram` with `outputs/<model-name>-architecture.drawio`;
3. `get_diagram` only when inspection is needed;
4. `edit_diagram` only for pure layout corrections;
5. `export_diagram`.

Default outputs:

- `outputs/<model-name>-source-analysis.json`;
- `outputs/<model-name>-architecture-ir.json`;
- `outputs/<model-name>-architecture.drawio`;
- `outputs/<model-name>-architecture.svg`;
- `outputs/<model-name>-architecture.png`.

If a multi-page export cannot produce one combined PNG, export pages separately
and report the actual paths.

# Completion report

Report:

- detected top-level model class;
- detected decoder or transformer layer class;
- detected Attention class;
- detected FFN or MoE classes;
- detected TP, PP and EP hints;
- unresolved external configuration or parser warnings;
- source-analysis JSON path;
- architecture-IR JSON path;
- Draw.io validation result;
- exported Draw.io, SVG, and PNG paths.
