---
name: vllm-model-architecture-diagram
description: Analyze a vLLM model adapter Python file and generate an editable Draw.io architecture diagram. Use when the user asks to inspect or visualize files under vllm/model_executor/models, explain model structure, show forward data flow, or draw Attention, MLP, MoE, TP, PP, EP, or checkpoint weight mappings.
compatibility: Requires Python 3, access to the input source file, and a connected Draw.io MCP server.
---

# Goal

Analyze a vLLM model adapter Python file and create an editable,
source-grounded Draw.io architecture diagram.

Do not invent modules, configuration values, tensor flows, dimensions or
parallelism features that are not supported by the supplied source code.

# Inputs

The user supplies one Python source path. The typical input is a file under:

`vllm/model_executor/models/*.py`

The user may optionally request `overview` or `full` detail. Default to
`overview`.

# Workflow

## 1. Validate the input

- Confirm that the path exists.
- Confirm that it is a `.py` file.
- Derive a filesystem-safe model name for output files.

## 2. Extract source facts

Locate this Skill's directory, then run its bundled extractor:

```text
python <skill-directory>/scripts/extract_architecture.py \
  <input-python-file> \
  --output outputs/<model-name>-source-analysis.json
```

Do not rewrite or replace the extractor before trying it.

## 3. Build the draft Architecture IR

Run the bundled IR builder:

```text
python <skill-directory>/scripts/build_architecture_ir.py \
  outputs/<model-name>-source-analysis.json \
  --output outputs/<model-name>-architecture-ir.json
```

The builder creates a conservative `overview` page only. It should identify
top-level ForCausalLM wrappers, `self.model`, embedding, repeated
`make_layers` decoder blocks, final norm, LM Head, LogitsProcessor, decoder
attention, Dense/MoE construction variants, and TP/PP/EP badges when these are
supported by source-analysis facts.

## 4. Read the extracted analysis and draft IR

Use the generated source-analysis JSON as the primary source of facts. Prefer
these fields:

- `classes`;
- `module_assignments`, filtered by `assignment_kind`;
- `layer_factories`;
- `forward_control_flows`;
- `conditions`;
- `parallelism_hints`;
- `weight_mappings`;
- `warnings`.

`forward_flows` and `weight_loading_hints` are compatibility/debug fields. Do
not use their flattened order when `forward_control_flows` contains branches or
loops.

Read `outputs/<model-name>-architecture-ir.json` and inspect `unresolved`.
Only supplement or correct the IR when source-analysis evidence supports the
change. Keep unresolved items when source or external config prevents a
deterministic answer.

When a relationship is unclear, inspect the corresponding source lines instead
of rereading unrelated parts of the file.

## 5. Validate Architecture IR

Run:

```text
python <skill-directory>/scripts/validate_architecture_ir.py \
  outputs/<model-name>-architecture-ir.json
```

Do not render Draw.io until validation succeeds. If validation fails, correct
the IR rather than bypassing the validator.

## 6. Render Draw.io deterministically

Run the bundled renderer:

```text
python <skill-directory>/scripts/render_drawio.py \
  outputs/<model-name>-architecture-ir.json \
  --output outputs/<model-name>-overview.drawio
```

Agent must not directly hand-write complete Draw.io XML. The renderer is the
only component that converts IR semantics into Draw.io cells.

Rendering rules:

- IR node IDs must map one-to-one to Draw.io semantic node IDs.
- IR edge IDs must map one-to-one to Draw.io semantic edge IDs.
- `parent_id` must map to Draw.io `parent`.
- Decorative cells must use the `decorative_` prefix.
- Dense/MoE variants are construction notes, not runtime decision nodes.
- TP, PP and EP are badge decorations, not compute nodes.
- Evidence remains in IR and is not shown in node titles.

## 7. Validate Draw.io XML

Run:

```text
python <skill-directory>/scripts/validate_drawio.py \
  outputs/<model-name>-architecture-ir.json \
  outputs/<model-name>-overview.drawio
```

Draw.io validation failure blocks export. Fix layout-only issues through the
renderer or a non-semantic MCP edit, then run `validate_drawio.py` again.

## 8. Call the Draw.io MCP

Use the Draw.io MCP only after both validators pass.

Allowed MCP work:

- open the generated diagram;
- inspect visual layout;
- fix pure layout issues;
- export.

Forbidden MCP work:

- adding semantic nodes;
- deleting semantic nodes;
- adding semantic edges;
- deleting semantic edges;
- changing edge source or target;
- reinterpreting architecture.

If MCP edits the `.drawio` file, run `validate_drawio.py` again before export.

Use the MCP tools in this order:

1. `start_session`;
2. `load_diagram` with `outputs/<model-name>-overview.drawio`;
3. `get_diagram` only when inspection is needed;
4. `edit_diagram` only for pure layout corrections;
5. `export_diagram`.

Default outputs:

- `outputs/<model-name>-source-analysis.json`;
- `outputs/<model-name>-architecture-ir.json`;
- `outputs/<model-name>-overview.drawio`;
- `outputs/<model-name>-overview.svg`;
- `outputs/<model-name>-overview.png`.

For a requested full diagram, replace `overview` with `full`.

## 9. Validate before completion

Before exporting or reporting completion, verify:

- Architecture IR validator passed;
- Draw.io validator passed;
- every IR node and edge exists in Draw.io;
- no extra non-decorative semantic Draw.io nodes or edges exist;
- every node parent matches IR `parent_id`;
- child nodes stay inside parent containers;
- repeated blocks show `repetition`;
- construction variants render as decorative notes;
- badges render as decorative cells;
- no concrete external config values were guessed.

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
