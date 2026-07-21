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

## 3. Read the extracted analysis

Use the generated JSON as the primary source of facts. Prefer these fields:

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

When a relationship is unclear, inspect the corresponding source lines instead
of rereading unrelated parts of the file.

## 4. Interpret vLLM semantics

Read only the relevant bundled references:

- `references/evidence-rules.md`;
- `references/vllm-patterns.md`;
- `references/architecture-rules.md`;
- `references/diagram-style.md`.

The extractor reports syntax-level evidence. Use the references to interpret
vLLM-specific symbols, but never let a pattern override the supplied source.

## 5. Write Architecture IR

Before generating Draw.io XML, write:

`outputs/<model-name>-architecture-ir.json`

The IR must follow:

`schemas/architecture-ir.schema.json`

Required top-level fields:

- `schema_version`;
- `model_name`;
- `detail_level`;
- `pages`;
- `unresolved`.

Every major node and every edge must include direct or derived source evidence.
Keep source line numbers in `evidence`, not in visible node titles.

Important semantic rules:

- preserve `if/else` branches from `forward_control_flows`;
- represent repeated `make_layers` output symbolically as `N × Decoder Layer`;
- represent residual addition with an explicit Add/Merge node;
- reconnect conditional branches to a common downstream node or output;
- represent TP, PP and EP as badges or annotations;
- keep checkpoint mappings outside runtime tensor flow;
- for `LogitsProcessor(self.lm_head, hidden_states)`, use a runtime edge from
  hidden states and a dependency edge from LM Head.

## 6. Validate Architecture IR

Run:

```text
python <skill-directory>/scripts/validate_architecture_ir.py \
  outputs/<model-name>-architecture-ir.json
```

Do not call Draw.io until validation succeeds. If validation fails, correct the
IR rather than bypassing the validator.

## 7. Generate the Draw.io document

Generate a complete, valid Draw.io XML document from the validated IR. Follow
`references/diagram-style.md`.

Use symbolic values when configuration is external:

- `N × Decoder Layer`;
- `E experts`;
- `Top-K routing`.

Use `html=0` in cell styles by default for SVG compatibility.

## 8. Call the Draw.io MCP

Use the Draw.io MCP tools in this order:

1. `start_session`;
2. `create_new_diagram` with the complete XML;
3. `get_diagram` only when inspection is needed;
4. `edit_diagram` for targeted corrections;
5. `export_diagram`.

Export:

- `outputs/<model-name>-overview.drawio`;
- `outputs/<model-name>-overview.svg`.

For a requested full diagram, replace `overview` with `full`.

## 9. Validate before completion

Before exporting, verify:

- the Architecture IR validator passed;
- every major node and edge has source evidence;
- mutually exclusive branches were not flattened into one path;
- loop-based decoder execution is visible when present;
- residual edges end at explicit Add/Merge nodes;
- conditional branches reconnect;
- parallel groups are annotations, not compute nodes;
- checkpoint mappings are separate from runtime flow;
- no concrete external config values were guessed;
- Draw.io XML uses stable IDs and `html=0` labels;
- the Draw.io XML opens successfully.

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
- exported Draw.io and SVG paths.
