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

If it fails, report the exact error. Inspect the source manually only for the
unsupported structure and record the limitation in `unresolved`.

## 3. Read the extracted analysis

Use the generated JSON as the primary source of facts. Review:

- `classes`;
- `module_assignments`;
- `forward_flows`;
- `conditions`;
- `parallelism_hints`;
- `weight_loading_hints`;
- `warnings`.

When a relationship is unclear, inspect the corresponding source lines instead
of rereading unrelated parts of the whole file.

## 4. Interpret vLLM semantics

Read only the relevant bundled references:

- `references/evidence-rules.md`;
- `references/vllm-patterns.md`;
- `references/diagram-style.md`.

The extractor reports syntax-level evidence. Use the references to interpret
vLLM-specific symbols, but never let a pattern override the supplied source.

## 5. Plan before drawing

Create a compact internal diagram plan containing:

- major nodes;
- runtime edges;
- residual edges;
- conditional branches;
- containment relationships;
- parallelism annotations;
- optional checkpoint mappings;
- unresolved items;
- source evidence lines for every major node.

For `overview`, use one page and no more than twelve major nodes.

Recommended main flow:

`Input Tokens → Embedding → N × Decoder Layer → Final Norm → LM Head → Logits`

Show decoder details below the main flow when supported by the source:

`Input Norm → Attention → Post-Attention Norm → Dense FFN or MoE`

## 6. Generate the Draw.io document

Generate a complete, valid Draw.io XML document. Follow
`references/diagram-style.md`.

Use symbolic values when configuration is external:

- `N × Decoder Layer`;
- `E experts`;
- `Top-K routing`.

Do not place `load_weights` relationships in the runtime forward path.

## 7. Call the Draw.io MCP

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

## 8. Validate before completion

Before exporting, verify:

- every major node has direct or derived source evidence;
- forward edges follow the source `forward` methods;
- real source conditions remain visible;
- residual connections are not omitted when directly present;
- parallel groups are annotations, not compute nodes;
- checkpoint mappings are separate from runtime flow;
- no concrete external config values were guessed;
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
- exported Draw.io and SVG paths.
