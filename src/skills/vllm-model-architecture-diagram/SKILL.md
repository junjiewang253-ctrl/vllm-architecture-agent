---
name: vllm-model-architecture-diagram
description: Analyze a vLLM model adapter Python file and generate an editable Draw.io model architecture diagram. Use when the user asks to inspect or visualize a Python file under vllm/model_executor/models, explain model structure, show forward data flow, or draw Attention, MLP, MoE, TP, PP, EP, or weight-loading relationships.
compatibility: Requires access to the input Python file and a connected Draw.io MCP server.
---

# Goal

Analyze a vLLM model adapter Python file and create an editable,
source-grounded Draw.io architecture diagram.

Do not invent modules, configuration values, tensor flows, or parallelism
features that are not supported by the source code.

# Workflow

1. Verify that the supplied path exists and is a Python file.

2. Read the complete source file.

3. Identify:

   - top-level model class;
   - base model class;
   - decoder or transformer layer;
   - instantiated child modules;
   - forward-call order;
   - residual paths;
   - conditional module selection;
   - Attention, MLP and MoE components;
   - TP, PP and EP features;
   - checkpoint weight mappings.

4. Separate these relationship types:

   - solid arrows: runtime tensor flow;
   - dashed arrows: module containment or configuration;
   - dotted arrows: checkpoint weight mappings.

5. For the initial version, generate one overview page with no more than
   twelve major nodes.

6. Use symbolic values such as:

   - N × Decoder Layer;
   - E experts;
   - Top-K routing.

   Do not guess values supplied by an external configuration.

7. Use the Draw.io MCP tools:

   - call `start_session`;
   - call `create_new_diagram` with complete Draw.io XML;
   - use `get_diagram` and `edit_diagram` only when corrections are needed;
   - call `export_diagram`.

8. Export:

   - `outputs/<model-name>-overview.drawio`;
   - `outputs/<model-name>-overview.svg`.

# Completion report

Report:

- detected top-level class;
- detected decoder class;
- detected Attention class;
- detected FFN or MoE classes;
- detected parallelism types;
- unresolved configuration;
- exported file paths.