---
name: vllm-model-architecture-diagram
description: Analyze a vLLM model adapter Python file and generate an editable Draw.io architecture diagram. Use when the user asks to inspect or visualize files under vllm/model_executor/models, explain model structure, show forward data flow, or draw Attention, MLP, MoE, TP, PP, EP, or checkpoint weight mappings.
compatibility: Requires Python 3, access to the input source file, and a connected Draw.io MCP server.
---

# Goal

Analyze a vLLM model adapter Python file and create a source-grounded,
deterministic Draw.io architecture diagram.

v0.9.1 default mode is `reviewed`: deterministic scripts build the baseline,
the current VSCode Codex session performs constrained Semantic Review and Visual
Review, writes structured Review/Patch JSON artifacts, and deterministic scripts
validate and apply the resulting patches. Agents must not directly hand-write
final Draw.io XML or use MCP to add, delete, or reinterpret semantic nodes and
edges.

# Modes

- `deterministic`: run extraction, inventory, baseline IR, Diagram View, layout,
  renderer and validators only. It may use `build_baseline_review_template.py`
  for deterministic Review/Patch templates.
- `reviewed`: default. Add Semantic Review, IR Patch, Visual Review, View Patch,
  coverage 0.2 and review lock before export. In the default interactive Skill
  flow, the active Codex session writes the Review/Patch JSON directly instead
  of calling an external Codex CLI.
- `exploratory`: may propose extra local files or experimental patches, but this
  version only documents the interface; do not auto-apply exploratory patches.

# Default Output

For `samples/hy_v3.py`, reviewed mode normally produces seven pages:

- Model Overview
- Decoder Layer Detail
- Attention Detail
- MoE Detail
- vLLM Adapter Integration
- Parallelism
- Weight Loading

The page plan is reviewable. Keep Overview and Adapter Integration by default;
keep detail pages only when source-analysis and inventory facts support them.

# Reviewed Workflow

The short instruction should work:

```text
Use $vllm-model-architecture-diagram to analyze samples/hy_v3.py and generate the default architecture diagram.
```

Run the pipeline in this order. Stop immediately when a semantic, view, Draw.io
or visual validator fails.

1. Validate input Python path and derive `<model-name>`.
2. Extract source facts:

```text
python <skill-directory>/scripts/extract_architecture.py <input.py> --output outputs/<model-name>-source-analysis.json
```

3. Build semantic inventory:

```text
python <skill-directory>/scripts/build_semantic_inventory.py outputs/<model-name>-source-analysis.json --output outputs/<model-name>-semantic-inventory.json
```

4. Build baseline Architecture IR:

```text
python <skill-directory>/scripts/build_architecture_ir.py outputs/<model-name>-source-analysis.json --output outputs/<model-name>-baseline-architecture-ir.json
```

5. Validate baseline IR and baseline coverage:

```text
python <skill-directory>/scripts/validate_architecture_ir.py outputs/<model-name>-baseline-architecture-ir.json
python <skill-directory>/scripts/validate_semantic_coverage.py outputs/<model-name>-source-analysis.json outputs/<model-name>-semantic-inventory.json outputs/<model-name>-baseline-architecture-ir.json --output outputs/<model-name>-baseline-semantic-coverage.json
```

6. Semantic Review Round 1:

The Agent reads the input source, source-analysis, semantic inventory, baseline
IR, baseline coverage and review references. The active VSCode Codex session
outputs these files directly:

- `outputs/<model-name>-semantic-review.json`
- `outputs/<model-name>-architecture-ir.patch.json`

Then validate and apply:

```text
python <skill-directory>/scripts/validate_semantic_review.py outputs/<model-name>-source-analysis.json outputs/<model-name>-semantic-inventory.json outputs/<model-name>-baseline-architecture-ir.json outputs/<model-name>-semantic-review.json outputs/<model-name>-architecture-ir.patch.json
python <skill-directory>/scripts/apply_ir_patch.py outputs/<model-name>-baseline-architecture-ir.json outputs/<model-name>-architecture-ir.patch.json --output outputs/<model-name>-reviewed-architecture-ir.json
```

`run_semantic_review.py` is an optional automation helper for mock tests or
future non-interactive integrations; do not make it the default VSCode Skill
path.

7. Validate reviewed semantics:

```text
python <skill-directory>/scripts/validate_architecture_ir.py outputs/<model-name>-reviewed-architecture-ir.json
python <skill-directory>/scripts/validate_semantic_coverage.py outputs/<model-name>-source-analysis.json outputs/<model-name>-semantic-inventory.json outputs/<model-name>-reviewed-architecture-ir.json --semantic-review outputs/<model-name>-semantic-review.json --output outputs/<model-name>-semantic-coverage.json
```

Reviewed mode requires `required.unresolved = 0` and `required.orphaned = 0`.
This means every required fact has been rendered, aggregated, documented as an
external boundary, excluded with reason, or otherwise explicitly disposed. It
does not mean every fact is a separate node.

8. Build baseline Diagram View, layout and Draw.io:

```text
python <skill-directory>/scripts/build_diagram_view.py outputs/<model-name>-reviewed-architecture-ir.json --output outputs/<model-name>-baseline-diagram-view.json
python <skill-directory>/scripts/layout_diagram.py outputs/<model-name>-baseline-diagram-view.json --output outputs/<model-name>-baseline-layout-plan.json
python <skill-directory>/scripts/render_drawio.py outputs/<model-name>-baseline-diagram-view.json --layout-plan outputs/<model-name>-baseline-layout-plan.json --output outputs/<model-name>-baseline-architecture.drawio
python <skill-directory>/scripts/validate_visual_layout.py outputs/<model-name>-reviewed-architecture-ir.json outputs/<model-name>-baseline-architecture.drawio --metrics-output outputs/<model-name>-baseline-layout-metrics.json
```

9. Visual Review Round 1:

The Agent may adjust labels, edge visibility, route classes, bundles, regions,
legends and boundary notes. It must not change semantic IDs, endpoints, phase,
ports or evidence. The active VSCode Codex session outputs:

- `outputs/<model-name>-visual-review.json`
- `outputs/<model-name>-diagram-view.patch.json`

```text
python <skill-directory>/scripts/validate_visual_review.py outputs/<model-name>-reviewed-architecture-ir.json outputs/<model-name>-baseline-diagram-view.json outputs/<model-name>-visual-review.json outputs/<model-name>-diagram-view.patch.json
python <skill-directory>/scripts/apply_view_patch.py outputs/<model-name>-baseline-diagram-view.json outputs/<model-name>-diagram-view.patch.json --output outputs/<model-name>-reviewed-diagram-view.json
```

`run_visual_review.py` is likewise optional automation, not the default
interactive workflow.

10. Final layout, render and validation:

```text
python <skill-directory>/scripts/layout_diagram.py outputs/<model-name>-reviewed-diagram-view.json --output outputs/<model-name>-layout-plan.json
python <skill-directory>/scripts/render_drawio.py outputs/<model-name>-reviewed-diagram-view.json --layout-plan outputs/<model-name>-layout-plan.json --output outputs/<model-name>-architecture.drawio
python <skill-directory>/scripts/validate_drawio.py outputs/<model-name>-reviewed-architecture-ir.json outputs/<model-name>-architecture.drawio --view outputs/<model-name>-reviewed-diagram-view.json --layout-plan outputs/<model-name>-layout-plan.json
python <skill-directory>/scripts/validate_visual_layout.py outputs/<model-name>-reviewed-architecture-ir.json outputs/<model-name>-architecture.drawio --metrics-output outputs/<model-name>-layout-metrics.json
```

11. Build review lock:

```text
python <skill-directory>/scripts/build_review_lock.py --source <input.py> --source-analysis outputs/<model-name>-source-analysis.json --semantic-inventory outputs/<model-name>-semantic-inventory.json --baseline-ir outputs/<model-name>-baseline-architecture-ir.json --semantic-review outputs/<model-name>-semantic-review.json --ir-patch outputs/<model-name>-architecture-ir.patch.json --reviewed-ir outputs/<model-name>-reviewed-architecture-ir.json --visual-review outputs/<model-name>-visual-review.json --view-patch outputs/<model-name>-diagram-view.patch.json --reviewed-view outputs/<model-name>-reviewed-diagram-view.json --output outputs/<model-name>-review-lock.json
```

12. Use Draw.io MCP only after validators pass. Allowed MCP actions are open,
inspect, export, and pure layout micro-adjustment that preserves semantic IDs,
ports and endpoints. Rerun Draw.io and visual validators after any MCP edit.

13. Build mentor package after exports exist:

```text
python tools/build_mentor_package.py --model-name <model-name> --outputs-dir outputs --destination dist/mentor-package
```

# Review Rules

- Direct evidence must cite structured source facts; import-only evidence cannot
  support behavior.
- Derived evidence must explain aggregation or derivation.
- External component internals must be `external` or `documented_external`, not
  `direct`.
- Semantic Review may aggregate low-level facts into high-level concepts, but
  must record target IDs and reasons.
- Visual Review may simplify display labels, hide noisy edge labels, add bundles
  and boundary notes, but must not change semantics.
- Semantic Review and Visual Review each have at most two rounds. Round 2 is
  only for validator errors or remaining critical/major findings.

# Diagram Rules

- Overview answers top-level inference flow and separates forward hidden-state
  production from compute logits.
- Decoder Detail separates hidden-state and residual lanes; do not reintroduce
  large Residual Handoff semantic nodes.
- Attention Detail separates Q, K, V and KV Cache; KV Cache read is an external
  boundary unless proven by the input file.
- MoE Detail shows Gate, Router, FusedMoE, routed experts, optional shared
  experts, restore shape and EP/EPLB hints without inventing All-to-All.
- Adapter Integration does not contain checkpoint mapping.
- Parallelism keeps TP, PP and EP independent.
- Weight Loading keeps wrapper and model `load_weights` flows separate.
- Do not draw Scheduler, Worker, EngineCore, Request Batching or other vLLM
  system components unless they appear in the input source.

# Completion Report

Report validator results, coverage matrix, review findings, patch counts,
deferred operations, review lock status, Draw.io/PNG/SVG paths, mentor package
path, and any remaining limitations.
