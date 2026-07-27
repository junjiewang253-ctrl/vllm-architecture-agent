# vLLM Model Architecture Diagram

Use this Skill when the user asks to analyze a vLLM model adapter file or a
registered vLLM architecture name and generate architecture diagrams.

The v2.0 workflow is Agent-native. Scripts collect and validate source context;
Codex is responsible for architecture understanding, page selection, diagram
design, Draw.io construction, visual review, and the final report.

## Scope

Default targets are files under:

```text
vllm/model_executor/models/*.py
```

The user may provide either:

- a Python file path; or
- a registry architecture name.

If an architecture name is supplied, resolve it through
`vllm/model_executor/models/registry.py` without importing vLLM.

## Default Output

Use one output directory per model:

```text
outputs/<model>/
├── source-context.json
├── architecture-plan.json
├── evidence.json
├── architecture.drawio
├── report.md
├── visual-review.md
└── images/
```

Do not generate legacy IR, Concept, View, Layout Plan, Patch, Lock, or Coverage
JSON by default.

## Step 1: Resolve Input

Run `vllm-arch prepare` or the underlying scripts:

```powershell
vllm-arch prepare `
  --repo-root <vllm-repo-root> `
  --input <model-file.py> `
  --outputs-dir outputs/<model>
```

For registry architecture input:

```powershell
vllm-arch prepare `
  --repo-root <vllm-repo-root> `
  --architecture <ArchitectureName> `
  --outputs-dir outputs/<model>
```

This creates `source-context.json`, `architecture-plan.template.json`, and
`evidence.template.json`.

## Step 2: Read Source

Read the target model file first. Use `source-context.json` as an index, not as
the architecture answer.

Default traversal budget:

- maximum follow depth: 3;
- default related file count: 20;
- suggested maximum related file count: 30.

Read related files only when needed to support a diagram claim or external
boundary. Stop when the main diagram claims are supported, an external component
boundary is reached, or the budget is reached.

Record every file actually read in `architecture-plan.json`.

## Step 3: Write Architecture Plan

Create `architecture-plan.json` from the template.

The plan describes what diagrams should communicate. It must not contain Draw.io
XML or exact coordinates.

For each page:

- state the engineering question;
- state the purpose;
- choose a view pattern;
- list source-backed claim IDs;
- outline the main story;
- list secondary topics;
- list external boundaries;
- set a detail budget.

Page selection is dynamic:

- always consider a Model Overview and Adapter Boundary;
- add Decoder, Attention, MoE, Weight Loading, Parallel Strategies, Multimodal,
  Pooling, Classification, Hybrid/Recurrent, Quantization, LoRA, or other pages
  only when source evidence supports them;
- do not create empty or irrelevant pages.

## Step 4: Write Evidence

Create `evidence.json`.

Use three confidence levels:

- `direct`: local source directly proves the claim;
- `derived`: multiple local facts support a higher-level claim;
- `external`: local source proves a call or dependency, but implementation is
  outside the target file or traversal boundary.

Direct evidence must cite concrete file line ranges and cannot rely only on
import lines. External claims must describe the local evidence and the boundary
that was not analyzed.

## Step 5: Validate Plan and Evidence

Run:

```powershell
vllm-arch validate `
  --context outputs/<model>/source-context.json `
  --plan outputs/<model>/architecture-plan.json `
  --evidence outputs/<model>/evidence.json
```

Fix any errors before drawing. Warnings should be addressed when they affect
diagram clarity.

## Step 6: Draw With Draw.io MCP

Use Draw.io MCP for diagram creation and editing. The Agent designs the diagram;
MCP is the drawing surface.

Required drawing rules:

- page names must match `architecture-plan.json`;
- runtime data flow uses primary arrows;
- configuration, capability, parallelism, and checkpoint loading must not be
  drawn as runtime tensor flow;
- external components must be placed behind a clear boundary;
- module implementation names belong in subtitles;
- source line numbers and long code expressions stay in `evidence.json` and
  `report.md`, not inside node titles;
- ordinary data-flow edge labels should be hidden unless the label disambiguates
  a branch;
- use a white, opaque page background.

Use `assets/drawio-style-template.drawio` as a style reference when helpful.

## Step 7: Visual Review

Export a first PNG draft. Inspect it and write `visual-review.md` covering:

- whether each page answers its question;
- text overlap;
- edges crossing through nodes;
- long or confusing lines;
- repeated labels;
- excessive whitespace;
- overly dense regions;
- concept-card pages that should become flows, maps, or boundaries.

Make at most two visual revision rounds. Revisions may adjust layout, grouping,
labels, edge visibility, page split/merge, and visual hierarchy. Revisions must
not invent new source behavior or promote external behavior to direct evidence.

## Step 8: Validate Draw.io

Run:

```powershell
vllm-arch validate `
  --context outputs/<model>/source-context.json `
  --plan outputs/<model>/architecture-plan.json `
  --evidence outputs/<model>/evidence.json `
  --drawio outputs/<model>/architecture.drawio
```

Do not export final images if validation fails.

## Step 9: Final Report

Write `report.md` with:

- target model and registry mapping;
- files read and traversal stops;
- model category;
- key architecture conclusions;
- page-by-page walkthrough;
- TP/PP/EP/LoRA/quantization or other capabilities;
- weight loading behavior, if relevant;
- external boundaries;
- unresolved items;
- intentionally omitted details;
- validation results and output paths.

Separate Source-proven, Derived, and External-boundary statements.

## Degradation

If the target is a helper or shared utility file, produce a boundary or utility
map rather than a full model diagram.

If only partial source context can be collected, report the partial status,
warnings, and next files to inspect. Never invent missing runtime behavior.

## Reference Files

Read these only as needed:

- `references/analysis-guide.md`
- `references/vllm-patterns.md`
- `references/page-playbook.md`
- `references/diagram-design-guide.md`
- `references/evidence-policy.md`
