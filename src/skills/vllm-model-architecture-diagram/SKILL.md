# vLLM Model Architecture Diagram

Use this Skill when the user asks to analyze a vLLM model adapter file or a
registered vLLM architecture name and generate a small set of complete,
high-density architecture diagrams.

The v2.1 workflow remains Agent-native. Scripts collect and validate source
structure; Codex reads the source, decides what matters, chooses pages, designs
the diagram, draws with Draw.io MCP, reviews the PNG, and writes the report.

## Scope

Default targets are files under:

```text
vllm/model_executor/models/*.py
```

The user may provide either a Python file path or a registry architecture name.
Resolve architecture names through `vllm/model_executor/models/registry.py`
without importing vLLM, torch, transformers, or CUDA-dependent modules.

## Default Output

Use one output directory per model:

```text
outputs/<model>/
  source-context.json
  architecture-plan.json
  evidence.json
  architecture.drawio
  report.md
  visual-review.md
  images/
```

Do not generate legacy IR, Concept Graph, View Graph, Layout Plan, Patch, Lock,
or Coverage JSON by default.

## Page Budget

Default detail level is `complete`.

For a full model, target four high-density composite pages and keep the hard
maximum at five pages. A helper or shared utility file may use one or two pages.

Do not stop after a simple overview. Do not create one page per class or method.
Use embedded detail regions inside a small number of pages.

When a page is dense, try these before adding a page:

1. nested containers;
2. embedded subgraphs;
3. component trees;
4. parameter panels;
5. variant tables;
6. mapping trees;
7. ports instead of low-information data objects;
8. capability badges;
9. side panels for config and metadata;
10. aggregation of repeated low-level steps;
11. fewer edge labels;
12. adjusted canvas size.

Do not delete core behavior just to make the page prettier.

## Step 1: Prepare Source Context

Run:

```powershell
vllm-arch prepare `
  --repo-root <vllm-repo-root> `
  --input <model-file.py> `
  --outputs-dir outputs/<model>
```

Or by registry architecture:

```powershell
vllm-arch prepare `
  --repo-root <vllm-repo-root> `
  --architecture <ArchitectureName> `
  --outputs-dir outputs/<model>
```

This creates `source-context.json`, `architecture-plan.template.json`, and
`evidence.template.json`.

## Step 2: Complete Target-File Review

Read the complete target Python file. Then inspect `source-context.json`:

- `source_coverage`;
- `classes`;
- `methods`;
- `module_functions`;
- `branches`;
- `weight_mappings`;
- `capability_signals`;
- `related_file_candidates`.

Use Source Context as a checklist, not as the final architecture answer.

You must review:

- every class source range;
- every method;
- every module-level function;
- every high or medium branch;
- every weight mapping;
- every detected capability.

No silent omission is allowed. Not everything must become a node, but everything
must receive a disposition in `architecture-plan.json`.

## Step 3: Traverse Related Source

Default traversal budget:

- maximum follow depth: 3;
- default related file count: 20;
- suggested maximum related file count: 30.

Read related files when they prove a claim, clarify an interface, or define an
external boundary. Stop when:

- the page claims are supported;
- the source reaches a stable imported component boundary;
- more recursion will not change the diagram;
- the traversal budget is reached.

Record every file read in `files_read` and explain important stops in
`traversal_notes`.

## Step 4: Plan Composite Pages

Fill `architecture-plan.json` before drawing.

For each page:

- state the engineering question;
- choose a `view_pattern`;
- list topics;
- provide at least one `detail_region`;
- map each region to `source_item_ids` and `claim_ids`;
- set a realistic detail budget.

Typical full-model clustering:

- Model Composition + End-to-End Execution;
- Repeated Block + Main Compute Mechanism;
- Specialized Subsystem, such as MoE, multimodal, pooling, or recurrent state;
- Parallelism + Configuration + Weight Loading + Boundaries.

Use a fifth page only if two visual revision rounds still cannot make the four
pages readable.

## Step 5: Fill Review Checklists

The plan template includes empty checklist items. Codex must complete them.

Class review:

- every class gets one entry;
- primary components cannot be excluded;
- rendered classes need page and region refs;
- excluded or unresolved classes need a concrete reason.

Method review:

- every method gets one entry;
- core methods must be `rendered_detail` or `rendered_aggregate`;
- each core method needs a claim and a page/region ref;
- trivial helpers can be excluded with reason.

Function review:

- every module-level function gets one entry;
- helper functions can be aggregated or excluded, but not ignored.

Coverage manifest:

- every high/medium branch;
- every weight mapping;
- every detected capability;
- external boundaries that affect diagram interpretation.

## Step 6: Write Evidence

Create `evidence.json` with schema version `2.1`.

Use:

- `direct`: local source directly proves the behavior;
- `derived`: multiple source facts support a higher-level claim;
- `external`: local code proves a delegation or dependency, but implementation
  is outside the reviewed boundary.

Direct claims cannot rely only on imports. Derived claims require derivation and
at least two evidence entries. External claims require an explicit boundary.

## Step 7: Validate Before Drawing

Run:

```powershell
vllm-arch validate `
  --context outputs/<model>/source-context.json `
  --plan outputs/<model>/architecture-plan.json `
  --evidence outputs/<model>/evidence.json
```

Fix errors before opening Draw.io. Treat coverage warnings as design feedback.

## Step 8: Draw With Draw.io MCP

Use Draw.io MCP for diagram creation and editing. Codex controls page design and
layout. Scripts do not render final diagrams.

Drawing rules:

- page names match the plan titles exactly;
- each detail region title appears on the matching page;
- runtime tensor flow uses primary arrows;
- construction/config dependencies use dashed or secondary lines;
- checkpoint loading is not runtime tensor flow;
- TP/PP/EP are panels or badges, not a serial data path;
- external components sit behind an explicit boundary;
- implementation names are subtitles;
- source line numbers and long expressions stay in Evidence and report;
- ordinary tensor edge labels are hidden unless needed for a branch;
- background is white and opaque.

Use `assets/drawio-style-template.drawio` as a style reference when useful.

## Step 9: Visual Review

Export the first PNG draft and write `visual-review.md`.

Check:

- whether every detail region appears in the diagram;
- whether a page is still only an overview;
- whether core methods appear only in the report but not in a diagram region;
- phase confusion between construction, runtime, loading, and parallelism;
- config lines drawn as tensor flow;
- external behavior drawn as local direct behavior;
- text overlap;
- edge-through-node problems;
- large empty spaces;
- whether grouping or folding would improve density.

Make at least one visual review pass for a formal example. Make at most two
revision rounds.

## Step 10: Validate Draw.io and Exports

Run:

```powershell
vllm-arch validate `
  --context outputs/<model>/source-context.json `
  --plan outputs/<model>/architecture-plan.json `
  --evidence outputs/<model>/evidence.json `
  --drawio outputs/<model>/architecture.drawio `
  --images-dir outputs/<model>/images
```

Do not treat the diagram as final if validation fails.

## Step 11: Final Report

Write `report.md` with:

- target model and registry mapping;
- files read and why traversal stopped;
- source coverage summary;
- class/method/function review summary;
- branch/mapping/capability coverage;
- page-by-page explanation;
- construction/runtime/loading/parallel separation;
- external boundaries;
- unresolved items;
- omitted details and reasons;
- validation results and output paths.

Separate Source-proven, Derived, and External-boundary statements.

## Degradation

For partial targets, generate one to five pages and clearly state what is
missing.

For helper or shared utility files, generate one or two utility/boundary pages.
Never force a full model flow onto helper code.

## References

Read these when needed:

- `references/analysis-guide.md`
- `references/vllm-patterns.md`
- `references/page-playbook.md`
- `references/diagram-design-guide.md`
- `references/evidence-policy.md`
