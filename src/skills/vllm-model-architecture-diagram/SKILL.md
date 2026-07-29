---
name: vllm-model-architecture-diagram
description: Analyze a vLLM model adapter Python file or registered architecture, review all indexed source structures, author evidence-backed architecture claims, and use Draw.io MCP to create one editable integrated architecture canvas with an expanded runtime path, loading mappings, external boundaries, PNG review, and structural validation. Use for vllm/model_executor/models source analysis or source-grounded architecture diagrams.
---

# vLLM Model Architecture Diagram

Analyze a vLLM adapter and produce a source-grounded engineering diagram. The
default deliverable is one large, connected, zoomable architecture canvas, not
a dashboard of concept cards and not one page per subsystem.

Default user instruction:

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，
生成默认架构图。
```

For an unambiguous local target, infer the workspace root, model slug and output
directory. Do not ask the user for page count, Plan paths or Evidence paths.

## Defaults

- `detail_level`: `complete`
- `diagram_strategy`: `integrated_single_canvas`
- `target_pages`: `1`
- `max_pages`: `1`
- logical canvas: at least `4200 x 2000`, usually about `5200 x 2500`
- maximum integrated aspect ratio: `2.5:1`
- PNG export: at least `2400 x 1200`
- visual review rounds: one required, two maximum
- output: `outputs/<model>/`

Use `composite_pages` only when one canvas remains unreadable after two genuine
MCP layout revisions. Record the reason in `integration_rationale`. A helper or
boundary-only file may use `compact_boundary`.

## Non-Negotiable Boundary

Scripts discover and validate source structure. You decide architecture
meaning, visual hierarchy and layout.

Never:

- generate final model semantics from a fixed model template;
- treat construction, configuration, loading or parallel metadata as runtime
  tensor flow;
- infer imported implementation internals without reading them;
- mark external behavior as direct local behavior;
- edit model source to make a diagram easier;
- manufacture `.drawio`, PNG or SVG files with Python, shell, templates or
  placeholder bytes;
- fall back when Draw.io MCP is unavailable.

## Required References

Read these before planning:

- `references/analysis-guide.md`
- `references/vllm-patterns.md`
- `references/evidence-policy.md`
- `references/page-playbook.md`
- `references/diagram-design-guide.md`
- `references/integrated-flow-standard.md`

When the repository examples are available, open both reference PNGs:

- `examples/integrated-flow/hy_v3/architecture.png`
- `examples/integrated-flow/qwen3_moe/architecture.png`

They define the expected integration and information density, not fixed model
content. Do not copy their model-specific nodes into another model.

## Workflow

### 1. Resolve and Prepare

Infer the workspace root from the nearest ancestor containing `pyproject.toml`
and this Skill. Resolve a file path directly or resolve an architecture name
statically through `registry.py`.

Run:

```powershell
vllm-arch prepare `
  --repo-root <workspace-root> `
  --input <target.py> `
  --outputs-dir outputs/<model> `
  --model-name <model>
```

Preparation creates:

- `source-context.json`
- `architecture-plan.template.json`
- `evidence.template.json`

The templates are checklists. They contain no final pages, nodes, edges or
coordinates.

### 2. Review the Complete Target File

Read the full Python file, then use `source-context.json` as a no-omission
checklist:

- every class, including nested classes;
- every method and module function;
- high and medium branches;
- architecture-level weight mapping groups;
- detected capabilities;
- related-file candidates.

Assign every indexed item a Plan disposition. A trivial item may be excluded
with a concrete reason. Aggregation is not a blanket escape hatch:

- render representative primary classes in detail;
- render at least 20% of Agent-reviewed core methods in detail, capped by the
  Validator;
- render representative high-relevance branches and mapping groups in detail;
- never map a large collection of core methods to one summary box.

Completeness means explicit review plus visible architectural depth, not one
node per method and not one region containing every source ID.

### 3. Traverse Only Useful Related Source

Open related files when they prove a major claim or clarify an external
boundary.

Default budget:

- depth: 3;
- normal limit: 20 files;
- suggested hard limit: 30 files.

Stop at a stable imported API boundary or when more traversal will not change
the architecture. Record every file read and the stop reason.

### 4. Write Evidence

Write `evidence.json` before the Plan is finalized. Keep repository files
repo-relative.

- `direct`: a local source range directly proves the statement;
- `derived`: at least two evidence entries plus an explicit derivation;
- `external`: local code proves delegation, while the unreviewed implementation
  remains outside the claim.

Direct claims cannot rely only on imports. Weight loading claims use phase
`loading`, not `runtime`. In complete mode, every Evidence `symbol` must match
the cited lines exactly or by its final Python identifier. Do not use wildcard
names, slash-separated summaries or prose as `symbol`.

### 5. Design the Integrated Story

Do not begin with pages or cards. First write one sentence:

> From which input, through which local transformations and imported
> boundaries, to which output does this adapter operate?

Then identify:

1. the runtime spine from input to returned model output;
2. mutually exclusive PP or input branches and their merge;
3. a representative repeated block expanded at its position in that spine;
4. compute details nested inside that block;
5. residual or bypass lanes;
6. the separate logits or pooling entrypoint when source defines one;
7. external runtime boundaries at the actual delegation points;
8. construction/configuration dependencies attached to affected components;
9. a lower loading plane whose mapping edges terminate at the components that
   receive those weights;
10. capability and parallel metadata as badges or compact side panels.

The result must read as one system. Do not draw four independent quadrants
called Overview, Attention, MoE and Loading.

An anchor with role `branch` must visibly split into at least two alternatives.
An anchor with role `merge` must visibly receive at least two alternatives.
Never use one box named "PP branch" or "output gate" to hide both paths.

#### Capability-Driven Depth

Use source categories, not model names:

- **Dense/repeated compute:** expand one local repeated block with its main
  compute path, residual/bypass lane and dense FFN when source proves them.
- **MoE:** show router inputs and logits, routed execution, FusedMoE as a
  container or external boundary, experts as contained members, EP metadata as
  dependency, and expert mappings in the loading plane.
- **Multimodal:** show modality-specific input branches, processor/placeholder
  handling, encoder entry, one expanded encoder block, feature
  merge/injection, local language-model orchestration and external decoder
  internals. Do not collapse EVS/MRoPE, DeepStack, eager/CUDA-graph selection
  or equivalent proven branches into one prose note.

An imported decoder, attention kernel or fused runtime may remain external.
Local orchestration around that boundary must still be expanded.

### 6. Write Architecture Plan 2.2

Write `architecture-plan.json` from the template:

- keep `schema_version` as `2.2`;
- keep the default `integrated_single_canvas` unless a documented exception is
  necessary;
- use one page with a concrete `main_story` of at least seven stages;
- use at least three evidence-backed detail regions;
- attach every region to a main-story anchor;
- map every rendered item to a real region;
- give structural regions at least four semantic anchors and three internal
  relationships;
- split a region when each visible anchor would otherwise stand for more than
  eight indexed source items;
- give each core-heavy region representative detailed methods instead of one
  detailed method surrounded by aggregate reviews.

#### Visual Contract

The page `visual_contract` is a drawing acceptance contract, not a second IR and
not a coordinate plan.

For each required visible element define a `required_anchor`:

- ID beginning with `visual:`;
- architecture label;
- representation such as node, container, storage, badge or merge;
- role and phase;
- semantic style role;
- claim IDs and source item IDs;
- optional parent container anchor.

For every required visible relation define a `required_relationship`:

- ID beginning with `visual-rel:`;
- source and target anchor IDs;
- kind;
- claim IDs.

Required relationship kinds:

- `runtime_flow`
- `residual_flow`
- `loading_mapping`
- `construction_dependency`
- `metadata_dependency`
- `external_delegation`

Use at least 18 anchors and 14 relationships for a complete full model. List
the ordered runtime spine in `main_story_anchor_ids`. Do not add an anchor for
every variable; anchors represent architecture-visible information.

Every detail region includes:

- an integration mode;
- `attach_to_anchor_id`;
- the visual anchors that realize the region.

The final canvas must use its space. Spread architecture anchors across the
runtime and loading bands instead of placing a few distant boxes around a large
blank center. Put loading dispatch below its receiving components so purple
mapping edges remain local. Keep the loading plane within about 20% of canvas
height below the runtime band, and keep ordinary loading destinations within
about 32% of canvas width from their dispatch stage.

### 7. Validate Before Drawing

Run:

```powershell
vllm-arch validate `
  --repo-root <workspace-root> `
  --context outputs/<model>/source-context.json `
  --plan outputs/<model>/architecture-plan.json `
  --evidence outputs/<model>/evidence.json
```

Fix all errors before opening Draw.io.

## Draw.io MCP Protocol

### 8. Start the Session

Confirm Draw.io MCP tools are available, then call `start_session`. If it fails,
stop and report the drawing stage as blocked. Do not synthesize replacement
files.

### 9. Create a Large Canvas

Use `create_new_diagram` and explicitly set:

- white background;
- landscape `pageWidth` and `pageHeight` satisfying the Plan canvas;
- no HTML labels;
- a title and one engineering question in separate text cells;
- enough spacing for labels at 12 px or larger.

For every required visual anchor, set the corresponding `mxCell` attribute:

```text
dataAnchor="visual:<id>"
```

For every required relationship edge, set:

```text
dataAnchor="visual-rel:<id>"
```

The Validator checks uniqueness, endpoints, topology and semantic line style.
Place every detail-region title immediately above or inside its actual
subgraph. A title at the opposite edge of the canvas does not realize a region.

### 10. Compose, Do Not Tile

Prefer a continuous left-to-right runtime spine. When a single row would exceed
the aspect-ratio limit or make labels unreadable, fold the same connected spine
once into a second runtime row at a semantic handoff such as feature merge or
decoder entry. Make the U-turn explicit with arrow direction; do not turn the
second row into a detached panel. Keep `main_story_anchor_ids` in execution
order even when the second row reads right-to-left. Expand the representative
decoder or repeated block in place with nested containers. Put optional branches
above or below their split and merge points. Use a distinct residual lane.

Place checkpoint loading below the runtime system as a mapping plane. Mapping
edges must point upward to actual projection, FFN, expert, embedding, head or
other destination components.

Avoid giant routing buses. If an edge requires a long detour, move or regroup
the source and target. Do not use far-away notes or mapping boxes to inflate
the content bounding box.

Use:

- blue solid arrows for runtime tensors;
- cyan solid arrows for residual/bypass paths;
- purple solid arrows for loading and mapping;
- warm dashed lines for construction/configuration;
- gray dashed lines for metadata/parallel effects;
- warm dashed nodes and edges for external boundaries;
- containers for ownership;
- badges for capabilities.

Ordinary tensor edges need no repeated labels. Implementation classes belong
in subtitles, not as the whole architecture story.

### 11. Export and Inspect Draft 1

Use Draw.io MCP `export_diagram` for:

- `outputs/<model>/architecture-draft1.drawio`;
- `outputs/<model>/architecture.drawio`;
- `outputs/<model>/images/<export_name>.png`;
- optional SVG for zoomable review.

Open the real PNG with the image viewer. A file check is not visual review.

Write `visual-review.md` with:

- Draft 1 findings;
- a `Geometry changes` section naming nodes/groups moved or resized and edges
  rerouted;
- the MCP edits actually made;
- final review;
- remaining risks.

Inspect at 100% and zoomed:

- can one trace input to output without jumping between disconnected panels?
- is the repeated block visibly expanded at its runtime location?
- are PP alternatives branches rather than serial steps?
- are detail regions connected to their declared anchors?
- do weight mappings terminate at real runtime components?
- are external boundaries at real delegation points?
- are construction/runtime/loading/parallel visually distinct?
- are labels readable, edges routed around nodes and blank areas intentional?

Make a region-level MCP revision after Draft 1 and export again. Move or resize
at least 15% of required anchors, with a minimum of four, and reroute at least
two required relationships. Adding notes, changing labels or moving one small
box does not count. Use at most two revision rounds.

### 12. Final Validation and Report

Run:

```powershell
vllm-arch validate `
  --repo-root <workspace-root> `
  --context outputs/<model>/source-context.json `
  --plan outputs/<model>/architecture-plan.json `
  --evidence outputs/<model>/evidence.json `
  --drawio outputs/<model>/architecture.drawio `
  --images-dir outputs/<model>/images `
  --visual-review outputs/<model>/visual-review.md
```

Do not report completion if Draw.io MCP was not used, a required anchor or
relationship is missing, PNGs were not opened, no revision occurred, or final
validation fails.

Write `report.md` only after validation. Include source coverage, review counts,
main architecture claims, external boundaries, omissions, visual review rounds,
exact validation result and output paths.

## Graceful Degradation

- Partial model: keep one integrated canvas, mark missing context, and reduce
  only unsupported detail.
- Helper/shared file: use `compact_boundary` with one or two focused pages.
- Exceptionally large model: attempt two integrated-layout revisions first;
  then use `composite_pages` only with a clear rationale and explicit
  cross-page reading order.
