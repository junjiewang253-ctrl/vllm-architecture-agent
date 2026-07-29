# Integrated Flow Standard

This reference defines the visual target for a complete model adapter. It
describes topology and quality, not a fixed model template.

## Reference Artifacts

When working in this repository, inspect:

- `examples/integrated-flow/hy_v3/architecture.png`
- `examples/integrated-flow/qwen3_moe/architecture.png`

Use the Draw.io or SVG versions when labels require zoom. The examples differ
semantically, but share one composition grammar.

## Core Principle

The diagram is one connected explanation:

```text
inputs
  -> entry alternatives and merge
  -> local repeated stack
       -> expanded representative block
            -> local preprocessing
            -> imported compute boundary
            -> specialized subsystem
            -> residual/bypass paths
  -> rank-dependent output
  -> returned hidden state or model output
  -> separate logits/pooling entrypoint when present

checkpoint/loading plane
  -> filter/remap/dispatch
  -> actual destination components above
```

Do not turn this into independent Overview, Attention, MoE and Loading boxes.
The detail belongs where it executes or where it configures a component.

## Five Visual Layers

### 1. Runtime Spine

Use a strong left-to-right route from external input to returned output when it
fits the readable aspect-ratio budget. For a complex adapter, one deliberate
fold into a second connected runtime row is preferred over an unreadably long
strip. Preserve execution order, show the U-turn explicitly, and keep the
second row connected to the same main story. Show
branch and merge points for mutually exclusive inputs, implementations or rank
behavior. Do not serialize alternatives.

The main story should remain traceable when all side annotations are ignored.

### 2. Inline Expansion

Expand one representative repeated block inside the stack container. Its entry
and exit stay on the main route. Nested subsystems sit at their real execution
positions.

Use local containers for:

- a repeated block;
- the primary compute mechanism;
- a routed or specialized subsystem;
- internal expert/member composition.

Containment means ownership, not runtime sequencing.

### 3. Auxiliary Lanes

Residual, bypass, cache, auxiliary hidden-state or optional paths use distinct
lanes. They split and rejoin visibly. Configuration choices that are fixed
during construction use dashed dependencies, not token-time diamonds.

### 4. External Boundaries

Place an imported runtime component at the local call site. Label it external
and use a warm dashed outline. Expand only behavior supported by files actually
read.

External storage such as a cache may use a cylinder. Separate locally proven
writes from external reads or scheduling.

### 5. Loading Plane

Put checkpoint ingestion below the runtime architecture. Show filtering,
renaming and dispatch as loading operations. Mapping must branch when target
families differ.

Purple mapping edges terminate at the runtime component or container that owns
the loaded parameter. Avoid a generic "model" sink when the source proves a
more specific destination.

## Canvas

For a complete full model:

- use one landscape page;
- logical size is at least `4200 x 2000`;
- `5200 x 2500` is a practical starting point;
- reserve the upper 55-65% for runtime and inline expansion;
- reserve the lower 25-35% for loading/mapping when present;
- keep a narrow header and legend band;
- use 12 px or larger for required semantic anchors;
- export at 2400 x 1200 or larger.

Changing the Draw.io page size is part of the task. Never squeeze this standard
onto the default 800 x 600 canvas.

## Visual Contract Authoring

The Plan names required architecture-visible objects without choosing
coordinates.

Use a required anchor for:

- main input and output;
- branch and merge points;
- repeated-stack and expanded-block containers;
- major local transformation stages;
- imported compute boundaries;
- storage or cache boundaries;
- mapping dispatch and destination families;
- capability or strategy panels when important.

Do not create anchors for every tensor alias, local variable or line of code.

Use `parent_anchor_id` to require semantic nesting. Use detail-region
`integration` to state where the region attaches. A region is invalid when its
anchors form an isolated island.

Relationship kinds carry line semantics:

| Kind | Meaning | Draw.io style |
| --- | --- | --- |
| `runtime_flow` | tensor/data execution | blue solid, prominent |
| `residual_flow` | residual or bypass | cyan solid, thinner |
| `loading_mapping` | checkpoint mapping | purple solid |
| `construction_dependency` | fixed construction/config | warm dashed |
| `metadata_dependency` | parallel/metadata influence | gray dashed |
| `external_delegation` | crossing local boundary | warm dashed |

Every required anchor and relationship must have a matching `dataAnchor` in
the Draw.io XML. This makes the semantic drawing contract auditable without
turning the Plan into a coordinate-level View Graph.

## Information Density

A complete integrated page normally contains:

- 7-14 main-story stages;
- 18-80 required semantic anchors;
- 14-110 visible relationships;
- 3-7 nested or attached detail regions;
- at least four semantic visual styles.

These are ranges, not quotas. Prefer aggregation when repeated low-level
operations convey the same architecture idea.

### Visible Depth Floor

Completeness review and visible architecture are related but not identical.
Use `rendered_aggregate` for repetitive helpers, not for every core method.

- render representative primary classes in detail;
- render a meaningful sample of core methods and high-relevance branches in
  detail;
- give structural regions at least four semantic anchors and three internal
  relationships;
- split a region when a handful of anchors would absorb dozens of unrelated
  source items;
- keep the average at eight or fewer indexed source items per visible
  semantic anchor;
- make each detected main subsystem visible as an internal subgraph, not a
  prose annotation.

For dense compute this usually means an expanded repeated block. For routed
compute it means router, routed execution and contained members. For
multimodal adapters it means modality input processing, encoder detail,
feature fusion/injection and local language-model orchestration.

### Canvas Occupancy

Large page dimensions alone do not create a high-density diagram. Required
anchors should occupy most of the width, both runtime and loading bands when
loading exists, and multiple areas of the canvas rather than a single thin
row.

- align loading sources below their destination components;
- keep the runtime-to-loading gap within roughly 20% of page height;
- keep ordinary loading mappings within roughly 32% of page width;
- avoid one mapping edge spanning most of the page diagonal;
- reduce page dimensions or regroup content when the center remains empty;
- expand source-proven local behavior before adding explanatory callouts;
- judge the exported PNG at normal display scale, not only at deep zoom.

## Anti-Patterns

Reject the draft when:

- the page is a grid of disconnected cards;
- detail is connected only by "see detail" arrows;
- a repeated block remains a black box despite available source evidence;
- parallel strategies appear as sequential runtime stages;
- construction alternatives appear as per-token branches;
- external internals appear as local direct behavior;
- checkpoint loading is drawn as tensor inference flow;
- mappings end at a generic summary box instead of receiving components;
- implementation names replace architecture labels;
- the page remains near the Draw.io default size;
- the diagram is technically complete only because the report contains the
  missing behavior.
- all primary classes and core methods are marked `rendered_aggregate`;
- a giant canvas passes only because distant notes inflate its bounding box;
- Draft 1 is "revised" only by adding prose.
- branch or merge boxes have only one visible alternative;
- title and engineering question are collapsed into one oversized line;
- detail-region headings are detached from the subgraphs they describe.

## Draft Review

At Draft 1, inspect the exported PNG and answer:

1. Can a reader trace one input-to-output route?
2. Are alternatives visibly exclusive and merged correctly?
3. Is the repeated block expanded in place?
4. Are specialized subsystems nested at their execution point?
5. Are loading mappings visibly connected to receiving components?
6. Are external boundaries explicit and conservatively worded?
7. Can each required detail region be found without searching?
8. Are text, edge routing and spacing readable at normal zoom?

Save Draft 1 as Draw.io XML. Make a region-level MCP edit that changes at least
15% of required anchor geometries, with a minimum of four, and reroutes at
least two required relationships. The final review should name moved/resized
groups and rerouted edges, not merely say that the diagram looks good or that
notes were added.
