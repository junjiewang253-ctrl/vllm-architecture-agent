# Page Playbook

Plan 2.2 defaults to one integrated architecture canvas for a complete model.
The page combines a continuous main story with detail regions expanded at their
real execution, configuration or loading positions.

## Default Page Strategy

Use `integrated_single_canvas` when the target is a full or partial model.

The page should contain:

- one runtime spine from input to returned output;
- an inline expanded representative repeated block;
- nested compute or specialized-subsystem detail;
- residual, cache or auxiliary lanes where source proves them;
- construction and parallel metadata attached to affected components;
- external boundaries at delegation points;
- a loading plane mapped into receiving runtime components;
- compact capability badges.

Do not place these as independent dashboard quadrants. A detail region must
attach to a main-story anchor through the Visual Contract.

Use `compact_boundary` for helper/shared files. Use `composite_pages` only after
two layout attempts show that a single canvas cannot remain readable.

## Main Story Grammar

Write concrete stages:

```text
input -> branch/merge -> local transformation -> repeated stack
      -> rank-dependent output -> returned state -> output entrypoint
```

Use at least seven stages for an integrated full model. Each stage has a
matching `main_story_anchor_id`, and required relationships must provide a
directed path between successive anchors.

Implementation names are subtitles. The main story uses architecture terms.

## Detail Region Integration

Every region declares:

- source item IDs;
- claim IDs;
- `integration.mode`;
- `integration.attach_to_anchor_id`;
- the required anchor IDs that realize it.

Use:

- `inline_expansion`: expand a repeated block on the runtime spine;
- `nested_container`: show owned components without fake flow arrows;
- `branch_merge`: show mutually exclusive or optional paths;
- `side_dependency`: attach construction, config or metadata;
- `loading_injection`: map loading output into runtime components;
- `boundary_crossing`: place imported behavior at a local call site.

An isolated region is invalid even if its title and evidence are present.

## Region Forms

### `component_tree`

Use for wrapper, base model, repeated block and owned component composition.
Containment is not runtime sequencing.

### `embedded_subgraph`

Use for attention preprocessing, residual handoffs, routed execution,
multimodal fusion or recurrent state updates.

### `parameter_panel`

Use for head partition, top-k, expert counts, quantization, prefixes, route
normalization and similar values. Do not create one node per parameter.

### `variant_table`

Use for construction-time variants. Never present a fixed implementation choice
as a per-token decision.

### `strategy_panel`

Use for TP, PP and EP metadata. Attach each strategy to affected components;
never draw TP -> PP -> EP.

### `mapping_tree`

Use for filter, rename, dispatch, packed/expert/regular branches and loader
fallback. Purple edges terminate at actual receiving components.

### `boundary_panel`

Use for imported attention, fused expert runtime, loaders, process groups,
LoRA runtime or external kernels. State what local source proves and what stays
external.

## Density

Typical complete integrated canvas:

- 18-80 required anchors;
- 14-110 required relationships;
- 3-7 detail regions;
- at least four semantic styles.

Aggregate repetitive details, but do not remove core behavior to satisfy a
visual preference.

## Capability-Driven Composite Grammar

These are subsystem grammars, not model templates.

### Dense or repeated compute

Keep the repeated stack on the runtime spine and expand one representative
block in place. Show local normalization, primary compute, residual/bypass and
dense FFN stages when source proves them. Imported attention internals may stay
external, but local projections and orchestration remain visible.

### Routed or MoE compute

Show data and router-logit inputs converging on routed execution. Represent
experts as contained members of the routed runtime, not stages after it.
Attach EP/EPLB metadata with dashed dependencies and map expert checkpoint
families into the expert container.

### Multimodal compute

Use modality-specific input branches that converge only after their real
processing:

```text
text ------------------------------\
image/video -> processor -> encoder -> feature merge/injection -> local LM
```

Expand one representative encoder block where it runs. Show local
placeholder/input validation, position metadata, feature merge, optional
injection and PP orchestration when present. A missing imported decoder file
does not justify hiding locally defined language-model control flow.

Do not compress proven correction or execution alternatives into prose cards.
Expose selection, compression/correction, position handling and ordered output
as a connected subgraph when the source contains them. Eager/CUDA-graph paths
must visibly split and rejoin the encoder route.

For every subsystem, a node labeled "branch" with one outgoing edge is not a
branch. A merge with one incoming edge is not a merge.

## Rejection Checklist

Reject the Plan or draft when:

- it is a collection of concept cards;
- the main story cannot be followed continuously;
- PP or input alternatives are serial;
- the repeated block stays opaque;
- a detail region has no connection to the main story;
- external behavior appears local;
- loading appears as runtime;
- mappings end at a generic summary node;
- the canvas remains near 800 x 600;
- crucial behavior exists only in `report.md`.
- all core methods are aggregated into one runtime region;
- a multimodal encoder, routed subsystem or repeated block is represented only
  by an annotation;
- a huge canvas contains a thin row of nodes and mostly blank space;
- the visual revision adds notes but does not change geometry or routing.
