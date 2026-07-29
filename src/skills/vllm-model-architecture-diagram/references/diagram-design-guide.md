# Diagram Design Guide

Use a white, opaque, landscape canvas. For a complete model, explicitly enlarge
the Draw.io page to at least `4200 x 2000`; about `5200 x 2500` is a useful
starting point.

## Composition

Header:

- architecture title;
- one engineering question;
- compact capability strip and legend.

Upper/main area:

- continuous runtime spine;
- input alternatives and merge;
- inline expanded repeated block;
- nested compute and specialized subsystem;
- rank-dependent output and output entrypoint.

Lower area:

- checkpoint/loading plane;
- filter, rename and dispatch;
- mapping edges that rise to receiving runtime components.

Side bands:

- parameter panels;
- parallel metadata;
- external boundary notes.

This is one composition. Avoid equally sized independent dashboard panels.

## Visual Semantics

| Meaning | Style |
| --- | --- |
| Runtime tensor/data | blue solid, 2-3 px |
| Residual/bypass | cyan solid, 1.5-2 px |
| Weight loading/mapping | purple solid, 2-2.5 px |
| Construction/config | orange dashed |
| Parallel/metadata | gray dashed |
| External delegation | orange/red dashed |
| Containment | swimlane/container, no arrow |
| Capability | compact rounded badge |
| Cache/storage | cylinder with explicit boundary |

Ordinary runtime edges need no labels. Label only decisions, mapping families,
boundary crossings or otherwise ambiguous relations.

## Nodes and Text

- architecture concept is the title;
- implementation class/method is the subtitle;
- title and subtitle use at most two lines each;
- required semantic anchors use at least 12 px text by default;
- page title uses 24-30 px;
- region title uses 16-20 px;
- code expressions and line references stay in Evidence/report;
- use short parameter rows instead of prose paragraphs.

Use ports or nearby branch nodes for Q/K/V-like data when separate cards add no
meaning. Use actual nodes when a path has distinct processing.

## Containers

Containers express ownership or implementation scope. A representative block
container should have visible input and output on the main route. Nested
containers must be geometrically contained, even when Draw.io cells use the
page root as their XML parent.

External components use a warm dashed outline and the word `EXTERNAL` or an
equally clear boundary label.

## Routing

- prefer orthogonal edges;
- keep the main route monotonic left to right when readable, or use one explicit
  connected U-turn into a second runtime row when a single row exceeds the
  aspect-ratio or label-readability budget;
- route branches above/below and align their merge;
- keep residual lanes parallel to the main path;
- route long loading mappings in a dedicated corridor;
- avoid using one edge segment for unrelated delegations;
- do not route through text or non-endpoint nodes.

## Visual Contract Attributes

Copy Plan IDs exactly:

```xml
<mxCell dataAnchor="visual:runtime-stage" ... />
<mxCell dataAnchor="visual-rel:stage-to-next" edge="1" ... />
```

Required anchor IDs are unique. Relationship edges use the exact planned source
and target cells. The validator also checks semantic style, canvas size,
content fill and export dimensions.

## Density Targets

One full-model integrated canvas usually has:

- 7-14 main-story stages;
- 18-80 required anchors;
- 14-110 visible edges;
- 3-7 detail regions;
- 4 or more visual styles.

Increase the page size instead of shrinking text below readability. Aggregate
repetitive low-level operations before considering a second page.

Large does not mean sparse. After layout:

- keep integrated canvas aspect ratio at or below 2.5:1;
- required anchors should span at least three quarters of page width;
- use both runtime and loading bands when loading is present;
- distribute meaningful anchors across the canvas instead of using remote
  notes to enlarge the bounding box;
- keep long loading mappings local by placing dispatch branches below their
  destinations;
- keep runtime-to-loading gap within about 20% of page height;
- keep ordinary loading mappings within about 32% of page width;
- prefer expanding a source-proven subsystem over filling space with prose.

Keep the page title and engineering question in separate header cells. Put
each detail-region title immediately above or inside its visible subgraph.

At Draft 1, compare the visual center of mass with the page center. If most
content forms one narrow row or leaves a dominant blank rectangle, regroup and
resize before final export.

## Acceptance Rubric

Score 0, 1 or 2 after opening the exported PNG:

| Category | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Connected story | No traceable route | Partial route | Input-to-output route is continuous |
| Inline detail | Generic cards | Some expansion | Repeated block and major subsystems are expanded in place |
| Semantic fidelity | Phases mixed | Mostly distinct | Runtime/config/loading/external are unambiguous |
| Topology | Misleading links | Minor issues | Branch, merge, containment and mappings are accurate |
| Readability | Tiny/colliding | Usable with friction | Labels and routes are clear |
| Information density | Sparse summary | Moderate | Complete without becoming a code dump |

Require at least 10/12 with no zero. Record specific Draft 1 defects and the
specific MCP geometry edits that addressed them. A label-only or note-only
change does not satisfy the revision requirement. Save Draft 1 and change at
least 15% of required anchor geometries plus two required routes.

## Reject

- card grids;
- default-size canvas;
- disconnected detail regions;
- generic loader or model sinks when specific destinations are known;
- construction variants drawn as token-time branches;
- TP/PP/EP serial chains;
- capability badges used as flow nodes;
- external internals shown as local direct behavior;
- core behavior omitted only for visual neatness.
