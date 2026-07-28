# Diagram Design Guide

Use a white opaque background and compact, readable hierarchy.

## Page Structure

Top: title and one engineering question.

Center: main flow or core structure.

Side/bottom: detail regions, parameter panels, capability badges and external
boundaries.

## Visual Semantics

- Runtime tensor flow: blue solid primary arrows.
- Residual/data bypass: thinner solid lines.
- Construction/config dependency: gray dashed lines.
- Weight mapping: purple mapping style.
- External delegation: red or orange dashed boundary crossing.
- Containment: containers, not arrows.
- Capability: badges, not flow nodes.
- Parallel strategy: independent panel.

## Density Targets

- primary flow nodes: 5 to 12;
- total semantic modules: 12 to 28;
- visible edges: 8 to 26;
- parameter panel: at most 10 items;
- title: at most two lines;
- subtitle: at most two lines.

Long Python expressions belong in Evidence and report, not in diagram nodes.

For a complete full-model page, eight semantic elements is the acceptance
minimum after excluding the page title, engineering question and region
headers. The minimum is a floor, not a target.

## Page Acceptance Rubric

Score each category 0, 1 or 2 after opening the exported PNG:

| Category | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Architecture story | No readable flow | Partial flow | Clear input, transformation, boundary and output |
| Detail coverage | Region-name cards | Some regions expanded | Every region has a subgraph or structured panel |
| Semantic distinction | One generic style | Partial distinction | Runtime/config/loading/external are visually distinct |
| Topology and grouping | Detached or misleading | Mostly connected | Branches, containers and panels express meaning |
| Readability | Collisions or tiny text | Readable with friction | Clear hierarchy and labels |
| Information density | Sparse summary | Moderate detail | Compact but complete engineering content |

Accept a page only at 10/12 or above with no zero. The visual review must name
specific observed defects and specific MCP edits; “looks good” is not a review.

## Avoid

- every class as a standalone card;
- every parameter as a node;
- TP to PP to EP serial chains;
- checkpoint loading as runtime tensor flow;
- wrapper and base loader methods drawn as direct serial calls without evidence;
- capability badges connected as fake runtime dependencies;
- construction variants drawn as token-time decisions;
- deleting core behavior only to make the page cleaner.
