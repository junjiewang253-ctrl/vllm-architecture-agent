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

## Avoid

- every class as a standalone card;
- every parameter as a node;
- TP to PP to EP serial chains;
- checkpoint loading as runtime tensor flow;
- wrapper and base loader methods drawn as direct serial calls without evidence;
- capability badges connected as fake runtime dependencies;
- construction variants drawn as token-time decisions;
- deleting core behavior only to make the page cleaner.
