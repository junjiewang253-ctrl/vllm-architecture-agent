# Page Playbook

v2.1 favors few, dense, composite pages. A page can include a main story plus
several detail regions.

## Composite Page Parts

Each page may contain:

- a main story;
- one to four detail regions;
- a parameter or config panel;
- an external boundary;
- a capability badge area.

Every detail region should map to source item IDs and claim IDs.

## Main Story Grammar

For a flow-oriented page, write the main story as concrete visible stages:

```text
input data -> local transformation -> component execution -> boundary -> output
```

Use 5 to 12 stages for runtime or branch pages and at least 4 stages for mapping
or state pages. Do not replace the sequence with three explanatory sentences.

Region headers organize the page; they do not count as architecture elements.
A region must contain at least two components/data/processes, or a structured
panel with at least three entries. A row of region-name cards is not an
architecture diagram.

## Detail Region Types

### component_tree

Use for wrapper/base/decoder/attention/MoE composition. Use containment, not
runtime arrows, for pure ownership.

### embedded_subgraph

Use for attention branch/merge, decoder residual flow, MoE routing, multimodal
fusion or recurrent state updates.

### parameter_panel

Use for symbolic config values such as top-k, head partition, expert counts,
quant config, prefixes, scoring functions and route normalization. Do not create
one node per parameter.

### variant_table

Use for construction-time variants such as dense versus MoE layers, first-rank
versus later-rank setup, or configured HPC versus fallback implementation.
Construction variants are not token-time runtime branches.

### strategy_panel

Use for TP, PP and EP. Strategies should be independent panels, not a fake
tensor-flow chain.

### mapping_tree

Use for checkpoint dispatch: stacked mapping, expert mapping, regular parameter
path, filter/skip policy and loader fallback. Dispatch is a tree, not a serial
chain.

### boundary_panel

Use for imported attention backends, fused expert runtimes, automatic loaders,
process groups, LoRA runtime and external kernels.

## Page Count

Full model: three to five pages. Prefer four.

Helper/shared module: one or two pages.

Do not make a page for every class or method. Aggregate details into regions
before adding pages.

## Anti-Minimality Check

Before drawing, reject a page design when any of these are true:

- the page can be represented by four to seven generic cards;
- a method name is the only explanation of its behavior;
- a runtime page has no explicit input or output;
- a branch page does not show its split and merge;
- a routed container does not show routing input plus contained execution;
- a mapping flow is a serial list instead of dispatch branches;
- a strategy page connects TP, PP and EP as if they were runtime stages;
- imported runtime components have no boundary treatment.
