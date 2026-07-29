# Qwen3 MoE Visual Review

## Draft 1

The first draft correctly separated the model from the HY V3 design, including
fixed Q/K processing, the FusedMoE internal-router path, sequence parallelism,
Eagle hidden-state capture and `WeightsMapper`. The FFN container title was too
long, and the model-loading entry label floated on a long purple edge.

## Revision 1

Using Draw.io MCP, the FFN title was shortened and its layer-selection rule was
moved into a compact construction node. The remote edge label was removed and
an explicit note documented the two independent local load entrypoints.

## Revision 2

The mapping families were redistributed horizontally across the lower loading
plane. Their purple injection lines now rise from the corresponding QKV,
Dense, Shared Expert, routed-expert and regular-parameter mappings instead of
leaving most of the page empty.

## Baseline Correction

The non-first PP path was detached from token inputs and connected to a
previous-rank `IntermediateTensors` source. `compute_logits` was inserted as a
separate wrapper entrypoint. The attention subgraph gained an explicit external
KV Cache with read/write interaction, and the decoder header gained a compact
MoE routing and load-balancing configuration strip.

## Final

- Architecture story: 2/2
- Detail coverage: 2/2
- Semantic distinction: 2/2
- Topology and grouping: 2/2
- Readability: 2/2 at full-size or zoomed SVG
- Information density: 2/2

The lower plane is visually secondary but remains connected to the live model
components. The main remaining risk is reduced-size preview legibility.
