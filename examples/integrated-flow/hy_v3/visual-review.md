# HY V3 Visual Review

## Draft 1

The integrated runtime story was readable from input to logits, and the
expanded decoder contained both Attention and configured Dense/MoE execution.
Two defects remained: construction dependencies crossed the decoder title bar,
and labels on long purple mapping edges floated inside the runtime layer.

## Revision 1

Using Draw.io MCP, the quantization, TP and EP configuration nodes were moved
inside the decoder container. Labels were removed from the five long mapping
edges, and a compact note was added inside the loading plane to explain their
destination semantics.

## Baseline Correction

The PP input branch was corrected so that non-first ranks receive
`IntermediateTensors` from a previous-rank source. An explicit `compute_logits`
entrypoint was inserted after model hidden states. Attention now contains an
external KV Cache with separate HPC write and Attention read edges. A compact
MoE configuration strip records routing, expert-bias, grouped-top-k,
normalization, scaling and shared-expert controls.

## Final

- Architecture story: 2/2
- Detail coverage: 2/2
- Semantic distinction: 2/2
- Topology and grouping: 2/2
- Readability: 2/2 at full-size or zoomed SVG
- Information density: 2/2

The result is one connected graph rather than four adjacent page summaries.
The remaining risk is that a reduced Markdown preview cannot expose every
label; the SVG and editable Draw.io file are the primary inspection formats.
