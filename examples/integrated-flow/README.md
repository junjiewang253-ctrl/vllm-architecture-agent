# Integrated Single-Canvas Reference

This directory is the manually designed visual baseline for the Plan 2.2
integrated-flow contract. It was drawn directly from the two source files
before that contract was implemented. It is not a four-page diagram compressed
into one page. Each model is represented as one connected, hierarchical
architecture flow:

1. the runtime spine runs continuously from model inputs to logits;
2. the repeated decoder layer is expanded in place;
3. attention and MoE details remain nested at their execution positions;
4. PP entry and exit behavior is part of the runtime topology;
5. TP, EP, quantization and capabilities attach to affected components;
6. checkpoint loading forms a lower mapping plane whose edges terminate at the
   runtime components that receive the weights;
7. imported vLLM behavior is shown as an explicit external boundary.
8. model `forward` output and the separate `compute_logits` entrypoint remain
   distinct;
9. KV cache storage and read/write boundaries are visible;
10. non-first pipeline ranks receive `IntermediateTensors` from the previous
    rank, not from the model's token-input branch.

## Files

- `hy_v3/architecture.drawio`: editable HY V3 reference.
- `hy_v3/architecture.png`: full-size raster export.
- `hy_v3/architecture.svg`: scalable export.
- `qwen3_moe/architecture.drawio`: editable Qwen3 MoE reference.
- `qwen3_moe/architecture.png`: full-size raster export.
- `qwen3_moe/architecture.svg`: scalable export.

Draft exports are retained to prove that each diagram was opened and revised
after visual inspection.

## Visual Semantics

- Blue solid arrows: runtime tensors.
- Cyan solid arrows: residual or auxiliary hidden-state paths.
- Purple solid arrows: checkpoint loading and weight mapping.
- Orange dashed arrows: construction or configuration dependencies.
- Red dashed nodes and arrows: imported vLLM implementation boundaries.
- Containers: ownership or composition, never runtime sequencing.
- Compact badges: capabilities and parallel strategies.

## Acceptance Standard

A future default Skill output should meet these conditions:

- one connected main story, not a dashboard of independent panels;
- an expanded representative repeated block with visible entry and exit;
- no isolated detail region unless it is explicitly auxiliary;
- all major external components located at the real delegation point;
- weight mapping connected to the actual destination component;
- construction-time alternatives not presented as token-time branches;
- mutually exclusive PP inputs merge before the local decoder slice;
- the non-first PP input originates at the previous PP rank;
- a separate `compute_logits` stage follows returned hidden states;
- attention pages expose KV-cache storage and external read/write behavior;
- MoE configuration includes routing, expert-count and load-balancing controls;
- readable at full-size PNG and naturally zoomable as SVG/Draw.io;
- at least one PNG-based visual review and one MCP edit.

Plan 2.2 now makes this topology the default target: one large canvas, explicit
visual anchors, required semantic relationships and Draw.io validation. The
examples remain visual references rather than production templates; their
model-specific nodes must not be copied into unrelated adapters.
