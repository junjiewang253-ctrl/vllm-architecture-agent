# HY V3 Visual Review

## Draft 1 Findings

- Page 2, `Decoder and Attention`: the residual initialization note overlapped the `Decoder Output` node, and the HPC/fallback edge labels crowded the attention branch nodes.
- Page 3, `MoE Architecture and Routing`: container helper edges crossed the FusedMoE composition panel, and the GateLinear to Router Logits route initially crossed node text.
- Page headers used a combined title/question line that was too visually heavy.

## Revision Applied

- Split page titles and engineering questions into separate header lines.
- Moved the Decoder residual note below the main decoder lane and increased the Decoder Output node height.
- Removed visible branch labels from the crowded attention edges and kept the branch meaning inside the nodes.
- Removed redundant containment edges inside the FusedMoE composition boundary.
- Re-routed the GateLinear to Router Logits connection as a short local arrow that no longer cuts through the main label.

## Final Review

- All four plan pages are present and exported as PNG.
- Every planned detail region title is visible in the corresponding page.
- The diagrams distinguish runtime tensor flow, construction/configuration dependency, loading flow and external boundaries.
- No large concept-card-only page remains; each page has a main story plus detail regions.
- Residual, Q/K/V fallback, V bypass, FusedMoE containment, independent TP/PP/EP panels and two separate weight-loading entrypoints are represented.

## Residual Risk

- The MoE runtime region is intentionally dense; the GateLinear, Router Logits and FusedMoE inputs are readable after routing cleanup, but this area is the tightest part of the example.
