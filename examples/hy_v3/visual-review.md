# HY V3 Visual Review

## Draft 1 Findings

- Page 1, `Model Architecture and Execution`: first-rank input preparation and non-first-rank `IntermediateTensors` were visually stacked, which could be read as a serial PP path.
- Page 1, `Class and Component Composition`: the component list did not make containment clear enough, and the optional embedding/LM-head tied weight was not visible.
- Page 2, `Decoder and Attention`: residual handling was only described by a note instead of a visible residual lane.
- Page 2, `Attention Forward Detail`: the HPC path reached the backend, but KV cache write/read boundary was not explicit enough; the first visual fix also produced a crowded edge label.
- Page 3, `MoE Architecture and Routing`: `GateLinear` appeared in both construction and runtime areas, which could imply two instances.
- Page 3, `MoE Architecture and Routing`: runtime `FusedMoE` and the composition/external runtime boundary needed an explicit relationship, and EP metadata needed to point to expert placement or the FusedMoE boundary.
- Page 4, `Parallelism, Configuration and Weight Loading`: the configuration panel included development wording about a missing LoRA-to-checkpoint edge; TP/PP/EP panels were too sentence-like.

## Revision Round 1

- Reworked Page 1 so first PP rank and non-first PP rank are side-by-side mutually exclusive entry modes that merge before the local decoder slice.
- Added dashed containment lines in the composition region and a dashed optional tied-weight relationship from embedding to the LM head.
- Added a residual bus on Page 2: residual initialization, input RMSNorm handoff, post-attention RMSNorm handoff and residual output.
- Added explicit KV cache write and external KV cache read boundary markers in the attention detail.
- Changed Page 3 construction text to `Router configuration / GateLinear(num_experts)` and connected it by a dashed configuration edge to the single runtime `GateLinear`.
- Added dashed delegation from runtime `FusedMoE` to the external fused runtime boundary, plus metadata edges from EP placement/update to the expert placement boundary.
- Removed the development note from Page 4 and rewrote TP/PP/EP panels as compact structured strategy summaries.

## Revision Round 2

- Removed the crowded `external read boundary` edge label on Page 2 and shortened the backend label to keep the fallback path readable.
- Shortened Page 4 strategy and capability panel text to avoid long wrapped paragraphs while preserving the same evidence-backed content.

## Final Review

- All four planned pages are present and exported as PNG with white backgrounds.
- Every planned detail region title appears in the corresponding Draw.io page.
- Runtime tensor flow, loading flow, construction/configuration dependency, parallel metadata and external delegation use distinct visual treatments.
- The final diagrams show PP entry alternatives, residual handoff, HPC/fallback attention, V bypass, KV cache boundary, FusedMoE containment, EP metadata, independent TP/PP/EP panels and two separate weight-loading entrypoints.
- No page is a concept-card-only summary; each page has a main story plus detail regions.

## Remaining Risk

- Pages 2 and 4 are intentionally dense. The high-density layout favors completeness, so small screens should use zoom rather than reading the exported PNG at thumbnail size.
