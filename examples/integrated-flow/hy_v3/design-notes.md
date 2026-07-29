# HY V3 Integrated Flow Notes

## Source-Grounded Story

- PP input alternatives and local layer execution:
  `samples/hy_v3.py:529-561`.
- Decoder residual handoffs and construction-time Dense/MoE selection:
  `samples/hy_v3.py:366-433`.
- Dense FFN internals:
  `samples/hy_v3.py:86-123`.
- QKV projection, configured HPC/fallback preprocessing, V bypass, external
  attention and output projection:
  `samples/hy_v3.py:224-363`.
- Local router logits, external FusedMoE execution, shared expert composition
  and output reshape:
  `samples/hy_v3.py:126-222`.
- EP/EPLB metadata and expert-map update:
  `samples/hy_v3.py:437-527`.
- Model-level stacked/expert/regular weight dispatch:
  `samples/hy_v3.py:563-651`.
- Wrapper filtering and AutoWeightsLoader boundary:
  `samples/hy_v3.py:654-725`.

## Integration Decisions

- `Decoder Stack` is expanded in place instead of linking to a separate detail
  page.
- Token inputs enter only the first PP rank. The non-first branch begins at a
  separate previous-rank `IntermediateTensors` source.
- The residual path is a second lane through both fused RMSNorm handoffs.
- HPC and fallback are labeled as configured implementations. The HPC path
  consumes `qkv`; the fallback path exposes local Q/K/V processing.
- `FusedMoE` is an external runtime boundary containing routed and optional
  shared expert execution, not a serial step before the experts.
- The MoE construction strip exposes expert count, top-k routing, sigmoid plus
  expert bias, grouped top-k, route normalization, routing scale and shared
  experts.
- `HpcRopeNorm` writes an explicit external KV Cache, and the external
  Attention backend reads that cache.
- Hidden states leave model `forward` before the separate `compute_logits`
  entrypoint applies `ParallelLMHead` and `LogitsProcessor`.
- Wrapper and model `load_weights` are independent local entrypoints.
- Purple mapping edges terminate at QKV, Dense FFN, FusedMoE and model
  parameters in the same graph.

## Remaining Limits

- KV-cache reads and internal attention scheduling remain external.
- FusedMoE dispatch/combine and AutoWeightsLoader delegation are not expanded
  beyond the local source boundary.
- The page is intentionally large and should be read through SVG or Draw.io
  zoom when inspecting fine detail.
