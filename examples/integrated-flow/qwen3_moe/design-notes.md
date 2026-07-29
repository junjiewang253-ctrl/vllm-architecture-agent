# Qwen3 MoE Integrated Flow Notes

## Source-Grounded Story

- PP entry/exit, local decoder slice and Eagle auxiliary hidden states:
  `samples/qwen3_moe.py:433-538`.
- Decoder residual handoffs and construction-time Dense/Sparse-MoE selection:
  `samples/qwen3_moe.py:357-431`.
- Dense MLP and optional shared-expert gate:
  `samples/qwen3_moe.py:85-127`.
- Q/K RMSNorm, RoPE, V bypass, external attention and output projection:
  `samples/qwen3_moe.py:254-355`.
- Sparse MoE internal-router path, fallback local router, optional sequence
  parallel chunk/all-gather and output shape restoration:
  `samples/qwen3_moe.py:130-251`.
- HF-to-vLLM `WeightsMapper` rules and model AutoWeightsLoader:
  `samples/qwen3_moe.py:433-538`.
- Wrapper capabilities, packed modules, tied embedding, logits and wrapper
  AutoWeightsLoader:
  `samples/qwen3_moe.py:541-657`.

## Integration Decisions

- The Qwen diagram does not reuse HY's HPC branch.
- Token inputs enter only the first PP rank; non-first ranks receive
  `IntermediateTensors` from a previous-rank source.
- Q and K always pass through RMSNorm and RoPE; V bypasses both local paths.
- KV Cache is an explicit external store connected to the imported Attention
  backend.
- `FusedMoE` internal routing is the configured primary path. The retained
  local-gate code is shown as a construction/fallback relation.
- Optional sequence parallel chunk and all-gather appear around FusedMoE.
- Eagle hidden-state capture is a cyan auxiliary lane, not the main logits
  path.
- Returned hidden states enter a separate `compute_logits` wrapper entrypoint
  before `ParallelLMHead` and `LogitsProcessor`.
- The MoE configuration strip records expert count, top-k, top-k
  normalization, shared-expert size, redundant experts and EPLB.
- Mapping families are spread across the loading plane and connect upward to
  the runtime components they populate.

## Remaining Limits

- Attention KV-cache internals, FusedMoE routing kernels, MoERunner expert
  loading and AutoWeightsLoader assignment remain external boundaries.
- The sample proves adapter wiring, not the internals of those imported
  components.
- Fine labels are intended for SVG or Draw.io zoom.
