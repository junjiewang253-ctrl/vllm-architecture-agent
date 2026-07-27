# HY V3 Architecture Example

## Target

- Model: `hy-v3`
- Source: `samples/hy_v3.py`
- Source Context schema: `2.1`
- Architecture Plan schema: `2.1`
- Evidence schema: `2.1`

The sample is outside a real vLLM repository checkout, so registry lookup is partial, but the target file itself was fully indexed.

## Source Coverage

- Classes indexed: 6 / 6
- Methods indexed: 20 / 20
- Module functions indexed: 0 / 0
- Branches catalogued: 37
- Weight mappings catalogued: 69
- Unindexed nodes: 0

## Review Coverage

- Class reviews: 6 total, 0 unreviewed
- Method reviews: 20 total, 0 unreviewed
- Function reviews: 0 total, 0 unreviewed
- Required high/medium branches: 30, covered: 37 manifest entries
- Weight mappings: 69 total, covered: 69
- Detected capabilities: 10, covered: 10
- Unresolved items: 0

## Evidence Summary

- Direct claims: 14
- Derived claims: 4
- External claims: 2

External claims are used for imported vLLM runtime internals. The local file proves delegation and arguments, but not backend implementation.

## Pages

1. `Model Architecture and Execution`
   - Regions: `Class and Component Composition`, `End-to-End Runtime`, `Output Boundary`
   - Shows wrapper/body composition, PP rank variants, local decoder execution, final add/norm, and logits processing.

2. `Decoder and Attention`
   - Regions: `Decoder Runtime`, `FFN Construction Variant`, `Dense FFN Internal Flow`, `Attention Construction Panel`, `Attention Forward Detail`
   - Shows hidden/residual lanes, construction-time Dense-vs-MoE selection, Dense FFN internals, attention configuration, HPC/fallback paths, V bypass and vLLM Attention boundary.

3. `MoE Architecture and Routing`
   - Regions: `Construction and Configuration`, `Runtime Routing`, `FusedMoE Composition`, `Expert Parallel Metadata`
   - Shows routing parameters, flatten/gate/router logits, FusedMoE as a container with routed and optional shared experts, and EP metadata.

4. `Parallelism, Configuration and Weight Loading`
   - Regions: `Tensor Parallel Strategy`, `Pipeline Parallel Strategy`, `Expert Parallel Strategy`, `Configuration and Capabilities`, `Wrapper Weight Loading`, `Model Weight Dispatch`
   - Shows independent TP/PP/EP panels, quantization and capability markers, wrapper AutoWeightsLoader lane, and model dispatch tree for stacked/expert/regular parameter paths.

## Correctness Notes

- Dense FFN and MoE are construction variants, not simultaneous runtime branches.
- `HpcRopeNorm.support()` is a construction-time implementation choice; the runtime page labels it as configured implementation.
- The HPC path receives the whole `qkv` tensor; it is not fed from the Q stream.
- Q and K may pass through optional Q/K RMSNorm; V bypasses Q/K Norm.
- KV cache read is marked as external vLLM Attention behavior.
- FusedMoE experts are contained inside the FusedMoE boundary, not drawn as a serial stage after FusedMoE.
- Wrapper and model `load_weights` are separate local entrypoints. AutoWeightsLoader delegation is external, and model loading is a dispatch tree rather than `stacked -> expert -> regular` serial flow.
- LoRA is represented as a capability marker only; there is no LoRA-to-checkpoint-loading edge.

## Validation

Commands run:

```powershell
python -m vllm_architecture_agent.cli validate --context examples\hy_v3\source-context.json --plan examples\hy_v3\architecture-plan.json --evidence examples\hy_v3\evidence.json
python -m vllm_architecture_agent.cli validate --context examples\hy_v3\source-context.json --plan examples\hy_v3\architecture-plan.json --evidence examples\hy_v3\evidence.json --drawio examples\hy_v3\architecture.drawio --images-dir examples\hy_v3\images
```

Results:

- Evidence validation: passed
- Architecture plan validation: passed
- Draw.io validation: passed

## Outputs

- Draw.io: `examples/hy_v3/architecture.drawio`
- PNGs:
  - `examples/hy_v3/images/model-architecture-and-execution.png`
  - `examples/hy_v3/images/decoder-and-attention.png`
  - `examples/hy_v3/images/moe-architecture-and-routing.png`
  - `examples/hy_v3/images/parallelism-configuration-and-weight-loading.png`
- Visual review: `examples/hy_v3/visual-review.md`
