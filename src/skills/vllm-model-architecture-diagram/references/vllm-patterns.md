# vLLM Patterns

Use these as recognition aids, not as final diagram templates.

Registry:

- model architectures are registered in model registry mappings;
- registry parsing must be static and must not import vLLM.

Common interfaces and capabilities:

- `SupportsPP` indicates pipeline-parallel support;
- `SupportsLoRA` indicates LoRA support;
- `SupportsMultiModal` indicates multimodal input handling;
- `MixtureOfExperts` indicates MoE integration;
- `support_torch_compile` indicates compile-related capability.

Common model building blocks:

- embeddings such as vocabulary-parallel embedding layers;
- tensor-parallel projections and row/column parallel linear layers;
- attention backend wrappers;
- fused MoE layers;
- logits processors and output heads;
- pooling or classification heads;
- recurrent or state-space blocks.

Common loading patterns:

- packed parameter mappings;
- stacked parameter mappings;
- expert parameter mappings;
- weight name filtering or remapping;
- automatic loaders and per-parameter weight loaders.

Treat imported runtime internals as external unless you actually read the
implementation file.
