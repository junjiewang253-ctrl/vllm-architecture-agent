# vLLM Patterns

Use these as recognition aids. They are not final diagram templates.

## Registry

Model architecture names are statically mapped in model registry dictionaries.
Do not import vLLM to resolve registry entries.

## Capabilities

- `SupportsPP`: pipeline-parallel support.
- `SupportsLoRA`: local LoRA capability marker.
- `SupportsMultiModal`: multimodal inputs.
- `MixtureOfExperts`: MoE integration.
- `support_torch_compile`: compile compatibility.

## Components

- embeddings and pooling heads;
- QKV, column, merged-column and row parallel projections;
- attention backend wrappers;
- normalization layers;
- dense FFN blocks;
- fused MoE blocks and routers;
- logits processors and output heads;
- multimodal encoders and fusion logic;
- recurrent or state-space blocks.

## Loading

Look for:

- packed module mappings;
- stacked parameter mappings;
- expert parameter mappings;
- name remap;
- skip/filter branches;
- `AutoWeightsLoader`;
- per-parameter `weight_loader`;
- `default_weight_loader`.

Treat imported runtime internals as external unless you actually read their
implementation files.
