# vLLM source patterns

Use this file to interpret facts extracted from vLLM model adapter files. The
extractor reports syntax-level evidence; these rules add vLLM-specific meaning.

## VocabParallelEmbedding

- Category: embedding.
- Meaning: vocabulary embedding partitioned for tensor parallel execution.
- Diagram label: `Token Embedding` with a `TP` badge.
- Do not draw it as a communication operation.

## ParallelLMHead

- Category: language-model output head.
- Meaning: vocabulary projection compatible with tensor parallel execution.
- Diagram label: `LM Head` with a `TP` badge.

## QKVParallelLinear

- Category: attention projection.
- Meaning: fused Q, K and V projection supporting tensor parallel execution.
- Diagram label: `QKV Projection` with a `TP` badge.
- Do not invent three separately instantiated Python modules.

## MergedColumnParallelLinear

- Category: merged feed-forward projection.
- Common use: fused gate and up projections.
- Diagram label should preserve the source attribute name, such as
  `gate_up_proj`.

## RowParallelLinear

- Category: tensor-parallel output projection.
- Diagram label: use the source role, such as `Attention Output Projection` or
  `Down Projection`, with a `TP` badge.

## RMSNorm

- Category: normalization.
- Draw as a normal forward-flow node when it is called in `forward`.

## Attention

- Category: paged attention / KV-cache operation in vLLM.
- Draw it only when instantiated or called by the source file.
- KV cache may be annotated, but do not invent cache shape or placement.

## FusedMoE

- Category: mixture-of-experts execution.
- Typical semantic flow: router logits → top-k routing → routed experts →
  combined output.
- Exact expert count and top-k value must remain symbolic when external.

## make_layers

- Category: repeated-layer construction helper.
- It may justify a symbolic `N × Decoder Layer` representation.
- Do not infer a numeric layer count without the relevant config value.

## PPMissingLayer and get_pp_group

- Category: pipeline-parallel partitioning.
- Represent as a `PP` annotation or partition boundary, not a runtime tensor
  processing node.

## get_ep_group

- Category: expert-parallel execution.
- Represent with an `EP` annotation near MoE experts.
