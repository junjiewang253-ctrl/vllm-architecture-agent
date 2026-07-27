# Architecture Walkthrough

## Model Overview

The overview page answers how the HY V3 adapter produces logits at the top
level. It shows input IDs or embeddings, token embedding, repeated decoder
layers, final residual add, final RMSNorm, hidden states, the LM head dependency
and logits processor output. Pipeline-parallel side notes describe first-rank,
non-first-rank, non-last-rank and last-rank behavior without pretending every
rank executes the same visible path.

## Decoder Layer Detail

The decoder page separates hidden-state and residual lanes. RMSNorm nodes expose
ports for both channels, replacing the older derived Residual Handoff circles.
Dense FFN and MoE are construction variants, not runtime branches.

## Attention Detail

The attention page separates Q, K, V and KV Cache. HPC fused processing starts
from the QKV projection output, while fallback Q and K paths optionally pass
through Q/K RMSNorm before RoPE. V bypasses Q/K normalization. KV Cache read is
shown as an external boundary because the local file invokes vLLM Attention but
does not implement its internal read behavior.

## MoE Detail

The MoE page shows flattening, gate linear, router logits, FusedMoE, restore
shape and output. Routed experts, optional shared experts, expert bias and EPLB
metadata are represented as source-grounded construction, dependency or
parallel-partition concepts.

## vLLM Adapter Integration

The integration page explains how HY V3 classes, vLLM configuration objects,
interfaces and execution components fit together. Capability concepts are not a
fake inheritance chain. Execution components are grouped by input, attention,
FFN/MoE and output.

## Parallelism

The parallelism page keeps tensor, pipeline and expert parallel facts in
separate lanes. It shows concrete local facts such as head partitioning,
make_layers, start/end layer range, PPMissingLayer, first/non-first/non-last/last
rank behavior, get_ep_group and expert metadata.

## Weight Loading

The weight loading page separates `HYV3ForCausalLM.load_weights` from
`HYV3Model.load_weights`. The wrapper path uses `_filter_weights` and
`AutoWeightsLoader`; the model path shows FP8 scale remapping before mapping
dispatch, stacked mapping, expert mapping, regular parameter loading, PP missing
filtering and default loader fallback.
