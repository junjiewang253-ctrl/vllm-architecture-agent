# Architecture Walkthrough

## 1. Model Architecture and Execution

Question: how do the wrapper, base body, pipeline ranks and logits path fit together in vLLM?

The page combines a component tree, runtime path and output boundary. It shows `HYV3ForCausalLM` containing `HYV3Model`, decoder layers, embedding, final norm, `ParallelLMHead` and `LogitsProcessor`. The PP entry modes are mutually exclusive: first rank prepares embeddings, non-first rank receives `IntermediateTensors`, and both merge before the local decoder slice.

Direct evidence comes from constructors, `HYV3Model.forward`, `HYV3ForCausalLM.forward` and `compute_logits`. `LogitsProcessor` internals are treated as external.

## 2. Decoder and Attention

Question: how does one decoder block run, and how is HY V3 attention adapted to vLLM attention?

The page shows hidden-state flow and a residual bus through the two fused RMSNorm handoffs. Dense FFN and MoE are construction variants, not simultaneous runtime branches. Attention construction records TP head partitioning, QKV and output projections, optional Q/K norm, RoPE, HpcRopeNorm and quantization.

The attention forward region distinguishes the configured HPC path from fallback Q/K/V processing. HPC receives the whole `qkv` tensor. V bypasses Q/K norm. KV cache write is local evidence; KV cache read is external backend behavior.

## 3. MoE Architecture and Routing

Question: how does `HYV3MoEFused` route tokens, configure experts and expose EP metadata?

The runtime path is hidden states to flattening, GateLinear, router logits, FusedMoE and restore shape. FusedMoE is shown as an external runtime boundary containing routed experts and optional shared experts, not as a serial stage after experts.

Construction parameters, expert bias, shared experts and EP metadata are shown as configuration or parallel metadata. EP metadata points to expert placement and FusedMoE boundary, not runtime tensor input.

## 4. Parallelism, Configuration and Weight Loading

Question: how are TP, PP, EP, configuration capabilities and two weight-loading paths organized?

TP, PP and EP are independent strategy panels. The configuration panel marks `quant_config`, `SupportsLoRA` and `torch.compile` without turning them into runtime tensor flow.

Weight loading has two lanes. `HYV3ForCausalLM.load_weights` delegates to `AutoWeightsLoader` through an external boundary. `HYV3Model.load_weights` uses FP8 name remap, dispatches stacked, expert and regular parameter paths, and applies shared skip/filter policies. The wrapper and model loaders are not drawn as a local serial call chain.

