# 01. HY V3 架构图讲解

HY V3 示例最终生成 4 张复合架构图。每张图都回答一个工程问题，并把源码证据分成 `direct`、`derived` 和 `external`。

## 1. Model Architecture and Execution

这页回答：`HYV3ForCausalLM`、`HYV3Model`、pipeline rank 和 logits 输出路径如何组合在一起？

主要区域：

- `Class and Component Composition`：展示 `HYV3ForCausalLM` 包含 `HYV3Model`，base model 内包含 embedding、local decoder layers、attention、Dense FFN / MoE 和 final RMSNorm；wrapper 侧包含 `ParallelLMHead` 与 `LogitsProcessor`。
- `End-to-End Runtime`：展示输入如何进入 local decoder slice，再根据 PP rank 决定返回 `IntermediateTensors` 或执行 final residual add + norm。
- `Output Boundary`：展示 hidden states 与 `ParallelLMHead` 进入 `LogitsProcessor` 并产生 logits。

重要修正：

- First PP rank input 和 non-first PP rank `IntermediateTensors` 是互斥入口，不是串行步骤。
- Embedding 与 LM Head 的 tied weight 是 optional dependency，不是 runtime tensor flow。
- `LogitsProcessor` 内部属于 external boundary。

## 2. Decoder and Attention

这页回答：一个 decoder block 如何运行，HY V3 attention 如何适配 vLLM Attention backend？

主要区域：

- `Decoder Runtime`：hidden states 和 residual 两条通道并行穿过 input fused RMSNorm、self attention、post-attention fused RMSNorm 和 FFN。
- `FFN Construction Variant`：前 K 层使用 Dense FFN，后续层使用 MoE。这是 construction-time variant，不是一次 forward 中同时执行。
- `Attention Construction Panel`：展示 TP head partition、QKV projection、output projection、optional Q/K RMSNorm、RoPE、HpcRopeNorm 和 quantization。
- `Attention Forward Detail`：展示 QKV projection 后的 HPC configured path 与 fallback Q/K/V path。

重要修正：

- HPC path 输入是完整 `qkv`，不是 Q stream。
- V bypass Q/K RMSNorm 和 Q/K RoPE 处理路径。
- KV cache write 有本地调用证据；KV cache read 是 vLLM Attention backend 内部行为，标为 external。
- Attention 输出经过 `RowParallelLinear`。

## 3. MoE Architecture and Routing

这页回答：`HYV3MoEFused` 如何路由 token、配置 experts，并暴露 EP metadata？

主要区域：

- `Construction and Configuration`：用 parameter panel 表示 `num_experts`、`top_k`、scoring、grouped top-k、scaling、shared expert 和 expert bias。
- `Runtime Routing`：hidden states flatten 后进入 GateLinear 生成 router logits；hidden states 与 router logits 一起进入 FusedMoE；输出恢复原 shape。
- `FusedMoE Composition`：FusedMoE 作为 external runtime boundary，内部包含 routed experts 和 optional shared experts。
- `Expert Parallel Metadata`：展示 EP group/rank/size、physical/local experts、redundant experts 和 EPLB。

重要修正：

- GateLinear 只作为一个 runtime component 出现，construction 区只记录配置依赖。
- Experts 是 FusedMoE 内部组成，不是 FusedMoE 之后的串行阶段。
- EP metadata 通过虚线影响 FusedMoE/expert placement，不画成 runtime tensor input。

## 4. Parallelism, Configuration and Weight Loading

这页回答：TP、PP、EP、配置能力和两套 weight loading 路径如何组织？

主要区域：

- `Tensor Parallel Strategy`：展示 Embedding、QKV、MergedColumn、Row、LM Head，以及 head partition。
- `Pipeline Parallel Strategy`：展示 `make_layers`、local layer range、`PPMissingLayer`、first/non-first input、last/non-last output 和 PP missing filter。
- `Expert Parallel Strategy`：展示 EP group、rank/size、physical/local experts、FusedMoE placement、redundant experts 和 EPLB。
- `Configuration and Capabilities`：展示 `quant_config`、SupportsLoRA 和 `torch.compile`，不把它们画成 runtime flow。
- `Weight Loading`：分成 wrapper lane 和 model lane。

重要修正：

- TP、PP、EP 是独立策略面板，不互相串联。
- `HYV3ForCausalLM.load_weights` 通过 `AutoWeightsLoader` external boundary 处理。
- `HYV3Model.load_weights` 是 dispatch tree：stacked mapping、expert mapping、regular path。
- LoRA capability 不再连接到 checkpoint loading。

## 如何阅读 evidence

`evidence.json` 中：

- `direct` 表示本地源码直接证明；
- `derived` 表示多个本地事实组合推导；
- `external` 表示本地源码只能证明边界调用或导入，外部组件内部行为没有在本文件中展开。
