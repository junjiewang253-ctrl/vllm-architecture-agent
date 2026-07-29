# 01. 一体化架构图讲解

v2.2 的默认目标是一张连续、可缩放的大画布。HY V3 与 Qwen3 MoE baseline
位于 `examples/integrated-flow/`。它们不是模型模板，而是信息组织和视觉质量
标准。

## 如何阅读整张图

上半部分是一条从输入到输出的 runtime spine。重复 Decoder Layer 在它的执行
位置原位展开，Attention、FFN、MoE、residual 和 cache 等细节嵌套其中。

下半部分是 checkpoint/loading plane。过滤、重命名和 dispatch 使用紫色，
映射线向上连接到真正接收权重的 QKV、Dense FFN、FusedMoE、LM Head 或其他
组件。

侧边或顶部的 TP、PP、EP、quantization 和 capability 使用 badge、parameter
panel 或虚线 dependency，不会伪装成 runtime tensor flow。

## 输入和 Pipeline Parallel

First PP rank 的 token/embedding 输入与 non-first rank 的
`IntermediateTensors` 是互斥入口。两条路径在 local decoder slice 前汇合，
而不是按顺序执行。

Decoder Stack 只表示本 rank 的 local layer range。Non-last rank 返回
`IntermediateTensors`，last rank 执行 final residual add 和 RMSNorm。

## 展开的 Decoder

代表性 Decoder Layer 保留主 hidden-state 路径和 residual lane。两次 fused
RMSNorm handoff 使用不同颜色，避免把 residual 只写成注释。

Dense 与 MoE 变体属于 construction-time 选择；图中用 variant/container 表达，
不会画成每个 token 都同时经过两条路径。

## Attention 边界

本地源码证明 projection、split、optional norm、RoPE 调用和 output projection。
导入的 vLLM Attention 使用暖色虚线 external boundary。

HY V3 baseline 还区分 configured HPC path 与 fallback Q/K/V path；Qwen3 baseline
则依据自己的源码表达 Q/K Norm、RoPE 和 V bypass。两个模型不会共享一个固定
Attention 模板。

KV Cache 使用独立 storage 形状。本地能证明的 write 与 external backend 的
read/scheduling 分开标注。

## MoE 组织

Runtime route 显示 hidden states、router logits 和 FusedMoE 的实际输入关系。
Experts 是 FusedMoE 容器内部组成，不是它之后的串行阶段。

EP/EPLB、expert placement 和 routing configuration 使用 metadata/config
dependency；它们不会作为 tensor input。

## 输出边界

模型 `forward` 返回的 hidden states 与 wrapper 的 `compute_logits` 保持独立。
只有当源码定义对应入口时，图才继续连接 `ParallelLMHead` 和
`LogitsProcessor`。外部 logits processing 内部逻辑不伪装成本地 direct。

## Evidence 含义

`evidence.json` 中：

- `direct`：本地源码直接证明；
- `derived`：多个本地事实组合推导，并记录 derivation；
- `external`：本地源码证明调用或边界，但没有假装已分析外部实现。

Plan 2.2 的每个 required visual anchor 和 relationship 都引用这些 Claim，并在
Draw.io `mxCell` 中使用相同 `dataAnchor`，使“证据里写了”和“图里真的画了”
可以被 Validator 对照检查。
