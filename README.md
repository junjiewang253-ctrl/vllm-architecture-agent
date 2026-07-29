# vLLM Architecture Agent

一个面向 VS Code Codex 的轻量 Agent Skill。它读取 vLLM 模型适配器源码，
由 Codex 理解执行逻辑并通过 Draw.io MCP 生成有源码证据、可编辑、可放大的
一体化架构图。

脚本只负责完整索引、Evidence 和结果验证；Codex 负责架构理解、信息组织和
视觉设计。生产代码不依赖 HY V3、Qwen 或其他模型的固定模板。

## 快速开始

在仓库根目录打开 PowerShell：

```powershell
.\tools\setup.ps1 -ConfigureDrawioMcp
```

安装完成后重启 VS Code Codex，然后输入：

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，生成默认架构图。
```

结果写入：

```text
outputs/hy-v3/
├── source-context.json
├── architecture-plan.json
├── evidence.json
├── architecture.drawio
├── report.md
├── visual-review.md
└── images/
```

`architecture.drawio` 可继续编辑，`images/` 用于直接查看，`report.md`
说明架构结论、证据边界和验证结果。

## 默认成品标准

默认不是四张独立详情页，也不是几张概念卡片，而是一张主动放大纸张尺寸的
连续工程画布：

- 从输入到输出有一条可追踪的 runtime spine；
- PP 等互斥入口先分支、再在本地执行前汇合；
- Decoder 或 repeated block 在主流程中原位展开；
- Attention、MoE、SSM 或多模态子系统嵌套在真实执行位置；
- residual、cache 和 optional path 使用独立支路；
- external vLLM 组件位于真实 delegation point；
- checkpoint loading 位于下方映射平面，并连到实际接收权重的组件；
- TP、PP、EP、LoRA、quantization 等作为策略、配置或 badge，不伪装成
  runtime tensor flow。

完整模型默认使用一张至少 `4200 x 2000` 的横向 Draw.io 页面，常用尺寸约为
`5200 x 2500`。只有经过两轮真实视觉调整后仍无法阅读，才允许说明理由拆页。

## 一体化效果参考

下面两张图是当前 Skill 的视觉和信息组织 baseline。点击可查看原图；可编辑
Draw.io 和 SVG 位于同一目录。

| HY V3 Integrated Flow | Qwen3 MoE Integrated Flow |
| --- | --- |
| [![HY V3 Integrated Flow](examples/integrated-flow/hy_v3/architecture.png)](examples/integrated-flow/hy_v3/architecture.png) | [![Qwen3 MoE Integrated Flow](examples/integrated-flow/qwen3_moe/architecture.png)](examples/integrated-flow/qwen3_moe/architecture.png) |

这两张图用于约束通用拓扑和质量，不是模型答案模板。分析其他模型时，Codex
仍必须依据对应源码自主选择模块、分支、边界和加载映射。

旧版四页 HY V3 交付示例仍保留用于比较：

| Model Architecture and Execution | Decoder and Attention |
| --- | --- |
| [![Model Architecture and Execution](examples/hy_v3/images/model-architecture-and-execution.png)](examples/hy_v3/images/model-architecture-and-execution.png) | [![Decoder and Attention](examples/hy_v3/images/decoder-and-attention.png)](examples/hy_v3/images/decoder-and-attention.png) |
| **MoE Architecture and Routing** | **Parallelism, Configuration and Weight Loading** |
| [![MoE Architecture and Routing](examples/hy_v3/images/moe-architecture-and-routing.png)](examples/hy_v3/images/moe-architecture-and-routing.png) | [![Parallelism, Configuration and Weight Loading](examples/hy_v3/images/parallelism-configuration-and-weight-loading.png)](examples/hy_v3/images/parallelism-configuration-and-weight-loading.png) |

## 它如何约束 Codex

`vllm-arch prepare` 会完整索引目标文件中的 Class、Method、Function、重要
Branch、Weight Mapping Group 和 Capability。Codex 必须逐项审阅，不进入图的
内容也要写明原因。

Architecture Plan 2.2 还会定义轻量 `visual_contract`：

- `visual:<id>`：必须出现在 Draw.io 中的架构锚点；
- `visual-rel:<id>`：必须出现的语义关系；
- detail region 挂载到哪一个主流程锚点；
- runtime、residual、loading、construction、metadata、external 的线型；
- 最小画布、字体、节点、边和导出分辨率；
- 核心 Class/Method、重要 Branch 和 Mapping Group 的最低可见细节比例；
- structural detail region 至少包含 4 个语义 anchor 和 3 条内部关系，
  且平均每个可见 anchor 最多承载约 8 个源码项；
- branch/merge 必须真的分叉和汇合；
- 标题、工程问题和 detail-region 标题必须具有清楚的空间层级；
- 画布轴向覆盖、网格占用、区域密度、loading 间距和长线限制。

每个对应 `mxCell` 都必须写入 `dataAnchor`。Validator 会核对锚点唯一性、
关系端点、主线连通、嵌套容器、外部边界、线型、画布尺寸和真实 PNG。
这比“节点数量大于某个值”更能阻止简陋卡片图，同时没有恢复固定页面、
坐标或确定性 Renderer。

完整模型不能把所有 Class 和 Method 都标成 `rendered_aggregate`。第一稿必须
保存 Draw.io XML；之后至少修改 15% 的 required anchor 几何并重路由两条
required relationship。只增加说明文字或移动一个小节点不算完成视觉复查。

## 完整工作流

一次默认运行会自动完成：

1. 推断工作区与输出目录；
2. 静态解析目标文件并生成 Source Context；
3. 完整阅读目标源码和必要相关文件；
4. 编写 Evidence 与 Architecture Plan 2.2；
5. 绘图前验证证据和审阅完整性；
6. 调用 Draw.io MCP 创建大尺寸一体化画布；
7. 导出真实 PNG；
8. 打开 PNG 做至少一轮视觉复查并通过 MCP 修改；
9. 验证 Draw.io、视觉锚点和导出文件；
10. 生成最终报告。

Draw.io MCP 不可用时，Skill 必须停止并说明阻塞，不会用脚本伪造
`.drawio` 或占位 PNG。

## 分析其他模型

直接指定文件：

```text
使用 $vllm-model-architecture-diagram 分析 vllm/model_executor/models/qwen2.py，生成默认架构图。
```

也可以指定 Registry architecture：

```text
使用 $vllm-model-architecture-diagram 分析 Qwen2ForCausalLM，生成默认架构图。
```

项目面向 `vllm/model_executor/models/*.py` 设计，但不会承诺对所有当前和未来
版本自动达到相同质量。不同模型会形成不同的 inline expansion：

- Dense Decoder：展开 repeated block、attention 和 dense FFN；
- MoE：加入 router、expert container 和 EP metadata；
- Multimodal：展开不同模态输入、processor/placeholder、代表性 encoder block、
  feature merge/injection 与本地语言模型编排；
- Embedding/Pooling：不强行生成 CausalLM logits flow；
- Hybrid/SSM：展开状态更新或 recurrent path；
- Helper 文件：降级为紧凑 boundary/component 图。

## 手动验证

运行全部测试：

```powershell
pytest
```

验证某次输出：

```powershell
vllm-arch validate `
  --repo-root . `
  --context outputs\<model>\source-context.json `
  --plan outputs\<model>\architecture-plan.json `
  --evidence outputs\<model>\evidence.json `
  --drawio outputs\<model>\architecture.drawio `
  --images-dir outputs\<model>\images `
  --visual-review outputs\<model>\visual-review.md
```

默认 CLI 仍只有四个辅助命令：

```text
vllm-arch list-models
vllm-arch prepare
vllm-arch validate
vllm-arch scan
```

日常使用 Skill 时不需要手工填写这些参数。

## 常见问题

**Codex 找不到 Skill**

重新运行 `.\tools\setup.ps1 -ConfigureDrawioMcp`，然后重启 VS Code Codex。
确认 `.agents/skills/vllm-model-architecture-diagram` 指向
`src/skills/vllm-model-architecture-diagram`。

**Draw.io MCP 不可用**

运行 `codex mcp list`。应能看到 `drawio`。若没有，重新运行安装脚本并重启
VS Code Codex。

**图仍然很简陋**

确认生成的 Plan 为 `schema_version: 2.2`，并使用
`diagram_strategy: integrated_single_canvas`。旧 Plan 2.1 仍可验证，但只用于
兼容已有交付，不会启用新的视觉锚点契约。正式输出应使用具备充分源码推理和
视觉规划能力的 Codex 模型；Validator 能阻止明显退化，但不能替代架构判断。

**没有真实 vLLM checkout**

可先使用 `samples/hy_v3.py` 和 `samples/qwen3_moe.py`。需要验证某个 vLLM
commit 的目录兼容性时，准备本地 checkout 后运行：

```powershell
vllm-arch scan --repo-root <vllm-root> --output outputs\compatibility-report.json
```

历史 compiler-style 流水线保存在 `legacy/compiler-pipeline-v1/`，不参与默认
运行。当前默认实现保持 Agent-native：脚本检查事实和成品契约，Codex 决定图。
