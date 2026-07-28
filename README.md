# vLLM Architecture Agent

> 一个面向 VS Code Codex 的轻量 Agent Skill：自动分析
> vLLM 模型适配器 Python 文件，并通过 Draw.io MCP 生成
> 有源码证据、可编辑、少页高密度的完整架构图。

Codex 负责阅读源码、理解模型、组织页面和设计 Draw.io 图；脚本负责完整索引、证据校验和输出验证。本项目不依赖 HY V3 固定模板，默认分析目标是 `vllm/model_executor/models/*.py` 下的模型适配器文件。

项目面向该目录设计，并通过通用源码索引、Agent 自主分析和降级机制支持不同模型类型；真实全目录兼容率需要在指定 vLLM commit 上扫描验证。

## 导师最快使用

### 第一步：运行安装脚本

在仓库根目录打开 PowerShell：

```powershell
.\tools\setup-mentor.ps1 -ConfigureDrawioMcp
```

这个脚本会检查 Python、执行 `pip install -e ".[dev]"`、创建本地 Codex Skill 链接，并在传入 `-ConfigureDrawioMcp` 时写入 Draw.io MCP 配置。

### 第二步：重启 VS Code Codex

安装或修改 MCP 配置后，请重启 VS Code Codex，让 Skill 和 Draw.io MCP 生效。

可选检查：

```powershell
codex mcp list
```

### 第三步：输入一句话

在 Codex 中输入：

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，生成默认架构图。
```

默认输出保存到：

```text
outputs/hy-v3/
  source-context.json
  architecture-plan.json
  evidence.json
  architecture.drawio
  report.md
  visual-review.md
  images/
    model-architecture-and-execution.png
    decoder-and-attention.png
    moe-architecture-and-routing.png
    parallelism-configuration-and-weight-loading.png
```

## 默认会做什么

默认模式是 `complete`，目标是生成 3 到 5 张少页、高密度、包含关键实现细节的架构图。HY V3 示例默认生成 4 张复合页面：

1. `Model Architecture and Execution`
2. `Decoder and Attention`
3. `MoE Architecture and Routing`
4. `Parallelism, Configuration and Weight Loading`

完整工作流包括：

1. 自动推断仓库根目录、模型名和输出目录。
2. 运行 `vllm-arch prepare` 生成 `source-context.json` 和空的 Plan/Evidence checklist。
3. Codex 完整阅读目标 Python 文件，审阅所有 Class、Method、模块级 Function、重要 Branch、Mapping Group 和 Capability。
4. Codex 按需阅读相关源码，遇到外部稳定组件时标记 external boundary。
5. Codex 编写 `architecture-plan.json` 和 `evidence.json`。
6. 脚本验证 Plan、Evidence 和源码路径。
7. Codex 使用 Draw.io MCP 绘制多页架构图。
8. 导出 PNG，进行至少一轮视觉复查和修改。
9. 运行最终验证，生成 `report.md`。

## 每个输出文件有什么用

- `source-context.json`：脚本生成的源码索引，包含 Class、Method、Branch、Mapping Group、Capability 和源码覆盖统计。
- `architecture-plan.json`：Codex 编写的架构计划，说明页面、工程问题、detail region、审阅状态和遗漏原因。
- `evidence.json`：Codex 编写的证据文件，记录每条主要架构主张对应的源码位置，并区分 `direct`、`derived`、`external`。
- `architecture.drawio`：可编辑 Draw.io 源文件。
- `report.md`：最终架构分析报告，包含页面说明、覆盖统计、验证结果和限制。
- `visual-review.md`：PNG 导出后的视觉复查记录，说明修改轮次和剩余风险。
- `images/*.png`：导师最容易直接查看的导出图片。

仓库内可移植 Golden Example 位于 `examples/hy_v3/`，同样包含这些文件和 4 张 PNG。

## 分析其他模型

在 Codex 中可以直接换目标文件：

```text
使用 $vllm-model-architecture-diagram 分析 vllm/model_executor/models/qwen2.py，生成默认架构图。
```

也可以使用 registry architecture 名称：

```text
使用 $vllm-model-architecture-diagram 分析 Qwen2ForCausalLM，生成默认架构图。
```

如果使用 architecture 名称，Skill 会通过 `vllm/model_executor/models/registry.py` 静态解析到模型文件。没有 MoE 证据时不会强行生成 MoE 页；helper/shared utility 文件会优雅降级为 1 到 2 张边界或说明图。

## 高级复现命令

导师通常不需要手工执行 CLI。需要复现或调试时，可使用：

```powershell
vllm-arch list-models --repo-root <vllm-root>

vllm-arch prepare `
  --input samples\hy_v3.py `
  --outputs-dir outputs\hy-v3 `
  --model-name hy-v3

vllm-arch validate `
  --repo-root . `
  --context examples\hy_v3\source-context.json `
  --plan examples\hy_v3\architecture-plan.json `
  --evidence examples\hy_v3\evidence.json `
  --drawio examples\hy_v3\architecture.drawio `
  --images-dir examples\hy_v3\images

vllm-arch scan `
  --repo-root <vllm-root> `
  --output outputs\compatibility-report.json
```

默认 CLI 只有四个子命令：`list-models`、`prepare`、`validate`、`scan`。

## 验证当前仓库

```powershell
pytest

vllm-arch validate `
  --repo-root . `
  --context examples\hy_v3\source-context.json `
  --plan examples\hy_v3\architecture-plan.json `
  --evidence examples\hy_v3\evidence.json `
  --drawio examples\hy_v3\architecture.drawio `
  --images-dir examples\hy_v3\images
```

当前测试覆盖路径可移植性、HY V3 示例可迁移验证、Method 重要性分类、Mapping Group 聚合、Capability 精度和 Draw.io 输出结构验证。

## 常见故障排查

**Codex 找不到 Skill**

重新运行：

```powershell
.\tools\setup-mentor.ps1
```

确认 `.agents/skills/vllm-model-architecture-diagram` 指向 `src/skills/vllm-model-architecture-diagram`，然后重启 VS Code Codex。

**Draw.io MCP 不可用**

运行：

```powershell
codex mcp list
```

如果没有 `drawio`，重新执行：

```powershell
.\tools\setup-mentor.ps1 -ConfigureDrawioMcp
```

然后重启 VS Code Codex。

**验证提示 Evidence 路径不存在**

确认在仓库根目录运行命令，并传入 `--repo-root .`。正式示例中的路径均为仓库相对路径，可以复制到其他目录后继续验证。

**没有真实 vLLM checkout**

可以先使用 `samples/hy_v3.py` 和 `examples/hy_v3/` 体验完整流程。真实全目录兼容性需要本地有 vLLM 仓库，并通过 `vllm-arch scan --repo-root <vllm-root>` 验证。

## Legacy

旧的 compiler-style 研究流水线保存在 `legacy/compiler-pipeline-v1/`。它只用于历史参考，不参与默认 Skill、CLI、测试或导师交付包。
