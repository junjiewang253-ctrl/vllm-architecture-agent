# 03. 复现指南

本指南面向拿到仓库或 mentor zip 的导师。推荐在 Windows PowerShell 中执行。

## 1. 安装

在仓库根目录运行：

```powershell
.\tools\setup-mentor.ps1 -ConfigureDrawioMcp
```

脚本会：

- 检查 Python；
- 执行 `python -m pip install -e ".[dev]"`；
- 创建 `.agents/skills/vllm-model-architecture-diagram` 到 Skill 源目录的 Junction；
- 写入 Draw.io MCP 配置；
- 打印下一步的一句话指令。

## 2. 重启 VS Code Codex

安装完成后必须重启 VS Code Codex。这样 Codex 才能重新加载 Skill 和 MCP 配置。

可选检查：

```powershell
codex mcp list
```

列表中应能看到 `drawio`。

## 3. 一句话启动

在 Codex 中输入：

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，生成默认架构图。
```

默认输出目录：

```text
outputs/hy-v3/
```

## 4. 查看结果

主要查看：

- `outputs/hy-v3/architecture.drawio`
- `outputs/hy-v3/images/*.png`
- `outputs/hy-v3/report.md`
- `outputs/hy-v3/visual-review.md`

可移植示例保存在：

```text
examples/hy_v3/
```

## 5. 手工验证

运行测试：

```powershell
pytest
```

验证 HY V3 Golden Example：

```powershell
vllm-arch validate `
  --repo-root . `
  --context examples\hy_v3\source-context.json `
  --plan examples\hy_v3\architecture-plan.json `
  --evidence examples\hy_v3\evidence.json `
  --drawio examples\hy_v3\architecture.drawio `
  --images-dir examples\hy_v3\images
```

期望输出包含：

```text
Evidence validation passed
Architecture plan validation passed
Draw.io validation passed
```

## 6. 分析其他 vLLM 模型

如果有本地 vLLM checkout，可以直接给文件：

```text
使用 $vllm-model-architecture-diagram 分析 vllm/model_executor/models/qwen2.py，生成默认架构图。
```

也可以给 registry architecture 名称：

```text
使用 $vllm-model-architecture-diagram 分析 Qwen2ForCausalLM，生成默认架构图。
```

如果目标只是 helper/shared utility 文件，Skill 会降级生成少量边界或说明图，而不会强行画完整 CausalLM 架构。

## 7. 常见问题

**PowerShell 禁止运行脚本**

可以在当前 PowerShell 会话临时允许本地脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

然后重新运行 `setup-mentor.ps1`。

**Draw.io MCP 没有出现**

重新运行：

```powershell
.\tools\setup-mentor.ps1 -ConfigureDrawioMcp
```

并重启 VS Code Codex。

**validate 找不到源码文件**

请确认当前目录是仓库根目录，并使用 `--repo-root .`。正式示例中的路径都是仓库相对路径。
