# Agent 集成说明

当前唯一的 Skill 源目录是：

`src/skills/vllm-model-architecture-diagram/`

不要把 Skill 源码复制到本目录。本目录只保存不同 Agent 宿主所需的配置示例、MCP 配置示例和集成说明。

## 前置依赖

- Python 3.10+；
- Node.js 20 LTS+，并确保 `npm`、`npx` 在 PATH 中；
- VS Code Codex，且已完成登录；
- 首次启动 MCP 时能够访问 npm 和默认 Draw.io Web UI。

Draw.io Desktop、PyTorch、CUDA 和完整 vLLM 安装都不是运行示例的前置条件。

## 推荐安装方式

在仓库根目录运行：

```powershell
.\tools\setup.ps1 -ConfigureDrawioMcp
```

脚本会：

- 检查 Python、Node.js、npm 和 npx；
- 执行 `pip install -e ".[dev]"`；
- 创建 `.agents/skills/vllm-model-architecture-diagram` 到 canonical Skill 的 Windows Junction；
- 在指定 `-ConfigureDrawioMcp` 时，将 Draw.io MCP 配置写入 Codex 配置文件；
- 打印重启 VS Code Codex 后的一句话启动指令。

## 开发者安装方式

如果只想安装包和 Skill 链接，不想自动修改全局 Codex MCP 配置，可以运行：

```powershell
.\tools\setup-codex-dev.ps1
```

随后手工检查：

```powershell
codex mcp list
```

如果 Draw.io MCP 还没有配置，可参考 `integrations/codex/config.toml.example`。

Codex 配置默认位于 `$HOME/.codex/config.toml`；设置了 `CODEX_HOME` 时位于
`$CODEX_HOME/config.toml`。修改后必须完整重启 VS Code Codex。

Claude Code 插件打包是后续工作，不属于当前 v2.2 默认交付范围。
