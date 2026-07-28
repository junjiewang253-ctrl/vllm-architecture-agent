param(
    [string]$RepoRoot = (Resolve-Path ".").Path
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

$repo = Resolve-Path $RepoRoot
$skillSource = Join-Path $repo "src\skills\vllm-model-architecture-diagram"
$skillLink = Join-Path $repo ".agents\skills\vllm-model-architecture-diagram"

if (-not (Test-Path (Join-Path $repo "pyproject.toml"))) {
    throw "未找到 pyproject.toml。请在仓库根目录运行，或传入 -RepoRoot。"
}
if (-not (Test-Path (Join-Path $skillSource "SKILL.md"))) {
    throw "未找到 Skill 源目录：$skillSource"
}

Write-Step "检查 Python"
python --version

Write-Step "安装 editable 开发包"
python -m pip install -e ".[dev]"

Write-Step "创建本地 Codex Skill 链接"
$linkParent = Split-Path $skillLink -Parent
New-Item -ItemType Directory -Force -Path $linkParent | Out-Null

if (Test-Path $skillLink) {
    $item = Get-Item $skillLink
    if ($item.LinkType -eq "Junction" -or $item.LinkType -eq "SymbolicLink") {
        if ((Resolve-Path $item.Target).Path -eq (Resolve-Path $skillSource).Path) {
            Write-Host "Skill 链接已经指向当前源码。"
        } else {
            throw "已有 Skill 链接指向其他位置：$($item.Target)"
        }
    } else {
        throw "Skill 路径已存在但不是链接：$skillLink"
    }
} else {
    New-Item -ItemType Junction -Path $skillLink -Target $skillSource | Out-Null
    Write-Host "已创建 Junction：$skillLink -> $skillSource"
}

Write-Step "Draw.io MCP 检查"
Write-Host "请运行以下命令确认 drawio MCP server 已配置："
Write-Host "  codex mcp list"
Write-Host "本开发脚本不会自动修改全局 Codex MCP 配置。"

Write-Step "下一步"
Write-Host "请重启 VS Code Codex，然后输入："
Write-Host "使用 `$vllm-model-architecture-diagram 分析 samples/hy_v3.py，生成默认架构图。"

