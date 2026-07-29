param(
    [string]$RepoRoot = (Resolve-Path ".").Path,
    [switch]$ConfigureDrawioMcp
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Assert-Command {
    param(
        [string]$Name,
        [string]$InstallHint
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "未找到命令 '$Name'。$InstallHint"
    }
}

function Assert-MinimumVersion {
    param(
        [string]$Name,
        [version]$Current,
        [version]$Minimum,
        [string]$InstallHint
    )

    if ($Current -lt $Minimum) {
        throw "$Name 版本过低：$Current；最低要求：$Minimum。$InstallHint"
    }
}

function Get-CodexConfigPath {
    if ($env:CODEX_HOME) {
        return Join-Path $env:CODEX_HOME "config.toml"
    }
    return Join-Path $HOME ".codex\config.toml"
}

function Ensure-SkillLink {
    param(
        [string]$Repo,
        [string]$SkillSource,
        [string]$SkillLink
    )

    $linkParent = Split-Path $SkillLink -Parent
    New-Item -ItemType Directory -Force -Path $linkParent | Out-Null

    if (Test-Path $SkillLink) {
        $item = Get-Item $SkillLink
        if ($item.LinkType -eq "Junction" -or $item.LinkType -eq "SymbolicLink") {
            if ((Resolve-Path $item.Target).Path -eq (Resolve-Path $SkillSource).Path) {
                Write-Host "Skill 链接已经存在并指向当前源码。"
                return
            }
            throw "已有 Skill 链接指向其他位置：$($item.Target)"
        }
        throw "Skill 路径已存在但不是链接：$SkillLink"
    }

    New-Item -ItemType Junction -Path $SkillLink -Target $SkillSource | Out-Null
    Write-Host "已创建 Skill Junction：$SkillLink -> $SkillSource"
}

function Ensure-DrawioMcpConfig {
    param([string]$ConfigPath)

    $configDir = Split-Path $ConfigPath -Parent
    New-Item -ItemType Directory -Force -Path $configDir | Out-Null

    $drawioBlock = @'

[mcp_servers.drawio]
command = "npx"
args = ["-y", "@next-ai-drawio/mcp-server@latest"]
startup_timeout_sec = 30
tool_timeout_sec = 120
enabled = true
'@

    if (Test-Path $ConfigPath) {
        $existing = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8
        if ($existing -match '(?m)^\s*\[mcp_servers\.drawio\]\s*$') {
            Write-Host "Codex 配置中已经存在 [mcp_servers.drawio]。"
            return
        }
        Add-Content -LiteralPath $ConfigPath -Value $drawioBlock -Encoding UTF8
        Write-Host "已追加 Draw.io MCP 配置：$ConfigPath"
        return
    }

    Set-Content -LiteralPath $ConfigPath -Value $drawioBlock.TrimStart() -Encoding UTF8
    Write-Host "已创建 Codex 配置并写入 Draw.io MCP：$ConfigPath"
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

Write-Step "检查前置依赖"
if ($PSVersionTable.PSVersion -lt [version]"5.1") {
    throw "PowerShell 版本过低：$($PSVersionTable.PSVersion)。最低要求为 5.1。"
}

Assert-Command -Name "python" -InstallHint "请安装 Python 3.10+ 并加入 PATH。"
Assert-Command -Name "node" -InstallHint "请安装 Node.js 20 LTS+ 并加入 PATH。"
Assert-Command -Name "npm" -InstallHint "npm 随 Node.js 安装，请重新安装 Node.js LTS。"
Assert-Command -Name "npx" -InstallHint "npx 随 Node.js 安装，请重新安装 Node.js LTS。"

$pythonVersionText = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 Python 版本。"
}
$nodeVersionText = (& node --version).Trim().TrimStart("v")

Assert-MinimumVersion -Name "Python" -Current ([version]$pythonVersionText) `
    -Minimum ([version]"3.10") -InstallHint "请安装 Python 3.10+。"
Assert-MinimumVersion -Name "Node.js" -Current ([version]$nodeVersionText) `
    -Minimum ([version]"20.0") -InstallHint "请安装 Node.js 20 LTS+。"

python --version
node --version
npm --version
npx --version

if (Get-Command "codex" -ErrorAction SilentlyContinue) {
    codex --version
} else {
    Write-Warning "未找到 codex CLI。VS Code Codex 仍可使用；请在扩展的 MCP 面板检查 drawio。"
}

Write-Step "安装项目和测试依赖"
Push-Location $repo
try {
    python -m pip install -e ".[dev]"
} finally {
    Pop-Location
}

Write-Step "创建 VS Code Codex 本地 Skill 链接"
Ensure-SkillLink -Repo $repo -SkillSource $skillSource -SkillLink $skillLink

if ($ConfigureDrawioMcp) {
    Write-Step "配置 Draw.io MCP"
    Ensure-DrawioMcpConfig -ConfigPath (Get-CodexConfigPath)
} else {
    Write-Step "跳过 Draw.io MCP 自动配置"
    Write-Host "如需自动写入 Draw.io MCP，请重新运行："
    Write-Host "  .\tools\setup-mentor.ps1 -ConfigureDrawioMcp"
}

Write-Step "安装完成"
Write-Host "请重启 VS Code Codex。"
Write-Host "重启后在 Codex 中输入："
Write-Host "使用 `$vllm-model-architecture-diagram 分析 samples/hy_v3.py，生成默认架构图。"
Write-Host ""
Write-Host "可选检查命令："
Write-Host "  codex mcp list"
Write-Host "  node --version"
Write-Host "  npx --version"

