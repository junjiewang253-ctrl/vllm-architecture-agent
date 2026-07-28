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
    throw "pyproject.toml not found. Run this script from the repository root or pass -RepoRoot."
}
if (-not (Test-Path (Join-Path $skillSource "SKILL.md"))) {
    throw "Canonical Skill source not found: $skillSource"
}

Write-Step "Checking Python"
python --version

Write-Step "Installing package in editable dev mode"
python -m pip install -e ".[dev]"

Write-Step "Creating local Codex Skill link"
$linkParent = Split-Path $skillLink -Parent
New-Item -ItemType Directory -Force -Path $linkParent | Out-Null

if (Test-Path $skillLink) {
    $item = Get-Item $skillLink
    if ($item.LinkType -eq "Junction" -or $item.LinkType -eq "SymbolicLink") {
        if ((Resolve-Path $item.Target).Path -eq (Resolve-Path $skillSource).Path) {
            Write-Host "Skill link already points to canonical source."
        } else {
            throw "Existing skill link points elsewhere: $($item.Target)"
        }
    } else {
        throw "Skill path exists and is not a link: $skillLink"
    }
} else {
    New-Item -ItemType Junction -Path $skillLink -Target $skillSource | Out-Null
    Write-Host "Created junction: $skillLink -> $skillSource"
}

Write-Step "Draw.io MCP check"
Write-Host "Run this command and confirm a drawio MCP server is configured:"
Write-Host "  codex mcp list"
Write-Host ""
Write-Host "This setup script does not modify global Codex MCP configuration."

Write-Step "Next step"
Write-Host "使用 `$vllm-model-architecture-diagram 分析 samples/hy_v3.py，"
Write-Host "生成默认架构图。"
