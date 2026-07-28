param(
    [string]$RepoRoot = (Resolve-Path ".").Path,
    [switch]$ConfigureDrawioMcp
)

$ErrorActionPreference = "Stop"
$installer = Join-Path $PSScriptRoot "setup-mentor.ps1"

if (-not (Test-Path -LiteralPath $installer)) {
    throw "安装脚本不存在：$installer"
}

if ($ConfigureDrawioMcp) {
    & $installer -RepoRoot $RepoRoot -ConfigureDrawioMcp
} else {
    & $installer -RepoRoot $RepoRoot
}

