$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

$SourceSkill = Join-Path `
    $RepoRoot `
    "src\skills\vllm-model-architecture-diagram"

$CodexSkillsRoot = Join-Path `
    $RepoRoot `
    ".agents\skills"

$CodexSkillLink = Join-Path `
    $CodexSkillsRoot `
    "vllm-model-architecture-diagram"

if (-not (Test-Path $SourceSkill)) {
    throw "Core skill directory does not exist: $SourceSkill"
}

New-Item `
    -ItemType Directory `
    -Force `
    -Path $CodexSkillsRoot | Out-Null

if (Test-Path $CodexSkillLink) {
    Write-Host "Codex skill link already exits:"
    Write-Host "  $CodexSkillLink"
    exit 0
}

New-Item `
    -ItemType Junction `
    -Path $CodexSkillLink `
    -Target $SourceSkill | Out-Null

Write-Host "Codex development skill linked successfully:"
Write-Host "  $CodexSkillLink"
Write-Host "  -> $SourceSkill"