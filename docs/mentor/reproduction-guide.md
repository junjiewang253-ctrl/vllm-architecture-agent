# Reproduction Guide

Run from the repository root in Windows PowerShell.

## Setup

```powershell
python -m pip install -e ".[dev]"
pytest
```

## Reviewed Mode

In the default VSCode Codex workflow, the active Codex session writes the
Semantic Review, IR Patch, Visual Review and Diagram View Patch JSON files.
Scripts then validate and apply those artifacts. No nested Codex CLI process is
required.

When the reviewed artifacts already exist, reproduce the checked build with:

```powershell
vllm-arch run `
  --mode reviewed `
  --input samples\hy_v3.py `
  --model-name hy-v3-v0.9.1 `
  --outputs-dir outputs `
  --semantic-review outputs\hy-v3-v0.9.1-semantic-review.json `
  --ir-patch outputs\hy-v3-v0.9.1-architecture-ir.patch.json `
  --visual-review outputs\hy-v3-v0.9.1-visual-review.json `
  --view-patch outputs\hy-v3-v0.9.1-diagram-view.patch.json
```

If those review artifacts are missing, run the Skill interactively so VSCode
Codex can produce them, then rerun the validators and patch application steps.

## Deterministic Mode

Deterministic mode does not use Agent-authored review. It uses deterministic
baseline review templates and is intended for CI/regression checks.

```powershell
vllm-arch run `
  --mode deterministic `
  --input samples\hy_v3.py `
  --model-name hy-v3-v0.9.1-deterministic `
  --outputs-dir outputs
```

## Package

After validators pass and PNG exports exist:

```powershell
python tools\build_mentor_package.py `
  --model-name hy-v3-v0.9.1 `
  --outputs-dir outputs `
  --destination dist\mentor-package
```
