# vLLM Architecture Agent

An Agent Skill that analyzes a vLLM model adapter Python file and generates a
source-grounded, editable Draw.io architecture diagram.

## Current Status

Version 0.9.1 keeps the deterministic extraction, IR, Diagram View, layout,
renderer and validators, and replaces simulated review templates with real
Agent-authored Review/Patch artifacts. In the normal VSCode Codex workflow, the
current Codex session writes the structured review JSON and patch JSON directly;
scripts only validate, apply and render those audited artifacts.

- Source Analysis 0.3 with stable fact IDs.
- Semantic Inventory and Semantic Coverage 0.2.
- Architecture IR 0.6 with fact-id evidence and semantic ports.
- Codex Semantic Review and audited Architecture IR Patch artifacts.
- Diagram View, Layout Plan and deterministic Draw.io rendering.
- Codex Visual Review and audited Diagram View Patch artifacts.
- Review Lock hashes for reproducible reviewed builds.
- Mentor package generation for submission candidates.

Codex does not hand-write final Draw.io XML and does not silently rewrite the
Architecture IR. The allowed path is:

```text
Codex judgment -> review JSON -> patch JSON -> validators -> patch applier
```

`run_semantic_review.py` and `run_visual_review.py` remain optional automation
helpers and test harness entry points. They are not the default Skill path.

## Default Pages

The HY V3 reviewed pipeline currently produces seven pages:

- Model Overview
- Decoder Layer Detail
- Attention Detail
- MoE Detail
- vLLM Adapter Integration
- Parallelism
- Weight Loading

The page set is planned from source-backed facts in reviewed mode. Empty pages
should not be generated for models that do not expose the relevant facts.

## Repository Layout

```text
src/skills/vllm-model-architecture-diagram/
  SKILL.md
  scripts/
    extract_architecture.py
    build_semantic_inventory.py
    build_architecture_ir.py
    run_semantic_review.py        optional automation/helper
    validate_semantic_review.py
    apply_ir_patch.py
    validate_semantic_coverage.py
    build_diagram_view.py
    run_visual_review.py          optional automation/helper
    validate_visual_review.py
    apply_view_patch.py
    layout_diagram.py
    render_drawio.py
    validate_drawio.py
    validate_visual_layout.py
    build_review_lock.py
    validate_review_lock.py
  references/
  schemas/

docs/mentor/                  Mentor-facing submission docs
tools/build_mentor_package.py Package builder
samples/                      Manual end-to-end inputs
tests/                        Regression tests
outputs/                      Generated analysis and diagrams
```

## Run Tests

```powershell
python -m pip install -e ".[dev]"
pytest
```

## Reviewed HY V3 Pipeline

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

When using the Skill interactively, VSCode Codex produces the four review/patch
files during the Semantic Review and Visual Review steps, then runs the
validators and patch appliers. `deterministic` mode remains available for CI and
baseline regression checks.

## Mentor Package

After all validators pass and page PNG exports are present:

```powershell
python tools\build_mentor_package.py --model-name hy-v3-v0.9.1 --outputs-dir outputs --destination dist\mentor-package
```

## Boundaries

v0.9.1 does not recursively analyze external vLLM components. Imported component
internals such as `Attention`, `FusedMoE` and `AutoWeightsLoader` are represented
as boundaries unless the input file itself proves the behavior.
