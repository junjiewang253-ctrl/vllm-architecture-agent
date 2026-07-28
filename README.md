# vLLM Architecture Agent

Agent-native Codex Skill for source-grounded vLLM model adapter architecture diagrams.

Recommended mentor demo command:

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，
生成默认架构图。
```

By default, the Skill runs in complete mode, reviews every indexed class and method in the target file, writes Architecture Plan and Evidence artifacts, draws with Draw.io MCP, performs at least one visual review pass, validates the result, and writes outputs to `outputs/hy-v3/`.

## What v2.1.1 Produces

For the HY V3 example, the default deliverable is four high-density composite diagrams:

1. `Model Architecture and Execution`
2. `Decoder and Attention`
3. `MoE Architecture and Routing`
4. `Parallelism, Configuration and Weight Loading`

The diagrams are not a simple overview and not a one-page-per-method dump. Codex decides the page clustering and visual design, while scripts only index source structure and validate evidence.

Default outputs:

```text
outputs/<model>/
  source-context.json
  architecture-plan.json
  evidence.json
  architecture.drawio
  report.md
  visual-review.md
  images/
```

The portable golden example lives in `examples/hy_v3/` and contains the same artifact types plus four PNG exports.

## Responsibilities

Scripts:

- resolve model files or registry architecture names;
- statically index classes, methods, functions, branches, mapping groups and capabilities;
- generate empty Plan/Evidence checklists;
- validate Architecture Plan, Evidence, Draw.io structure and PNG exports;
- scan model directories without importing vLLM, torch, transformers or CUDA.

Codex:

- reads the complete target file and any needed related files;
- decides the architecture interpretation and page set;
- fills `architecture-plan.json` and `evidence.json`;
- draws and edits Draw.io pages through MCP;
- performs visual review and writes `report.md`.

## Installation For Codex Development

On Windows PowerShell:

```powershell
.\tools\setup-codex-dev.ps1
```

The script installs the package in editable mode and creates `.agents/skills/vllm-model-architecture-diagram` as a local link to the canonical Skill source.

Check Draw.io MCP separately:

```powershell
codex mcp list
```

## Advanced Reproduction

The default CLI intentionally has only four commands:

```powershell
vllm-arch list-models --repo-root <vllm-root>
vllm-arch prepare --input samples\hy_v3.py --outputs-dir outputs\hy-v3 --model-name hy-v3
vllm-arch validate --repo-root . --context examples\hy_v3\source-context.json --plan examples\hy_v3\architecture-plan.json --evidence examples\hy_v3\evidence.json --drawio examples\hy_v3\architecture.drawio --images-dir examples\hy_v3\images
vllm-arch scan --repo-root <vllm-root> --output outputs\compatibility-report.json
```

`prepare` creates `architecture-plan.template.json` and `evidence.template.json`. Codex fills the final Plan and Evidence after source review.

## Validation

Run:

```powershell
pytest
```

Current v2.1.1 tests include portable paths, relocatable HY V3 example validation, method importance precision, mapping group aggregation, capability precision and Draw.io validation.

## Legacy

The old compiler-style pipeline is archived under `legacy/compiler-pipeline-v1/`. It is retained for history only and is not imported by the default v2.x Skill, CLI or tests.
