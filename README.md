# vLLM Architecture Agent

Agent-native Skill for source-grounded vLLM model adapter architecture diagrams.

Version 2.1 generates a small set of complete, high-density architecture
diagrams. The default is not a simple overview and not a pile of one-page-per
method details. Codex reviews the complete target file, chooses 3 to 5 composite
pages for full models, and draws with Draw.io MCP.

## Short Instruction

```text
使用 $vllm-model-architecture-diagram 以 complete 模式分析 <模型文件或 Architecture 名称>，
生成 3～5 张少页、高密度、包含关键实现细节的架构图。
```

## What Scripts Do

Scripts keep the workflow grounded:

- resolve a model file or registry architecture;
- index every class, method, module-level function, branch, mapping and
  capability candidate in the target file;
- generate a review checklist;
- validate the Agent-authored plan and evidence;
- validate Draw.io structure and exported images;
- scan model directories without importing vLLM.

Scripts do not choose final pages, semantic nodes, edges or layout.

## What Codex Does

Codex reads the source, completes class/method/function review, decides page
clustering, writes `architecture-plan.json` and `evidence.json`, draws with
Draw.io MCP, reviews the first PNG draft, and writes `report.md`.

Content that does not enter a diagram must still have a reason in the plan. No
silent omission is allowed.

## CLI

The default `vllm-arch` command exposes only four commands:

```powershell
vllm-arch list-models --repo-root D:\path\to\vllm
vllm-arch prepare --repo-root D:\path\to\vllm --input D:\path\to\vllm\vllm\model_executor\models\model_file.py --outputs-dir outputs\model
vllm-arch prepare --repo-root D:\path\to\vllm --architecture SomeForCausalLM --outputs-dir outputs\some-model
vllm-arch validate --context outputs\model\source-context.json --plan outputs\model\architecture-plan.json --evidence outputs\model\evidence.json --drawio outputs\model\architecture.drawio --images-dir outputs\model\images
vllm-arch scan --repo-root D:\path\to\vllm --output outputs\compatibility-report.json
```

The CLI never imports vLLM, torch, transformers or CUDA-dependent modules.

## Outputs

Default per-model output:

```text
outputs/<model>/
├── source-context.json
├── architecture-plan.json
├── evidence.json
├── architecture.drawio
├── report.md
├── visual-review.md
└── images/
```

`prepare` creates `architecture-plan.template.json` and
`evidence.template.json`. Codex fills the final Plan and Evidence after reading
source.

## Compatibility

v2.1 targets `vllm/model_executor/models/*.py` and supports file input or
registry architecture names. It handles decoder-only, MoE, embedding/pooling,
classification, multimodal, speculative, hybrid/recurrent, attention-free,
custom loading and helper/shared modules by producing different Source Context
and letting Codex choose different page sets.

## External Boundaries

External runtime internals are not direct claims unless Codex actually reads the
relevant local file. Imported attention backends, fused expert runtimes,
automatic loaders, schedulers, workers, engine internals and kernels must be
documented as external boundaries when not analyzed.

## Legacy

The v0.4 through v1.2 compiler-style pipeline is preserved under:

```text
legacy/compiler-pipeline-v1/
```

It is retained for history and comparison only. It is not the default Skill,
not the default CLI, and not part of default pytest.
