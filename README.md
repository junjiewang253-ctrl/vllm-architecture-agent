# vLLM Architecture Agent

Agent-native Skill for source-grounded vLLM model adapter architecture diagrams.

Version 2.0 is a subtractive refactor. The default workflow no longer runs the
old compiler pipeline of IR, concept graphs, view graphs, layout plans, patches,
locks, and deterministic Draw.io rendering. Scripts now collect and validate
mechanical facts; Codex reads source, chooses pages, designs the diagram, draws
with Draw.io MCP, reviews the PNG, and writes the report.

## Short Instruction

```text
使用 $vllm-model-architecture-diagram 分析 <vLLM 模型文件或 Architecture 名称>，生成架构图。
```

## CLI

The default `vllm-arch` command exposes only four commands:

```powershell
vllm-arch list-models --repo-root D:\path\to\vllm
vllm-arch prepare --repo-root D:\path\to\vllm --input D:\path\to\vllm\vllm\model_executor\models\model_file.py --outputs-dir outputs\model
vllm-arch prepare --repo-root D:\path\to\vllm --architecture SomeForCausalLM --outputs-dir outputs\some-model
vllm-arch validate --context outputs\model\source-context.json --plan outputs\model\architecture-plan.json --evidence outputs\model\evidence.json --drawio outputs\model\architecture.drawio
vllm-arch scan --repo-root D:\path\to\vllm --output outputs\compatibility-report.json
```

The CLI never imports vLLM, torch, or transformers when resolving registry
entries.

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
`evidence.template.json`. Codex fills them as `architecture-plan.json` and
`evidence.json` after reading source.

## Compatibility

v2.0 targets `vllm/model_executor/models/*.py` and supports file input or
registry architecture names. The Source Context collector recognizes candidate
categories such as decoder-only text generation, embedding/pooling,
classification, multimodal, speculative, hybrid/recurrent, attention-free,
custom loading, tensor parallel, pipeline parallel, expert parallel, LoRA,
quantization, and helper/shared modules.

Compatibility means graceful handling, not identical pages for every model.
Codex decides page sets dynamically from source evidence.

## External Boundaries

External runtime internals are not claimed as direct source facts unless Codex
actually reads the relevant local file. Imported attention backends, fused expert
runtimes, automatic loaders, kernels, schedulers, workers, and engine internals
must be documented as external boundaries when not analyzed.

## Legacy

The v0.4 through v1.2 compiler-style pipeline is preserved under:

```text
legacy/compiler-pipeline-v1/
```

It is retained for history and comparison only. It is not the default Skill,
not the default CLI, and not part of default pytest.
