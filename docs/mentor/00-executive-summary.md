# Executive Summary

vLLM Architecture Agent is a Codex Skill for turning vLLM model adapter source into evidence-grounded architecture diagrams.

The project does not use fixed model templates. Scripts only resolve targets, index source structure and validate artifacts. Codex reads the target file, decides the architecture interpretation, designs pages, draws with Draw.io MCP, reviews exported PNGs and writes the report.

The v2.1.1 mentor release focuses on a portable HY V3 example:

- 6 / 6 classes indexed and reviewed;
- 20 / 20 methods indexed and reviewed;
- 12 architecture-level weight mapping groups covered;
- 9 detected capabilities covered;
- 4 high-density composite pages;
- relocatable example validation;
- Draw.io and PNG exports.

The recommended demo command is:

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，
生成默认架构图。
```

The Skill defaults to complete review, four composite pages, at least one visual revision round and final validation.

