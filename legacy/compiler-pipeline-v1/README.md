# Compiler Pipeline v1 Legacy Archive

This directory preserves the v0.4 through v1.2 compiler-style pipeline for
historical reference and regression archaeology.

The archived pipeline includes the previous Skill body, deterministic source
analysis, semantic inventories, source fact graphs, Architecture IR, Concept and
Design graphs, View compilation, layout plans, renderers, patch/review tooling,
legacy tests, and development reports.

It is not part of the v2.0 default workflow. The default `vllm-arch` entry point
and the canonical Skill now use the lightweight Agent-native workflow in
`src/skills/vllm-model-architecture-diagram/`.

Use this archive only when comparing older research results or intentionally
running legacy experiments. Do not import it from production v2.0 scripts.
