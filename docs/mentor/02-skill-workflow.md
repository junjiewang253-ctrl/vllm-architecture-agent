# Skill Workflow

User sentence:

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，
生成默认架构图。
```

Workflow:

1. Infer workspace root, target path, model slug and output directory.
2. Run `vllm-arch prepare` to generate Source Context and empty checklists.
3. Codex reads the complete target file and reviews every indexed class, method and relevant structure.
4. Codex reads related files only when needed to prove a claim or define an external boundary.
5. Codex writes `architecture-plan.json`.
6. Codex writes `evidence.json` with direct, derived and external claims.
7. Validators run before drawing.
8. Codex draws pages with Draw.io MCP.
9. Codex exports PNGs, reviews them and performs up to two visual revisions.
10. Validators run again with Draw.io and image checks.
11. Codex writes `report.md`.

Scripts do not decide final architecture semantics or layout. They only index and validate.

