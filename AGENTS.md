# Repository instructions

## Single source of truth

- The canonical v2.0 Skill is `src/skills/vllm-model-architecture-diagram/`.
- Do not edit `.agents/skills/` as source; it is a local Codex development link.
- Do not edit generated packages under `dist/`.
- The old compiler-style research pipeline is archived in
  `legacy/compiler-pipeline-v1/` and must not be imported by the default v2.0
  workflow.

## v2.0 Architecture

Keep these concerns separate:

1. Static model target resolution.
2. Lightweight Source Context collection.
3. Codex source traversal and architecture planning.
4. Evidence authoring and validation.
5. Draw.io MCP diagram creation and visual review.
6. Structural Draw.io validation.
7. Compatibility scanning.

Scripts are allowed to decide what source facts exist and whether files are
valid. Scripts must not decide final architecture pages, semantic nodes, or
layout. Codex is the architecture designer.

## Implementation rules

- Use Python's standard `ast` module for static parsing.
- Do not import vLLM, torch, transformers, or CUDA-dependent modules in default
  resolver, context, or scan scripts.
- Do not infer concrete config values from missing config files.
- Do not treat checkpoint loading as runtime tensor flow.
- Do not hardcode final pages, nodes, edges, or coordinates for any model.
- Do not let scripts generate Architecture IR, Concept Graph, View Graph,
  Layout Plan, Review Patch, Review Lock, or Coverage files by default.
- Draw.io XML may be created by Codex through Draw.io MCP, but it must follow
  `architecture-plan.json` and `evidence.json`.
- Do not let Draw.io edits invent source behavior or promote external behavior
  to direct evidence.
- Put cross-Agent core behavior in the Skill directory.
- Put Codex-, Claude Code-, or other Agent-specific packaging under
  `integrations/`.

## Default CLI

Only these commands are default:

```text
vllm-arch list-models
vllm-arch prepare
vllm-arch validate
vllm-arch scan
```

Do not reintroduce default `architect`, `deterministic`, `reviewed`,
`finalize`, patch, lock, IR, or renderer commands.

## Validation

After changing v2.0 scripts or CLI:

```text
pytest
```

For local fixture-like validation:

```text
vllm-arch prepare --repo-root <repo> --architecture <ArchitectureName> --outputs-dir outputs/<model>
vllm-arch scan --repo-root <repo> --output outputs/compatibility-report.json
```

When a real vLLM checkout is available, set `VLLM_REPO_ROOT` and run pytest to
include the optional full-directory scan test.
