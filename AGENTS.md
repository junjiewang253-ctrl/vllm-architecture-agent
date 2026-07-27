# Repository instructions

## Single source of truth

- The canonical v2.1 Skill is `src/skills/vllm-model-architecture-diagram/`.
- Do not edit `.agents/skills/` as source; it is a local Codex development link.
- Do not edit generated packages under `dist/`.
- The old compiler-style research pipeline is archived in
  `legacy/compiler-pipeline-v1/` and must not be imported by the default v2.x
  workflow.

## v2.1 Architecture

Keep these concerns separate:

1. Static model target resolution.
2. Complete Source Context indexing.
3. Codex complete target-file review.
4. Architecture Plan and Evidence authoring.
5. Class/Method/Function and manifest completeness validation.
6. Draw.io MCP diagram creation and visual review.
7. Structural Draw.io and export validation.
8. Compatibility scanning.

Scripts discover and validate source structure. Codex decides architecture
meaning, page clustering, visual design and layout.

## Implementation rules

- Use Python's standard `ast` module for static parsing.
- Do not import vLLM, torch, transformers or CUDA-dependent modules in default
  scripts.
- Do not infer concrete config values from missing config files.
- Do not treat checkpoint loading as runtime tensor flow.
- Do not hardcode final pages, nodes, edges or coordinates for any model.
- Do not restore Architecture IR, Concept Graph, View Graph, deterministic
  renderer, layout plan, patch, lock or coverage pipeline as the default path.
- Do not silently omit indexed classes, methods, module functions, important
  branches, mappings or capabilities from the Plan review.
- Do not reduce source review scope to make tests pass.
- Do not create one page per class or method; use 3 to 5 composite pages for
  full models.
- Do not delete core behavior for visual neatness; aggregate it into detail
  regions, panels, tables, ports or badges.
- Draw.io XML may be created by Codex through Draw.io MCP, but it must follow
  `architecture-plan.json` and `evidence.json`.
- Do not let Draw.io edits invent source behavior or promote external behavior
  to direct evidence.
- Put cross-Agent core behavior in the Skill directory.
- Put Agent-specific packaging under `integrations/`.

## Default CLI

Only these commands are default:

```text
vllm-arch list-models
vllm-arch prepare
vllm-arch validate
vllm-arch scan
```

Do not reintroduce default `architect`, `deterministic`, `reviewed`,
`finalize`, patch, lock, IR, concept, view or renderer commands.

## Validation

After changing v2.x scripts or CLI:

```text
pytest
```

When a real vLLM checkout is available, set `VLLM_REPO_ROOT` and run pytest to
include the optional full-directory scan test.
