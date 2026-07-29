# Repository instructions

## Single source of truth

- The canonical v2.2 Skill is `src/skills/vllm-model-architecture-diagram/`.
- Do not edit `.agents/skills/` as source; it is a local Codex development link.
- Do not edit generated packages under `dist/` except for the requested mentor zip.
- The old compiler-style research pipeline is archived in
  `legacy/compiler-pipeline-v1/` and must not be imported by the default v2.x
  workflow.

## v2.2 Architecture

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

The default mentor workflow starts from one user sentence:

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，生成默认架构图。
```

For local files, infer repo root and output directory automatically. Store
repository-internal artifact paths as repo-relative paths so examples remain
portable across machines.

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
- Default complete full-model output is one large
  `integrated_single_canvas`, not a dashboard of subsystem pages.
- Expand the representative repeated block in place on a continuous runtime
  spine. Put loading below it and map weights into actual receiving components.
- Do not keep the Draw.io default page size. Plan 2.2 integrated output requires
  an explicit landscape canvas of at least 4200 x 2000.
- Do not create one page per class or method. Use `composite_pages` only after
  two genuine MCP layout revisions fail to keep one canvas readable.
- Do not delete core behavior for visual neatness; aggregate it into detail
  regions, panels, tables, ports or badges.
- Plan 2.2 required anchors and relationships must be copied to Draw.io cells
  as exact `dataAnchor` attributes. Do not satisfy them with invisible or
  decorative cells.
- Detail regions must attach to the declared main-story anchor. Required
  architecture anchors may not form isolated card islands.
- Draw.io XML may be created by Codex through Draw.io MCP, but it must follow
  `architecture-plan.json` and `evidence.json`.
- Draw.io MCP is mandatory for final diagram creation. Never use Python,
  PowerShell, shell redirection or templates to manufacture `.drawio` XML, and
  never create placeholder PNG/SVG exports.
- If `start_session` or another required Draw.io MCP tool is unavailable, stop
  and report the drawing stage as blocked instead of falling back.
- A visual review requires opening each real exported PNG and making at least
  one MCP-based revision after Draft 1.
- Do not let Draw.io edits invent source behavior or promote external behavior
  to direct evidence.
- Do not write developer-machine absolute paths into examples, Evidence, Plan,
  reports or mentor deliverables when the file is inside the repository.
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
