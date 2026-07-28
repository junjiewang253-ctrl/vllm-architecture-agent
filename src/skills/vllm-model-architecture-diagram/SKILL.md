# vLLM Model Architecture Diagram

Use this Skill when the user asks to analyze a vLLM model adapter file or a registered architecture name and generate architecture diagrams.

Default user instruction:

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，
生成默认架构图。
```

For a valid target file, do not ask for repo root, output directory, page count or validation paths. Infer sensible defaults and execute the full workflow.

## Default Parameters

- `detail_level`: `complete`
- `target_pages`: 4
- `max_pages`: 5
- `visual_review_rounds_min`: 1
- `visual_review_rounds_max`: 2
- default HY V3 output: `outputs/hy-v3/`

## Scope

Default targets are vLLM model adapter files under:

```text
vllm/model_executor/models/*.py
```

Samples such as `samples/hy_v3.py` are also valid. The user may provide a Python path or a registry architecture name. Registry resolution must be static and must not import vLLM, torch, transformers or CUDA-dependent modules.

## Workspace Root

Infer the workspace root in this order:

1. nearest ancestor with `pyproject.toml` and `src/skills/vllm-model-architecture-diagram/SKILL.md`;
2. nearest ancestor of the target file;
3. current working directory.

For `samples/hy_v3.py`, the default is equivalent to:

```powershell
vllm-arch prepare `
  --repo-root <workspace-root> `
  --input samples/hy_v3.py `
  --outputs-dir outputs/hy-v3 `
  --model-name hy-v3
```

Do not require the user to type this command.

## Workflow

### 1. Parse Request

- Identify the target file or architecture name.
- Confirm the target exists or can be resolved.
- Generate a stable model slug.
- Create `outputs/<model>/`.
- Overwrite only known artifacts for this model; do not delete unrelated user files.

### 2. Prepare Source Context

Run `vllm-arch prepare` with inferred defaults.

This creates:

- `source-context.json`;
- `architecture-plan.template.json`;
- `evidence.template.json`.

The templates are checklists only. They must not contain final pages, model-specific answers, nodes, edges or layout.

### 3. Complete Target-File Review

Read the full target Python file. Then use `source-context.json` as a checklist:

- `source_coverage`;
- `classes`;
- `methods`;
- `module_functions`;
- `branches`;
- `weight_mappings`;
- `capability_signals`;
- `related_file_candidates`.

Review every class, method and module-level function. Review all high/medium branches, all architecture-level mapping groups and all detected capabilities.

Do not silently omit anything. Not every item must become a node, but every item must receive a disposition in `architecture-plan.json`.

### 4. Traverse Related Source

Use `related_file_candidates` when additional files prove a claim or clarify an external boundary.

Default budget:

- max depth: 3;
- default related files: 20;
- suggested max: 30.

Stop when the local source reaches a stable imported component boundary, more recursion will not change the diagram, or the budget is reached. Record files read in `files_read` and stopping reasons in `traversal_notes`.

### 5. Write Architecture Plan

Write `architecture-plan.json` before drawing.

For full models, design 3 to 5 composite pages, usually 4. Do not create one page per class or method. Use detail regions, component trees, embedded subgraphs, parameter panels, variant tables, mapping trees, boundary panels, ports and badges.

Each page must include:

- engineering question;
- purpose;
- topics;
- view pattern;
- claim IDs;
- at least one detail region;
- detail budget.

Each rendered class, method, function, branch, mapping group or capability must map to a real page and region.

### 6. Write Evidence

Write `evidence.json` with schema version `2.1`.

Use repo-relative paths for repository files, for example `samples/hy_v3.py`.

Evidence confidence:

- `direct`: local source directly proves the behavior;
- `derived`: multiple evidence entries support a higher-level claim;
- `external`: local source proves delegation or dependency, but implementation is outside the reviewed boundary.

Direct claims cannot rely only on imports. Derived claims require a derivation. External claims require an explicit boundary.

### 7. Validate Before Drawing

Run:

```powershell
vllm-arch validate `
  --repo-root <workspace-root> `
  --context outputs/<model>/source-context.json `
  --plan outputs/<model>/architecture-plan.json `
  --evidence outputs/<model>/evidence.json
```

If validation fails, fix Plan or Evidence before opening Draw.io.

### 8. Draw With Draw.io MCP

Use Draw.io MCP. Do not use the legacy deterministic renderer.

Rules:

- call `start_session`;
- create or edit pages matching Plan titles exactly;
- use one page per planned page, no unrelated extra pages;
- include every detail region title visibly on its page;
- runtime tensor flow uses primary arrows;
- construction/config dependencies use dashed or secondary lines;
- checkpoint loading is not runtime tensor flow;
- TP/PP/EP are independent panels or badges, not a serial flow;
- external components sit behind explicit boundaries;
- implementation names are subtitles or compact labels;
- long expressions and line numbers stay in Evidence/report;
- background is white and opaque.

### 9. Export Draft PNGs

Export one PNG per page into `outputs/<model>/images/`.

PNG exports must not show Draw.io editor grid and should use clear readable fonts.

### 10. Visual Review

View every PNG and write `visual-review.md`.

Check:

- every detail region appears;
- pages are not only overview cards;
- core methods appear in diagram regions, not only in report;
- construction/runtime/loading/parallel are visually distinct;
- config and capabilities are not shown as tensor flow;
- external behavior is not shown as local direct behavior;
- no obvious text overlap;
- no obvious edge-through-node problems;
- no huge empty spaces;
- no misleading serial chain for mutually exclusive branches.

Make at least one visual revision pass and at most two.

### 11. Final Validation

Run:

```powershell
vllm-arch validate `
  --repo-root <workspace-root> `
  --context outputs/<model>/source-context.json `
  --plan outputs/<model>/architecture-plan.json `
  --evidence outputs/<model>/evidence.json `
  --drawio outputs/<model>/architecture.drawio `
  --images-dir outputs/<model>/images
```

Do not report completion if validation fails.

### 12. Report

Write `report.md` after final validation succeeds.

The report must include:

- target model and registry status;
- files read and stop reasons;
- source coverage;
- class/method/function review summary;
- branch/mapping/capability coverage;
- page-by-page walkthrough;
- direct/derived/external distinctions;
- external boundaries;
- unresolved or omitted content and reasons;
- exact validation commands and results;
- output paths.

Do not hand-write “passed” before validators actually pass.

## Degradation

For partial targets, produce 1 to 5 pages and clearly state missing context.

For helper/shared utility files, produce 1 or 2 utility/boundary pages. Do not force a full model flow onto helper code.

## References

Read as needed:

- `references/analysis-guide.md`
- `references/vllm-patterns.md`
- `references/page-playbook.md`
- `references/diagram-design-guide.md`
- `references/evidence-policy.md`
