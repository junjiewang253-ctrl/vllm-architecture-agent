# Validation and Limitations

## Validation Model

Each extracted structured fact has a stable `fact_id`. Architecture IR evidence
prefers `fact_ids` over bare line numbers. Direct evidence must cite behavior
facts such as calls, assignments, branches, returns or weight-loading stages.
Import facts cannot alone prove behavior.

Codex Semantic Review assigns one disposition to every required fact. The final
coverage matrix distinguishes:

- `rendered_direct`
- `rendered_derived`
- `aggregated`
- `documented_external`
- `excluded`
- `unresolved`
- `orphaned`

Reviewed mode requires required unresolved and orphaned counts to be zero.

Architecture IR Patch and Diagram View Patch carry base hashes. Apply scripts
refuse stale inputs and add audit metadata. Review Lock records the artifact
hashes needed to reproduce reviewed IR, reviewed View, layout and Draw.io from
the saved patches.

v0.9.1 review artifacts and optional review logs preserve the prompts or review
context used by Codex, so the review decision can be audited alongside the JSON
artifacts.

## Boundaries

The current implementation does not recursively analyze imported vLLM
components. It does not claim direct proof of vLLM Attention internals,
FusedMoE communication, AutoWeightsLoader internals, Scheduler, Worker,
EngineCore or request batching. These must be left out or documented as external
boundaries.

The HY V3 builder remains more specialized than a complete multi-model product.
v0.9 improves the review and submission workflow, not general vLLM model
coverage.
