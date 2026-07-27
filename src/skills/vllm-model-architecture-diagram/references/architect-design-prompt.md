# Architect Design Prompt v1

You are designing engineering architecture diagrams, not summaries.

Inputs:

- Python source file
- `architect-brief.json`
- `architecture-concept.json`
- `boundary-report.json`
- `diagram-grammar.md`
- `page-patterns.md`

Output:

- `outputs/<model>-architecture-design.json`

Rules:

- Use `architecture-design.schema.json` version `1.0`.
- Set `author.type` to `agent`.
- Every non-annotation node must have `concept_refs`, `fact_refs`, and ports unless it is a strategy or boundary marker.
- Every semantic edge must have `concept_refs`, `fact_refs`, source/target ports, kind, phase, display, and route_hint.
- Do not create modules or behavior not supported by source facts.
- Do not mark imported external component internals as direct local behavior.
- Do not draw Draw.io XML.

For every page:

1. State the engineering question.
2. Define the main reading path.
3. Define branches and merge points.
4. Distinguish runtime flow from configuration, parallelism and mapping.
5. Distinguish local adapter behavior from external implementation.
6. Decide which facts should be components, data objects, processes, containers, badges or annotations.
7. Avoid turning every concept into a node.
8. Avoid a shared summary node as the target of unrelated strategies.
9. Keep page semantics source-grounded.
10. Use implementation names as subtitles, not as the primary architecture story.
