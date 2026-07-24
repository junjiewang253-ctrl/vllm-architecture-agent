# Agent Review Rules

v0.9 uses the rule:

```text
Use scripts to constrain trusted boundaries; use Agent review for judgments that
are hard to fully encode as deterministic rules.
```

## Semantic Review Inputs

Read these before producing `semantic-review.json` or
`architecture-ir.patch.json`:

- input Python source;
- source-analysis JSON;
- semantic inventory JSON;
- baseline Architecture IR;
- baseline semantic coverage;
- `references/evidence-rules.md`;
- `references/vllm-patterns.md`;
- `references/architecture-rules.md`.

## Semantic Review May

- classify required facts as rendered, aggregated, documented external,
  excluded, or unresolved;
- aggregate low-level facts into a high-level architecture node, edge,
  annotation, badge, page, or capability concept;
- add derived concepts when they cite source facts and explain the derivation;
- mark imported component behavior as an external boundary;
- decide page keep/add/remove/merge/split actions;
- resolve baseline unresolved items only when a reliable disposition exists.

## Semantic Review Must Not

- invent modules, runtime flow or configuration values not supported by the
  input file;
- use import facts as the only evidence for behavior;
- promote external component internals to direct evidence;
- change checkpoint loading into runtime tensor flow;
- exclude facts in bulk to obtain zero unresolved;
- write final Draw.io XML;
- change source code to fit the diagram.

## Patch Requirements

Every semantic patch operation needs a reason, confidence and evidence facts.
Operations below confidence `0.70` are deferred by default. Critical and major
findings must be patched or explicitly deferred with a reason.

`unresolved=0` means every required fact has a reviewed disposition. It does not
mean every fact is drawn as an independent node.

## Visual Review Inputs

Read these before producing `visual-review.json` or `diagram-view.patch.json`:

- reviewed Architecture IR;
- baseline Diagram View;
- baseline Layout Plan;
- baseline layout metrics;
- baseline Draw.io;
- page screenshots when available.

## Visual Review May

- shorten display labels;
- hide noisy edge labels;
- add bundles, legends, boundary notes and page notes;
- adjust region, lane, preferred size and route hints;
- mark redundant semantic edges as hidden visual edges while preserving cells.

## Visual Review Must Not

- add or remove semantic nodes;
- add or remove semantic edges;
- change edge source, target, source port or target port;
- change phase, kind, evidence or source facts;
- hide required optional/external boundary markers;
- use MCP to bypass validators.

## Round Limits

Semantic Review and Visual Review each allow at most two rounds. Round 2 is only
for validator failures, remaining unresolved required facts, or critical/major
findings that were not resolved in Round 1.
