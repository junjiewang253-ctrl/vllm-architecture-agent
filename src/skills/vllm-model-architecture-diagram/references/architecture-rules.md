# Architecture IR rules

The Architecture IR is the semantic contract between source analysis and
Draw.io rendering. It must be written to disk and validated before any diagram
is created.

## Required files

For an input model named `hy-v3`, produce:

- `outputs/hy-v3-source-analysis.json`
- `outputs/hy-v3-architecture-ir.json`
- `outputs/hy-v3-overview.drawio`
- `outputs/hy-v3-overview.svg`

## Node rules

- Use stable lowercase IDs with underscores.
- Keep labels semantic and short.
- Put the Python class or attribute in `subtitle`.
- Store source lines in `evidence`; do not stack line numbers in the visible
  node label.
- Use `badges` for `TP`, `PP` and `EP`.
- Use a `decision` node only for a real source condition.
- Represent residual addition with an explicit `add` or `merge` node.

## Edge rules

- `runtime`: normal forward tensor flow.
- `residual`: skip path ending at an `add` or `merge` node.
- `dependency`: a runtime dependency that is not itself an executed tensor
  stage, for example LM Head passed into LogitsProcessor.
- `conditional_true` and `conditional_false`: mutually exclusive branches.
- `weight_mapping`: checkpoint-to-runtime parameter mapping only.

Conditional branches must reconnect to a common downstream node or output.
Do not serialize mutually exclusive calls as a single linear sequence.

## Evidence

Every major node and every edge requires at least one evidence object. Examples:

```json
{"type": "direct", "line": 452}
```

```json
{"type": "derived", "lines": [457, 459, 547, 550]}
```

## Overview limit

An overview page contains at most twelve major nodes. Notes and containers do
not count toward this limit.
