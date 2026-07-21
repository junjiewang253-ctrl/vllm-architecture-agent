# Draw.io diagram style

## Page structure

For the first overview, use one page with no more than twelve major nodes.
Use a left-to-right main tensor flow and place decoder details below the main
flow.

## Shapes

- Rounded rectangle: runtime module or meaningful operation.
- Container: top-level model or repeated decoder block.
- Diamond: a real source-code condition only.
- Note: unresolved external configuration or concise parallelism annotation.

## Edges

- Solid edge: runtime tensor flow.
- Solid edge labelled `Residual`: residual connection.
- Dashed edge: containment, construction or configuration relationship.
- Dotted edge: checkpoint weight mapping.

## Labels

- Prefer semantic labels, with the Python class or attribute as a subtitle.
- Add compact `TP`, `PP` and `EP` badges rather than turning parallel groups
  into forward-flow nodes.
- Use `N ×`, `E experts` and `Top-K` when values are external.

## Layout checks

- Avoid crossing the main tensor-flow edges.
- Keep checkpoint mappings separate from forward flow.
- Do not place more than one long paragraph inside a node.
- Ensure all node IDs are stable so later `edit_diagram` calls can target them.
