# Draw.io diagram style

## Page structure

For the first overview, use one page with no more than twelve major nodes.
Use a left-to-right main tensor flow and place decoder details below the main
flow.

## Shapes

- Rounded rectangle: runtime module or meaningful operation.
- Container: top-level model or repeated decoder block.
- Diamond: a real source-code condition only.
- Circle or compact rectangle: explicit Add/Merge operation.
- Note: unresolved external configuration or concise parallelism annotation.

## Edges

- Solid edge: runtime tensor flow.
- Solid skip edge labelled `Residual`: residual connection ending at Add/Merge.
- Dashed edge: containment, construction or dependency relationship.
- Dotted edge: checkpoint weight mapping.
- Conditional edges must be labelled `True` and `False` and reconnect to a
  common downstream node or output.

## Labels

- Prefer semantic labels, with the Python class or attribute as a subtitle.
- Keep source line numbers in Architecture IR evidence or Draw.io metadata, not
  in the visible title.
- Add compact `TP`, `PP` and `EP` badges rather than turning parallel groups
  into forward-flow nodes.
- Use `N ×`, `E experts` and `Top-K` when values are external.

## SVG compatibility

Use Draw.io styles with `html=0` by default. Avoid HTML labels and
`foreignObject` so exported SVG files render in browsers, Markdown viewers,
PPT tools and image converters.

Recommended base style:

```text
rounded=1;whiteSpace=wrap;html=0;
```

## Layout checks

- Avoid crossing the main tensor-flow edges.
- Keep checkpoint mappings separate from forward flow.
- Do not place more than one long paragraph inside a node.
- Ensure all node IDs are stable so later `edit_diagram` calls can target them.
