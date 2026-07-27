# Diagram Grammar

Architecture Design is a blueprint, not Draw.io XML.

Node semantic kinds:

- `data`: tensors, weights, logits, cache entries.
- `component`: local source-backed modules and adapter classes.
- `process`: transformations such as split, route, remap, filter or restore shape.
- `storage`: checkpoint, loaded set or cache-like storage.
- `external_component`: imported component boundary.
- `strategy`: TP, PP, EP or capability strategy.
- `mapping`: checkpoint name/parameter mapping.
- `merge`: explicit merge/add operation.
- `boundary`: local/external divider or adapter boundary.
- `annotation`: explanatory note grounded in facts.

Edge kinds:

- `tensor_flow`: runtime tensor/data flow.
- `residual_flow`: residual-state lane.
- `dependency`: construction/configuration dependency.
- `mapping`: checkpoint/parameter mapping.
- `parallel_effect`: TP/PP/EP effect.
- `delegation`: local adapter delegates to external/imported component.
- `external_boundary`: local facts prove only the boundary, not internals.
- `control`: branch or optional path condition.
- `containment`: parent/child or container membership.

Display rules:

- Ordinary tensor flow should not show labels.
- Long Python expressions stay in evidence/facts, not node text.
- Implementation class names belong in subtitles.
- Capabilities are badges or small notes, not fake inheritance chains.
