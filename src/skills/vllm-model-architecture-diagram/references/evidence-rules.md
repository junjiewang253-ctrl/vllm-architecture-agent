# Source evidence rules

Every major diagram node and edge must be grounded in the supplied source file.

## Direct evidence

A class, module, condition or call is explicitly present in the source.

Examples:

- `self.self_attn = HYV3Attention(...)`
- `hidden_states = self.self_attn(hidden_states)`

Direct evidence may be drawn as a normal node or edge. Preserve the relevant
source line in the analysis or completion report.

## Derived evidence

A representation combines multiple direct facts without adding a new factual
claim.

Examples:

- `make_layers(...)` plus a decoder-layer constructor can be represented as
  `N × Decoder Layer`.
- a source-level if/else assigning `DenseMLP` or `FusedMoE` can be represented
  by a conditional branch.

Derived facts must stay symbolic when values come from external configuration.

## Unresolved evidence

The file references an external config, helper, imported implementation or
runtime value that is not available in the supplied input.

Rules:

- do not guess;
- place the item in `unresolved`;
- use a short annotation only when it helps explain the visible architecture.

## Prohibited assumptions

- Do not add components based only on knowledge of the model family.
- Do not turn imports into diagram nodes unless they are instantiated or used.
- Do not represent `load_weights` mappings as runtime tensor flow.
- Do not invent exact layer counts, expert counts, top-k values or tensor
  dimensions.
