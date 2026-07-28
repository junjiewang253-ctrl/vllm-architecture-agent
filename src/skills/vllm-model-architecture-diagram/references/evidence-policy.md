# Evidence Policy

Evidence schema version is `2.1`.

## Confidence

`direct`: local source directly constructs, calls, branches, returns, filters,
maps, loads or assigns.

`derived`: multiple local facts support a higher-level architecture statement.
Derived claims require a non-empty `derivation`.

`external`: local code proves a dependency or delegation, but the implementation
is outside the reviewed boundary. External claims require `external_boundary`.

## Source Item IDs

Claims should reference relevant source item IDs:

- class IDs;
- method IDs;
- function IDs;
- branch IDs;
- mapping IDs;
- capability IDs.

## Rules

- Direct claims cannot rely only on imports.
- Derived claims need at least two evidence entries.
- Evidence file paths must be part of `files_read`.
- Repository-internal evidence paths must be repo-relative, such as
  `samples/hy_v3.py`, so examples can be validated after relocation.
- Line ranges must be valid and should include the cited symbol when possible.
- Checkpoint loading is not runtime tensor flow.
- Symbolic config expressions remain symbolic; do not invent concrete values.
- External behavior can be drawn as a boundary, not as local direct behavior.
