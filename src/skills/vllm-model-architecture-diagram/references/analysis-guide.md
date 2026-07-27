# Analysis Guide

Use `source-context.json` as a complete reading checklist. It is not the final
architecture answer.

## Complete Source Review

1. Read the complete target Python file.
2. Check `source_coverage`; all classes, methods and module functions should be
   indexed.
3. Read every class source range from `start_line` to `end_line`.
4. Review every method in the Method Catalog.
5. Review every module-level function.
6. Review high and medium branches.
7. Review all weight mappings.
8. Review every detected capability.
9. Give every item a rendered, aggregated, external, excluded, or unresolved
   disposition in the plan.

Do not assume a short class is irrelevant. Do not ignore module-level functions;
they may contain weight mapping, input processing, filtering, or registry logic.

## Phase Separation

Separate:

- construction: `__init__`, module assignments, layer factories, variants;
- runtime: `forward`, `compute_logits`, pooling, encoding, recurrent state;
- loading: `load_weights`, remap, dispatch, filtering, loader calls;
- parallel: tensor, pipeline, expert process-group and rank behavior;
- capability: mixins, decorators, compile hooks, LoRA or multimodal support.

One page may contain multiple phases, but use regions and line styles to keep
them visually distinct.

## Method Importance

Core methods:

- change the main execution path;
- choose model submodules;
- process parallel ranks;
- implement custom weight loading;
- update expert layout;
- fuse multimodal inputs;
- produce logits, pooling, classification or recurrent output.

Supporting methods:

- prepare inputs;
- reshape tensors;
- update metadata;
- filter, remap or adapt names;
- help a core method.

Trivial methods:

- simple property access;
- very short helpers with no architectural consequence.

Importance is a script-provided candidate. Codex makes the final review call and
records the reason.

## Traversal

Read related files only when they support a claim or boundary. Stop when an
imported component becomes a stable external API boundary or additional
recursion will not alter the diagram.
