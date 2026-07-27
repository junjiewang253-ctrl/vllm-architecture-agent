# Page Patterns

## `pipeline`

Use for a main left-to-right execution story. Put the primary data path in the
middle, dependency nodes above or below, and side-rank branches off the main
path. Do not turn TP/PP/EP into main runtime nodes.

## `block_with_residual`

Use for transformer blocks with hidden-state and residual lanes. Hidden-state
flow stays on the upper lane. Residual state stays on the lower lane. RMSNorm
nodes expose hidden/residual ports. Avoid large loopback residual arcs.

## `branch_merge`

Use for Attention-like pages. The branch point and merge point must be clear.
HPC and fallback paths must not be serially connected. External attention
backend belongs in a boundary region.

## `routed_container`

Use for MoE. Router and Top-K stay on the main path. `FusedMoE` is a container;
experts are inside the container rather than serial stages after it.

## `mapping_dispatch`

Use for checkpoint loading. Separate entrypoint lanes. After dispatch, mapping
branches should be tree-like and converge at loader/result nodes.

## `strategy_matrix`

Use for TP/PP/EP. Independent panels, no tensor-flow edges between strategies,
and no shared summary sink.

## `boundary_map`

Use for local adapter versus external vLLM runtime. Local nodes on the left,
boundary in the middle, external nodes on the right. Delegation source nodes
should be concrete classes or methods.
