# Page Playbook

Use pages to answer engineering questions. Do not create a page merely because a
class or concept exists.

Patterns:

- `pipeline`: left-to-right or top-to-bottom runtime flow.
- `block`: internal structure of one repeated or reusable block.
- `branch_merge`: alternative paths that converge at a backend or merge point.
- `routed_container`: routing into an internal container, such as experts.
- `mapping_flow`: checkpoint or name mapping from source weights to parameters.
- `strategy_matrix`: independent strategy panels, such as parallelism modes.
- `boundary_map`: local adapter versus external runtime responsibility.
- `multimodal_pipeline`: separate modality paths and fusion.
- `state_machine`: recurrent, streaming, or state update behavior.
- `component_map`: relationships for helper or shared modules.

Common mistakes:

- turning capability names into a fake runtime chain;
- generating a MoE page without MoE evidence;
- making configuration dependencies look like tensor flow;
- drawing external internals as if the target file directly implemented them;
- filling a page with concept cards and no input, transformation, boundary, or
  output.
