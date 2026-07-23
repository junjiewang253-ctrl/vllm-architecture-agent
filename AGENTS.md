# Repository instructions

## Single source of truth

- The canonical Skill is `src/skills/vllm-model-architecture-diagram/`.
- Do not edit `.agents/skills/` as source; it is a local Codex development link.
- Do not edit generated packages under `dist/`.

## Architecture

Keep these concerns separate:

1. Python source fact extraction.
2. vLLM semantic interpretation.
3. Architecture/diagram planning.
4. Diagram View and deterministic layout planning.
5. Draw.io rendering and MCP interaction.

Use source-analysis JSON between source parsing and Architecture IR building.
Architecture IR is the only semantic source for Draw.io rendering.
Diagram View may decide presentation, ports, lanes and route hints, but must
not change source semantics.

## Implementation rules

- Use Python's standard `ast` module for the initial extractor.
- Preserve source line numbers for extracted facts.
- Do not infer concrete values from missing config files.
- Do not treat checkpoint loading as runtime tensor flow.
- Do not let Agents hand-write complete Draw.io XML; use `render_drawio.py`.
- Do not add or remove semantic nodes or edges through Draw.io MCP.
- Build and validate Diagram View before rendering v0.8 diagrams.
- Use `layout_diagram.py` for deterministic coordinates and routed waypoints.
- Put cross-Agent core behavior in the Skill directory.
- Put Codex-, Claude Code- or other Agent-specific packaging under
  `integrations/`.

## Validation

After changing extractor code:

```text
pytest
python src/skills/vllm-model-architecture-diagram/scripts/extract_architecture.py samples/simple_model.py --output outputs/simple-model-source-analysis.json
```

Test the small synthetic model before testing a complex vLLM adapter.

After changing IR, renderer, or Draw.io validation code:

```text
pytest
python src/skills/vllm-model-architecture-diagram/scripts/extract_architecture.py samples/hy_v3.py --output outputs/hy-v3-source-analysis.json
python src/skills/vllm-model-architecture-diagram/scripts/build_architecture_ir.py outputs/hy-v3-source-analysis.json --output outputs/hy-v3-architecture-ir.json
python src/skills/vllm-model-architecture-diagram/scripts/validate_architecture_ir.py outputs/hy-v3-architecture-ir.json
python src/skills/vllm-model-architecture-diagram/scripts/build_diagram_view.py outputs/hy-v3-architecture-ir.json --output outputs/hy-v3-diagram-view.json
python src/skills/vllm-model-architecture-diagram/scripts/validate_diagram_view.py outputs/hy-v3-architecture-ir.json outputs/hy-v3-diagram-view.json
python src/skills/vllm-model-architecture-diagram/scripts/layout_diagram.py outputs/hy-v3-diagram-view.json --output outputs/hy-v3-layout-plan.json
python src/skills/vllm-model-architecture-diagram/scripts/render_drawio.py outputs/hy-v3-diagram-view.json --layout-plan outputs/hy-v3-layout-plan.json --output outputs/hy-v3-architecture.drawio
python src/skills/vllm-model-architecture-diagram/scripts/validate_drawio.py outputs/hy-v3-architecture-ir.json outputs/hy-v3-architecture.drawio --view outputs/hy-v3-diagram-view.json --layout-plan outputs/hy-v3-layout-plan.json
python src/skills/vllm-model-architecture-diagram/scripts/validate_visual_layout.py outputs/hy-v3-architecture-ir.json outputs/hy-v3-architecture.drawio --metrics-output outputs/hy-v3-layout-metrics.json
```
