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
4. Draw.io rendering and MCP interaction.

Use a source-analysis JSON between source parsing and Agent reasoning. A later
version may add a separate architecture IR between reasoning and rendering.

## Implementation rules

- Use Python's standard `ast` module for the initial extractor.
- Preserve source line numbers for extracted facts.
- Do not infer concrete values from missing config files.
- Do not treat checkpoint loading as runtime tensor flow.
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
