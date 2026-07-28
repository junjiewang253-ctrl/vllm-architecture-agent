# Limitations And Future Work

Limitations:

- Imported vLLM runtime internals are marked external unless Codex explicitly reads those local files.
- The bundled HY V3 sample is outside a full vLLM checkout, so registry context is partial while target-file indexing is full.
- Layout is Agent-designed through Draw.io MCP; identical semantic review can still produce small visual differences across manual runs.
- The Skill validates structure, evidence and exports, but it does not perform pixel-level image analysis.
- Full `vllm/model_executor/models` compatibility requires a real local vLLM checkout and is not claimed unless scan is executed.

Future work:

- Run regular compatibility scans against current vLLM.
- Add more portable golden examples for dense, multimodal, embedding/pooling and hybrid models.
- Add optional visual linting based on exported PNG analysis.
- Add Claude Code packaging while keeping the canonical Skill source unchanged.
