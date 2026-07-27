# Analysis Guide

Start with the target model file and use `source-context.json` as a navigation
aid. It is an index, not the answer.

Identify the wrapper class, base model class, repeated layer construction,
forward path, output/logits path, and checkpoint loading path when present.

Separate phases:

- construction: modules assigned in `__init__` and layer factory choices;
- runtime: calls in `forward`, pooling, encoding, or logits computation;
- loading: `load_weights`, mapping tables, filters, remaps, and loaders;
- parallel: rank/group setup, layer partitioning, tensor-parallel layers, and
  expert-parallel metadata.

Traversal budget:

- read the target file completely;
- follow local model imports first;
- follow interfaces and utility files when they prove capability claims;
- follow instantiated layer files only to define an external boundary or stable
  API contract;
- stop at stable external components unless their implementation is essential to
  the user-requested diagram.

Record every file read and why traversal stopped.
