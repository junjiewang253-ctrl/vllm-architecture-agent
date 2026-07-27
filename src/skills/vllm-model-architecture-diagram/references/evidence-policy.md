# Evidence Policy

Confidence levels:

- `direct`: the cited source line directly performs the construction, call,
  branch, return, mapping, or assignment.
- `derived`: the claim combines multiple direct facts. State the reasoning.
- `external`: local code proves a dependency or delegation, but the behavior
  happens in an imported component or a file not traversed.

Rules:

- direct claims cannot rely only on import lines;
- derived claims need at least two evidence entries;
- external claims must say what local code proves and what remains outside the
  inspected boundary;
- do not infer concrete config values when only symbolic config access exists;
- checkpoint loading is not runtime tensor flow;
- source line references belong in `evidence.json` and `report.md`, not in node
  titles.
