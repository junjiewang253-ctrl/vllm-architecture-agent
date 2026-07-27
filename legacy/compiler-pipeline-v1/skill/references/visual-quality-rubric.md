# Visual Quality Rubric

An acceptable vLLM adapter architecture diagram should:

- answer one engineering question per page;
- show a readable primary path where the page has runtime or mapping flow;
- keep ordinary tensor edges unlabeled;
- keep long source expressions out of the visible diagram;
- keep nodes inside page bounds and containers;
- avoid visible node overlap and text overflow;
- avoid edge segments through non-endpoint nodes;
- keep branch/merge pages aligned around a clear merge node;
- keep strategy panels independent;
- separate local adapter behavior from external vLLM runtime behavior;
- retain semantic IDs and evidence through Design, View, Layout and Draw.io.

Visual review may adjust labels, subtitles, preferred sizes, route hints,
badge visibility, regions and lanes. It must not change source/target,
phase, concept references, fact references or external boundary semantics.
