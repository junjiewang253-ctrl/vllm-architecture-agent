# Reproduction Guide

Run from the repository root in Windows PowerShell.

```powershell
python -m pip install -e ".[dev]"
pytest
python src\skills\vllm-model-architecture-diagram\scripts\extract_architecture.py samples\hy_v3.py --output outputs\hy-v3-v0.9-source-analysis.json
python src\skills\vllm-model-architecture-diagram\scripts\build_semantic_inventory.py outputs\hy-v3-v0.9-source-analysis.json --output outputs\hy-v3-v0.9-semantic-inventory.json
python src\skills\vllm-model-architecture-diagram\scripts\build_architecture_ir.py outputs\hy-v3-v0.9-source-analysis.json --output outputs\hy-v3-v0.9-baseline-architecture-ir.json
python src\skills\vllm-model-architecture-diagram\scripts\validate_architecture_ir.py outputs\hy-v3-v0.9-baseline-architecture-ir.json
python src\skills\vllm-model-architecture-diagram\scripts\validate_semantic_coverage.py outputs\hy-v3-v0.9-source-analysis.json outputs\hy-v3-v0.9-semantic-inventory.json outputs\hy-v3-v0.9-baseline-architecture-ir.json --output outputs\hy-v3-v0.9-baseline-semantic-coverage.json
python src\skills\vllm-model-architecture-diagram\scripts\build_semantic_review.py outputs\hy-v3-v0.9-source-analysis.json outputs\hy-v3-v0.9-semantic-inventory.json outputs\hy-v3-v0.9-baseline-architecture-ir.json outputs\hy-v3-v0.9-baseline-semantic-coverage.json --source-file samples\hy_v3.py --review-output outputs\hy-v3-v0.9-semantic-review.json --patch-output outputs\hy-v3-v0.9-architecture-ir.patch.json
python src\skills\vllm-model-architecture-diagram\scripts\validate_semantic_review.py outputs\hy-v3-v0.9-source-analysis.json outputs\hy-v3-v0.9-semantic-inventory.json outputs\hy-v3-v0.9-baseline-architecture-ir.json outputs\hy-v3-v0.9-semantic-review.json outputs\hy-v3-v0.9-architecture-ir.patch.json
python src\skills\vllm-model-architecture-diagram\scripts\apply_ir_patch.py outputs\hy-v3-v0.9-baseline-architecture-ir.json outputs\hy-v3-v0.9-architecture-ir.patch.json --output outputs\hy-v3-v0.9-reviewed-architecture-ir.json
python src\skills\vllm-model-architecture-diagram\scripts\validate_architecture_ir.py outputs\hy-v3-v0.9-reviewed-architecture-ir.json
python src\skills\vllm-model-architecture-diagram\scripts\validate_semantic_coverage.py outputs\hy-v3-v0.9-source-analysis.json outputs\hy-v3-v0.9-semantic-inventory.json outputs\hy-v3-v0.9-reviewed-architecture-ir.json --semantic-review outputs\hy-v3-v0.9-semantic-review.json --output outputs\hy-v3-v0.9-semantic-coverage.json
python src\skills\vllm-model-architecture-diagram\scripts\build_diagram_view.py outputs\hy-v3-v0.9-reviewed-architecture-ir.json --output outputs\hy-v3-v0.9-baseline-diagram-view.json
python src\skills\vllm-model-architecture-diagram\scripts\layout_diagram.py outputs\hy-v3-v0.9-baseline-diagram-view.json --output outputs\hy-v3-v0.9-baseline-layout-plan.json
python src\skills\vllm-model-architecture-diagram\scripts\render_drawio.py outputs\hy-v3-v0.9-baseline-diagram-view.json --layout-plan outputs\hy-v3-v0.9-baseline-layout-plan.json --output outputs\hy-v3-v0.9-baseline-architecture.drawio
python src\skills\vllm-model-architecture-diagram\scripts\validate_visual_layout.py outputs\hy-v3-v0.9-reviewed-architecture-ir.json outputs\hy-v3-v0.9-baseline-architecture.drawio --metrics-output outputs\hy-v3-v0.9-baseline-layout-metrics.json
python src\skills\vllm-model-architecture-diagram\scripts\build_visual_review.py outputs\hy-v3-v0.9-reviewed-architecture-ir.json outputs\hy-v3-v0.9-baseline-diagram-view.json outputs\hy-v3-v0.9-baseline-layout-plan.json outputs\hy-v3-v0.9-baseline-layout-metrics.json outputs\hy-v3-v0.9-baseline-architecture.drawio --review-output outputs\hy-v3-v0.9-visual-review.json --patch-output outputs\hy-v3-v0.9-diagram-view.patch.json
python src\skills\vllm-model-architecture-diagram\scripts\validate_visual_review.py outputs\hy-v3-v0.9-reviewed-architecture-ir.json outputs\hy-v3-v0.9-baseline-diagram-view.json outputs\hy-v3-v0.9-visual-review.json outputs\hy-v3-v0.9-diagram-view.patch.json
python src\skills\vllm-model-architecture-diagram\scripts\apply_view_patch.py outputs\hy-v3-v0.9-baseline-diagram-view.json outputs\hy-v3-v0.9-diagram-view.patch.json --output outputs\hy-v3-v0.9-reviewed-diagram-view.json
python src\skills\vllm-model-architecture-diagram\scripts\layout_diagram.py outputs\hy-v3-v0.9-reviewed-diagram-view.json --output outputs\hy-v3-v0.9-layout-plan.json
python src\skills\vllm-model-architecture-diagram\scripts\render_drawio.py outputs\hy-v3-v0.9-reviewed-diagram-view.json --layout-plan outputs\hy-v3-v0.9-layout-plan.json --output outputs\hy-v3-v0.9-architecture.drawio
python src\skills\vllm-model-architecture-diagram\scripts\validate_drawio.py outputs\hy-v3-v0.9-reviewed-architecture-ir.json outputs\hy-v3-v0.9-architecture.drawio --view outputs\hy-v3-v0.9-reviewed-diagram-view.json --layout-plan outputs\hy-v3-v0.9-layout-plan.json
python src\skills\vllm-model-architecture-diagram\scripts\validate_visual_layout.py outputs\hy-v3-v0.9-reviewed-architecture-ir.json outputs\hy-v3-v0.9-architecture.drawio --metrics-output outputs\hy-v3-v0.9-layout-metrics.json
```

Then use Draw.io MCP to open `outputs/hy-v3-v0.9-architecture.drawio` and
export the seven pages to PNG/SVG. Build the mentor package after exports exist.
