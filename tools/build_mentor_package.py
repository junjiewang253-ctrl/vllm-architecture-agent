#!/usr/bin/env python3
"""Build a mentor submission package from reviewed v0.9 artifacts."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PAGE_SLUGS = [
    "overview",
    "decoder-layer",
    "attention",
    "moe",
    "adapter-integration",
    "parallelism",
    "weight-loading",
]

ANALYSIS_FILES = {
    "source-analysis.json": "{model}-source-analysis.json",
    "semantic-inventory.json": "{model}-semantic-inventory.json",
    "baseline-architecture-ir.json": "{model}-baseline-architecture-ir.json",
    "baseline-semantic-coverage.json": "{model}-baseline-semantic-coverage.json",
    "semantic-review.json": "{model}-semantic-review.json",
    "architecture-ir.patch.json": "{model}-architecture-ir.patch.json",
    "reviewed-architecture-ir.json": "{model}-reviewed-architecture-ir.json",
    "semantic-coverage.json": "{model}-semantic-coverage.json",
    "baseline-diagram-view.json": "{model}-baseline-diagram-view.json",
    "visual-review.json": "{model}-visual-review.json",
    "diagram-view.patch.json": "{model}-diagram-view.patch.json",
    "reviewed-diagram-view.json": "{model}-reviewed-diagram-view.json",
    "layout-plan.json": "{model}-layout-plan.json",
    "layout-metrics.json": "{model}-layout-metrics.json",
    "review-lock.json": "{model}-review-lock.json",
}


def _copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise ValueError(f"required package artifact is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_optional(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_package(model_name: str, outputs_dir: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "diagrams").mkdir(parents=True)
    (destination / "analysis").mkdir(parents=True)
    (destination / "source").mkdir(parents=True)
    (destination / "reproduction").mkdir(parents=True)

    _copy_required(outputs_dir / f"{model_name}-architecture.drawio", destination / "diagrams" / "architecture.drawio")
    for slug in PAGE_SLUGS:
        _copy_optional(outputs_dir / f"{model_name}-{slug}.png", destination / "diagrams" / f"{slug}.png")
        _copy_optional(outputs_dir / f"{model_name}-{slug}.svg", destination / "diagrams" / f"{slug}.svg")
    for package_name, output_template in ANALYSIS_FILES.items():
        _copy_required(outputs_dir / output_template.format(model=model_name), destination / "analysis" / package_name)
    _copy_required(Path("samples/hy_v3.py"), destination / "source" / "hy_v3.py")

    for name in ("executive-summary.md", "architecture-walkthrough.md", "validation-and-limitations.md", "reproduction-guide.md"):
        _copy_required(Path("docs/mentor") / name, destination / name)
    _copy_required(Path("docs/mentor/reproduction-guide.md"), destination / "reproduction" / "README.md")

    commands_ps1 = destination / "reproduction" / "commands.ps1"
    commands_sh = destination / "reproduction" / "commands.sh"
    commands_ps1.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "pytest",
                "python src/skills/vllm-model-architecture-diagram/scripts/extract_architecture.py samples/hy_v3.py --output outputs/hy-v3-v0.9-source-analysis.json",
                "python src/skills/vllm-model-architecture-diagram/scripts/build_semantic_inventory.py outputs/hy-v3-v0.9-source-analysis.json --output outputs/hy-v3-v0.9-semantic-inventory.json",
                "python src/skills/vllm-model-architecture-diagram/scripts/build_architecture_ir.py outputs/hy-v3-v0.9-source-analysis.json --output outputs/hy-v3-v0.9-baseline-architecture-ir.json",
                "python src/skills/vllm-model-architecture-diagram/scripts/validate_semantic_coverage.py outputs/hy-v3-v0.9-source-analysis.json outputs/hy-v3-v0.9-semantic-inventory.json outputs/hy-v3-v0.9-reviewed-architecture-ir.json --semantic-review outputs/hy-v3-v0.9-semantic-review.json --output outputs/hy-v3-v0.9-semantic-coverage.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    commands_sh.write_text(commands_ps1.read_text(encoding="utf-8").replace("$ErrorActionPreference = 'Stop'\n", "set -euo pipefail\n"), encoding="utf-8")
    (destination / "reproduction" / "config.toml.example").write_text('mode = "reviewed"\nmodel_name = "hy-v3-v0.9"\n', encoding="utf-8")
    (destination / "README.md").write_text(
        "# vLLM Architecture Agent v0.9 Mentor Package\n\n"
        "This package contains reviewed HY V3 semantic analysis artifacts, deterministic Draw.io diagrams, validation outputs, and reproduction commands.\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mentor package.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_package(args.model_name, args.outputs_dir, args.destination)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote mentor package to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
