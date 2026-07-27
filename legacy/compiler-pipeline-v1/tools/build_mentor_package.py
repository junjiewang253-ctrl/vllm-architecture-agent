#!/usr/bin/env python3
"""Build a mentor submission package from v1.2 architect artifacts."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PAGE_FILES = {
    "model-overview.png": "{model}-model-overview.png",
    "decoder-block.png": "{model}-decoder-block.png",
    "attention-adaptation.png": "{model}-attention-adaptation.png",
    "moe-execution.png": "{model}-moe-execution.png",
    "checkpoint-loading.png": "{model}-checkpoint-loading.png",
    "parallel-strategies.png": "{model}-parallel-strategies.png",
    "adapter-boundary.png": "{model}-adapter-boundary.png",
}

ANALYSIS_FILES = {
    "source-analysis.json": "{model}-source-analysis.json",
    "semantic-inventory.json": "{model}-semantic-inventory.json",
    "source-fact-graph.json": "{model}-source-fact-graph.json",
    "architecture-concept.json": "{model}-architecture-concept.json",
    "boundary-report.json": "{model}-boundary-report.json",
    "architect-brief.json": "{model}-architect-brief.json",
    "architecture-design.json": "{model}-architecture-design.json",
    "architecture-view.json": "{model}-architecture-view.json",
    "layout-plan.json": "{model}-layout-plan.json",
    "layout-metrics.json": "{model}-layout-metrics.json",
    "architecture-report.md": "{model}-architecture-report.md",
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
    for package_name, output_template in PAGE_FILES.items():
        _copy_optional(outputs_dir / output_template.format(model=model_name), destination / "diagrams" / package_name)
        _copy_optional(outputs_dir / output_template.format(model=model_name).replace(".png", ".svg"), destination / "diagrams" / package_name.replace(".png", ".svg"))
    for package_name, output_template in ANALYSIS_FILES.items():
        _copy_required(outputs_dir / output_template.format(model=model_name), destination / "analysis" / package_name)
    _copy_required(Path("samples/hy_v3.py"), destination / "source" / "hy_v3.py")

    summary = destination / "README.md"
    summary.write_text(
        "\n".join(
            [
                "# vLLM Architecture Agent Mentor Package",
                "",
                "This package contains v1.2 Agent Architect outputs for HY V3.",
                "",
                "## Contents",
                "",
                "- `diagrams/architecture.drawio`: seven-page Draw.io file",
                "- `diagrams/*.png`: exported page images when Draw.io MCP export succeeded",
                "- `analysis/`: source facts, concepts, Agent Design, View, layout metrics and report",
                "- `source/hy_v3.py`: analyzed source sample",
                "- `reproduction/commands.ps1`: PowerShell reproduction commands",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (destination / "executive-summary.md").write_text(
        "# Executive Summary\n\n"
        "v1.2 separates deterministic evidence extraction from Agent-authored architecture design. "
        "The final diagrams are rendered from Architecture View Graph, not from concept cards or hand-written XML.\n",
        encoding="utf-8",
    )
    commands = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"vllm-arch prepare --input samples\\hy_v3.py --model-name {model_name} --outputs-dir outputs",
            "# Codex Agent writes outputs/<model>-architecture-design.json",
            f"vllm-arch finalize --design outputs\\{model_name}-architecture-design.json --model-name {model_name} --outputs-dir outputs --source-file samples\\hy_v3.py",
            f"python tools\\build_mentor_package.py --model-name {model_name} --outputs-dir outputs --destination outputs\\mentor-package-v1.2",
        ]
    )
    (destination / "reproduction" / "commands.ps1").write_text(commands + "\n", encoding="utf-8")
    (destination / "reproduction" / "commands.sh").write_text(
        commands.replace("$ErrorActionPreference = 'Stop'", "set -euo pipefail").replace("\\", "/") + "\n",
        encoding="utf-8",
    )
    (destination / "reproduction" / "config.toml.example").write_text(
        f'mode = "architect"\nmodel_name = "{model_name}"\n',
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
