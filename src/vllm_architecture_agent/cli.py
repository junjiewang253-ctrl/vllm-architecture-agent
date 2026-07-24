"""CLI for the vLLM Architecture Agent reviewed and deterministic pipelines."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "src" / "skills" / "vllm-model-architecture-diagram"
SCRIPTS = SKILL / "scripts"


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=str(ROOT), text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}: {' '.join(command)}")


def _model_name(input_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    return input_path.stem.replace("_", "-")


def _paths(outputs_dir: Path, model: str) -> dict[str, Path]:
    return {
        "source_analysis": outputs_dir / f"{model}-source-analysis.json",
        "inventory": outputs_dir / f"{model}-semantic-inventory.json",
        "baseline_ir": outputs_dir / f"{model}-baseline-architecture-ir.json",
        "baseline_coverage": outputs_dir / f"{model}-baseline-semantic-coverage.json",
        "semantic_review": outputs_dir / f"{model}-semantic-review.json",
        "ir_patch": outputs_dir / f"{model}-architecture-ir.patch.json",
        "reviewed_ir": outputs_dir / f"{model}-reviewed-architecture-ir.json",
        "coverage": outputs_dir / f"{model}-semantic-coverage.json",
        "baseline_view": outputs_dir / f"{model}-baseline-diagram-view.json",
        "baseline_layout": outputs_dir / f"{model}-baseline-layout-plan.json",
        "baseline_drawio": outputs_dir / f"{model}-baseline-architecture.drawio",
        "baseline_metrics": outputs_dir / f"{model}-baseline-layout-metrics.json",
        "visual_review": outputs_dir / f"{model}-visual-review.json",
        "view_patch": outputs_dir / f"{model}-diagram-view.patch.json",
        "reviewed_view": outputs_dir / f"{model}-reviewed-diagram-view.json",
        "layout": outputs_dir / f"{model}-layout-plan.json",
        "metrics": outputs_dir / f"{model}-layout-metrics.json",
        "drawio": outputs_dir / f"{model}-architecture.drawio",
        "lock": outputs_dir / f"{model}-review-lock.json",
        "source_fact_graph": outputs_dir / f"{model}-source-fact-graph.json",
        "architecture_design": outputs_dir / f"{model}-architecture-design.json",
        "architecture_concept": outputs_dir / f"{model}-architecture-concept.json",
        "architecture_view": outputs_dir / f"{model}-architecture-view.json",
        "boundary_report": outputs_dir / f"{model}-boundary-report.json",
        "mentor_report": outputs_dir / f"{model}-mentor-report.md",
        "architecture_report": outputs_dir / f"{model}-architecture-report.md",
    }


def _copy_if_needed(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _use_review_pair(
    *,
    review_name: str,
    review_flag: str,
    patch_flag: str,
    explicit_review: Path | None,
    explicit_patch: Path | None,
    expected_review: Path,
    expected_patch: Path,
) -> None:
    """Use VSCode-Codex-produced Review/Patch artifacts for reviewed mode."""

    if explicit_review or explicit_patch:
        if not explicit_review or not explicit_patch:
            raise RuntimeError(f"{review_name} review and patch paths must be provided together")
        if not explicit_review.exists():
            raise RuntimeError(f"{review_name} review file does not exist: {explicit_review}")
        if not explicit_patch.exists():
            raise RuntimeError(f"{review_name} patch file does not exist: {explicit_patch}")
        _copy_if_needed(explicit_review, expected_review)
        _copy_if_needed(explicit_patch, expected_patch)
        return

    if expected_review.exists() and expected_patch.exists():
        return

    raise RuntimeError(
        f"reviewed mode requires VSCode Codex to write {expected_review} and {expected_patch} "
        f"before this stage, or pass explicit {review_flag}/{patch_flag} paths. "
        "Use --mode deterministic when no Agent review is needed."
    )


def run_pipeline(args: argparse.Namespace) -> None:
    input_path = args.input
    if not input_path.exists() or input_path.suffix != ".py":
        raise RuntimeError(f"input must be an existing .py file: {input_path}")
    model = _model_name(input_path, args.model_name)
    paths = _paths(args.outputs_dir, model)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    _run([py, str(SCRIPTS / "extract_architecture.py"), str(input_path), "--output", str(paths["source_analysis"])])
    _run([py, str(SCRIPTS / "build_semantic_inventory.py"), str(paths["source_analysis"]), "--output", str(paths["inventory"])])
    if args.mode == "architect":
        _run([py, str(SCRIPTS / "build_source_fact_graph.py"), str(paths["source_analysis"]), "--output", str(paths["source_fact_graph"])])
        _run([
            py,
            str(SCRIPTS / "run_architect_review.py"),
            str(paths["source_fact_graph"]),
            "--source-analysis",
            str(paths["source_analysis"]),
            "--semantic-inventory",
            str(paths["inventory"]),
            "--output",
            str(paths["architecture_concept"]),
        ])
        _run([py, str(SCRIPTS / "run_view_architect.py"), str(paths["architecture_concept"]), str(paths["source_fact_graph"]), "--output", str(paths["architecture_view"])])
        _run([py, str(SCRIPTS / "build_boundary_report.py"), str(paths["architecture_concept"]), "--output", str(paths["boundary_report"])])
        _run([
            py,
            str(SCRIPTS / "validate_architecture_quality.py"),
            str(paths["source_fact_graph"]),
            str(paths["architecture_concept"]),
            str(paths["architecture_view"]),
            "--boundary-report",
            str(paths["boundary_report"]),
        ])
        _run([py, str(SCRIPTS / "validate_architecture_view.py"), str(paths["architecture_view"]), "--architecture-concept", str(paths["architecture_concept"]), "--source-fact-graph", str(paths["source_fact_graph"])])
        _run([py, str(SCRIPTS / "apply_view_layout.py"), str(paths["architecture_view"]), "--output", str(paths["layout"])])
        _run([py, str(SCRIPTS / "render_drawio.py"), str(paths["architecture_view"]), "--layout-plan", str(paths["layout"]), "--output", str(paths["drawio"])])
        _run([py, str(SCRIPTS / "build_mentor_report.py"), str(paths["architecture_concept"]), str(paths["architecture_view"]), str(paths["boundary_report"]), "--output", str(paths["architecture_report"])])
        print(f"vllm-arch architect run completed: {paths['drawio']}")
        return
    _run([py, str(SCRIPTS / "build_architecture_ir.py"), str(paths["source_analysis"]), "--output", str(paths["baseline_ir"])])
    _run([py, str(SCRIPTS / "validate_architecture_ir.py"), str(paths["baseline_ir"])])
    _run([
        py,
        str(SCRIPTS / "validate_semantic_coverage.py"),
        str(paths["source_analysis"]),
        str(paths["inventory"]),
        str(paths["baseline_ir"]),
        "--output",
        str(paths["baseline_coverage"]),
    ])

    if args.mode == "reviewed":
        if args.mock_semantic_review:
            semantic_cmd = [
                py,
                str(SCRIPTS / "run_semantic_review.py"),
                str(paths["source_analysis"]),
                str(paths["inventory"]),
                str(paths["baseline_ir"]),
                str(paths["baseline_coverage"]),
                "--source-file",
                str(input_path),
                "--review-output",
                str(paths["semantic_review"]),
                "--patch-output",
                str(paths["ir_patch"]),
            ]
            semantic_cmd.extend(["--mock-response", str(args.mock_semantic_review)])
            _run(semantic_cmd)
        else:
            _use_review_pair(
                review_name="semantic",
                review_flag="--semantic-review",
                patch_flag="--ir-patch",
                explicit_review=args.semantic_review,
                explicit_patch=args.ir_patch,
                expected_review=paths["semantic_review"],
                expected_patch=paths["ir_patch"],
            )
    else:
        _run([
            py,
            str(SCRIPTS / "build_baseline_review_template.py"),
            "semantic",
            str(paths["source_analysis"]),
            str(paths["inventory"]),
            str(paths["baseline_ir"]),
            str(paths["baseline_coverage"]),
            "--source-file",
            str(input_path),
            "--review-output",
            str(paths["semantic_review"]),
            "--patch-output",
            str(paths["ir_patch"]),
        ])

    _run([
        py,
        str(SCRIPTS / "validate_semantic_review.py"),
        str(paths["source_analysis"]),
        str(paths["inventory"]),
        str(paths["baseline_ir"]),
        str(paths["semantic_review"]),
        str(paths["ir_patch"]),
    ])
    _run([py, str(SCRIPTS / "apply_ir_patch.py"), str(paths["baseline_ir"]), str(paths["ir_patch"]), "--output", str(paths["reviewed_ir"])])
    _run([py, str(SCRIPTS / "validate_architecture_ir.py"), str(paths["reviewed_ir"])])
    _run([
        py,
        str(SCRIPTS / "validate_semantic_coverage.py"),
        str(paths["source_analysis"]),
        str(paths["inventory"]),
        str(paths["reviewed_ir"]),
        "--semantic-review",
        str(paths["semantic_review"]),
        "--output",
        str(paths["coverage"]),
    ])

    _run([py, str(SCRIPTS / "build_diagram_view.py"), str(paths["reviewed_ir"]), "--output", str(paths["baseline_view"])])
    _run([py, str(SCRIPTS / "layout_diagram.py"), str(paths["baseline_view"]), "--output", str(paths["baseline_layout"])])
    _run([
        py,
        str(SCRIPTS / "render_drawio.py"),
        str(paths["baseline_view"]),
        "--layout-plan",
        str(paths["baseline_layout"]),
        "--output",
        str(paths["baseline_drawio"]),
    ])
    _run([
        py,
        str(SCRIPTS / "validate_visual_layout.py"),
        str(paths["reviewed_ir"]),
        str(paths["baseline_drawio"]),
        "--metrics-output",
        str(paths["baseline_metrics"]),
    ])

    if args.mode == "reviewed":
        if args.mock_visual_review:
            visual_cmd = [
                py,
                str(SCRIPTS / "run_visual_review.py"),
                str(paths["reviewed_ir"]),
                str(paths["baseline_view"]),
                str(paths["baseline_metrics"]),
                "--review-output",
                str(paths["visual_review"]),
                "--patch-output",
                str(paths["view_patch"]),
            ]
            visual_cmd.extend(["--mock-response", str(args.mock_visual_review)])
            _run(visual_cmd)
        else:
            _use_review_pair(
                review_name="visual",
                review_flag="--visual-review",
                patch_flag="--view-patch",
                explicit_review=args.visual_review,
                explicit_patch=args.view_patch,
                expected_review=paths["visual_review"],
                expected_patch=paths["view_patch"],
            )
    else:
        _run([
            py,
            str(SCRIPTS / "build_baseline_review_template.py"),
            "visual",
            str(paths["reviewed_ir"]),
            str(paths["baseline_view"]),
            str(paths["baseline_layout"]),
            str(paths["baseline_metrics"]),
            str(paths["baseline_drawio"]),
            "--review-output",
            str(paths["visual_review"]),
            "--patch-output",
            str(paths["view_patch"]),
        ])

    _run([py, str(SCRIPTS / "validate_visual_review.py"), str(paths["reviewed_ir"]), str(paths["baseline_view"]), str(paths["visual_review"]), str(paths["view_patch"])])
    _run([py, str(SCRIPTS / "apply_view_patch.py"), str(paths["baseline_view"]), str(paths["view_patch"]), "--output", str(paths["reviewed_view"])])
    _run([py, str(SCRIPTS / "validate_diagram_view.py"), str(paths["reviewed_ir"]), str(paths["reviewed_view"])])
    _run([py, str(SCRIPTS / "layout_diagram.py"), str(paths["reviewed_view"]), "--output", str(paths["layout"])])
    _run([py, str(SCRIPTS / "render_drawio.py"), str(paths["reviewed_view"]), "--layout-plan", str(paths["layout"]), "--output", str(paths["drawio"])])
    _run([py, str(SCRIPTS / "validate_drawio.py"), str(paths["reviewed_ir"]), str(paths["drawio"]), "--view", str(paths["reviewed_view"]), "--layout-plan", str(paths["layout"])])
    _run([py, str(SCRIPTS / "validate_visual_layout.py"), str(paths["reviewed_ir"]), str(paths["drawio"]), "--metrics-output", str(paths["metrics"])])
    _run([
        py,
        str(SCRIPTS / "build_review_lock.py"),
        "--source",
        str(input_path),
        "--source-analysis",
        str(paths["source_analysis"]),
        "--semantic-inventory",
        str(paths["inventory"]),
        "--baseline-ir",
        str(paths["baseline_ir"]),
        "--semantic-review",
        str(paths["semantic_review"]),
        "--ir-patch",
        str(paths["ir_patch"]),
        "--reviewed-ir",
        str(paths["reviewed_ir"]),
        "--visual-review",
        str(paths["visual_review"]),
        "--view-patch",
        str(paths["view_patch"]),
        "--reviewed-view",
        str(paths["reviewed_view"]),
        "--output",
        str(paths["lock"]),
    ])
    print(f"vllm-arch run completed: {paths['drawio']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vllm-arch")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--input", type=Path, default=Path("samples/hy_v3.py"))
    run.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    run.add_argument("--model-name")
    run.add_argument("--mode", choices=["deterministic", "reviewed", "architect"], default="reviewed")
    run.add_argument("--semantic-review", type=Path)
    run.add_argument("--ir-patch", type=Path)
    run.add_argument("--visual-review", type=Path)
    run.add_argument("--view-patch", type=Path)
    run.add_argument("--mock-semantic-review", type=Path)
    run.add_argument("--mock-visual-review", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "run":
            run_pipeline(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
