"""Default v2.0 CLI for the Agent-native vLLM Architecture Skill."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from vllm_architecture_agent import __version__


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_SCRIPTS = REPO_ROOT / "src" / "skills" / "vllm-model-architecture-diagram" / "scripts"


def _load_script(name: str) -> Any:
    path = SKILL_SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"vllm_arch_agent_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_table(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        print(
            f"{entry['architecture']}\t{entry['module']}\t"
            f"{entry['class_name']}\t{entry['category']}"
        )


def cmd_list_models(args: argparse.Namespace) -> int:
    resolver = _load_script("resolve_model_target")
    entries, warnings = resolver.list_registry_models(args.repo_root)
    if args.filter:
        needle = args.filter.lower()
        entries = [
            entry
            for entry in entries
            if needle in entry["architecture"].lower()
            or needle in entry["module"].lower()
            or needle in entry["class_name"].lower()
        ]
    payload = {"models": entries, "warnings": warnings}
    if args.output:
        _write_json(args.output, payload)
    elif args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_table(entries)
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    resolver = _load_script("resolve_model_target")
    collector = _load_script("collect_source_context")
    resolution = resolver.resolve_model_target(
        args.repo_root,
        input_path=args.input,
        architecture=args.architecture,
    )
    context = collector.collect_source_context(args.repo_root, resolution)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or args.architecture
    if not model_name:
        target = context.get("target", {}).get("target_file") or "model"
        model_name = Path(str(target)).stem
    _write_json(args.outputs_dir / "source-context.json", context)
    _write_json(args.outputs_dir / "architecture-plan.template.json", collector.plan_template(context, model_name))
    _write_json(args.outputs_dir / "evidence.template.json", collector.evidence_template(context))
    print(f"Prepared Source Context in {args.outputs_dir}")
    return 0 if context.get("classification", {}).get("status") != "unsupported" else 2


def cmd_validate(args: argparse.Namespace) -> int:
    failed = False
    context = None
    plan = None
    evidence = None
    if args.context and args.context.exists():
        context = json.loads(args.context.read_text(encoding="utf-8"))
    if args.plan and args.plan.exists():
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.evidence and args.evidence.exists():
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        evidence_mod = _load_script("validate_evidence")
        errors, warnings, summary = evidence_mod.validate_evidence(evidence, context=context, plan=plan)
        for warning in warnings:
            print(f"WARNING: {warning}")
        if errors:
            failed = True
            for error in errors:
                print(f"ERROR: {error}")
        else:
            print(f"Evidence validation passed: {json.dumps(summary, sort_keys=True)}")
    if args.plan and args.plan.exists():
        plan_mod = _load_script("validate_architecture_plan")
        errors, warnings = plan_mod.validate_plan(plan, evidence=evidence, context=context)
        for warning in warnings:
            print(f"WARNING: {warning}")
        if errors:
            failed = True
            for error in errors:
                print(f"ERROR: {error}")
        else:
            print("Architecture plan validation passed")
    if args.drawio and args.drawio.exists():
        drawio_mod = _load_script("validate_drawio")
        errors = drawio_mod.validate_drawio(args.drawio, plan=plan, images=args.image or [], images_dir=args.images_dir)
        if errors:
            failed = True
            for error in errors:
                print(f"ERROR: {error}")
        else:
            print("Draw.io validation passed")
    if not any((args.context, args.plan, args.evidence, args.drawio)):
        print("Nothing to validate; provide at least one input path.", file=sys.stderr)
        return 2
    return 1 if failed else 0


def cmd_scan(args: argparse.Namespace) -> int:
    scanner = _load_script("scan_models_directory")
    report = scanner.scan_models_directory(args.repo_root)
    _write_json(args.output, report)
    print(f"Compatibility report written to {args.output}")
    return 0 if report["summary"]["failures"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-arch", description="Agent-native vLLM architecture helper CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_models = subparsers.add_parser("list-models", help="Statically list registry models")
    list_models.add_argument("--repo-root", required=True, type=Path)
    list_models.add_argument("--filter", default="")
    list_models.add_argument("--json", action="store_true")
    list_models.add_argument("--output", type=Path)
    list_models.set_defaults(func=cmd_list_models)

    prepare = subparsers.add_parser("prepare", help="Resolve a target and collect Source Context")
    prepare.add_argument("--repo-root", required=True, type=Path)
    target = prepare.add_mutually_exclusive_group(required=True)
    target.add_argument("--input", type=Path)
    target.add_argument("--architecture")
    prepare.add_argument("--outputs-dir", required=True, type=Path)
    prepare.add_argument("--model-name")
    prepare.set_defaults(func=cmd_prepare)

    validate = subparsers.add_parser("validate", help="Validate context, plan, evidence, and Draw.io outputs")
    validate.add_argument("--context", type=Path)
    validate.add_argument("--plan", type=Path)
    validate.add_argument("--evidence", type=Path)
    validate.add_argument("--drawio", type=Path)
    validate.add_argument("--image", action="append", type=Path, default=[])
    validate.add_argument("--images-dir", type=Path)
    validate.set_defaults(func=cmd_validate)

    scan = subparsers.add_parser("scan", help="Scan vllm/model_executor/models compatibility")
    scan.add_argument("--repo-root", required=True, type=Path)
    scan.add_argument("--output", required=True, type=Path)
    scan.set_defaults(func=cmd_scan)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
