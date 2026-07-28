"""Build the v2.1.1 mentor delivery zip.

The package intentionally excludes the archived legacy compiler pipeline and
local generated output directories. It contains enough source, tests, examples
and docs for a mentor to install, validate and inspect the current Skill.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


DEFAULT_ZIP = Path("dist/vllm-architecture-agent-v2.1.1-mentor.zip")

INCLUDE_PATHS = (
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "src/vllm_architecture_agent",
    "src/skills/vllm-model-architecture-diagram",
    "integrations",
    "tools/setup-codex-dev.ps1",
    "samples/hy_v3.py",
    "samples/simple_model.py",
    "examples/hy_v3",
    "docs/mentor",
    "docs/development/v2.1.1-mentor-delivery-report.md",
    "tests/fixtures",
    "tests/unit",
)

EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def _should_include(path: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if path.name.endswith(".egg-info"):
        return False
    if path.name == "config.toml.bak":
        return False
    return True


def iter_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in INCLUDE_PATHS:
        path = repo_root / relative
        if not path.exists():
            continue
        if path.is_file():
            if _should_include(path):
                files.append(path)
            continue
        for child in path.rglob("*"):
            if child.is_file() and _should_include(child):
                files.append(child)
    return sorted(files, key=lambda item: item.as_posix())


def build_package(repo_root: Path, destination: Path) -> Path:
    repo_root = repo_root.resolve()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = iter_files(repo_root)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(repo_root).as_posix())
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--destination", type=Path, default=DEFAULT_ZIP)
    args = parser.parse_args(argv)

    destination = build_package(args.repo_root, args.destination)
    print(f"Mentor package written: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
