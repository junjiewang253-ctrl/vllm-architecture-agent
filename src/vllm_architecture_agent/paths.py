"""Portable path helpers for vLLM architecture artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def infer_repo_root(
    *,
    explicit: Path | None = None,
    start: Path | None = None,
    context_path: Path | None = None,
) -> Path:
    """Infer the workspace root without relying on machine-specific paths."""

    if explicit is not None:
        return explicit.resolve()
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start if start.is_dir() else start.parent)
    if context_path is not None:
        candidates.append(context_path if context_path.is_dir() else context_path.parent)
    candidates.append(Path.cwd())
    for candidate in candidates:
        for parent in [candidate.resolve(), *candidate.resolve().parents]:
            if (
                (parent / "pyproject.toml").exists()
                and (parent / "src" / "skills" / "vllm-model-architecture-diagram" / "SKILL.md").exists()
            ):
                return parent
    return candidates[0].resolve()


def to_artifact_path(path: str | Path | None, repo_root: Path) -> str | None:
    """Store repo-internal paths as portable POSIX-style relative paths."""

    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        relative = candidate.resolve().relative_to(repo_root.resolve())
        return relative.as_posix() or "."
    except ValueError:
        return str(candidate.resolve())


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    """Resolve an artifact path relative to repo_root unless already absolute."""

    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def scrub_repo_absolute_strings(value: Any, repo_root: Path) -> Any:
    """Remove machine-specific repo-root prefixes from nested JSON payloads."""

    if isinstance(value, dict):
        return {key: scrub_repo_absolute_strings(item, repo_root) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_repo_absolute_strings(item, repo_root) for item in value]
    if not isinstance(value, str):
        return value
    root = repo_root.resolve()
    normalized = value.replace("\\", "/")
    root_forward = str(root).replace("\\", "/")
    if root_forward in normalized:
        normalized = normalized.replace(root_forward + "/", "")
        normalized = normalized.replace(root_forward, ".")
        return normalized
    return value


def portable_resolution(resolution: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Return a copy of a resolver payload suitable for JSON artifacts."""

    payload = dict(resolution)
    for key in ("repo_root", "models_root", "target_file"):
        if payload.get(key):
            payload[key] = to_artifact_path(payload[key], repo_root)
    payload = scrub_repo_absolute_strings(payload, repo_root)
    return payload
