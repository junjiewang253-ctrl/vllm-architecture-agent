"""Shared helpers for Codex review runners."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} root must be an object")
    return data


def stable_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first top-level JSON object from plain text or markdown."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        candidates: list[str] = []
        in_fence = False
        current: list[str] = []
        for line in lines:
            if line.strip().startswith("```"):
                if in_fence:
                    candidates.append("\n".join(current))
                    current = []
                    in_fence = False
                else:
                    in_fence = True
                continue
            if in_fence:
                current.append(line)
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    if start < 0:
        raise ValueError("Codex response does not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(stripped[start:index + 1])
                if not isinstance(data, dict):
                    raise ValueError("extracted JSON must be an object")
                return data
    raise ValueError("Codex response contains an unterminated JSON object")


def run_codex(prompt: str, *, codex_bin: str, cwd: Path, timeout_seconds: int) -> str:
    output_path = cwd / "outputs" / ".codex-review-last-message.json.tmp"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        codex_bin,
        "exec",
        "--cd",
        str(cwd),
        "--sandbox",
        "read-only",
        "--output-last-message",
        str(output_path),
        prompt,
    ]
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        raise ValueError(
            "Codex review command failed "
            f"(exit {completed.returncode}): {stderr or stdout}"
        )
    if output_path.exists():
        return output_path.read_text(encoding="utf-8")
    return completed.stdout


def read_codex_or_mock(
    prompt: str,
    *,
    mock_response: Path | None,
    codex_bin: str,
    cwd: Path,
    timeout_seconds: int,
) -> str:
    if mock_response is not None:
        return mock_response.read_text(encoding="utf-8")
    return run_codex(prompt, codex_bin=codex_bin, cwd=cwd, timeout_seconds=timeout_seconds)


def write_failure(path: Path, message: str, prompt: str | None = None) -> None:
    payload: dict[str, Any] = {"status": "failed", "error": message}
    if prompt is not None:
        payload["prompt"] = prompt
    stable_write_json(path, payload)
