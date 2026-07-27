#!/usr/bin/env python3
"""Deterministic text measurement approximations for diagram layout."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextBox:
    width: float
    height: float
    line_count: int


def _wrap_lines(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word[:max_chars]
    if current:
        lines.append(current)
    return lines


def measure_node_text(
    title: str,
    subtitle: str = "",
    *,
    max_width: float = 220.0,
    title_font: float = 13.0,
    subtitle_font: float = 11.0,
    badges: int = 0,
    padding_x: float = 14.0,
    padding_y: float = 10.0,
) -> TextBox:
    """Approximate a node's minimum text box.

    The function is deterministic and intentionally conservative. It separates
    display text from exact source expressions, which should remain in evidence
    rather than on the diagram.
    """

    max_chars = max(8, int((max_width - padding_x * 2) / 6.4))
    title_lines = _wrap_lines(title, max_chars)[:2]
    subtitle_lines = _wrap_lines(subtitle, max_chars)[:2] if subtitle else []
    line_width = max([len(line) * 6.4 for line in title_lines + subtitle_lines] or [0.0])
    badge_height = 20.0 if badges else 0.0
    height = padding_y * 2 + len(title_lines) * (title_font + 5.0) + len(subtitle_lines) * (subtitle_font + 4.0) + badge_height
    width = min(max_width, max(96.0, line_width + padding_x * 2))
    return TextBox(width=round(width, 2), height=round(height, 2), line_count=len(title_lines) + len(subtitle_lines))


def edge_label_is_short(label: str, *, max_chars: int = 18) -> bool:
    return len(label.strip()) <= max_chars
