"""Text/string helpers for SubBakeAgent.

Extracted from ``_core.py``. Pure and near-pure functions for reference
extraction, command splitting, and text manipulation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._core import SubBakeAgent

from .trace import REFERENCE_RE


def extract_references(agent: SubBakeAgent, line: str) -> list[Path]:
    """Extract file-path references (@path or bare path) from a line."""
    paths: list[Path] = []
    for match in REFERENCE_RE.finditer(line):
        raw = next(group for group in match.groups() if group is not None)
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = agent.cwd / path
        paths.append(path)
    return paths


def remove_references(line: str) -> str:
    """Strip all file-path references from a line."""
    return REFERENCE_RE.sub("", line)


def split_command(line: str) -> tuple[str | None, str]:
    """Split a line into (command, rest). Returns (None, line) if not a command."""
    if not line.startswith("/"):
        return None, line
    command, _, rest = line.partition(" ")
    return command.lower(), rest


def content_after_references(line: str) -> str:
    """Return the line content after stripping references and action verbs."""
    cleaned = remove_references(line).strip()
    cleaned = re.sub(r"^(创建|新建|追加|替换)\s*", "", cleaned).strip()
    cleaned = re.sub(r"^(create|append|replace)\b", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def search_pattern_from_text(line: str) -> str:
    """Extract a search pattern from natural language, stripping references and markers."""
    cleaned = remove_references(line).strip()
    for marker in ("搜索", "查找", "search", "find"):
        match = re.search(rf"\b{re.escape(marker)}\b|{re.escape(marker)}", cleaned, flags=re.IGNORECASE)
        if match is not None:
            return cleaned[match.end():].strip(" ：:=,，。")
    return cleaned
