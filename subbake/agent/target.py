"""Translation target resolution for SubBakeAgent.

Extracted from ``_core.py``. Resolves source paths, infers languages,
and enriches translation arguments for the target subtitle file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._core import SubBakeAgent

from subbake.editing import is_generated_subtitle
from subbake.series import SUPPORTED_SUBTITLE_SUFFIXES, discover_series_files


def source_path_for_translation_reference(path: Path) -> Path | None:
    """Determine the source subtitle path from a reference (handles generated files)."""
    if is_generated_subtitle(path):
        for marker in (".translated.", ".bilingual."):
            if marker in path.name:
                return path.with_name(path.name.replace(marker, ".", 1))
    if path.suffix.lower() in SUPPORTED_SUBTITLE_SUFFIXES:
        return path
    return None


def infer_target_from_user_language(text: str) -> str | None:
    """Infer the target translation language from the user's natural language query.

    When the user doesn't explicitly specify a target language, detect the
    language they are writing in and use it as the translation target.

    Detection priority: Japanese kana > Korean hangul > Chinese CJK > Latin.
    """
    stripped = text.strip()
    if not stripped:
        return None
    # Japanese (Hiragana + Katakana) — checked first because Japanese text also
    # uses CJK ideographs shared with Chinese
    if re.search(r"[぀-ゟ]", stripped) or re.search(r"[゠-ヿ]", stripped):
        return "Japanese"
    # Korean (Hangul Syllables) — checked before CJK because Korean text also
    # uses hanja (CJK ideographs)
    if re.search(r"[가-힯]", stripped):
        return "Korean"
    # Chinese characters (CJK Unified Ideographs)
    if re.search(r"[一-鿿]", stripped):
        return "Chinese"
    # Latin script → default to English
    if re.search(r"[a-zA-Z]", stripped):
        return "English"
    return None


def translation_arguments_for_target(
    agent: SubBakeAgent,
    *,
    path: Path,
    arguments: dict[str, Any],
    series: bool,
    user_message: str = "",
) -> dict[str, Any]:
    """Enrich translation arguments with inferred target language."""
    enriched = dict(arguments)
    if "target_language" not in enriched:
        source_language = str(enriched.get("source_language") or "").strip()
        if not source_language:
            source_language = infer_source_language_for_target(agent, path, series=series) or ""
        target_language = target_language_for_bilingual_pair_from_arguments(
            enriched,
            source_language=source_language,
        )
        if target_language is not None:
            enriched["target_language"] = target_language

        # If still no target language, infer from the user's query language
        if "target_language" not in enriched:
            inferred = infer_target_from_user_language(user_message)
            if inferred is not None:
                enriched["target_language"] = inferred
    return enriched


def target_language_for_bilingual_pair_from_arguments(
    arguments: dict[str, Any],
    *,
    source_language: str,
) -> str | None:
    """Determine the target language for a bilingual pair."""
    if not bool(arguments.get("bilingual")):
        return None
    if source_language == "Chinese":
        return "English"
    if source_language == "English":
        return "Chinese"
    return None


def infer_source_language_for_target(agent: SubBakeAgent, path: Path, *, series: bool) -> str | None:
    """Infer the source language by reading the first subtitle file."""
    candidate = first_source_file(agent, path, series=series)
    if candidate is None:
        return None
    try:
        text = candidate.read_text(encoding="utf-8", errors="ignore")[:12000]
    except OSError:
        return None
    if re.search(r"[一-鿿]", text):
        return "Chinese"
    if re.search(r"[A-Za-z]", text):
        return "English"
    return None


def first_source_file(agent: SubBakeAgent, path: Path, *, series: bool) -> Path | None:
    """Get the first source subtitle file for a given path."""
    if not series:
        return path if path.exists() else None
    suffixes = None
    try:
        files = discover_series_files(path, recursive=False, suffixes=suffixes)
    except Exception:
        return None
    return files[0] if files else None
