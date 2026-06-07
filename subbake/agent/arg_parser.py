"""Argument parsing from natural language text.

Extracted from agent.py. All functions take explicit parameters,
making them independently testable with zero agent coupling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from subbake.series import SUPPORTED_SUBTITLE_SUFFIXES
from subbake.title_matching import normalize_title_text, title_tokens_from_text as _title_tokens

REFERENCE_RE = re.compile(r"@(?:\"([^\"]+)\"|'([^']+)'|(\S+))")


def language_phrases() -> tuple[tuple[str, str], ...]:
    """Return (phrase, language_name) pairs for language detection."""
    return (
        ("简体中文", "Chinese"), ("中文", "Chinese"), ("汉语", "Chinese"),
        ("chinese", "Chinese"), ("zh-cn", "Chinese"), ("zh", "Chinese"),
        ("繁体中文", "Traditional Chinese"), ("traditional chinese", "Traditional Chinese"),
        ("英文", "English"), ("英语", "English"), ("english", "English"), ("en", "English"),
        ("日文", "Japanese"), ("日语", "Japanese"), ("japanese", "Japanese"), ("ja", "Japanese"),
        ("韩文", "Korean"), ("韩语", "Korean"), ("korean", "Korean"), ("ko", "Korean"),
        ("法文", "French"), ("法语", "French"), ("french", "French"), ("fr", "French"),
        ("德文", "German"), ("德语", "German"), ("german", "German"), ("de", "German"),
        ("西班牙文", "Spanish"), ("西班牙语", "Spanish"), ("spanish", "Spanish"), ("es", "Spanish"),
        ("葡萄牙文", "Portuguese"), ("葡萄牙语", "Portuguese"), ("portuguese", "Portuguese"), ("pt", "Portuguese"),
        ("俄文", "Russian"), ("俄语", "Russian"), ("russian", "Russian"), ("ru", "Russian"),
        ("意大利文", "Italian"), ("意大利语", "Italian"), ("italian", "Italian"), ("it", "Italian"),
    )


def output_format_patterns() -> tuple[str, ...]:
    """Return regex patterns for detecting output format in text."""
    return (
        r"(?:输出|生成|保存|导出|产出).{0,12}\.?(srt|vtt|txt)(?:\s*格式|\s*文件|\s*字幕)?",
        r"\.?(srt|vtt|txt).{0,8}(?:格式|输出)",
        r"(?:output|export|save).{0,12}\.?(srt|vtt|txt)\b",
    )


def resolve_user_path(value: str, *, cwd: Path) -> Path:
    """Resolve a user-provided path, making it absolute relative to *cwd*."""
    if not value:
        raise ValueError("Tool path argument is required.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path


def series_suffixes_from_argument(value: object) -> set[str] | None:
    """Validate and normalise series input suffixes."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        raw_suffixes = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_suffixes = [str(item) for item in value]
    else:
        raise ValueError("Series suffixes must be a string or list of strings.")
    suffixes = {f".{suffix.strip().lower().lstrip('.')}" for suffix in raw_suffixes if suffix.strip()}
    unsupported = suffixes - SUPPORTED_SUBTITLE_SUFFIXES
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported series input suffix: {names}")
    return suffixes or None


def bool_argument(value: object, name: str) -> bool | None:
    """Parse a boolean from a tool argument value."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Tool argument {name} must be a boolean.")


def output_format_from_argument(value: object) -> str | None:
    """Validate and normalise an output-format argument."""
    if value is None or value == "":
        return None
    output_format = str(value).strip().lower().lstrip(".")
    if f".{output_format}" not in SUPPORTED_SUBTITLE_SUFFIXES:
        raise ValueError(f"Unsupported output format: {output_format}")
    return output_format


def bilingual_requested(line: str) -> bool:
    """Check if the user requested bilingual output."""
    lowered = line.casefold()
    return any(word in lowered for word in ("双语", "中英", "bilingual", "dual-language", "dual language"))


def monolingual_requested(line: str) -> bool:
    """Check if the user explicitly requested monolingual output."""
    lowered = line.casefold()
    return any(word in lowered for word in ("不要双语", "非双语", "单语", "only translation", "translation only"))


def source_language_from_text(line: str) -> str | None:
    """Extract source language from natural language text."""
    lowered = line.casefold()
    explicit = re.search(r"(?:source(?: language)?|源语言)\s*[:=：]\s*([a-zA-Z][a-zA-Z_-]*)", line)
    if explicit is not None:
        return explicit.group(1)
    if any(word in lowered for word in ("中文字幕", "中文源字幕", "中文原字幕", "原文中文")):
        return "Chinese"
    if any(word in lowered for word in ("英文字幕", "英语字幕", "英文源字幕", "英文原字幕", "原文英文", "原文英语")):
        return "English"
    for phrase, language in language_phrases():
        if re.search(rf"(?:从|from\s+)\s*{re.escape(phrase)}", lowered):
            return language
    return None


def target_language_for_bilingual_pair(
    line: str,
    *,
    source_language: str | None,
) -> str | None:
    """Infer target language from a bilingual pair mention like '中英'."""
    lowered = line.casefold()
    if any(word in lowered for word in ("中英", "中文英文", "chinese english", "chinese-english")):
        if source_language == "Chinese":
            return "English"
        if source_language == "English":
            return "Chinese"
    if any(word in lowered for word in ("英中", "英文中文", "english chinese", "english-chinese")):
        if source_language == "English":
            return "Chinese"
        if source_language == "Chinese":
            return "English"
    return None


def target_language_from_text(line: str, *, source_language: str | None = None) -> str | None:
    """Extract target language from natural language text."""
    lowered = line.casefold()
    explicit = re.search(r"(?:target(?: language)?|目标语言)\s*[:=：]\s*([a-zA-Z][a-zA-Z_-]*)", line)
    if explicit is not None:
        return explicit.group(1)
    for phrase, language in language_phrases():
        if re.search(rf"(?:翻译|译|translate).{{0,12}}(?:成|到|为|to)\s*{re.escape(phrase)}", lowered):
            return language
        if re.search(rf"(?:译成|翻成)\s*{re.escape(phrase)}", lowered):
            return language
    return target_language_for_bilingual_pair(line, source_language=source_language)


def output_format_from_text(line: str) -> str | None:
    """Extract output format (.srt/.vtt/.txt) from natural language text."""
    lowered = line.casefold()
    for pattern in output_format_patterns():
        match = re.search(pattern, lowered)
        if match is not None:
            return match.group(1)
    return None


def line_without_output_format_phrases(line: str) -> str:
    """Remove output-format phrases from *line*."""
    cleaned = line.casefold()
    for pattern in output_format_patterns():
        cleaned = re.sub(pattern, " ", cleaned)
    return cleaned


def series_suffixes_from_text(line: str) -> set[str] | None:
    """Detect subtitle file suffixes mentioned in *line*."""
    lowered = line_without_output_format_phrases(line).casefold()
    suffixes = {
        suffix
        for suffix in SUPPORTED_SUBTITLE_SUFFIXES
        if re.search(rf"(?<![a-z0-9])\.?{re.escape(suffix.lstrip('.'))}(?![a-z0-9])", lowered)
    }
    return suffixes or None


def title_tokens_from_text(line: str) -> list[str]:
    """Extract title tokens from *line* for fuzzy file-name matching."""
    cleaned = REFERENCE_RE.sub("", line)
    alias_tokens = _title_tokens(cleaned)
    spans = re.findall(r"[A-Za-z0-9][A-Za-z0-9 ._'()-]{2,}", cleaned)
    best = " ".join(alias_tokens)
    for span in spans:
        normalized = normalize_title_text(span)
        tokens = [
            token
            for token in normalized.split()
            if token not in {"the", "a", "an", "to", "of", "and", "or", "can", "could"}
        ]
        if len(tokens) >= 2 and len(normalized) > len(best):
            best = normalized
    return [
        token
        for token in best.split()
        if token not in {"the", "a", "an", "to", "of", "and", "or", "can", "could"}
    ]


def translation_arguments_from_text(line: str) -> dict[str, Any]:
    """Extract translation arguments from natural language *line*."""
    lowered = line.casefold()
    arguments: dict[str, Any] = {}

    suffixes = series_suffixes_from_text(line)
    if suffixes is not None:
        arguments["suffixes"] = sorted(suffixes)
    if any(word in lowered for word in ("递归", "子目录", "recursive", "subdir")):
        arguments["recursive"] = True
    if any(word in lowered for word in ("覆盖", "重新翻译", "overwrite", "retranslate")):
        arguments["overwrite"] = True
    if any(word in lowered for word in ("dry run", "dry-run", "只规划", "预览")):
        arguments["dry_run"] = True
    if any(word in lowered for word in ("快速", "fast mode", "fast")):
        arguments["fast"] = True
    if any(word in lowered for word in ("不要复核", "不复核", "no final review", "without final review")):
        arguments["final_review"] = False
    if monolingual_requested(line):
        arguments["bilingual"] = False
    elif bilingual_requested(line):
        arguments["bilingual"] = True

    source_language = source_language_from_text(line)
    if source_language is not None:
        arguments["source_language"] = source_language
    target_language = target_language_from_text(line, source_language=source_language)
    if target_language is not None:
        arguments["target_language"] = target_language
    output_format = output_format_from_text(line)
    if output_format is not None:
        arguments["output_format"] = output_format
    return arguments


def arguments_with_text_overrides(arguments: dict[str, Any], original: str) -> dict[str, Any]:
    """Merge tool *arguments* with overrides extracted from natural language."""
    merged = {key: value for key, value in arguments.items() if value is not None}
    merged.update(translation_arguments_from_text(original))
    return merged


def translation_values_for_tool(arguments: dict[str, Any], base_values: dict[str, Any]) -> dict[str, Any]:
    """Build translation config values from tool *arguments* on top of *base_values*."""
    values = dict(base_values)
    boolean_keys = {"bilingual", "dry_run", "fast", "final_review", "resume", "cache", "agent"}
    for key in boolean_keys:
        if key in arguments:
            parsed = bool_argument(arguments[key], key)
            if parsed is not None:
                values[key] = parsed

    for key in ("source_language", "target_language"):
        value = arguments.get(key)
        if value is not None and str(value).strip():
            values[key] = str(value).strip()

    output_format = output_format_from_argument(arguments.get("output_format"))
    if output_format is not None:
        values["output_format"] = output_format
    return values
