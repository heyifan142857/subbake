from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from subbake.series import SUPPORTED_SUBTITLE_SUFFIXES
from subbake.title_matching import normalize_title_text, title_query_variants, title_tokens_from_text


DISCOVERY_TOOL_NAMES = frozenset(
    {
        "list_files",
        "search_files",
        "recent_translations",
        "candidate_subtitles",
        "read_file_preview",
    }
)
MUTATING_TOOL_NAMES = frozenset(
    {
        "translate_file",
        "translate_series",
        "edit_subtitle",
        "create_file",
        "append_file",
        "replace_in_file",
        "rename_path",
        "delete_file",
    }
)
GENERATED_SUBTITLE_MARKERS = (".translated.", ".bilingual.")
MEDIA_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}
PROTECTED_PATH_PARTS = {".git", ".hg", ".svn", ".venv", "venv", ".subbake", "__pycache__"}
STOPWORDS = {"the", "a", "an", "to", "of", "and", "or", "can", "could"}


@dataclass(slots=True)
class AgentLoopStep:
    action: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class AgentObservation:
    tool_name: str
    arguments: dict[str, Any]
    preview: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "preview": self.preview,
            "data": self.data,
        }


@dataclass(slots=True)
class AgentLoopState:
    original_user_message: str
    max_steps: int = 5
    current_mode: str = "chat"
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    steps: list[AgentLoopStep] = field(default_factory=list)
    observations: list[AgentObservation] = field(default_factory=list)

    def to_context(self) -> dict[str, Any]:
        return {
            "original_user_message": self.original_user_message,
            "max_steps": self.max_steps,
            "current_mode": self.current_mode,
            "allowed_tools": list(self.allowed_tools),
            "steps": [step.to_dict() for step in self.steps],
            "observations": [observation.to_dict() for observation in self.observations],
            "remaining_steps": max(self.max_steps - len(self.steps), 0),
        }


@dataclass(frozen=True, slots=True)
class FileCandidate:
    path: str
    kind: str
    suffix: str
    score: float
    match_reason: str
    inferred_source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "suffix": self.suffix,
            "score": self.score,
            "match_reason": self.match_reason,
            "inferred_source_path": self.inferred_source_path,
        }


def rank_file_candidates(
    paths: list[Path],
    query: str,
    *,
    project_root: Path,
    limit: int = 20,
) -> list[FileCandidate]:
    candidates = [
        candidate
        for path in paths
        if (candidate := score_file_candidate(path, query, project_root=project_root)) is not None
    ]
    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            _kind_sort_order(candidate.kind),
            len(candidate.path),
            candidate.path.casefold(),
        )
    )
    return candidates[:limit]


def score_file_candidate(path: Path, query: str, *, project_root: Path) -> FileCandidate | None:
    if _is_protected(path, project_root=project_root):
        return None

    kind = classify_candidate_path(path)
    if kind is None:
        return None

    inferred_source = infer_generated_source_path(path)
    relative = _relative_path(path, project_root=project_root)
    source_relative = (
        _relative_path(inferred_source, project_root=project_root)
        if inferred_source is not None
        else None
    )

    haystack_parts = [relative.as_posix(), path.name, path.stem]
    if inferred_source is not None:
        haystack_parts.extend([source_relative.as_posix(), inferred_source.name, inferred_source.stem])
    normalized_haystack = normalize_title_text(" ".join(haystack_parts))
    compact_name = re.sub(r"\s+", "", path.name.casefold())
    compact_query = re.sub(r"\s+", "", query.casefold())

    score = 0.0
    reasons: list[str] = []

    if compact_query and compact_query in compact_name:
        score = max(score, 96.0)
        reasons.append("filename contains query")

    variants = [
        variant
        for variant in (normalize_title_text(item) for item in title_query_variants(query))
        if variant
    ]
    for variant in variants:
        if variant in normalized_haystack:
            score = max(score, 78.0 + min(len(variant), 40) / 2)
            reasons.append(f"title variant '{variant}'")

    tokens = candidate_query_tokens(query)
    if tokens:
        matched = [token for token in tokens if token in normalized_haystack]
        if len(matched) == len(tokens):
            score = max(score, 92.0 + len(tokens) * 8)
            reasons.append("all title tokens")
        elif matched:
            score = max(score, 18.0 * len(matched))
            reasons.append(f"{len(matched)}/{len(tokens)} title tokens")

    if score <= 0:
        return None

    if kind == "source":
        score += 28.0
    elif kind in {"translated", "bilingual"}:
        score += 4.0
        if inferred_source is not None and inferred_source.exists():
            score -= 12.0
    elif kind == "media":
        score -= 24.0
    elif kind == "directory":
        score -= 36.0

    return FileCandidate(
        path=relative.as_posix(),
        kind=kind,
        suffix=path.suffix.lower(),
        score=round(score, 2),
        match_reason=", ".join(_dedupe(reasons)) or "matched query",
        inferred_source_path=source_relative.as_posix() if source_relative is not None else None,
    )


def candidate_query_tokens(query: str) -> list[str]:
    tokens = title_tokens_from_text(query)
    if not tokens:
        tokens = normalize_title_text(query).split()
    return [token for token in tokens if token and token not in STOPWORDS]


def classify_candidate_path(path: Path) -> str | None:
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_SUBTITLE_SUFFIXES:
        name = path.name
        if ".bilingual." in name:
            return "bilingual"
        if ".translated." in name:
            return "translated"
        return "source"
    if suffix in MEDIA_SUFFIXES:
        return "media"
    return None


def infer_generated_source_path(path: Path) -> Path | None:
    for marker in GENERATED_SUBTITLE_MARKERS:
        if marker in path.name:
            return path.with_name(path.name.replace(marker, ".", 1))
    return None


def executable_subtitle_path(candidate: FileCandidate) -> str:
    if candidate.inferred_source_path:
        return candidate.inferred_source_path
    return candidate.path


def strong_subtitle_candidates(
    candidates: list[FileCandidate],
    *,
    score_margin: float = 8.0,
) -> list[FileCandidate]:
    subtitle_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind in {"source", "translated", "bilingual"}
    ]
    if not subtitle_candidates:
        return []
    top_score = subtitle_candidates[0].score
    return [
        candidate
        for candidate in subtitle_candidates
        if candidate.score >= top_score - score_margin
    ]


def format_candidate_lines(candidates: list[FileCandidate]) -> str:
    if not candidates:
        return "No candidates."
    lines: list[str] = []
    for candidate in candidates:
        line = (
            f"{candidate.path} kind={candidate.kind} suffix={candidate.suffix or '-'} "
            f"score={candidate.score:g} reason={candidate.match_reason}"
        )
        if candidate.inferred_source_path:
            line = f"{line} inferred_source_path={candidate.inferred_source_path}"
        lines.append(line)
    return "\n".join(lines)


def _relative_path(path: Path, *, project_root: Path) -> Path:
    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return path


def _is_protected(path: Path, *, project_root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return True
    return bool(PROTECTED_PATH_PARTS.intersection(relative.parts))


def _kind_sort_order(kind: str) -> int:
    return {
        "source": 0,
        "translated": 1,
        "bilingual": 1,
        "media": 2,
        "directory": 3,
    }.get(kind, 9)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
