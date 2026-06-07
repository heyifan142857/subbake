"""Discovery tool execution for SubBakeAgent.

Extracted from ``_core.py``. Handles the non-mutating discovery tools
(list_files, search_files, candidate_subtitles, recent_translations,
read_file_preview) used by the agent loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._core import SubBakeAgent

from subbake.file_ops import FileOperationGuard

from .arg_parser import bool_argument, resolve_user_path
from .loop import (
    DISCOVERY_TOOL_NAMES,
    AgentObservation,
    FileCandidate,
    classify_candidate_path,
    executable_subtitle_path,
    format_candidate_lines,
    rank_file_candidates,
    strong_subtitle_candidates,
)


def summarize_observation(observation: AgentObservation) -> str:
    """Compress an observation into a concise summary string for LLM context."""
    if observation.tool_name == "list_files":
        files = observation.data.get("files", [])
        kinds: dict[str, int] = {}
        for f in files:
            k = str(f.get("kind", "file"))
            kinds[k] = kinds.get(k, 0) + 1
        parts = [f"{v} {k} file(s)" for k, v in sorted(kinds.items())]
        return f"{len(files)} items: {', '.join(parts)}"
    if observation.tool_name == "search_files":
        candidates = observation.data.get("candidates", [])
        if candidates:
            top = candidates[:3]
            top_paths = [str(c.get("path", "")) for c in top]
            return f"{len(candidates)} candidate(s), top: {', '.join(top_paths)}"
        matches = observation.data.get("matches", [])
        return f"{len(matches)} match(es)" if matches else "no matches"
    if observation.tool_name == "candidate_subtitles":
        candidates = observation.data.get("candidates", [])
        if not candidates:
            return "no subtitle candidates"
        top = [str(c.get("path", "")) for c in candidates[:3]]
        return f"{len(candidates)} candidate(s): {', '.join(top)}"
    if observation.tool_name == "recent_translations":
        records = observation.data.get("translations", [])
        if not records:
            return "no recent translations"
        first = records[0]
        return f"{len(records)} recent: {str(first.get('tool_name', ''))} {str(first.get('path', ''))}"
    if observation.tool_name == "read_file_preview":
        path = str(observation.data.get("path", ""))
        text = str(observation.data.get("text", ""))
        return f"preview {path} ({len(text)} chars)"
    return observation.preview


def run_discovery_tool_call(
    agent: SubBakeAgent,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    original: str,
) -> AgentObservation:
    """Execute a discovery tool and return the observation."""
    if tool_name not in DISCOVERY_TOOL_NAMES:
        raise ValueError(f"Agent loop discovery cannot run mutating tool: {tool_name}")

    if tool_name == "list_files":
        path = resolve_user_path(str(arguments.get("path") or "."), cwd=agent.cwd)
        recursive = bool_argument(arguments.get("recursive"), "recursive") or False
        files = FileOperationGuard(project_root=agent.project_root).list_files(path, recursive=recursive)
        items = [
            {
                "path": str(item.relative_to(agent.project_root)),
                "kind": classify_candidate_path(item) or ("directory" if item.is_dir() else "file"),
                "suffix": item.suffix.lower(),
            }
            for item in files
        ]
        obs = AgentObservation(
            tool_name=tool_name,
            arguments={"path": str(path), "recursive": recursive},
            preview=f"{len(items)} files",
            data={"files": items},
        )
        obs.context_summary = summarize_observation(obs)
        return obs

    if tool_name == "search_files":
        path = resolve_user_path(str(arguments.get("path") or "."), cwd=agent.cwd)
        pattern = str(arguments.get("pattern") or arguments.get("query") or "").strip()
        if not pattern:
            pattern = agent._search_pattern_from_text(original)
        if not pattern:
            obs = AgentObservation(
                tool_name=tool_name,
                arguments={"path": str(path), "pattern": pattern},
                preview="no search pattern",
                data={"pattern": pattern, "candidates": [], "matches": []},
            )
            obs.context_summary = summarize_observation(obs)
            return obs
        candidates = rank_candidates_in_path(agent, path, pattern, limit=20)
        data: dict[str, Any] = {"pattern": pattern, "candidates": [candidate.to_dict() for candidate in candidates]}
        if not candidates:
            matches = FileOperationGuard(project_root=agent.project_root).search_files(path, pattern)
            data["matches"] = matches
            preview = f"{len(matches)} text matches" if matches else "no matches"
        else:
            preview = candidate_observation_preview(candidates)
        obs = AgentObservation(
            tool_name=tool_name,
            arguments={"path": str(path), "pattern": pattern},
            preview=preview,
            data=data,
        )
        obs.context_summary = summarize_observation(obs)
        return obs

    if tool_name == "candidate_subtitles":
        path = resolve_user_path(str(arguments.get("path") or "."), cwd=agent.cwd)
        query = str(arguments.get("query") or "").strip() or original
        candidates = rank_candidates_in_path(agent, path, query, limit=20)
        obs = AgentObservation(
            tool_name=tool_name,
            arguments={"path": str(path), "query": query},
            preview=candidate_observation_preview(candidates),
            data={"query": query, "candidates": [candidate.to_dict() for candidate in candidates]},
        )
        obs.context_summary = summarize_observation(obs)
        return obs

    if tool_name == "recent_translations":
        records = recent_translation_records(agent)
        preview = (
            f"{len(records)} recent translations: {records[0]['path']}"
            if records
            else "no recent translations"
        )
        obs = AgentObservation(
            tool_name=tool_name,
            arguments={},
            preview=preview,
            data={"translations": records},
        )
        obs.context_summary = summarize_observation(obs)
        return obs

    if tool_name == "read_file_preview":
        path = resolve_user_path(str(arguments.get("path") or ""), cwd=agent.cwd)
        limit = int(arguments.get("limit") or 2000)
        text = FileOperationGuard(project_root=agent.project_root).read_file(path, limit=limit)
        obs = AgentObservation(
            tool_name=tool_name,
            arguments={"path": str(path), "limit": limit},
            preview=f"preview {path.relative_to(agent.project_root)} ({len(text)} chars)",
            data={"path": str(path.relative_to(agent.project_root)), "text": text},
        )
        obs.context_summary = summarize_observation(obs)
        return obs

    raise ValueError(f"Unsupported discovery tool: {tool_name}")


def format_discovery_observation_for_user(agent: SubBakeAgent, observation: AgentObservation) -> str:
    """Format an observation into a human-readable string."""
    if observation.tool_name == "list_files":
        files = [
            item.get("path")
            for item in observation.data.get("files", [])
            if isinstance(item, dict) and item.get("path")
        ]
        return "\n".join(str(path) for path in files)
    if observation.tool_name in {"search_files", "candidate_subtitles"}:
        candidates = [
            FileCandidate(
                path=str(item.get("path") or ""),
                kind=str(item.get("kind") or ""),
                suffix=str(item.get("suffix") or ""),
                score=float(item.get("score") or 0),
                match_reason=str(item.get("match_reason") or ""),
                inferred_source_path=(
                    str(item.get("inferred_source_path"))
                    if item.get("inferred_source_path") is not None
                    else None
                ),
            )
            for item in observation.data.get("candidates", [])
            if isinstance(item, dict)
        ]
        if candidates:
            return format_candidate_lines(candidates)
        matches = observation.data.get("matches")
        if isinstance(matches, list) and matches:
            return "\n".join(str(match) for match in matches)
        return "No matches."
    if observation.tool_name == "recent_translations":
        records = observation.data.get("translations", [])
        if not isinstance(records, list) or not records:
            return "No recent translations."
        return "\n".join(
            f"{record.get('tool_name')}: {record.get('path')}"
            for record in records
            if isinstance(record, dict)
        )
    if observation.tool_name == "read_file_preview":
        return str(observation.data.get("text") or "")
    return observation.preview


def rank_candidates_in_path(agent: SubBakeAgent, path: Path, query: str, *, limit: int) -> list[FileCandidate]:
    """Rank file candidates matching a query in the given path."""
    guard = FileOperationGuard(project_root=agent.project_root)
    files = guard.list_files(path, recursive=True, limit=500)
    return rank_file_candidates(files, query, project_root=agent.project_root, limit=limit)


def candidate_observation_preview(candidates: list[FileCandidate]) -> str:
    """Build a one-line preview for a candidate observation."""
    if not candidates:
        return "no candidates"
    strong = strong_subtitle_candidates(candidates)
    if len(strong) == 1:
        return f"selected {executable_subtitle_path(strong[0])}"
    return f"{len(candidates)} candidates, top {candidates[0].path}"


def recent_translation_records(agent: SubBakeAgent, *, limit: int = 5) -> list[dict[str, Any]]:
    """Return the most recent translation events from the session."""
    records: list[dict[str, Any]] = []
    for event in reversed(agent.session.events):
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        if event.get("kind") == "translate_file" and data.get("input_path"):
            records.append(
                {
                    "tool_name": "translate_file",
                    "path": str(data["input_path"]),
                    "output_path": data.get("output_path"),
                    "bilingual": bool(data.get("bilingual")),
                    "source_language": data.get("source_language"),
                    "target_language": data.get("target_language"),
                }
            )
        elif event.get("kind") == "series" and data.get("path"):
            records.append(
                {
                    "tool_name": "translate_series",
                    "path": str(data["path"]),
                    "suffixes": data.get("suffixes"),
                    "recursive": data.get("recursive"),
                    "bilingual": bool(data.get("bilingual")),
                    "source_language": data.get("source_language"),
                    "target_language": data.get("target_language"),
                }
            )
        if len(records) >= limit:
            break
    return records
