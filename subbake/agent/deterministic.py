"""Deterministic decision rules for SubBakeAgent.

Extracted from ``_core.py``. Keyword-based natural language matching
that maps common user requests directly to tool calls without LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._core import SubBakeAgent

from .arg_parser import series_suffixes_from_text, translation_arguments_from_text
from .text_helpers import content_after_references, extract_references, search_pattern_from_text


def deterministic_decision_from_line(agent: SubBakeAgent, line: str) -> dict[str, Any] | None:
    """Match common user requests to tool calls by keyword, bypassing the LLM agent loop."""
    lowered = line.casefold()
    references = extract_references(agent, line)

    def decision(tool_name: str, arguments: dict[str, Any], message: str) -> dict[str, Any]:
        if agent.session.mode == "plan":
            return {
                "action": "plan",
                "message": f"Plan:\n- {message}",
                "tool_calls": [{"tool_name": tool_name, "arguments": arguments}],
            }
        return {
            "action": "tool_call",
            "message": message,
            "tool_name": tool_name,
            "arguments": arguments,
        }

    if any(word in lowered for word in ("删除", "delete", "remove")) and len(references) == 1:
        return decision("delete_file", {"path": str(references[0])}, "Deleting file.")
    if any(word in lowered for word in ("改名", "重命名", "rename")) and len(references) >= 2:
        return decision(
            "rename_path",
            {"old_path": str(references[0]), "new_path": str(references[1])},
            "Renaming file.",
        )
    if any(word in lowered for word in ("追加", "append")) and len(references) == 1:
        return decision(
            "append_file",
            {"path": str(references[0]), "content": content_after_references(line)},
            "Appending file.",
        )
    if any(word in lowered for word in ("替换", "replace")) and len(references) == 1 and "=>" in line:
        old, _, new = content_after_references(line).partition("=>")
        return decision(
            "replace_in_file",
            {"path": str(references[0]), "old": old.strip(), "new": new.strip()},
            "Replacing text.",
        )
    if any(word in lowered for word in ("创建", "新建", "create")) and len(references) == 1:
        return decision(
            "create_file",
            {"path": str(references[0]), "content": content_after_references(line)},
            "Creating file.",
        )
    series_request = directory_series_request(agent, line, references)
    if series_request is not None:
        return decision("translate_series", series_request, "Translating subtitle series.")
    if (
        not references
        and any(word in lowered for word in ("当前目录", "目录下", "current directory", "cwd"))
        and any(word in lowered for word in ("有什么", "列", "查看", "读取", "list", "show", "read"))
    ):
        return decision("list_files", {"path": ".", "recursive": False}, "Listing files.")
    return None


def search_request(agent: SubBakeAgent, line: str, references: list[Path]) -> dict[str, Any] | None:
    """Check if the user is requesting a file search."""
    lowered = line.casefold()
    if not any(word in lowered for word in ("搜索", "查找", "search", "find")):
        return None
    path = references[0] if references else agent.cwd
    pattern = search_pattern_from_text(line)
    if not pattern:
        return None
    return {"path": str(path), "pattern": pattern}


def directory_series_request(agent: SubBakeAgent, line: str, references: list[Path]) -> dict[str, Any] | None:
    """Check if the user is requesting to translate a directory/series of subtitle files."""
    lowered = line.casefold()
    if not any(word in lowered for word in ("翻译", "translate")):
        return None

    referenced_folder = len(references) == 1 and references[0].is_dir()
    current_directory = not references and any(
        word in lowered
        for word in ("当前目录", "目录下", "current directory", "cwd")
    )
    if not referenced_folder and not current_directory:
        return None

    suffixes = series_suffixes_from_text(line)
    broad_series_request = suffixes is not None or any(
        word in lowered
        for word in ("都", "全部", "所有", "系列", "season", "series", "all")
    )
    if not broad_series_request:
        return None

    arguments: dict[str, Any] = {
        "path": str(references[0]) if referenced_folder else ".",
        "recursive": any(word in lowered for word in ("递归", "子目录", "recursive", "subdir")),
        "overwrite": any(word in lowered for word in ("覆盖", "重新翻译", "overwrite", "retranslate")),
        "dry_run": any(word in lowered for word in ("dry run", "dry-run", "只规划", "预览")),
    }
    if suffixes is not None:
        arguments["suffixes"] = sorted(suffixes)
    arguments.update(translation_arguments_from_text(line))
    return arguments
