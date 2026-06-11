"""Deterministic decision rules for SubBakeAgent.

Extracted from ``_core.py``. Keyword-based natural language matching
that maps common user requests directly to tool calls without LLM.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._core import SubBakeAgent

from .arg_parser import series_suffixes_from_text, translation_arguments_from_text
from .text_helpers import content_after_references, extract_references, search_pattern_from_text

WHISPER_MODEL_NAMES = ("large-v3", "medium", "small", "base", "tiny")


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

    whisper_request = manage_whisper_request(line)
    if whisper_request is not None:
        action = str(whisper_request["action"])
        messages = {
            "install": "Preparing to install whisper.cpp.",
            "download_model": "Downloading whisper.cpp model.",
            "update": "Preparing to update whisper.cpp.",
            "uninstall": "Preparing to uninstall whisper.cpp.",
            "status": "Checking whisper.cpp status.",
        }
        if action == "download_model":
            messages[action] = f"Download whisper.cpp model `{whisper_request.get('model', 'small')}`."
        return decision("manage_whisper", whisper_request, messages.get(action, "Managing whisper.cpp."))

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


def manage_whisper_request(line: str) -> dict[str, Any] | None:
    lowered = line.casefold()
    mentions_whisper = any(token in lowered for token in ("whisper", "whisper.cpp", "whisper cpp"))
    mentions_model_name = _whisper_model_name_in_text(line) is not None
    mentions_model_download = (
        mentions_model_name
        and any(word in lowered for word in ("下载", "download"))
        and not any(word in lowered for word in ("字幕", "subtitle", "文件", "file"))
    )
    if not mentions_whisper and not mentions_model_download:
        return None

    action: str | None = None
    if any(word in lowered for word in ("状态", "查看", "检查", "status", "version", "版本")):
        action = "status"
    elif any(word in lowered for word in ("更新", "升级", "update", "upgrade")):
        action = "update"
    elif any(word in lowered for word in ("卸载", "移除", "删除", "uninstall", "remove")):
        action = "uninstall"
    elif mentions_model_download or (
        any(word in lowered for word in ("下载", "download"))
        and any(word in lowered for word in ("模型", "model"))
        and not any(word in lowered for word in ("安装", "接入", "setup", "install"))
    ):
        action = "download_model"
    elif any(word in lowered for word in ("安装", "下载", "接入", "install", "download", "setup")):
        action = "install"

    if action is None:
        return None

    arguments: dict[str, Any] = {"action": action}
    if action == "install":
        arguments["version"] = _whisper_version_from_text(line)
    if action in {"install", "download_model"}:
        arguments["model"] = _whisper_model_from_text(line)
    if action == "uninstall":
        arguments["keep_models"] = any(
            phrase in lowered
            for phrase in ("保留模型", "保留 model", "保留 models", "keep model", "keep models")
        )
    return arguments


def _whisper_model_from_text(line: str) -> str:
    return _whisper_model_name_in_text(line) or "small"


def _whisper_model_name_in_text(line: str) -> str | None:
    lowered = line.casefold()
    for model in WHISPER_MODEL_NAMES:
        if model in lowered:
            return model
    return None


def _whisper_version_from_text(line: str) -> str:
    match = re.search(r"\bv\d+(?:\.\d+)+(?:[-._a-zA-Z0-9]*)?\b", line)
    if match:
        return match.group(0)
    if re.search(r"\blatest\b", line, flags=re.IGNORECASE) or "最新版" in line or "最新" in line:
        return "latest"
    return "latest"
