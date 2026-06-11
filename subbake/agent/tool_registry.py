"""Tool registry: category mappings and tool spec builders.

Pure data module — no dependency on the agent or console.
"""

from __future__ import annotations

from typing import Any

TOOL_CATEGORIES: dict[str, list[str]] = {
    "translate_file": ["translate_file"],
    "translate_series": ["translate_series"],
    "transcribe": ["transcribe_audio"],
    "manage_whisper": ["manage_whisper"],
    "edit_subtitle": ["edit_subtitle"],
    "diagnose": ["diagnose_path", "diagnose_text"],
    "file_operation": ["create_file", "append_file", "replace_in_file", "rename_path", "delete_file"],
    "browse": ["list_files", "search_files", "read_file", "read_file_preview", "candidate_subtitles"],
    "profile": ["switch_profile", "list_profiles"],
    "chat": [],
}

ALWAYS_AVAILABLE_TOOLS: tuple[str, ...] = (
    "list_files", "search_files", "read_file_preview", "recent_translations", "candidate_subtitles",
)

_TRANSCRIBE_AUDIO_SPEC: dict[str, Any] = {
    "name": "transcribe_audio",
    "mutating": True,
    "category": "transcribe",
    "schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the audio or video file to transcribe."},
            "transcriber": {"type": "string", "description": "Transcription provider: whisper_api or whisper_cpp.", "enum": ["whisper_api", "whisper_cpp"]},
            "language": {"type": "string", "description": "Source language hint (e.g., 'en', 'zh', 'ja')."},
            "output_format": {"type": "string", "description": "Output format: srt, vtt, or txt.", "enum": ["srt", "vtt", "txt"]},
            "dry_run": {"type": "boolean", "description": "Only plan without calling the transcription API."},
        },
        "required": ["path"],
    },
}

_MANAGE_WHISPER_SPEC: dict[str, Any] = {
    "name": "manage_whisper",
    "mutating": True,
    "category": "manage_whisper",
    "schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform: install, download_model, update, uninstall, status. install/download_model/update/uninstall require user approval with /approve.",
                "enum": ["install", "download_model", "update", "uninstall", "status"],
            },
            "version": {
                "type": "string",
                "description": "Version to install (e.g. 'latest' or 'v1.7.0'). Default: latest.",
            },
            "model": {
                "type": "string",
                "description": "GGML model to download. Default: small.",
            },
            "keep_models": {
                "type": "boolean",
                "description": "When uninstalling, keep downloaded GGML model files.",
            },
        },
        "required": ["action"],
    },
}

ALL_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "translate_file",
        "mutating": True,
        "category": "translate_file",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the subtitle file to translate."},
                "bilingual": {"type": "boolean", "description": "Output bilingual subtitles (source + translation)."},
                "target_language": {"type": "string", "description": "Target language for the translation."},
                "source_language": {"type": "string", "description": "Source language; auto-detect if omitted."},
                "output_format": {"type": "string", "description": "Output format: srt, vtt, or txt.", "enum": ["srt", "vtt", "txt"]},
                "dry_run": {"type": "boolean", "description": "Only plan batches without calling the model."},
                "fast": {"type": "boolean", "description": "Skip quality review for speed."},
                "final_review": {"type": "boolean", "description": "Enable final consistency review."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "translate_series",
        "mutating": True,
        "category": "translate_series",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the folder containing subtitle files."},
                "recursive": {"type": "boolean", "description": "Search subdirectories for subtitle files."},
                "overwrite": {"type": "boolean", "description": "Overwrite existing translated output files."},
                "dry_run": {"type": "boolean", "description": "Only plan batches without calling the model."},
                "suffixes": {"type": "array", "items": {"type": "string"}, "description": "File extensions to include, e.g. .srt .vtt .txt."},
                "bilingual": {"type": "boolean", "description": "Output bilingual subtitles."},
                "target_language": {"type": "string", "description": "Target language."},
                "source_language": {"type": "string", "description": "Source language."},
                "output_format": {"type": "string", "description": "Output format: srt, vtt, or txt.", "enum": ["srt", "vtt", "txt"]},
                "fast": {"type": "boolean", "description": "Skip quality review for speed."},
                "final_review": {"type": "boolean", "description": "Enable final consistency review."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "edit_subtitle",
        "mutating": True,
        "category": "edit_subtitle",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the generated subtitle file to edit."},
                "instruction": {"type": "string", "description": "Edit instruction describing the changes to make."},
            },
            "required": ["path", "instruction"],
        },
    },
    {
        "name": "diagnose_path",
        "mutating": False,
        "category": "diagnose",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the failure log or subtitle file to diagnose."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "diagnose_text",
        "mutating": False,
        "category": "diagnose",
        "schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Error text or log content to analyze."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "read_file",
        "mutating": False,
        "category": "browse",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "mutating": False,
        "category": "browse",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list."},
                "recursive": {"type": "boolean", "description": "List files in subdirectories."},
            },
        },
    },
    {
        "name": "search_files",
        "mutating": False,
        "category": "browse",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to search."},
                "pattern": {"type": "string", "description": "Search term or pattern."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "recent_translations",
        "mutating": False,
        "category": "browse",
        "schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "candidate_subtitles",
        "mutating": False,
        "category": "browse",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to search for subtitle files."},
                "query": {"type": "string", "description": "Search query to match subtitle titles."},
            },
        },
    },
    {
        "name": "read_file_preview",
        "mutating": False,
        "category": "browse",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to preview."},
                "limit": {"type": "integer", "description": "Maximum number of characters to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "create_file",
        "mutating": True,
        "category": "file_operation",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path for the new file."},
                "content": {"type": "string", "description": "Initial file content."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "append_file",
        "mutating": True,
        "category": "file_operation",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to append to."},
                "content": {"type": "string", "description": "Content to append."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "replace_in_file",
        "mutating": True,
        "category": "file_operation",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to modify."},
                "old": {"type": "string", "description": "Text to replace."},
                "new": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "rename_path",
        "mutating": True,
        "category": "file_operation",
        "schema": {
            "type": "object",
            "properties": {
                "old_path": {"type": "string", "description": "Current file path."},
                "new_path": {"type": "string", "description": "New file path."},
            },
            "required": ["old_path", "new_path"],
        },
    },
    {
        "name": "delete_file",
        "mutating": True,
        "category": "file_operation",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to delete."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_profiles",
        "mutating": False,
        "category": "profile",
        "schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "switch_profile",
        "mutating": False,
        "category": "profile",
        "schema": {
            "type": "object",
            "properties": {
                "profile": {"type": "string", "description": "Name of the profile to switch to."},
            },
            "required": ["profile"],
        },
    },
    _TRANSCRIBE_AUDIO_SPEC,
    _MANAGE_WHISPER_SPEC,
]


def build_tool_specs(categories: list[str] | None = None) -> list[dict[str, Any]]:
    """Return tool specs filtered by category names.

    When *categories* is None, returns all specs.
    Always includes ``ALWAYS_AVAILABLE_TOOLS`` in the result.
    """
    if categories is None:
        return list(ALL_TOOL_SPECS)
    allowed: set[str] = set()
    for cat in categories:
        allowed.update(TOOL_CATEGORIES.get(cat, []))
    allowed.update(ALWAYS_AVAILABLE_TOOLS)
    return [t for t in ALL_TOOL_SPECS if t["name"] in allowed]
