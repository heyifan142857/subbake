from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text

from subbake import __version__
from .loop import (
    DISCOVERY_TOOL_NAMES,
    AgentLoopState,
    AgentLoopStep,
    AgentObservation,
    FileCandidate,
    classify_candidate_path,
    executable_subtitle_path,
    format_candidate_lines,
    rank_file_candidates,
    strong_subtitle_candidates,
)
from .ui import (
    print_file_completion,
    print_file_op_result,
    print_help as _print_help_fn,
    print_series_completion,
    print_series_summary,
    print_tool_call_preview,
    print_translation_start,
    render_mode_label,
)
from .session import AgentSession, AgentSessionStore, SESSION_VERSION
from .arg_parser import (
    arguments_with_text_overrides,
    bilingual_requested,
    bool_argument,
    language_phrases,
    line_without_output_format_phrases,
    monolingual_requested,
    output_format_from_argument,
    output_format_from_text,
    output_format_patterns,
    resolve_user_path,
    series_suffixes_from_argument,
    series_suffixes_from_text,
    source_language_from_text,
    target_language_for_bilingual_pair,
    target_language_from_text,
    title_tokens_from_text,
    translation_arguments_from_text,
    translation_values_for_tool,
)
from .tool_registry import (
    ALWAYS_AVAILABLE_TOOLS,
    TOOL_CATEGORIES,
    build_tool_specs,
)
from .trace import (
    AGENT_COMMANDS,
    PICKER_CANCEL_TOKEN,
    PickerChoice,
    REFERENCE_RE,
    _AgentLoopTrace,
    _current_completion,
    _default_api_key_env,
    _language_phrases,
    _matching_picker_choices,
    _now_iso,
    _output_format_patterns,
    _picker_choice,
    _picker_choices,
    _picker_display_parts,
    _picker_prompt,
    _picker_toolbar,
    _prepend_default_profile,
    _prompt_toolkit_inline_picker,
    _prompt_toolkit_inline_text,
    _prompt_toolkit_prompt,
    _resolve_picker_selection,
    _resolve_text_prompt_value,
    _short_title,
    _slash_command_completer,
    _slash_command_matches,
    _text_prompt,
    _text_prompt_matches,
    _text_prompt_toolbar,
    _toml_key,
    _toml_string,
    _trace_arguments,
    _trace_value,
    _unique_slash_command_match,
    _verify_write_text,
)
from subbake.config import (
    AppConfig,
    TRANSLATE_CONFIG_KEYS,
    load_app_config,
    resolve_command_config,
)
# discover_config_path, discover_project_config_path, global_config_candidates
# are accessed via _config module reference so that tests can patch the source module.
from subbake import config as _config
from subbake.diagnostics import diagnose_path, diagnose_text, format_diagnostic_report
from subbake.editing import edit_generated_subtitle, is_generated_subtitle
from subbake.file_ops import FileOpResult, FileOperationGuard
from subbake.pipeline import SubtitlePipeline
from subbake.models import build_backend
from subbake.models.base_model import MockBackend
from subbake.runtime_options import (
    build_pipeline_options,
    merge_translation_values,
)
# build_backend_from_values is accessed via _runtime_options module reference
# so that tests can patch the source module.
from subbake import runtime_options as _runtime_options
from subbake.series import (
    SUPPORTED_SUBTITLE_SUFFIXES,
    discover_series_files,
    resolve_series_output_path,
    translate_series,
)
from subbake.title_matching import normalize_title_text, title_tokens_from_text
from subbake.ui import Dashboard

AGENT_LOOP_MAX_STEPS = 5

CONFIDENCE_LOW_THRESHOLD = 0.4
CONFIDENCE_MEDIUM_THRESHOLD = 0.7
CONFIDENCE_MIN_OBSERVATIONS = 2

# TOOL_CATEGORIES, ALWAYS_AVAILABLE_TOOLS imported from .tool_registry
# REFERENCE_RE imported from .trace
# AGENT_COMMANDS imported from .trace
NEW_PROFILE_VALUE = "__subbake_new_profile__"
CONFIG_BOOTSTRAP_CREATE = "create"
CONFIG_BOOTSTRAP_SKIP = "skip"
PICKER_CANCEL_TOKEN = "__subbake_picker_cancelled__"  # also in agent_trace; kept for internal use
PROFILE_PROVIDER_OPTIONS = ("mock", "openai", "anthropic", "gemini", "openai-compatible")
PROFILE_TARGET_LANGUAGE_OPTIONS = ("Chinese", "zh", "en", "ja", "ko", "fr", "es", "de")
PROFILE_API_KEY_ENV_OPTIONS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


class SubBakeAgent:
    def __init__(self, *, console: Console, resume: bool = False) -> None:
        self.console = console
        self.cwd = Path.cwd()
        self.config_path = _config.discover_config_path()
        self.config = self._load_config(self.config_path)
        project_config_path = _config.discover_project_config_path()
        self.project_root = project_config_path.parent if project_config_path is not None else self.cwd
        self.store = AgentSessionStore(self.project_root)
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self.profile = self._initial_profile()
        self.values = self._values_for_profile(self.profile)
        self.session = self._load_or_create_session(resume=resume)
        if self.session.profile != self.profile:
            self.session.profile = self.profile
        self.store.save(self.session)

    def run(self) -> None:
        self.console.print(f"[bold green]SubBake agent {__version__}[/bold green]  /help for commands, /exit to quit")
        if self.config_path is not None:
            self.console.print(f"[bold green]Config:[/bold green] {self.config_path}")
        if self.profile is not None:
            self.console.print(f"[bold green]Profile:[/bold green] {self.profile}")
        self._maybe_offer_config_bootstrap()

        while True:
            try:
                line = self._read_line()
            except EOFError:
                self.console.print("")
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                should_continue = self._handle_line(stripped)
            except Exception as exc:
                self.console.print(f"[bold red]Error:[/bold red] {exc}")
                self._record_event("error", stripped, {"error": str(exc)})
                should_continue = True
            self.store.save(self.session)
            if not should_continue:
                break

    def _handle_line(self, line: str) -> bool:
        command, rest = self._split_command(line)
        if command in {"/exit", "/quit"}:
            return False
        if command == "/help":
            self._print_help()
            self._record_event("help", line)
            return True
        if command == "/clear":
            self.session = self.store.create(
                cwd=self.cwd,
                profile=self.profile,
                config_path=self.config_path,
            )
            self.store.save(self.session)
            self.console.print("[bold green]Started a new agent session.[/bold green]")
            return True
        if command in {"/model", "/profile", "/profiles"}:
            self._handle_profile_command(rest)
            self._record_event("profile", line, {"profile": self.profile})
            return True
        if command == "/session":
            self._handle_session_command(rest)
            self._record_event("session", line, {"session_id": self.session.id})
            return True
        if command == "/sessions":
            self._print_sessions()
            self._record_event("sessions", line)
            return True
        if command == "/resume":
            self._resume_latest_session()
            return True
        if command == "/plan":
            self._handle_plan_command(rest)
            return True
        if command == "/approve":
            self._approve_pending_plan()
            return True
        if command == "/reject":
            self._reject_pending_plan()
            return True
        if command == "/undo":
            self._undo_last_operation()
            return True
        if command is not None:
            self.console.print("Unknown command. Use /help for available agent controls.")
            self._record_event("unknown_command", line, {"command": command})
            return True

        self._handle_conversational_line(line)
        return True

    def _handle_conversational_line(self, line: str) -> None:
        self._record_event("user", line)
        decision = self._deterministic_decision_from_line(line)
        if decision is None:
            intent = self._classify_intent(line)
            if intent is not None:
                decision = self._intent_to_decision(intent, line)
            else:
                decision = self._run_agent_loop(line)
        self._handle_decision(decision, original=line)

    def _deterministic_decision_from_line(self, line: str) -> dict[str, Any] | None:
        lowered = line.casefold()
        references = self._extract_references(line)

        def decision(tool_name: str, arguments: dict[str, Any], message: str) -> dict[str, Any]:
            if self.session.mode == "plan":
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
                {"path": str(references[0]), "content": self._content_after_references(line)},
                "Appending file.",
            )
        if any(word in lowered for word in ("替换", "replace")) and len(references) == 1 and "=>" in line:
            old, _, new = self._content_after_references(line).partition("=>")
            return decision(
                "replace_in_file",
                {"path": str(references[0]), "old": old.strip(), "new": new.strip()},
                "Replacing text.",
            )
        if any(word in lowered for word in ("创建", "新建", "create")) and len(references) == 1:
            return decision(
                "create_file",
                {"path": str(references[0]), "content": self._content_after_references(line)},
                "Creating file.",
            )
        series_request = self._directory_series_request(line, references)
        if series_request is not None:
            return decision("translate_series", series_request, "Translating subtitle series.")
        if (
            not references
            and any(word in lowered for word in ("当前目录", "目录下", "current directory", "cwd"))
            and any(word in lowered for word in ("有什么", "列", "查看", "读取", "list", "show", "read"))
        ):
            return decision("list_files", {"path": ".", "recursive": False}, "Listing files.")
        return None

    def _content_after_references(self, line: str) -> str:
        cleaned = self._remove_references(line).strip()
        cleaned = re.sub(r"^(创建|新建|追加|替换)\s*", "", cleaned).strip()
        cleaned = re.sub(r"^(create|append|replace)\b", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def _search_request(self, line: str, references: list[Path]) -> dict[str, Any] | None:
        lowered = line.casefold()
        if not any(word in lowered for word in ("搜索", "查找", "search", "find")):
            return None
        path = references[0] if references else self.cwd
        pattern = self._search_pattern_from_text(line)
        if not pattern:
            return None
        return {"path": str(path), "pattern": pattern}

    def _search_pattern_from_text(self, line: str) -> str:
        cleaned = self._remove_references(line).strip()
        for marker in ("搜索", "查找", "search", "find"):
            match = re.search(rf"\b{re.escape(marker)}\b|{re.escape(marker)}", cleaned, flags=re.IGNORECASE)
            if match is not None:
                return cleaned[match.end():].strip(" ：:=,，。")
        return cleaned

    def _directory_series_request(self, line: str, references: list[Path]) -> dict[str, Any] | None:
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

        suffixes = self._series_suffixes_from_text(line)
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
        arguments.update(self._translation_arguments_from_text(line))
        return arguments

    def _series_suffixes_from_text(self, line: str) -> set[str] | None:
        return series_suffixes_from_text(line)

    def _translation_retarget_request(
        self,
        line: str,
        references: list[Path],
    ) -> tuple[str, dict[str, Any]] | None:
        if not self._retarget_translation_requested(line):
            return None

        arguments = self._translation_arguments_from_text(line)
        if not self._has_translation_retarget_options(arguments):
            return None

        if references:
            target = references[0]
            if target.is_dir():
                return "translate_series", self._translation_arguments_for_target(
                    path=target,
                    arguments={"path": str(target), **arguments},
                    series=True,
                )
            source_path = self._source_path_for_translation_reference(target)
            if source_path is not None:
                return "translate_file", self._translation_arguments_for_target(
                    path=source_path,
                    arguments={"path": str(source_path), **arguments},
                    series=False,
                )
            return None

        title_target = self._translation_source_from_title_text(line)
        if title_target is not None:
            return "translate_file", self._translation_arguments_for_target(
                path=title_target,
                arguments={"path": str(title_target), **arguments},
                series=False,
            )

        latest = self._latest_translation_tool_call()
        if latest is None:
            return None
        tool_name, latest_arguments = latest
        merged_arguments = {**latest_arguments, **arguments}
        return tool_name, self._translation_arguments_for_target(
            path=Path(str(merged_arguments["path"])),
            arguments=merged_arguments,
            series=tool_name == "translate_series",
        )

    def _retarget_translation_requested(self, line: str) -> bool:
        lowered = line.casefold()
        if not any(
            word in lowered
            for word in (
                "变成",
                "变为",
                "改成",
                "改为",
                "换成",
                "做成",
                "重新生成",
                "再生成",
                "rerender",
                "re-render",
                "make",
            )
        ):
            return False
        return any(word in lowered for word in ("字幕", "subtitle", "翻译", "translation", "双语", "bilingual"))

    def _has_translation_retarget_options(self, arguments: dict[str, Any]) -> bool:
        return any(
            key in arguments
            for key in ("bilingual", "target_language", "source_language", "output_format")
        )

    def _source_path_for_translation_reference(self, path: Path) -> Path | None:
        if is_generated_subtitle(path):
            for marker in (".translated.", ".bilingual."):
                if marker in path.name:
                    return path.with_name(path.name.replace(marker, ".", 1))
        if path.suffix.lower() in SUPPORTED_SUBTITLE_SUFFIXES:
            return path
        return None

    def _translation_source_from_title_text(self, line: str) -> Path | None:
        tokens = self._title_tokens_from_text(line)
        if not tokens:
            return None
        candidates: list[Path] = []
        for path in self.cwd.glob("*"):
            source_path = self._source_path_for_translation_reference(path)
            if source_path is None:
                continue
            if source_path in candidates:
                continue
            normalized_name = normalize_title_text(source_path.stem)
            if all(token in normalized_name for token in tokens):
                candidates.append(source_path)
        if not candidates:
            return None
        candidates.sort(key=lambda path: (is_generated_subtitle(path), len(path.name), path.name.casefold()))
        return candidates[0]

    def _title_tokens_from_text(self, line: str) -> list[str]:
        return title_tokens_from_text(line)

    def _latest_translation_tool_call(self) -> tuple[str, dict[str, Any]] | None:
        for event in reversed(self.session.events):
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            if event.get("kind") == "translate_file" and data.get("input_path"):
                return "translate_file", {"path": str(data["input_path"])}
            if event.get("kind") == "series" and data.get("path"):
                arguments: dict[str, Any] = {"path": str(data["path"])}
                if data.get("suffixes"):
                    arguments["suffixes"] = data["suffixes"]
                if data.get("recursive") is not None:
                    arguments["recursive"] = data["recursive"]
                return "translate_series", arguments
        return None

    def _translation_arguments_for_target(
        self,
        *,
        path: Path,
        arguments: dict[str, Any],
        series: bool,
    ) -> dict[str, Any]:
        enriched = dict(arguments)
        if "target_language" not in enriched:
            source_language = str(enriched.get("source_language") or "").strip()
            if not source_language:
                source_language = self._infer_source_language_for_target(path, series=series) or ""
            target_language = self._target_language_for_bilingual_pair_from_arguments(
                enriched,
                source_language=source_language,
            )
            if target_language is not None:
                enriched["target_language"] = target_language
        return enriched

    def _target_language_for_bilingual_pair_from_arguments(
        self,
        arguments: dict[str, Any],
        *,
        source_language: str,
    ) -> str | None:
        if not bool(arguments.get("bilingual")):
            return None
        if source_language == "Chinese":
            return "English"
        if source_language == "English":
            return "Chinese"
        return None

    def _infer_source_language_for_target(self, path: Path, *, series: bool) -> str | None:
        candidate = self._first_source_file(path, series=series)
        if candidate is None:
            return None
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")[:12000]
        except OSError:
            return None
        if re.search(r"[\u4e00-\u9fff]", text):
            return "Chinese"
        if re.search(r"[A-Za-z]", text):
            return "English"
        return None

    def _first_source_file(self, path: Path, *, series: bool) -> Path | None:
        if not series:
            return path if path.exists() else None
        suffixes = None
        try:
            files = discover_series_files(path, recursive=False, suffixes=suffixes)
        except Exception:
            return None
        return files[0] if files else None

    VALID_INTENT_CATEGORIES = frozenset({
        "translate_file", "translate_series", "edit_subtitle",
        "diagnose", "file_operation", "browse", "profile", "chat",
    })

    def _classify_intent(self, line: str) -> dict[str, Any] | None:
        """Lightweight intent classification. Returns intent dict or None to skip gate."""
        backend = _runtime_options.build_backend_from_values(self.values)
        if backend is None:
            return self._fallback_intent_classification(line)
        if isinstance(backend, MockBackend):
            return self._mock_classify_intent(line)

        context = {
            "message": line,
            "cwd": str(self.cwd),
            "profile": self.profile,
            "recent_events": self.session.events[-4:],
        }
        system_prompt = (
            "You are a classifier for a subtitle translation agent. "
            "Classify the user's request into exactly one category and extract parameters.\n"
            "Categories:\n"
            "- translate_file: Translate a single subtitle file\n"
            "- translate_series: Translate a series/folder of subtitle files\n"
            "- edit_subtitle: Edit/post-process an already-generated subtitle\n"
            "- diagnose: Analyze failure logs or subtitle files\n"
            "- file_operation: Create, append, replace, rename, or delete files\n"
            "- browse: List, search, or read files\n"
            "- profile: Switch or list model profiles\n"
            "- chat: General conversation, no tool needed\n\n"
            "Return JSON only with: category, confidence (0-1), parameters (dict), and reason.\n"
            "Extract file paths, language names, format preferences from natural language."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        try:
            payload, _ = backend.generate_json(messages)
        except Exception:
            return None

        if not isinstance(payload, dict) or "category" not in payload:
            return None
        category = str(payload.get("category", ""))
        if category not in self.VALID_INTENT_CATEGORIES:
            return None
        return {
            "category": category,
            "parameters": dict(payload.get("parameters", {})),
            "confidence": float(payload.get("confidence", 0.5)),
            "reason": str(payload.get("reason", "")),
        }

    def _mock_classify_intent(self, line: str) -> dict[str, Any] | None:
        """Keyword-based intent classification for mock backend.
        Only returns classification when confident enough to skip the agent loop.
        Returns None for ambiguous cases, letting existing agent loop handle them."""

        lowered = line.casefold()
        references = self._extract_references(line)
        has_refs = bool(references)
        has_dir = any(r.is_dir() for r in references)
        has_file = any(r.is_file() for r in references)

        first_ref = str(references[0]) if references else ""

        # Clear translation with directory reference → skip agent loop
        if has_dir and any(w in lowered for w in ("翻译", "translate", "series")):
            return {"category": "translate_series", "parameters": {"path": first_ref}, "confidence": 0.9, "reason": "Directory reference"}
        # Edit with reference — must come before translate check to avoid
        # matching "translate" inside "translated" in file paths
        if has_refs and any(w in lowered for w in ("编辑", "修改", "edit", "fix", "change")):
            return {"category": "edit_subtitle", "parameters": {"path": first_ref}, "confidence": 0.9, "reason": "Edit reference"}  # fmt: skip
        # Clear translation with file reference → skip agent loop
        if has_file and any(
            lowered.startswith(w) or f" {w}" in lowered
            for w in ("翻译", "translate")
        ):
            return {"category": "translate_file", "parameters": {"path": first_ref}, "confidence": 0.9, "reason": "File reference"}
        # Explicit diagnose with reference → skip agent loop
        if has_refs and any(w in lowered for w in ("诊断", "分析", "diagnose", "error", "log", "failure")):
            return {"category": "diagnose", "parameters": {"path": first_ref}, "confidence": 0.9, "reason": "Diagnosis reference"}  # fmt: skip
        # Everything else → let agent loop handle
        return None

    def _fallback_intent_classification(self, line: str) -> dict[str, Any] | None:
        """Fallback when no backend is available. Only fires for clear cases with references."""
        lowered = line.casefold()
        references = self._extract_references(line)
        has_refs = bool(references)
        if has_refs and any(w in lowered for w in ("翻译", "translate")):
            paths = [str(r) for r in references]
            if len(references) == 1 and references[0].is_dir():
                return {"category": "translate_series", "parameters": {"path": paths[0]}, "confidence": 0.5, "reason": "Fallback: translate dir"}  # fmt: skip
            return {"category": "translate_file", "parameters": {"path": paths[0]}, "confidence": 0.5, "reason": "Fallback: translate ref"}  # fmt: skip
        return None

    def _intent_to_decision(self, intent: dict[str, Any], line: str) -> dict[str, Any]:
        """Convert intent classification to an agent decision."""
        category = intent["category"]
        params = intent["parameters"]
        reason = intent.get("reason", "")

        if category == "chat":
            return {"action": "respond", "message": "How can I help with your subtitles?"}

        allowed_tools: set[str] = set(ALWAYS_AVAILABLE_TOOLS)
        allowed_tools.update(TOOL_CATEGORIES.get(category, []))

        intent_confidence = intent.get("confidence", 0.5)
        if intent_confidence < 0.4:
            return {"action": "ask_user", "message": f"I'm not sure what you want to do. Could you clarify?\n\n({reason})"}

        pre_args = self._prepopulate_args_from_intent_parameters(params)

        if intent_confidence >= 0.85 and self._has_required_args(category, pre_args):
            return {
                "action": "final_tool_call",
                "tool_name": self._category_to_default_tool(category),
                "arguments": pre_args,
                "message": reason or f"Proceeding with {category}.",
                "confidence": intent_confidence,
                "reason": reason,
            }

        state = AgentLoopState(
            original_user_message=line,
            max_steps=AGENT_LOOP_MAX_STEPS,
            current_mode=self.session.mode,
            allowed_tools=tuple(sorted(allowed_tools)),
            pre_populated_arguments=pre_args,
            intent_hint={
                "category": category,
                "confidence": intent_confidence,
                "reason": reason,
            },
        )
        return self._run_agent_loop(line, state=state)

    def _prepopulate_args_from_intent_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Convert intent-extracted parameters into tool argument format."""
        args: dict[str, Any] = {}
        for key in ("path", "target_language", "source_language", "output_format", "pattern", "query", "content", "old", "new", "old_path", "new_path", "instruction", "text"):
            if key in parameters:
                args[key] = parameters[key]
        for key in ("bilingual", "recursive", "overwrite", "dry_run", "fast", "final_review"):
            if key in parameters:
                args[key] = bool(parameters[key])
        return args

    def _has_required_args(self, category: str, args: dict[str, Any]) -> bool:
        if category in {"translate_file", "diagnose"}:
            return "path" in args
        if category == "edit_subtitle":
            return "path" in args and "instruction" in args
        if category == "translate_series":
            return "path" in args
        return False

    def _category_to_default_tool(self, category: str) -> str:
        mapping = {
            "translate_file": "translate_file",
            "translate_series": "translate_series",
            "edit_subtitle": "edit_subtitle",
            "diagnose": "diagnose_path",
            "file_operation": "create_file",
            "browse": "list_files",
            "profile": "list_profiles",
        }
        return mapping.get(category, "list_files")

    def _apply_confidence_gate(
        self,
        decision: dict[str, Any],
        state: AgentLoopState,
    ) -> dict[str, Any] | None:
        """Check LLM confidence and gate mutating actions. Returns a modified decision or None to proceed.
        Only gates final_tool_call (mutating) — discovery tool_call actions pass through unchanged."""
        action = str(decision.get("action") or "").strip()
        if action not in {"final_tool_call"}:
            return None

        raw_confidence = decision.get("confidence")
        if not isinstance(raw_confidence, (int, float)):
            return None

        confidence = float(raw_confidence)
        reason = str(decision.get("reason") or "")
        num_observations = len(state.observations)

        if confidence < CONFIDENCE_LOW_THRESHOLD:
            return {"action": "respond", "message": "I need more information to proceed confidently. Could you please clarify your request?"}

        if confidence < CONFIDENCE_MEDIUM_THRESHOLD and num_observations < CONFIDENCE_MIN_OBSERVATIONS:
            message = reason or f"I think I should {action} but I am not entirely sure."
            return {"action": "ask_user", "message": f"{message}\n\nCan you confirm this is what you want?"}

        return None

    def _run_agent_loop(self, line: str, *, state: AgentLoopState | None = None) -> dict[str, Any]:
        if state is None:
            state = AgentLoopState(
                original_user_message=line,
                max_steps=AGENT_LOOP_MAX_STEPS,
                current_mode=self.session.mode,
                allowed_tools=tuple(tool["name"] for tool in self._tool_specs()),
            )
        trace = _AgentLoopTrace(console=self.console, interactive=self.interactive)
        trace.start()
        for _ in range(state.max_steps):
            with trace.thinking():
                decision = self._decide_loop_next_action(state, show_status=False)
            state.steps.append(self._loop_step_from_decision(decision))
            action = str(decision.get("action") or "").strip()

            gated = self._apply_confidence_gate(decision, state)
            if gated is not None:
                state.steps.append(self._loop_step_from_decision(gated))
                trace.final(gated)
                return gated

            if action == "tool_call":
                tool_name = str(decision.get("tool_name") or "").strip()
                arguments = dict(decision.get("arguments") or {})
                if tool_name not in DISCOVERY_TOOL_NAMES:
                    final_decision = self._decision_from_final_tool_call(
                        {
                            **decision,
                            "action": "final_tool_call",
                            "tool_name": tool_name,
                            "arguments": arguments,
                        },
                        original=line,
                    )
                    trace.final(final_decision)
                    return final_decision
                trace.tool(tool_name, arguments)
                observation = self._run_discovery_tool_call(tool_name, arguments, original=line)
                state.observations.append(observation)
                trace.observe(observation.preview)
                continue

            if action == "final_tool_call":
                final_decision = self._decision_from_final_tool_call(decision, original=line)
                trace.final(final_decision)
                return final_decision

            if action in {"respond", "ask_user", "plan"}:
                trace.final(decision)
                return decision

            raise ValueError(f"Unsupported agent loop action: {action or '<missing>'}")

        message = f"Agent loop stopped after {state.max_steps} steps without a final action."
        trace.observe(message)
        return {"action": "respond", "message": message}

    def _decide_loop_next_action(self, state: AgentLoopState, *, show_status: bool = True) -> dict[str, Any]:
        backend = _runtime_options.build_backend_from_values(self.values)
        if backend is None:
            raise RuntimeError("Agent conversation requires a model backend.")
        messages = self._build_agent_loop_decision_messages(state)
        if self.interactive and show_status:
            with self.console.status("Model thinking...", spinner="dots"):
                payload, _ = backend.generate_json(messages)
        else:
            payload, _ = backend.generate_json(messages)
        if not isinstance(payload, dict):
            raise ValueError("Agent loop decision must be a JSON object.")
        action = str(payload.get("action") or "").strip()
        if action not in {"respond", "ask_user", "tool_call", "final_tool_call", "plan"}:
            raise ValueError(f"Unsupported agent loop decision action: {action or '<missing>'}")
        return payload

    def _build_agent_loop_decision_messages(self, state: AgentLoopState) -> list[dict[str, str]]:
        all_tools = self._tool_specs()
        if state.allowed_tools:
            allowed_set = set(state.allowed_tools)
            filtered_tools = [t for t in all_tools if t["name"] in allowed_set]
        else:
            filtered_tools = all_tools
        context = {
            **state.to_context(),
            "user_message": state.original_user_message,
            "mode": self.session.mode,
            "profile": self.profile,
            "cwd": str(self.cwd),
            "project_root": str(self.project_root),
            "references": self._reference_context(state.original_user_message),
            "recent_events": self.session.events[-8:],
            "tools": filtered_tools,
        }
        system_prompt = (
            "You are SubBake's bounded local project agent.\n"
            "Return valid JSON only.\n"
            "Choose one action: respond, ask_user, tool_call, final_tool_call, or plan.\n"
            "Use tool_call only for safe discovery tools: list_files, search_files, "
            "recent_translations, candidate_subtitles, read_file_preview.\n"
            "Use final_tool_call when enough evidence supports one executable tool.\n"
            "Discovery before mutation is required when the target path is uncertain.\n"
            "Never claim a file exists unless it is present in references or tool observations.\n"
            "If multiple strong subtitle candidates remain, return ask_user with the choices.\n"
            "In plan mode, discovery tools may run; executable final actions will be stored for approval.\n"
            "When the context includes an intent_hint, trust its category and reason "
            "unless observations contradict it — do not re-derive the user's intent from scratch.\n"
        )
        user_prompt = (
            "TASK_START\n"
            "agent_loop_decide\n"
            "TASK_END\n"
            'Return JSON with "action", "message", "reason", and "confidence" when applicable. '
            'For tool_call or final_tool_call include "tool_name" and "arguments". '
            'For plan include "tool_calls".\n'
            "AGENT_LOOP_CONTEXT_JSON_START\n"
            f"{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}\n"
            "AGENT_LOOP_CONTEXT_JSON_END\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _loop_step_from_decision(self, decision: dict[str, Any]) -> AgentLoopStep:
        confidence: float | None = None
        raw_confidence = decision.get("confidence")
        if isinstance(raw_confidence, int | float):
            confidence = float(raw_confidence)
        return AgentLoopStep(
            action=str(decision.get("action") or ""),
            tool_name=str(decision.get("tool_name") or "") or None,
            arguments=dict(decision.get("arguments") or {}),
            reason=str(decision.get("reason") or ""),
            confidence=confidence,
        )

    def _decision_from_final_tool_call(self, decision: dict[str, Any], *, original: str) -> dict[str, Any]:
        tool_name = str(decision.get("tool_name") or "").strip()
        if not tool_name:
            raise ValueError("Executable agent decision is missing tool_name.")
        arguments = self._enrich_executable_arguments(
            tool_name,
            dict(decision.get("arguments") or {}),
            original=original,
        )
        message = str(decision.get("message") or decision.get("reason") or "Executing tool.").strip()
        if self.session.mode == "plan":
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

    def _enrich_executable_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        original: str,
    ) -> dict[str, Any]:
        enriched = self._arguments_with_text_overrides(arguments, original)
        if tool_name == "translate_file" and enriched.get("path"):
            requested_path = self._resolve_user_path(str(enriched["path"]))
            source_path = self._source_path_for_translation_reference(requested_path) or requested_path
            enriched["path"] = str(source_path)
            return self._translation_arguments_for_target(
                path=source_path,
                arguments=enriched,
                series=False,
            )
        if tool_name == "translate_series" and enriched.get("path"):
            requested_path = self._resolve_user_path(str(enriched["path"]))
            enriched["path"] = str(requested_path)
            return self._translation_arguments_for_target(
                path=requested_path,
                arguments=enriched,
                series=True,
            )
        return enriched

    def _summarize_observation(self, observation: AgentObservation) -> str:
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

    def _run_discovery_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        original: str,
    ) -> AgentObservation:
        if tool_name not in DISCOVERY_TOOL_NAMES:
            raise ValueError(f"Agent loop discovery cannot run mutating tool: {tool_name}")

        if tool_name == "list_files":
            path = self._resolve_user_path(str(arguments.get("path") or "."))
            recursive = self._bool_argument(arguments.get("recursive"), "recursive") or False
            files = FileOperationGuard(project_root=self.project_root).list_files(path, recursive=recursive)
            items = [
                {
                    "path": str(item.relative_to(self.project_root)),
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
            obs.context_summary = self._summarize_observation(obs)
            return obs

        if tool_name == "search_files":
            path = self._resolve_user_path(str(arguments.get("path") or "."))
            pattern = str(arguments.get("pattern") or arguments.get("query") or "").strip()
            if not pattern:
                pattern = self._search_pattern_from_text(original)
            if not pattern:
                obs = AgentObservation(
                    tool_name=tool_name,
                    arguments={"path": str(path), "pattern": pattern},
                    preview="no search pattern",
                    data={"pattern": pattern, "candidates": [], "matches": []},
                )
                obs.context_summary = self._summarize_observation(obs)
                return obs
            candidates = self._rank_candidates_in_path(path, pattern, limit=20)
            data: dict[str, Any] = {"pattern": pattern, "candidates": [candidate.to_dict() for candidate in candidates]}
            if not candidates:
                matches = FileOperationGuard(project_root=self.project_root).search_files(path, pattern)
                data["matches"] = matches
                preview = f"{len(matches)} text matches" if matches else "no matches"
            else:
                preview = self._candidate_observation_preview(candidates)
            obs = AgentObservation(
                tool_name=tool_name,
                arguments={"path": str(path), "pattern": pattern},
                preview=preview,
                data=data,
            )
            obs.context_summary = self._summarize_observation(obs)
            return obs

        if tool_name == "candidate_subtitles":
            path = self._resolve_user_path(str(arguments.get("path") or "."))
            query = str(arguments.get("query") or "").strip() or original
            candidates = self._rank_candidates_in_path(path, query, limit=20)
            obs = AgentObservation(
                tool_name=tool_name,
                arguments={"path": str(path), "query": query},
                preview=self._candidate_observation_preview(candidates),
                data={"query": query, "candidates": [candidate.to_dict() for candidate in candidates]},
            )
            obs.context_summary = self._summarize_observation(obs)
            return obs

        if tool_name == "recent_translations":
            records = self._recent_translation_records()
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
            obs.context_summary = self._summarize_observation(obs)
            return obs

        if tool_name == "read_file_preview":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            limit = int(arguments.get("limit") or 2000)
            text = FileOperationGuard(project_root=self.project_root).read_file(path, limit=limit)
            obs = AgentObservation(
                tool_name=tool_name,
                arguments={"path": str(path), "limit": limit},
                preview=f"preview {path.relative_to(self.project_root)} ({len(text)} chars)",
                data={"path": str(path.relative_to(self.project_root)), "text": text},
            )
            obs.context_summary = self._summarize_observation(obs)
            return obs

        raise ValueError(f"Unsupported discovery tool: {tool_name}")

    def _format_discovery_observation_for_user(self, observation: AgentObservation) -> str:
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

    def _rank_candidates_in_path(self, path: Path, query: str, *, limit: int) -> list[FileCandidate]:
        guard = FileOperationGuard(project_root=self.project_root)
        files = guard.list_files(path, recursive=True, limit=500)
        return rank_file_candidates(files, query, project_root=self.project_root, limit=limit)

    def _candidate_observation_preview(self, candidates: list[FileCandidate]) -> str:
        if not candidates:
            return "no candidates"
        strong = strong_subtitle_candidates(candidates)
        if len(strong) == 1:
            return f"selected {executable_subtitle_path(strong[0])}"
        return f"{len(candidates)} candidates, top {candidates[0].path}"

    def _recent_translation_records(self, *, limit: int = 5) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for event in reversed(self.session.events):
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

    def _handle_decision(self, decision: dict[str, Any], *, original: str) -> None:
        action = decision["action"]
        if action == "final_tool_call":
            self._handle_decision(self._decision_from_final_tool_call(decision, original=original), original=original)
            return
        if action == "respond":
            message = str(decision.get("message") or "")
            self.console.print(message or "Done.")
            self._record_event("assistant", message, {"decision": action})
            return
        if action == "ask_user":
            message = str(decision.get("message") or "I need more information.")
            self.console.print(message)
            self._record_event("ask_user", message, {"decision": action})
            return
        if action == "plan":
            self._store_plan(decision, original=original)
            return
        if action == "tool_call":
            if self.session.mode == "plan":
                self._store_plan(
                    {
                        "action": "plan",
                        "message": decision.get("message") or "Proposed tool action.",
                        "tool_calls": [
                            {
                                "tool_name": decision.get("tool_name"),
                                "arguments": decision.get("arguments") or {},
                            }
                        ],
                    },
                    original=original,
                )
                return
            tool_name = str(decision.get("tool_name") or "")
            arguments = dict(decision.get("arguments") or {})
            message = str(decision.get("message") or "").strip()
            if message and not self._tool_prints_own_progress(tool_name):
                self.console.print(message)
            result = self._run_tool_call(
                tool_name,
                arguments,
                original=original,
            )
            if result:
                self.console.print(result)
            return
        raise ValueError(f"Unsupported agent decision action: {action}")

    def _run_tool_call(self, tool_name: str, arguments: dict[str, Any], *, original: str) -> str:
        arguments = self._arguments_with_text_overrides(arguments, original)
        if tool_name == "translate_file":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            self._translate_file(path, original=original, arguments=arguments)
            return ""
        if tool_name == "translate_series":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            self._translate_series_tool(
                path,
                original=original,
                recursive=self._bool_argument(arguments.get("recursive"), "recursive") or False,
                overwrite=self._bool_argument(arguments.get("overwrite"), "overwrite") or False,
                dry_run=self._bool_argument(arguments.get("dry_run"), "dry_run") or False,
                suffixes=self._series_suffixes_from_argument(arguments.get("suffixes")),
                arguments=arguments,
            )
            return ""
        if tool_name == "diagnose_path":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            self._diagnose_path(path, original=original)
            return ""
        if tool_name == "diagnose_text":
            text = str(arguments.get("text") or "")
            report = diagnose_text(text)
            self.console.print(format_diagnostic_report(report))
            self._record_event("diagnose_text", original, {"diagnosis": report.diagnosis})
            return ""
        if tool_name == "edit_subtitle":
            target_path = self._resolve_user_path(str(arguments.get("path") or ""))
            instruction = str(arguments.get("instruction") or "").strip()
            self._edit_generated_subtitle(target_path=target_path, instruction=instruction, original=original)
            return ""
        if tool_name == "switch_profile":
            self._handle_profile_command(str(arguments.get("profile") or ""))
            return ""
        if tool_name == "list_profiles":
            self._print_profiles()
            return ""

        guard = FileOperationGuard(project_root=self.project_root)
        if tool_name == "read_file":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            return guard.read_file(path)
        if tool_name == "list_files":
            path = self._resolve_user_path(str(arguments.get("path") or "."))
            files = guard.list_files(path, recursive=bool(arguments.get("recursive", False)))
            return "\n".join(str(item.relative_to(self.project_root)) for item in files)
        if tool_name == "search_files":
            observation = self._run_discovery_tool_call(tool_name, arguments, original=original)
            return self._format_discovery_observation_for_user(observation)
        if tool_name in {"recent_translations", "candidate_subtitles", "read_file_preview"}:
            observation = self._run_discovery_tool_call(tool_name, arguments, original=original)
            return self._format_discovery_observation_for_user(observation)
        if tool_name == "create_file":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            result = guard.create_file(path, str(arguments.get("content") or ""))
            self._print_file_op_result(result)
            self._record_file_op_event(result, original)
            return ""
        if tool_name == "append_file":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            result = guard.append_file(path, str(arguments.get("content") or ""))
            self._print_file_op_result(result)
            self._record_file_op_event(result, original)
            return ""
        if tool_name == "replace_in_file":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            result = guard.replace_in_file(
                path,
                str(arguments.get("old") or ""),
                str(arguments.get("new") or ""),
            )
            self._print_file_op_result(result)
            self._record_file_op_event(result, original)
            return ""
        if tool_name == "rename_path":
            old_path = self._resolve_user_path(str(arguments.get("old_path") or ""))
            new_path = self._resolve_user_path(str(arguments.get("new_path") or ""))
            result = guard.rename_path(old_path, new_path)
            self._print_file_op_result(result)
            self._record_file_op_event(result, original)
            return ""
        if tool_name == "delete_file":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            result = guard.delete_file(path)
            self._print_file_op_result(result)
            self._record_file_op_event(result, original)
            return ""

        raise ValueError(f"Unsupported agent tool: {tool_name or '<missing>'}")

    def _tool_prints_own_progress(self, tool_name: str) -> bool:
        return tool_name in {"translate_file", "translate_series"}

    def _translate_series_tool(
        self,
        path: Path,
        *,
        original: str,
        recursive: bool,
        overwrite: bool,
        dry_run: bool,
        suffixes: set[str] | None,
        arguments: dict[str, Any],
    ) -> None:
        files = discover_series_files(path, recursive=recursive, suffixes=suffixes)
        if not files:
            suffix_label = ", ".join(sorted(suffixes)) if suffixes else "subtitle"
            self.console.print(f"[bold yellow]No {suffix_label} files found.[/bold yellow]")
            self._record_event("series_empty", original, {"path": str(path), "suffixes": sorted(suffixes) if suffixes else None})
            return
        values = self._translation_values_for_tool(arguments)
        if dry_run:
            values["dry_run"] = True
        self._print_translation_start(
            original=original,
            values=values,
            file_count=len(files),
            suffixes={file_path.suffix.lower() for file_path in files},
            series=True,
        )
        self.console.print(
            "[bold green]Series:[/bold green] "
            f"{len(files)} file(s), profile={self.profile or 'default'}, "
            f"target={values['target_language']}"
        )
        undo_snapshots: dict[Path, tuple[bool, Path | None]] = {}
        if not bool(values["dry_run"]):
            for file_path in files:
                output_path = resolve_series_output_path(
                    file_path,
                    output_format=values.get("output_format"),
                    bilingual=bool(values["bilingual"]),
                )
                if output_path.exists() and not overwrite:
                    continue
                undo_snapshots[output_path] = self._translation_output_undo_snapshot(output_path)
        result = translate_series(
            root=path,
            values=values,
            backend_factory=lambda: _runtime_options.build_backend_from_values(values),
            console=self.console,
            recursive=recursive,
            overwrite=overwrite,
            suffixes=suffixes,
        )
        self._print_series_summary(result)
        self._print_series_completion(result)
        self._record_event(
            "series",
            original,
            {
                "path": str(path),
                "processed": result.processed_count,
                "skipped": result.skipped_count,
                "failures": result.failure_count,
                "suffixes": sorted(suffixes) if suffixes else None,
                "recursive": recursive,
                "bilingual": bool(values["bilingual"]),
                "source_language": str(values["source_language"]),
                "target_language": str(values["target_language"]),
                "output_format": values.get("output_format"),
            },
        )
        operation_group_id = uuid.uuid4().hex
        for item in result.processed:
            if item.output_path is None:
                continue
            existed_before, backup_path = undo_snapshots.get(item.output_path, (False, None))
            self._record_translation_output_file_operation(
                output_path=item.output_path,
                original=original,
                existed_before=existed_before,
                backup_path=backup_path,
                group_id=operation_group_id,
            )

    def _edit_generated_subtitle(self, *, target_path: Path, instruction: str, original: str) -> None:
        if not instruction:
            raise ValueError("Subtitle edit needs an instruction.")
        if not is_generated_subtitle(target_path):
            raise ValueError(
                "Agent edits are limited to generated subtitles such as *.translated.* or *.bilingual.*."
            )
        values = dict(self.values)
        values["dry_run"] = False
        backend = _runtime_options.build_backend_from_values(values)
        if backend is None:
            raise RuntimeError("Subtitle edits require a model backend.")
        result = edit_generated_subtitle(
            target_path=target_path,
            instruction=instruction,
            backend=backend,
            values=values,
            project_root=self.project_root,
        )
        self.console.print(f"[bold green]Edited:[/bold green] {result.target_path}")
        self.console.print(f"[bold green]Backup:[/bold green] {result.backup_path}")
        if result.translation_memory_path is not None:
            self.console.print(f"[bold green]Translation memory:[/bold green] {result.translation_memory_path}")
        if result.edit_notes:
            self.console.print(f"[bold green]Notes:[/bold green] {result.edit_notes}")
        self._record_event(
            "edit",
            original,
            {
                "target_path": str(result.target_path),
                "backup_path": str(result.backup_path),
            },
        )

    def _undo_last_operation(self) -> None:
        """Undo the most recent file operation by restoring from its backup."""
        undo_target = self._latest_file_operation_to_undo()
        if undo_target is None:
            self.console.print("[bold yellow]Nothing to undo.[/bold yellow]")
            self._record_event("undo", "/undo", {"result": "nothing_to_undo"})
            return

        undo_targets = self._file_operation_group_targets(undo_target)
        undone_targets: list[dict[str, Any]] = []
        for target in undo_targets:
            if not self._undo_file_operation(target):
                return
            target["undone"] = True
            undone_targets.append(target)

        action = str(undo_target.get("action") or "")
        path = Path(str(undo_target.get("path") or ""))
        self._record_event(
            "undo",
            "/undo",
            {"result": "ok", "action": action, "path": str(path), "count": len(undone_targets)},
        )

    def _latest_file_operation_to_undo(self) -> dict[str, Any] | None:
        for event in reversed(self.session.events):
            if event.get("kind") != "file_operation":
                continue
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            if data.get("undone"):
                continue
            return data
        return None

    def _file_operation_group_targets(self, undo_target: dict[str, Any]) -> list[dict[str, Any]]:
        group_id = str(undo_target.get("group_id") or "")
        if not group_id:
            return [undo_target]
        targets: list[dict[str, Any]] = []
        for event in self.session.events:
            if event.get("kind") != "file_operation":
                continue
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            if data.get("undone"):
                continue
            if data.get("group_id") == group_id:
                targets.append(data)
        return list(reversed(targets))

    def _undo_file_operation(self, undo_target: dict[str, Any]) -> bool:
        action = str(undo_target.get("action") or "")
        path = Path(str(undo_target.get("path") or ""))
        new_path_str = str(undo_target.get("new_path") or "")
        backup_str = str(undo_target.get("backup_path") or "")

        if action == "created":
            if path.exists():
                path.unlink()
                self.console.print(f"[bold green]Undo created:[/bold green] deleted {path}")
            else:
                self.console.print(f"[bold yellow]Created file no longer exists:[/bold yellow] {path}")

        elif action in {"appended", "modified", "renamed"}:
            backup = Path(backup_str) if backup_str else None
            if backup is None or not backup.exists():
                self.console.print(
                    f"[bold yellow]Backup not found for {action}:[/bold yellow] {path}. Cannot undo."
                )
                self._record_event("undo", "/undo", {"result": "backup_missing", "path": str(path)})
                return False
            if action == "renamed":
                new_path = Path(new_path_str) if new_path_str else None
                if new_path is not None and new_path.exists():
                    new_path.unlink()
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, path)
                self.console.print(f"[bold green]Undo renamed:[/bold green] restored {path}")
            else:
                shutil.copy2(backup, path)
                self.console.print(f"[bold green]Undo {action}:[/bold green] restored {path}")

        elif action == "deleted":
            backup = Path(backup_str) if backup_str else None
            if backup is None or not backup.exists():
                self.console.print(
                    f"[bold yellow]Backup not found for deleted file:[/bold yellow] {path}. Cannot undo."
                )
                self._record_event("undo", "/undo", {"result": "backup_missing", "path": str(path)})
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, path)
            self.console.print(f"[bold green]Undo deleted:[/bold green] restored {path}")

        else:
            self.console.print(f"[bold yellow]Unknown operation type:[/bold yellow] {action}")
            self._record_event("undo", "/undo", {"result": "unknown_action", "action": action})
            return False

        return True

    def _record_file_op_event(self, result: FileOpResult, original: str) -> None:
        self._record_event(
            "file_operation",
            original,
            {
                "action": result.action,
                "path": str(result.path),
                "new_path": str(result.new_path) if result.new_path is not None else None,
                "backup_path": str(result.backup_path) if result.backup_path is not None else None,
            },
        )

    def _translation_output_undo_snapshot(self, output_path: Path) -> tuple[bool, Path | None]:
        if not output_path.exists():
            return False, None
        return True, self._backup_translation_output_for_undo(output_path)

    def _backup_translation_output_for_undo(self, output_path: Path) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        resolved_output = output_path.resolve()
        resolved_root = self.project_root.resolve()
        try:
            relative = resolved_output.relative_to(resolved_root)
        except ValueError:
            digest = hashlib.sha1(str(resolved_output).encode("utf-8")).hexdigest()[:12]
            relative = Path("__external__") / digest / output_path.name
        backup_path = self.project_root / ".subbake" / "agent" / "backups" / timestamp / relative
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if backup_path.exists():
            backup_path = backup_path.with_name(
                f"{backup_path.stem}-{output_path.stat().st_mtime_ns}{backup_path.suffix}"
            )
        shutil.copy2(output_path, backup_path)
        return backup_path

    def _record_translation_output_file_operation(
        self,
        *,
        output_path: Path,
        original: str,
        existed_before: bool,
        backup_path: Path | None,
        group_id: str | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "action": "modified" if existed_before else "created",
            "path": str(output_path),
        }
        if backup_path is not None:
            data["backup_path"] = str(backup_path)
        if group_id is not None:
            data["group_id"] = group_id
        self._record_event("file_operation", original, data)

    def _reference_context(self, line: str) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        for path in self._extract_references(line):
            references.append(
                {
                    "raw": str(path),
                    "path": str(path),
                    "exists": path.exists(),
                    "is_dir": path.is_dir(),
                    "is_file": path.is_file(),
                    "suffix": path.suffix.lower(),
                    "name": path.name,
                    "generated_subtitle": is_generated_subtitle(path),
                }
            )
        return references

    def _tool_specs(self, categories: list[str] | None = None) -> list[dict[str, Any]]:
        return build_tool_specs(categories)

    def _resolve_user_path(self, value: str) -> Path:
        return resolve_user_path(value, cwd=self.cwd)

    def _series_suffixes_from_argument(self, value: object) -> set[str] | None:
        return series_suffixes_from_argument(value)

    def _arguments_with_text_overrides(self, arguments: dict[str, Any], original: str) -> dict[str, Any]:
        return arguments_with_text_overrides(arguments, original)

    def _translation_values_for_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return translation_values_for_tool(arguments, self.values)

    def _bool_argument(self, value: object, name: str) -> bool | None:
        return bool_argument(value, name)

    def _output_format_from_argument(self, value: object) -> str | None:
        return output_format_from_argument(value)

    def _translation_arguments_from_text(self, line: str) -> dict[str, Any]:
        return translation_arguments_from_text(line)

    def _bilingual_requested(self, line: str) -> bool:
        return bilingual_requested(line)

    def _monolingual_requested(self, line: str) -> bool:
        return monolingual_requested(line)

    def _target_language_from_text(self, line: str, *, source_language: str | None = None) -> str | None:
        return target_language_from_text(line, source_language=source_language)

    def _source_language_from_text(self, line: str) -> str | None:
        return source_language_from_text(line)

    def _target_language_for_bilingual_pair(
        self,
        line: str,
        *,
        source_language: str | None,
    ) -> str | None:
        return target_language_for_bilingual_pair(line, source_language=source_language)

    def _output_format_from_text(self, line: str) -> str | None:
        return output_format_from_text(line)

    def _line_without_output_format_phrases(self, line: str) -> str:
        return line_without_output_format_phrases(line)

    def _print_translation_start(
        self,
        *,
        original: str,
        values: dict[str, Any],
        file_count: int,
        suffixes: set[str],
        series: bool,
        path: Path | None = None,
    ) -> None:
        print_translation_start(
            self.console, values=values, file_count=file_count,
            suffixes=suffixes, series=series, path=path, original=original,
        )

    def _print_file_completion(self, *, output_path: Path | None, dry_run: bool) -> None:
        print_file_completion(self.console, output_path=output_path, dry_run=dry_run)

    def _print_series_completion(self, result) -> None:
        print_series_completion(self.console, result)

    def _file_count_label(self, file_count: int, suffixes: set[str]) -> str:
        from .ui import _file_count_label
        return _file_count_label(file_count, suffixes)

    def _render_mode_label(self, original: str, values: dict[str, Any]) -> str:
        return render_mode_label(original, values)

    def _store_plan(self, decision: dict[str, Any], *, original: str) -> None:
        tool_calls = [
            call
            for call in decision.get("tool_calls") or []
            if isinstance(call, dict)
        ]
        plan = {
            "message": str(decision.get("message") or "Proposed plan."),
            "tool_calls": tool_calls,
            "created_at": _now_iso(),
            "original": original,
        }
        self.session.pending_plan = plan
        self.console.print("<proposed_plan>")
        self.console.print(plan["message"])
        if tool_calls:
            self.console.print("─" * 40)
            for call in tool_calls:
                self._print_tool_call_preview(call)
            self.console.print("─" * 40)
        self.console.print("</proposed_plan>")
        self.console.print("Use /approve to execute this plan, or /reject to discard it.")
        self._record_event("plan", original, {"tool_calls": tool_calls})

    def _print_tool_call_preview(self, call: dict[str, Any]) -> None:
        print_tool_call_preview(self.console, call)

    def _print_content_preview(self, content: str) -> None:
        from .ui import _print_content_preview
        _print_content_preview(self.console, content)

    def _print_replace_preview(self, old: str, new: str) -> None:
        from .ui import _print_replace_preview
        _print_replace_preview(self.console, old, new)

    def _print_translation_options(self, arguments: dict[str, Any]) -> None:
        from .ui import _print_translation_options
        _print_translation_options(self.console, arguments)

    def _handle_plan_command(self, rest: str) -> None:
        option = rest.strip().lower()
        if option in {"off", "false", "0", "disable"}:
            self.session.mode = "chat"
            self.console.print("[bold green]Plan mode off.[/bold green]")
            self._record_event("mode", "/plan off", {"mode": "chat"})
            return
        self.session.mode = "plan"
        self.console.print("[bold green]Plan mode on.[/bold green] Mutating tools will be proposed, not executed.")
        self._record_event("mode", "/plan", {"mode": "plan"})

    def _approve_pending_plan(self) -> None:
        plan = self.session.pending_plan
        if not plan:
            self.console.print("[bold yellow]No pending plan to approve.[/bold yellow]")
            return
        tool_calls = [
            call
            for call in plan.get("tool_calls") or []
            if isinstance(call, dict)
        ]
        if not tool_calls:
            self.console.print("[bold yellow]Pending plan has no executable tool calls.[/bold yellow]")
            self.session.pending_plan = None
            return
        for call in tool_calls:
            result = self._run_tool_call(
                str(call.get("tool_name") or ""),
                dict(call.get("arguments") or {}),
                original=str(plan.get("original") or "/approve"),
            )
            if result:
                self.console.print(result)
        self.session.pending_plan = None
        self._record_event("approve", "/approve", {"executed": len(tool_calls)})

    def _reject_pending_plan(self) -> None:
        if self.session.pending_plan is None:
            self.console.print("[bold yellow]No pending plan to reject.[/bold yellow]")
            return
        self.session.pending_plan = None
        self.console.print("[bold green]Pending plan discarded.[/bold green]")
        self._record_event("reject", "/reject")

    def _translate_file(self, path: Path, *, original: str, arguments: dict[str, Any] | None = None) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Subtitle file not found: {path}")
        values = self._translation_values_for_tool(arguments or {})
        self._print_translation_start(
            original=original,
            values=values,
            file_count=1,
            suffixes={path.suffix.lower()},
            series=False,
            path=path,
        )
        backend = _runtime_options.build_backend_from_values(values)
        options = build_pipeline_options(
            input_path=path,
            output_path=None,
            values=values,
        )
        pipeline = SubtitlePipeline(
            backend=backend,
            options=options,
            dashboard=Dashboard(console=self.console),
        )
        existed_before, backup_path = (
            self._translation_output_undo_snapshot(pipeline.output_path)
            if not options.dry_run
            else (False, None)
        )
        result = pipeline.run()
        if result.dry_run:
            self.console.print(f"[bold yellow]Dry run:[/bold yellow] {len(result.planned_batches)} batch(es) planned.")
            self._print_file_completion(output_path=None, dry_run=True)
        else:
            self.console.print(f"[bold green]Output:[/bold green] {result.output_path}")
            self._print_file_completion(output_path=result.output_path, dry_run=False)
        self._record_event(
            "translate_file",
            original,
            {
                "input_path": str(path),
                "output_path": str(result.output_path) if result.output_path else None,
                "bilingual": bool(values["bilingual"]),
                "source_language": str(values["source_language"]),
                "target_language": str(values["target_language"]),
                "output_format": values.get("output_format"),
            },
        )
        if result.output_path is not None and not result.dry_run:
            self._record_translation_output_file_operation(
                output_path=result.output_path,
                original=original,
                existed_before=existed_before,
                backup_path=backup_path,
            )

    def _print_file_op_result(self, result: FileOpResult) -> None:
        print_file_op_result(self.console, result)

    def _diagnose_path(self, path: Path, *, original: str) -> None:
        report = diagnose_path(path)
        self.console.print(format_diagnostic_report(report))
        self._record_event("diagnose_path", original, {"path": str(path), "diagnosis": report.diagnosis})

    def _handle_profile_command(self, rest: str) -> None:
        profile_name = rest.strip()
        if not profile_name:
            if self.interactive:
                self._open_profile_picker()
            else:
                self._print_profiles(include_new=True)
            return
        if profile_name.casefold() == "new":
            self._create_profile_interactively()
            return
        self._switch_profile(profile_name)

    def _switch_profile(self, profile_name: str) -> None:
        if self.config is None or profile_name not in self.config.profiles:
            raise ValueError(f"Config profile '{profile_name}' was not found.")
        self.profile = profile_name
        self.values = self._values_for_profile(profile_name)
        self.session.profile = profile_name
        self.console.print(
            f"[bold green]Profile switched:[/bold green] {profile_name} "
            f"({self.values['provider']} / {self.values['model']})"
        )

    def _open_profile_picker(self) -> None:
        options: list[tuple[str, str]] = []
        if self.config is not None:
            for name in sorted(self.config.profiles):
                values = self._values_for_profile(name)
                marker = "* " if name == self.profile else ""
                options.append((name, f"{marker}{name}: {values['provider']} / {values['model']}"))
        options.append((NEW_PROFILE_VALUE, "new"))
        selected = self._select_from_list(
            "Model profile",
            options,
            default=self.profile if self.profile is not None else NEW_PROFILE_VALUE,
        )
        if selected == NEW_PROFILE_VALUE:
            self._create_profile_interactively()
        elif selected:
            self._switch_profile(selected)

    def _create_profile_interactively(self) -> None:
        if not self.interactive:
            self.console.print("Profile creation is available from the interactive /profile picker.")
            return
        config_path = self._config_path_for_profile_write()
        profile_name_prompt = self._prompt_text("New profile", "Profile name", default="")
        if profile_name_prompt is None or not profile_name_prompt.strip():
            self._cancel_profile_creation()
            return
        profile_name = profile_name_prompt.strip()
        if self.config is not None and profile_name in self.config.profiles:
            raise ValueError(f"Config profile '{profile_name}' already exists.")

        provider_default = str(self.values.get("provider") or "mock")
        provider_prompt = self._prompt_text(
            "New profile",
            "Provider",
            default=provider_default,
            completions=PROFILE_PROVIDER_OPTIONS,
        )
        if provider_prompt is None:
            self._cancel_profile_creation()
            return
        provider = provider_prompt.strip() or provider_default

        model_default = str(self.values.get("model") or "mock-zh")
        model_prompt = self._prompt_text("New profile", "Model", default=model_default)
        if model_prompt is None:
            self._cancel_profile_creation()
            return
        model = model_prompt.strip() or model_default

        api_key_env_default = _default_api_key_env(provider)
        api_key_env_prompt = self._prompt_text(
            "New profile",
            "API key environment variable",
            default=api_key_env_default,
            completions=PROFILE_API_KEY_ENV_OPTIONS,
        )
        if api_key_env_prompt is None:
            self._cancel_profile_creation()
            return
        api_key_env = api_key_env_prompt.strip()

        base_url_prompt = self._prompt_text("New profile", "Base URL", default="")
        if base_url_prompt is None:
            self._cancel_profile_creation()
            return
        base_url = base_url_prompt.strip()

        target_language_default = str(self.values.get("target_language") or "Chinese")
        target_language_prompt = self._prompt_text(
            "New profile",
            "Target language",
            default=target_language_default,
            completions=PROFILE_TARGET_LANGUAGE_OPTIONS,
        )
        if target_language_prompt is None:
            self._cancel_profile_creation()
            return
        target_language = target_language_prompt.strip() or target_language_default

        profile_values = {
            "provider": provider,
            "model": model,
            "api_key_env": api_key_env,
            "base_url": base_url,
            "target_language": target_language,
        }
        self._append_profile_to_config(config_path, profile_name, profile_values)
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.session.config_path = str(config_path)
        self._switch_profile(profile_name)
        self.console.print(f"[bold green]Profile created:[/bold green] {profile_name} in {config_path}")

    def _cancel_profile_creation(self) -> None:
        self.console.print("[bold yellow]Profile creation cancelled.[/bold yellow]")

    def _maybe_offer_config_bootstrap(self) -> None:
        if not self.interactive or self.config is not None:
            return
        selected = self._select_from_list(
            "No SubBake config found",
            [
                (CONFIG_BOOTSTRAP_CREATE, "create a model profile"),
                (CONFIG_BOOTSTRAP_SKIP, "continue with mock defaults"),
            ],
            default=CONFIG_BOOTSTRAP_CREATE,
        )
        if selected == CONFIG_BOOTSTRAP_CREATE:
            self._create_profile_interactively()
            self._record_event("config_bootstrap", "create", {"config_path": self.session.config_path})
            return
        self.console.print("[bold yellow]No config created.[/bold yellow] Continuing with built-in mock defaults.")
        self._record_event("config_bootstrap", "skip")

    def _config_path_for_profile_write(self) -> Path:
        if self.config_path is not None:
            return self.config_path
        candidates = _config.global_config_candidates()
        if not candidates:
            raise RuntimeError("No global config path is available on this platform.")
        return candidates[0]

    def _append_profile_to_config(self, path: Path, profile_name: str, values: dict[str, str]) -> None:
        path = path.expanduser()
        if path.exists():
            existing_config = load_app_config(path)
            if profile_name in existing_config.profiles:
                raise ValueError(f"Config profile '{profile_name}' already exists.")
            should_set_default = existing_config.default_profile is None and not existing_config.profiles
            content = path.read_text(encoding="utf-8")
        else:
            should_set_default = True
            content = ""
        prefix = content
        if should_set_default:
            prefix = _prepend_default_profile(prefix, profile_name)
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        if prefix and not prefix.endswith("\n\n"):
            prefix += "\n"

        lines = [f"[profiles.{_toml_key(profile_name)}]"]
        for key in ("provider", "model", "api_key_env", "base_url", "target_language"):
            value = str(values.get(key) or "").strip()
            if value:
                lines.append(f"{key} = {_toml_string(value)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = prefix + "\n".join(lines) + "\n"
        path.write_text(content, encoding="utf-8")
        _verify_write_text(path, content)

    def _print_profiles(self, *, include_new: bool = False) -> None:
        if self.config is None or not self.config.profiles:
            self.console.print("No configured profiles were found. Using built-in mock defaults.")
            if include_new:
                self.console.print("Use /profile in an interactive terminal and choose new to create one.")
            return
        self.console.print("[bold green]Profiles:[/bold green]")
        for name in sorted(self.config.profiles):
            marker = "*" if name == self.profile else " "
            values = self._values_for_profile(name)
            self.console.print(f" {marker} {name}: {values['provider']} / {values['model']}")
        if include_new:
            self.console.print("   new: create a new model profile from the interactive picker")

    def _print_sessions(self) -> None:
        sessions = self.store.list_sessions()[-10:]
        if not sessions:
            self.console.print("No agent sessions found.")
            return
        self.console.print("[bold green]Recent sessions:[/bold green]")
        for path in sessions:
            try:
                session = self.store.load(path)
            except Exception:
                continue
            marker = "*" if session.id == self.session.id else " "
            title = self._session_title(session)
            self.console.print(
                f" {marker} {session.id}  profile={session.profile or 'default'}  {title}"
            )

    def _handle_session_command(self, rest: str) -> None:
        key = rest.strip()
        if key:
            self._switch_session_by_key(key)
            return
        options = self._session_options(limit=30)
        if not options:
            self.console.print("No agent sessions found.")
            return
        if self.interactive:
            selected = self._select_from_list("Sessions", options, default=self.session.id)
            if selected:
                self._switch_session_by_key(selected)
            return
        self._print_sessions()

    def _switch_session_by_key(self, key: str) -> None:
        path = self._find_session_path(key)
        if path is None:
            raise ValueError(f"Agent session '{key}' was not found.")
        session = self.store.load(path)
        self._activate_session(session)
        self.console.print(f"[bold green]Session switched:[/bold green] {session.id}  {self._session_title(session)}")

    def _find_session_path(self, key: str) -> Path | None:
        for path in self.store.list_sessions():
            if key in {path.stem, path.name, str(path)}:
                return path
        candidate = Path(key).expanduser()
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        if candidate.exists():
            return candidate
        return None

    def _activate_session(self, session: AgentSession) -> None:
        self.session = session
        if session.config_path:
            config_path = Path(session.config_path).expanduser()
            if config_path.exists():
                self.config_path = config_path
                self.config = self._load_config(config_path)
        if session.profile is not None:
            if self.config is not None and session.profile in self.config.profiles:
                self.profile = session.profile
                self.values = self._values_for_profile(session.profile)
            else:
                self.console.print(
                    f"[bold yellow]Session profile unavailable:[/bold yellow] {session.profile}. "
                    f"Keeping {self.profile or 'default'}."
                )
                self.session.profile = self.profile
        elif self.config is None or not self.config.profiles:
            self.profile = None
            self.values = self._values_for_profile(None)

    def _session_options(self, *, limit: int) -> list[tuple[str, str]]:
        options: list[tuple[str, str]] = []
        for path in reversed(self.store.list_sessions()[-limit:]):
            try:
                session = self.store.load(path)
            except Exception:
                continue
            marker = "* " if session.id == self.session.id else ""
            updated = session.updated_at.replace("T", " ")[:19]
            label = f"{marker}{self._session_title(session)}  ({session.profile or 'default'}, {updated})"
            options.append((session.id, label))
        return options

    def _session_title(self, session: AgentSession) -> str:
        for event in session.events:
            data = event.get("data")
            if isinstance(data, dict) and isinstance(data.get("summary"), str) and data["summary"].strip():
                return _short_title(data["summary"])
        for event in session.events:
            if event.get("kind") == "user" and isinstance(event.get("input"), str) and event["input"].strip():
                return _short_title(event["input"])
        return "New session"

    def _resume_latest_session(self) -> None:
        latest = self.store.latest()
        if latest is None:
            self.console.print("[bold yellow]No previous agent session found.[/bold yellow]")
            return
        self._activate_session(latest)
        self.console.print(f"[bold green]Resumed session:[/bold green] {latest.id}  {self._session_title(latest)}")
        self._record_event("resume", "/resume", {"session_id": latest.id})

    def _print_help(self) -> None:
        _print_help_fn(self.console)

    def _print_series_summary(self, result) -> None:
        print_series_summary(self.console, result)
        for item in result.skipped[:5]:
            self.console.print(f"  skipped {item.input_path}: {item.reason}")
        for item in result.failures[:5]:
            self.console.print(f"  failed {item.input_path}: {item.reason}")

    def _load_config(self, path: Path | None) -> AppConfig | None:
        if path is None:
            return None
        return load_app_config(path)

    def _initial_profile(self) -> str | None:
        if self.config is None or not self.config.profiles:
            return None
        latest = self.store.latest()
        if latest is not None and latest.profile in self.config.profiles:
            return latest.profile
        if self.config.default_profile is not None and self.config.default_profile in self.config.profiles:
            return self.config.default_profile
        if len(self.config.profiles) == 1:
            return next(iter(self.config.profiles))
        names = sorted(self.config.profiles)
        return self._choose(
            "Choose profile",
            [(name, name) for name in names],
            default=names[0],
        )

    def _values_for_profile(self, profile: str | None) -> dict[str, Any]:
        if self.config is None:
            return merge_translation_values()
        config_values, _ = resolve_command_config(
            self.config,
            profile=profile,
            allowed_keys=TRANSLATE_CONFIG_KEYS,
        )
        return merge_translation_values(config_values)

    def _load_or_create_session(self, *, resume: bool) -> AgentSession:
        if resume:
            latest = self.store.latest()
            if latest is not None:
                self.console.print(f"[bold green]Resumed session:[/bold green] {latest.id}")
                return latest
            self.console.print("[bold yellow]No previous agent session found. Starting a new one.[/bold yellow]")
        return self.store.create(
            cwd=self.cwd,
            profile=self.profile,
            config_path=self.config_path,
        )

    def _record_event(self, kind: str, input_text: str, data: dict[str, Any] | None = None) -> None:
        self.session.events.append(
            {
                "kind": kind,
                "input": input_text,
                "data": data or {},
                "created_at": _now_iso(),
            }
        )

    def _read_line(self) -> str:
        mode = "|plan" if self.session.mode == "plan" else ""
        prompt = f"sbake[{self.profile or 'default'}{mode}]> "
        prompt_toolkit_prompt = _prompt_toolkit_prompt()
        if self.interactive and prompt_toolkit_prompt is not None:
            return prompt_toolkit_prompt(
                prompt,
                completer=_slash_command_completer(),
                key_bindings=self._plan_toggle_key_bindings(),
            )
        return self.console.input(prompt, markup=False)

    def _plan_toggle_key_bindings(self):
        try:
            from prompt_toolkit.filters import has_completions
            from prompt_toolkit.key_binding import KeyBindings
        except Exception:
            return None

        key_bindings = KeyBindings()

        @key_bindings.add("s-tab")
        def _toggle(event) -> None:
            self._toggle_plan_mode()
            event.app.exit(result="")

        def _do_complete(event) -> None:
            buffer = event.current_buffer
            text = buffer.document.text_before_cursor
            if text.startswith("/"):
                match = _unique_slash_command_match(text)
                if match is not None:
                    buffer.delete_before_cursor(len(text))
                    buffer.insert_text(match)
                    return
                matches = _slash_command_matches(text)
                if len(matches) > 1 and buffer.complete_state is None:
                    buffer.start_completion(select_first=True)
                    return
            if buffer.complete_state is None:
                buffer.start_completion(select_first=True)
                return
            completion = buffer.complete_state.current_completion
            if completion is not None:
                buffer.apply_completion(completion)

        key_bindings.add("tab")(_do_complete)

        @key_bindings.add("down", filter=has_completions)
        def _next_completion(event) -> None:
            event.current_buffer.complete_next()

        @key_bindings.add("up", filter=has_completions)
        def _previous_completion(event) -> None:
            event.current_buffer.complete_previous()

        @key_bindings.add("enter", filter=has_completions)
        def _accept_completion(event) -> None:
            buffer = event.current_buffer
            complete_state = buffer.complete_state
            completion = complete_state.current_completion if complete_state is not None else None
            if completion is not None:
                buffer.apply_completion(completion)
            event.app.exit(result=buffer.text)

        return key_bindings

    def _toggle_plan_mode(self) -> None:
        if self.session.mode == "plan":
            self.session.mode = "chat"
        else:
            self.session.mode = "plan"
        self._record_event("mode", "<shift-tab>", {"mode": self.session.mode})
        self.store.save(self.session)

    def _select_from_list(self, title: str, options: list[tuple[str, str]], *, default: str) -> str | None:
        if not options:
            return None
        if self.interactive:
            selected = _prompt_toolkit_inline_picker(title, options, default=default)
            if selected == PICKER_CANCEL_TOKEN:
                return None
            if selected:
                return str(selected)
        return self._choose(title, options, default=default)

    def _prompt_text(
        self,
        title: str,
        text: str,
        *,
        default: str,
        completions: tuple[str, ...] = (),
    ) -> str | None:
        if self.interactive:
            answer = _prompt_toolkit_inline_text(title, text, default=default, completions=completions)
            if answer == PICKER_CANCEL_TOKEN:
                return None
            if answer is not None:
                return str(answer)
        if not self.interactive:
            return default
        suffix = f" [default: {default}]" if default else ""
        answer = self.console.input(f"{text}{suffix}: ").strip()
        return answer or default

    def _choose(self, title: str, options: list[tuple[str, str]], *, default: str) -> str:
        if not self.interactive:
            return default
        self.console.print(f"[bold green]{title}[/bold green]")
        for index, (_, label) in enumerate(options, start=1):
            self.console.print(f"  {index}. {label}")
        answer = self.console.input(f"Choice [default: {default}]: ").strip()
        if not answer:
            return default
        if answer.isdigit():
            option_index = int(answer) - 1
            if 0 <= option_index < len(options):
                return options[option_index][0]
        allowed = {value for value, _ in options}
        if answer in allowed:
            return answer
        return default

    def _extract_references(self, line: str) -> list[Path]:
        paths: list[Path] = []
        for match in REFERENCE_RE.finditer(line):
            raw = next(group for group in match.groups() if group is not None)
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.cwd / path
            paths.append(path)
        return paths

    def _remove_references(self, line: str) -> str:
        return REFERENCE_RE.sub("", line)

    def _split_command(self, line: str) -> tuple[str | None, str]:
        if not line.startswith("/"):
            return None, line
        command, _, rest = line.partition(" ")
        return command.lower(), rest


def start_interactive_agent(*, console: Console, resume: bool = False) -> None:
    SubBakeAgent(console=console, resume=resume).run()

