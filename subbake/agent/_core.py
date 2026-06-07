from __future__ import annotations

import json
import sys
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
)
from .ui import (
    print_file_completion,
    print_file_op_result,
    print_help as _print_help_fn,
    print_series_completion,
    print_series_summary,
    print_translation_start,
    render_mode_label,
    select_from_list,
)
from .session import AgentSession, AgentSessionStore, SESSION_VERSION
from .arg_parser import (
    arguments_with_text_overrides,
    bilingual_requested,
    bool_argument,
    line_without_output_format_phrases,
    monolingual_requested,
    output_format_from_argument,
    output_format_from_text,
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
from .executor import (
    edit_generated_subtitle as _edit_generated_subtitle,
    translate_file as _exec_translate_file,
    translate_series_tool as _exec_translate_series_tool,
)
from .intent import (
    apply_confidence_gate as _apply_confidence_gate,
    classify_intent as _classify_intent,
    intent_to_decision as _intent_to_decision,
)
from .undo import undo_last_operation as _undo_last_operation
from .deterministic import deterministic_decision_from_line as _deterministic_decision_from_line
from .profile import (
    handle_profile_command as _handle_profile_command,
    initial_profile as _initial_profile,
    offer_config_bootstrap as _offer_config_bootstrap,
    print_profiles as _print_profiles,
    values_for_profile as _values_for_profile,
)
from .plan import (
    approve_pending_plan as _approve_pending_plan,
    build_plan_toggle_key_bindings as _build_plan_toggle_key_bindings,
    handle_plan_command as _handle_plan_command,
    reject_pending_plan as _reject_pending_plan,
    store_plan as _store_plan,
    toggle_plan_mode as _toggle_plan_mode,
)
from .discovery import (
    format_discovery_observation_for_user as _format_discovery_observation_for_user,
    run_discovery_tool_call as _run_discovery_tool_call,
)
from .target import (
    source_path_for_translation_reference,
    translation_arguments_for_target,
)
from .session_ops import (
    handle_session_command as _handle_session_command,
    load_or_create_session as _load_or_create_session,
    print_sessions as _print_sessions,
    resume_latest_session as _resume_latest_session,
)
from .text_helpers import (
    extract_references as _extract_references,
    split_command as _split_command,
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
    _AgentLoopTrace,
    _current_completion,
    _language_phrases,
    _matching_picker_choices,
    _now_iso,
    _output_format_patterns,
    _picker_choice,
    _picker_choices,
    _picker_display_parts,
    _picker_prompt,
    _picker_toolbar,
    _prompt_toolkit_prompt,
    _resolve_picker_selection,
    _resolve_text_prompt_value,
    _short_title,
    _slash_command_completer,
    _slash_command_matches,
    _text_prompt,
    _text_prompt_matches,
    _text_prompt_toolbar,
    _trace_arguments,
    _trace_value,
    _unique_slash_command_match,
)
from subbake.config import (
    AppConfig,
    load_app_config,
)
# discover_config_path, discover_project_config_path, global_config_candidates
# are accessed via _config module reference so that tests can patch the source module.
from subbake import config as _config
from subbake.diagnostics import diagnose_path, diagnose_text, format_diagnostic_report
from subbake.editing import is_generated_subtitle
from subbake.file_ops import FileOpResult, FileOperationGuard
# build_backend_from_values is accessed via _runtime_options module reference
# so that tests can patch the source module.
from subbake import runtime_options as _runtime_options
from subbake.models.base_model import MockBackend
from subbake.title_matching import title_tokens_from_text

AGENT_LOOP_MAX_STEPS = 5


# TOOL_CATEGORIES, ALWAYS_AVAILABLE_TOOLS imported from .tool_registry
# AGENT_COMMANDS imported from .trace
PICKER_CANCEL_TOKEN = "__subbake_picker_cancelled__"  # also in agent_trace; kept for internal use


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
        self.profile = _initial_profile(self)
        self.values = _values_for_profile(self, self.profile)
        self.session = _load_or_create_session(self, resume=resume)
        if self.session.profile != self.profile:
            self.session.profile = self.profile
        self.store.save(self.session)

    def run(self) -> None:
        self.console.print(f"[bold green]SubBake agent {__version__}[/bold green]  /help for commands, /exit to quit")
        if self.config_path is not None:
            self.console.print(f"[bold green]Config:[/bold green] {self.config_path}")
        if self.profile is not None:
            self.console.print(f"[bold green]Profile:[/bold green] {self.profile}")
        _offer_config_bootstrap(self)

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
        command, rest = _split_command(line)
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
            _handle_profile_command(self, rest)
            self._record_event("profile", line, {"profile": self.profile})
            return True
        if command == "/session":
            _handle_session_command(self, rest)
            self._record_event("session", line, {"session_id": self.session.id})
            return True
        if command == "/sessions":
            _print_sessions(self)
            self._record_event("sessions", line)
            return True
        if command == "/resume":
            _resume_latest_session(self)
            return True
        if command == "/plan":
            _handle_plan_command(self, rest)
            return True
        if command == "/approve":
            _approve_pending_plan(self)
            return True
        if command == "/reject":
            _reject_pending_plan(self)
            return True
        if command == "/undo":
            _undo_last_operation(self)
            return True
        if command is not None:
            self.console.print("Unknown command. Use /help for available agent controls.")
            self._record_event("unknown_command", line, {"command": command})
            return True

        self._handle_conversational_line(line)
        return True

    def _handle_conversational_line(self, line: str) -> None:
        self._record_event("user", line)
        decision = _deterministic_decision_from_line(self, line)
        if decision is None:
            intent = _classify_intent(self, line)
            if intent is not None:
                if intent.get("category") == "chat":
                    self._handle_chat(line)
                    return
                decision = _intent_to_decision(
                    self, intent, line,
                    run_agent_loop=self._run_agent_loop,
                    agent_loop_max_steps=AGENT_LOOP_MAX_STEPS,
                )
            else:
                decision = self._run_agent_loop(line)
        self._handle_decision(decision, original=line)

    def _handle_chat(self, line: str) -> None:
        """Handle casual chat / non-tool user input with an LLM response."""
        backend = _runtime_options.build_backend_from_values(self.values)
        if backend is None or isinstance(backend, MockBackend):
            fallback = "How can I help with your subtitles?"
            self.console.print(fallback)
            self._record_event("assistant", fallback, {"decision": "respond"})
            return

        messages = self._build_chat_messages(line)
        try:
            if self.interactive:
                with self.console.status("thinking", spinner="dots"):
                    payload, _ = backend.generate_json(messages)
            else:
                payload, _ = backend.generate_json(messages)
            response = str(payload.get("message", "") or "").strip()
            if not response:
                response = "How can I help with your subtitles?"
        except Exception:
            response = "How can I help with your subtitles?"

        if response:
            self.console.print(response)
            self._record_event("assistant", response, {"decision": "respond"})

    def _build_chat_messages(self, line: str) -> list[dict[str, str]]:
        """Build a chat message list with conversation history for casual chat."""
        system_prompt = (
            "你是 SubBake，一个友好的字幕翻译助手。你可以帮助用户翻译字幕文件（.srt、.vtt、.txt），"
            "也可以闲聊。请用用户使用的语言自然、简洁地回复。\n"
            "返回 JSON 格式：{\"message\": \"你的回复\"}\n"
            "You are SubBake, a friendly subtitle translation assistant. You help users translate "
            "subtitle files (.srt, .vtt, .txt) using AI models. You can also chat casually.\n"
            "Respond in the user's language naturally and concisely.\n"
            "Return JSON: {\"message\": \"your response\"}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Pull conversation history from session events (user + assistant pairs only)
        relevant = [
            e for e in self.session.events
            if e.get("kind") in ("user", "assistant")
        ]
        for event in relevant[-16:]:
            kind = event.get("kind", "")
            text = event.get("input", "")
            if kind == "user":
                messages.append({"role": "user", "content": text})
            elif kind == "assistant":
                messages.append({"role": "assistant", "content": text})

        return messages



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

            gated = _apply_confidence_gate(decision, state)
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
                observation = _run_discovery_tool_call(self, tool_name, arguments, original=line)
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
            "recent_events": self.session.events[-20:],
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
            source_path = source_path_for_translation_reference(requested_path) or requested_path
            enriched["path"] = str(source_path)
            return translation_arguments_for_target(
                self,
                path=source_path,
                arguments=enriched,
                series=False,
                user_message=original,
            )
        if tool_name == "translate_series" and enriched.get("path"):
            requested_path = self._resolve_user_path(str(enriched["path"]))
            enriched["path"] = str(requested_path)
            return translation_arguments_for_target(
                self,
                path=requested_path,
                arguments=enriched,
                series=True,
                user_message=original,
            )
        return enriched

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
            _store_plan(self, decision, original=original)
            return
        if action == "tool_call":
            if self.session.mode == "plan":
                _store_plan(self,
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
            _exec_translate_file(self, path, original=original, arguments=arguments)
            return ""
        if tool_name == "translate_series":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            _exec_translate_series_tool(
                self,
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
            _edit_generated_subtitle(self, target_path=target_path, instruction=instruction, original=original)
            return ""
        if tool_name == "switch_profile":
            _handle_profile_command(self, str(arguments.get("profile") or ""))
            return ""
        if tool_name == "list_profiles":
            _print_profiles(self)
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
            observation = _run_discovery_tool_call(self, tool_name, arguments, original=original)
            return _format_discovery_observation_for_user(self, observation)
        if tool_name in {"recent_translations", "candidate_subtitles", "read_file_preview"}:
            observation = _run_discovery_tool_call(self, tool_name, arguments, original=original)
            return _format_discovery_observation_for_user(self, observation)
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

    def _reference_context(self, line: str) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        for path in _extract_references(self, line):
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

    def _print_file_op_result(self, result: FileOpResult) -> None:
        print_file_op_result(self.console, result)

    def _diagnose_path(self, path: Path, *, original: str) -> None:
        report = diagnose_path(path)
        self.console.print(format_diagnostic_report(report))
        self._record_event("diagnose_path", original, {"path": str(path), "diagnosis": report.diagnosis})

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

    def _record_event(self, kind: str, input_text: str, data: dict[str, Any] | None = None) -> None:
        self.session.events.append(
            {
                "kind": kind,
                "input": input_text,
                "data": data or {},
                "created_at": _now_iso(),
            }
        )

    def _build_input_history(self):
        """Build InMemoryHistory from session events for up/down arrow navigation."""
        try:
            from prompt_toolkit.history import InMemoryHistory
        except Exception:
            return None
        history = InMemoryHistory()
        count = 0
        for event in self.session.events:
            if count >= 100:
                break
            if event.get('kind') == 'user':
                text = event.get('input', '')
                if text and text.strip():
                    history.append_string(text)
                    count += 1
        return history

    def _read_line(self) -> str:
        mode = "|plan" if self.session.mode == "plan" else ""
        prompt = f"sbake[{self.profile or 'default'}{mode}]> "
        prompt_toolkit_prompt = _prompt_toolkit_prompt()
        if self.interactive and prompt_toolkit_prompt is not None:
            return prompt_toolkit_prompt(
                prompt,
                completer=_slash_command_completer(),
                key_bindings=_build_plan_toggle_key_bindings(self),
                history=self._build_input_history(),
            )
        return self.console.input(prompt, markup=False)


def start_interactive_agent(*, console: Console, resume: bool = False) -> None:
    SubBakeAgent(console=console, resume=resume).run()

