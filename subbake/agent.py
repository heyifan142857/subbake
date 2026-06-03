from __future__ import annotations

import json
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from subbake.config import (
    AppConfig,
    TRANSLATE_CONFIG_KEYS,
    discover_config_path,
    discover_project_config_path,
    global_config_candidates,
    load_app_config,
    resolve_command_config,
)
from subbake.diagnostics import diagnose_path, diagnose_text, format_diagnostic_report
from subbake.editing import edit_generated_subtitle, is_generated_subtitle
from subbake.file_ops import FileOpResult, FileOperationGuard
from subbake.pipeline import SubtitlePipeline
from subbake.runtime_options import (
    build_backend_from_values,
    build_pipeline_options,
    merge_translation_values,
)
from subbake.series import discover_series_files, translate_series
from subbake.ui import Dashboard

SESSION_VERSION = 1
REFERENCE_RE = re.compile(r"@(?:\"([^\"]+)\"|'([^']+)'|(\S+))")
AGENT_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "show agent help"),
    ("/model", "choose a model profile"),
    ("/profile", "choose a model profile"),
    ("/session", "choose a session"),
    ("/sessions", "list recent sessions"),
    ("/clear", "start a new session"),
    ("/plan", "turn plan mode on"),
    ("/plan off", "turn plan mode off"),
    ("/approve", "execute the pending plan"),
    ("/reject", "discard the pending plan"),
    ("/resume", "resume the latest session"),
    ("/exit", "quit"),
    ("/quit", "quit"),
)
NEW_PROFILE_VALUE = "__subbake_new_profile__"
CONFIG_BOOTSTRAP_CREATE = "create"
CONFIG_BOOTSTRAP_SKIP = "skip"
PICKER_CANCEL_TOKEN = "__subbake_picker_cancelled__"
PROFILE_PROVIDER_OPTIONS = ("mock", "openai", "anthropic", "gemini", "openai-compatible")
PROFILE_TARGET_LANGUAGE_OPTIONS = ("Chinese", "zh", "en", "ja", "ko", "fr", "es", "de")
PROFILE_API_KEY_ENV_OPTIONS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


@dataclass(slots=True)
class PickerChoice:
    value: str
    label: str
    completion_text: str
    display: str
    meta: str
    search_text: str


@dataclass(slots=True)
class AgentSession:
    id: str
    created_at: str
    updated_at: str
    cwd: str
    profile: str | None = None
    config_path: str | None = None
    mode: str = "chat"
    pending_plan: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SESSION_VERSION,
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cwd": self.cwd,
            "profile": self.profile,
            "config_path": self.config_path,
            "mode": self.mode,
            "pending_plan": self.pending_plan,
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentSession":
        return cls(
            id=str(data["id"]),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
            cwd=str(data.get("cwd") or Path.cwd()),
            profile=data.get("profile") if isinstance(data.get("profile"), str) else None,
            config_path=data.get("config_path") if isinstance(data.get("config_path"), str) else None,
            mode=str(data.get("mode") or "chat"),
            pending_plan=data.get("pending_plan") if isinstance(data.get("pending_plan"), dict) else None,
            events=list(data.get("events") or []),
        )


class AgentSessionStore:
    def __init__(self, project_root: Path) -> None:
        self.root = project_root / ".subbake" / "agent" / "sessions"

    def create(self, *, cwd: Path, profile: str | None, config_path: Path | None) -> AgentSession:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
        return AgentSession(
            id=session_id,
            created_at=_now_iso(),
            updated_at=_now_iso(),
            cwd=str(cwd),
            profile=profile,
            config_path=str(config_path) if config_path is not None else None,
            mode="chat",
        )

    def save(self, session: AgentSession) -> Path:
        session.updated_at = _now_iso()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(session.id)
        path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def latest(self) -> AgentSession | None:
        sessions = self.list_sessions()
        if not sessions:
            return None
        return self.load(sessions[-1])

    def list_sessions(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("*.json"))

    def load(self, path: Path) -> AgentSession:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != SESSION_VERSION:
            raise ValueError(f"Unsupported agent session version in {path}.")
        return AgentSession.from_dict(data)

    def path_for(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"


class SubBakeAgent:
    def __init__(self, *, console: Console, resume: bool = False) -> None:
        self.console = console
        self.cwd = Path.cwd()
        self.config_path = discover_config_path()
        self.config = self._load_config(self.config_path)
        project_config_path = discover_project_config_path()
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
        self.console.print("[bold green]SubBake agent[/bold green]  /help for commands, /exit to quit")
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
        if command is not None:
            self.console.print("Unknown command. Use /help for available agent controls.")
            self._record_event("unknown_command", line, {"command": command})
            return True

        self._handle_conversational_line(line)
        return True

    def _handle_conversational_line(self, line: str) -> None:
        self._record_event("user", line)
        decision = self._deterministic_decision_from_line(line) or self._decide_next_action(line)
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
        return None

    def _content_after_references(self, line: str) -> str:
        cleaned = self._remove_references(line).strip()
        cleaned = re.sub(r"^(创建|新建|追加|替换)\s*", "", cleaned).strip()
        cleaned = re.sub(r"^(create|append|replace)\b", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def _decide_next_action(self, line: str) -> dict[str, Any]:
        backend = build_backend_from_values(self.values)
        if backend is None:
            raise RuntimeError("Agent conversation requires a model backend.")
        messages = self._build_agent_decision_messages(line)
        payload, _ = backend.generate_json(messages)
        if not isinstance(payload, dict):
            raise ValueError("Agent decision must be a JSON object.")
        action = str(payload.get("action") or "").strip()
        if action not in {"respond", "tool_call", "plan", "ask_user"}:
            raise ValueError(f"Unsupported agent decision action: {action or '<missing>'}")
        return payload

    def _build_agent_decision_messages(self, line: str) -> list[dict[str, str]]:
        context = {
            "user_message": line,
            "mode": self.session.mode,
            "profile": self.profile,
            "cwd": str(self.cwd),
            "project_root": str(self.project_root),
            "references": self._reference_context(line),
            "recent_events": self.session.events[-8:],
            "tools": self._tool_specs(),
        }
        system_prompt = (
            "You are SubBake's conversational subtitle agent.\n"
            "Return valid JSON only.\n"
            "Choose one action: respond, ask_user, plan, or tool_call.\n"
            "Use tools for concrete work. Do not invent file contents or command results.\n"
            "When mode is plan, never return tool_call; return plan with tool_calls instead.\n"
            "Low-level file operations are internal tools, not user slash commands.\n"
            "If a requested mutation is ambiguous, return ask_user or plan instead of tool_call.\n"
        )
        user_prompt = (
            "TASK_START\n"
            "agent_decide\n"
            "TASK_END\n"
            'Return JSON with action plus "message". For tool_call include "tool_name" and "arguments". '
            'For plan include "tool_calls".\n'
            "AGENT_CONTEXT_JSON_START\n"
            f"{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}\n"
            "AGENT_CONTEXT_JSON_END\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _handle_decision(self, decision: dict[str, Any], *, original: str) -> None:
        action = decision["action"]
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
            result = self._run_tool_call(
                str(decision.get("tool_name") or ""),
                dict(decision.get("arguments") or {}),
                original=original,
            )
            message = str(decision.get("message") or "").strip()
            if message:
                self.console.print(message)
            if result:
                self.console.print(result)
            return
        raise ValueError(f"Unsupported agent decision action: {action}")

    def _run_tool_call(self, tool_name: str, arguments: dict[str, Any], *, original: str) -> str:
        if tool_name == "translate_file":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            self._translate_file(path, original=original)
            return ""
        if tool_name == "translate_series":
            path = self._resolve_user_path(str(arguments.get("path") or ""))
            self._translate_series_tool(
                path,
                original=original,
                recursive=bool(arguments.get("recursive", False)),
                overwrite=bool(arguments.get("overwrite", False)),
                dry_run=bool(arguments.get("dry_run", False)),
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
            path = self._resolve_user_path(str(arguments.get("path") or "."))
            pattern = str(arguments.get("pattern") or "")
            return "\n".join(guard.search_files(path, pattern)) or "No matches."
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

    def _translate_series_tool(
        self,
        path: Path,
        *,
        original: str,
        recursive: bool,
        overwrite: bool,
        dry_run: bool,
    ) -> None:
        files = discover_series_files(path, recursive=recursive)
        if not files:
            self.console.print("[bold yellow]No subtitle files found.[/bold yellow]")
            self._record_event("series_empty", original, {"path": str(path)})
            return
        self.console.print(
            "[bold green]Series:[/bold green] "
            f"{len(files)} file(s), profile={self.profile or 'default'}, "
            f"target={self.values['target_language']}"
        )
        values = dict(self.values)
        if dry_run:
            values["dry_run"] = True
        result = translate_series(
            root=path,
            values=values,
            backend_factory=lambda: build_backend_from_values(values),
            console=self.console,
            recursive=recursive,
            overwrite=overwrite,
        )
        self._print_series_summary(result)
        self._record_event(
            "series",
            original,
            {
                "path": str(path),
                "processed": result.processed_count,
                "skipped": result.skipped_count,
                "failures": result.failure_count,
            },
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
        backend = build_backend_from_values(values)
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

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [
            {"name": "translate_file", "mutating": True, "args": ["path"]},
            {"name": "translate_series", "mutating": True, "args": ["path", "recursive", "overwrite", "dry_run"]},
            {"name": "edit_subtitle", "mutating": True, "args": ["path", "instruction"]},
            {"name": "diagnose_path", "mutating": False, "args": ["path"]},
            {"name": "diagnose_text", "mutating": False, "args": ["text"]},
            {"name": "read_file", "mutating": False, "args": ["path"]},
            {"name": "list_files", "mutating": False, "args": ["path", "recursive"]},
            {"name": "search_files", "mutating": False, "args": ["path", "pattern"]},
            {"name": "create_file", "mutating": True, "args": ["path", "content"]},
            {"name": "append_file", "mutating": True, "args": ["path", "content"]},
            {"name": "replace_in_file", "mutating": True, "args": ["path", "old", "new"]},
            {"name": "rename_path", "mutating": True, "args": ["old_path", "new_path"]},
            {"name": "delete_file", "mutating": True, "args": ["path"]},
            {"name": "list_profiles", "mutating": False, "args": []},
            {"name": "switch_profile", "mutating": False, "args": ["profile"]},
        ]

    def _resolve_user_path(self, value: str) -> Path:
        if not value:
            raise ValueError("Tool path argument is required.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.cwd / path
        return path

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
        self.console.print("</proposed_plan>")
        self.console.print("Use /approve to execute this plan, or /reject to discard it.")
        self._record_event("plan", original, {"tool_calls": tool_calls})

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

    def _translate_file(self, path: Path, *, original: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Subtitle file not found: {path}")
        values = dict(self.values)
        backend = build_backend_from_values(values)
        options = build_pipeline_options(
            input_path=path,
            output_path=None,
            values=values,
        )
        result = SubtitlePipeline(
            backend=backend,
            options=options,
            dashboard=Dashboard(console=self.console),
        ).run()
        if result.dry_run:
            self.console.print(f"[bold yellow]Dry run:[/bold yellow] {len(result.planned_batches)} batch(es) planned.")
        else:
            self.console.print(f"[bold green]Output:[/bold green] {result.output_path}")
        self._record_event(
            "translate_file",
            original,
            {
                "input_path": str(path),
                "output_path": str(result.output_path) if result.output_path else None,
            },
        )

    def _print_file_op_result(self, result: FileOpResult) -> None:
        if result.action == "renamed" and result.new_path is not None:
            self.console.print(f"[bold green]Renamed:[/bold green] {result.path} -> {result.new_path}")
        else:
            self.console.print(f"[bold green]{result.action.title()}:[/bold green] {result.path}")
        if result.backup_path is not None:
            self.console.print(f"[bold green]Backup:[/bold green] {result.backup_path}")

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
        candidates = global_config_candidates()
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
        path.write_text(prefix + "\n".join(lines) + "\n", encoding="utf-8")

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
        self.console.print(
            "\n".join(
                [
                    "Ask naturally:",
                    "  翻译 @episode01.srt",
                    "  把 @Season01 翻译成中文",
                    "  分析 @.subbake/runs/.../failure.json",
                    "  把 notes.txt 改名成 glossary-notes.txt",
                    "  创建 @notes.txt 记录 Alice 译作爱丽丝",
                    "",
                    "Controls:",
                    "  Tab                     complete slash commands",
                    "  Shift+Tab               toggle plan mode",
                    "  /model or /profile      choose a model profile",
                    "  /model <profile>        switch profile directly",
                    "  /session                choose a previous session",
                    "  /plan                   enter plan mode",
                    "  /plan off               return to chat mode",
                    "  /approve                execute the pending plan",
                    "  /reject                 discard the pending plan",
                    "  /clear                  start a new agent session",
                    "  /sessions               list recent sessions",
                    "  /resume                 resume the latest session",
                    "  /exit                   quit",
                ]
            )
        )

    def _print_series_summary(self, result) -> None:
        self.console.print(
            "[bold green]Series result:[/bold green] "
            f"{result.processed_count} processed, {result.skipped_count} skipped, {result.failure_count} failed"
        )
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

        @key_bindings.add("tab")
        def _complete(event) -> None:
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


def _prompt_toolkit_prompt():
    try:
        from prompt_toolkit import prompt
    except Exception:
        return None

    def _run(prompt_text: str, *, completer=None, key_bindings=None) -> str:
        return prompt(
            prompt_text,
            completer=completer,
            complete_while_typing=True,
            key_bindings=key_bindings,
        )

    return _run


def _slash_command_completer():
    try:
        from prompt_toolkit.completion import Completer, Completion
    except Exception:
        return None

    class SlashCommandCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            query = text.casefold()
            for command, meta in AGENT_COMMANDS:
                if command.casefold().startswith(query):
                    yield Completion(
                        command,
                        start_position=-len(text),
                        display=command,
                        display_meta=meta,
                    )

    return SlashCommandCompleter()


def _slash_command_matches(query: str) -> list[str]:
    normalized = query.casefold()
    return [
        command
        for command, _ in AGENT_COMMANDS
        if command.casefold().startswith(normalized)
    ]


def _unique_slash_command_match(query: str) -> str | None:
    matches = _slash_command_matches(query)
    if len(matches) == 1:
        return matches[0]
    return None


def _prompt_toolkit_inline_picker(title: str, options: list[tuple[str, str]], *, default: str) -> str | None:
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.application.current import get_app
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.filters import has_completions
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.shortcuts import CompleteStyle
        from prompt_toolkit.styles import Style
    except Exception:
        return None

    choices = _picker_choices(options, default=default)

    class InlinePickerCompleter(Completer):
        def get_completions(self, document, complete_event):
            for choice in _matching_picker_choices(document.text_before_cursor, choices):
                yield Completion(
                    choice.completion_text,
                    start_position=-len(document.text_before_cursor),
                    display=choice.display,
                    display_meta=choice.meta,
                )

    key_bindings = KeyBindings()

    @key_bindings.add("down", filter=has_completions)
    def _next_completion(event) -> None:
        event.current_buffer.complete_next()

    @key_bindings.add("up", filter=has_completions)
    def _previous_completion(event) -> None:
        event.current_buffer.complete_previous()

    @key_bindings.add("tab")
    def _complete_or_accept(event) -> None:
        buffer = event.current_buffer
        completion = _current_completion(buffer)
        if completion is not None:
            buffer.apply_completion(completion)
            return
        buffer.start_completion(select_first=True)

    @key_bindings.add("enter")
    def _accept(event) -> None:
        buffer = event.current_buffer
        completion = _current_completion(buffer)
        if completion is not None:
            buffer.apply_completion(completion)
        event.app.exit(result=buffer.text)

    @key_bindings.add("escape")
    @key_bindings.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=PICKER_CANCEL_TOKEN)

    style = Style.from_dict(
        {
            "completion-menu.completion": "",
            "completion-menu.completion.current": "reverse bold",
            "completion-menu.meta.completion": "fg:ansibrightblack",
            "completion-menu.meta.completion.current": "reverse bold",
            "bottom-toolbar": "reverse",
        }
    )

    try:
        raw_selection = prompt(
            _picker_prompt(title),
            completer=InlinePickerCompleter(),
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            key_bindings=key_bindings,
            reserve_space_for_menu=8,
            bottom_toolbar=_picker_toolbar(title),
            style=style,
            pre_run=lambda: get_app().current_buffer.start_completion(select_first=True),
        )
    except KeyboardInterrupt:
        return PICKER_CANCEL_TOKEN
    if raw_selection == PICKER_CANCEL_TOKEN:
        return PICKER_CANCEL_TOKEN
    return _resolve_picker_selection(raw_selection, choices, default=default) or PICKER_CANCEL_TOKEN


def _current_completion(buffer):
    complete_state = buffer.complete_state
    if complete_state is None:
        return None
    return complete_state.current_completion


def _picker_choices(options: list[tuple[str, str]], *, default: str) -> list[PickerChoice]:
    choices = [_picker_choice(value, label) for value, label in options]
    default_index = next((index for index, choice in enumerate(choices) if choice.value == default), None)
    if default_index is None:
        return choices
    return [choices[default_index], *choices[:default_index], *choices[default_index + 1 :]]


def _matching_picker_choices(query: str, choices: list[PickerChoice]) -> list[PickerChoice]:
    normalized = query.strip().casefold()
    if not normalized:
        return choices
    return [choice for choice in choices if normalized in choice.search_text]


def _picker_choice(value: str, label: str) -> PickerChoice:
    display, meta = _picker_display_parts(label)
    completion_text = label if value.startswith("__subbake_") else value
    search_text = f"{value} {label} {display} {meta}".casefold()
    return PickerChoice(
        value=value,
        label=label,
        completion_text=completion_text,
        display=display,
        meta=meta,
        search_text=search_text,
    )


def _picker_display_parts(label: str) -> tuple[str, str]:
    if "  (" in label and label.endswith(")"):
        display, _, meta = label.rpartition("  ")
        return display, meta
    if ": " in label:
        display, meta = label.split(": ", 1)
        return display, meta
    return label, ""


def _resolve_picker_selection(raw_selection: str, choices: list[PickerChoice], *, default: str) -> str | None:
    selection = raw_selection.strip()
    if not selection:
        return default
    normalized = selection.casefold()
    for choice in choices:
        if normalized in {
            choice.value.casefold(),
            choice.label.casefold(),
            choice.completion_text.casefold(),
        }:
            return choice.value
    matches = [choice for choice in choices if normalized in choice.search_text]
    if len(matches) == 1:
        return matches[0].value
    return None


def _picker_prompt(title: str) -> str:
    lowered = title.casefold()
    if "session" in lowered:
        return "session> "
    if "profile" in lowered or "model" in lowered:
        return "profile> "
    if "config" in lowered:
        return "config> "
    return "choose> "


def _picker_toolbar(title: str) -> str:
    return f" {title}: type to filter, Tab/Enter to select, Esc to cancel "


def _prompt_toolkit_inline_text(
    title: str,
    text: str,
    *,
    default: str,
    completions: tuple[str, ...],
) -> str | None:
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.filters import has_completions
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.shortcuts import CompleteStyle
        from prompt_toolkit.styles import Style
    except Exception:
        return None

    class InlineTextCompleter(Completer):
        def get_completions(self, document, complete_event):
            query = document.text_before_cursor.strip()
            for value in _text_prompt_matches(query, completions):
                yield Completion(
                    value,
                    start_position=-len(document.text_before_cursor),
                )

    key_bindings = KeyBindings()

    @key_bindings.add("down", filter=has_completions)
    def _next_completion(event) -> None:
        event.current_buffer.complete_next()

    @key_bindings.add("up", filter=has_completions)
    def _previous_completion(event) -> None:
        event.current_buffer.complete_previous()

    @key_bindings.add("tab")
    def _complete(event) -> None:
        buffer = event.current_buffer
        query = buffer.document.text_before_cursor
        matches = _text_prompt_matches(query, completions)
        if len(matches) == 1:
            buffer.delete_before_cursor(len(query))
            buffer.insert_text(matches[0])
            return
        completion = _current_completion(buffer)
        if completion is not None:
            buffer.apply_completion(completion)
            return
        buffer.start_completion(select_first=True)

    @key_bindings.add("enter")
    def _accept(event) -> None:
        buffer = event.current_buffer
        completion = _current_completion(buffer)
        if completion is not None:
            buffer.apply_completion(completion)
        event.app.exit(result=buffer.text)

    @key_bindings.add("escape")
    @key_bindings.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=PICKER_CANCEL_TOKEN)

    style = Style.from_dict(
        {
            "completion-menu.completion": "",
            "completion-menu.completion.current": "reverse bold",
            "bottom-toolbar": "reverse",
        }
    )

    try:
        raw_value = prompt(
            _text_prompt(text),
            completer=InlineTextCompleter() if completions else None,
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            key_bindings=key_bindings,
            reserve_space_for_menu=6,
            bottom_toolbar=_text_prompt_toolbar(title, text, default),
            style=style,
        )
    except KeyboardInterrupt:
        return PICKER_CANCEL_TOKEN
    if raw_value == PICKER_CANCEL_TOKEN:
        return PICKER_CANCEL_TOKEN
    return _resolve_text_prompt_value(raw_value, default=default)


def _text_prompt_matches(query: str, completions: tuple[str, ...]) -> list[str]:
    normalized = query.strip().casefold()
    if not completions:
        return []
    if not normalized:
        return list(completions)
    return [value for value in completions if value.casefold().startswith(normalized)]


def _resolve_text_prompt_value(raw_value: str, *, default: str) -> str:
    if raw_value == "":
        return default
    return raw_value


def _default_api_key_env(provider: str) -> str:
    normalized = provider.strip().casefold()
    if normalized in {"openai", "openai-compatible", "compatible"}:
        return "OPENAI_API_KEY"
    if normalized == "anthropic":
        return "ANTHROPIC_API_KEY"
    if normalized == "gemini":
        return "GEMINI_API_KEY"
    return ""


def _text_prompt(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().casefold())
    labels = {
        "profile name": "profile name",
        "provider": "provider",
        "model": "model",
        "api key environment variable": "api key env",
        "base url": "base url",
        "target language": "target language",
    }
    return f"{labels.get(normalized, normalized or 'value')}> "


def _text_prompt_toolbar(title: str, text: str, default: str) -> str:
    default_text = f", default: {default}" if default else ""
    return f" {title} / {text}{default_text}: Enter accepts, Tab completes, Esc cancels "


def _toml_key(value: str) -> str:
    if re.match(r"^[A-Za-z0-9_-]+$", value):
        return value
    return json.dumps(value, ensure_ascii=False)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _prepend_default_profile(content: str, profile_name: str) -> str:
    default_line = f"default_profile = {_toml_string(profile_name)}\n"
    if not content:
        return default_line
    return f"{default_line}\n{content}"


def _short_title(value: str, *, limit: int = 72) -> str:
    title = " ".join(value.strip().split())
    if len(title) <= limit:
        return title
    return f"{title[: limit - 3]}..."


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
