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
    load_app_config,
    resolve_command_config,
)
from subbake.diagnostics import diagnose_path, diagnose_text, format_diagnostic_report
from subbake.editing import edit_generated_subtitle, is_generated_subtitle
from subbake.pipeline import SubtitlePipeline
from subbake.runtime_options import (
    build_backend_from_values,
    build_pipeline_options,
    merge_translation_values,
)
from subbake.series import SUPPORTED_SUBTITLE_SUFFIXES, discover_series_files, translate_series
from subbake.ui import Dashboard

SESSION_VERSION = 1
REFERENCE_RE = re.compile(r"@(?:\"([^\"]+)\"|'([^']+)'|(\S+))")
EDIT_WORDS = ("修改", "修正", "改一下", "edit", "fix", "apply")
DIAGNOSE_WORDS = ("错误", "报错", "日志", "失败", "diagnose", "analyze", "analyse", "log", "error")


@dataclass(slots=True)
class AgentSession:
    id: str
    created_at: str
    updated_at: str
    cwd: str
    profile: str | None = None
    config_path: str | None = None
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
        if command == "/sessions":
            self._print_sessions()
            self._record_event("sessions", line)
            return True
        if command == "/resume":
            self._resume_latest_session()
            return True
        if command == "/edit":
            self._handle_edit(line=rest, explicit=True)
            return True

        references = self._extract_references(line)
        if references and self._is_edit_intent(line):
            self._handle_edit(line=line, explicit=False)
            return True
        if references:
            self._handle_references(line=line, references=references)
            return True
        if self._is_diagnostic_intent(line):
            self._handle_diagnostic_text_or_latest(line)
            return True

        self.console.print("No action detected. Use @file, @folder, /model, /profile, /clear, or /help.")
        self._record_event("unknown", line)
        return True

    def _handle_references(self, *, line: str, references: list[Path]) -> None:
        diagnostic_intent = self._is_diagnostic_intent(line)
        for path in references:
            if diagnostic_intent:
                self._diagnose_path(path, original=line)
                continue
            if path.is_dir():
                self._translate_folder(path, original=line)
                continue
            if path.suffix.lower() in SUPPORTED_SUBTITLE_SUFFIXES:
                self._translate_file(path, original=line)
                continue
            self._diagnose_path(path, original=line)

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

    def _translate_folder(self, path: Path, *, original: str) -> None:
        files = discover_series_files(path, recursive=False)
        if not files:
            self.console.print("[bold yellow]No subtitle files found.[/bold yellow]")
            self._record_event("series_empty", original, {"path": str(path)})
            return
        self.console.print(
            "[bold green]Series:[/bold green] "
            f"{len(files)} file(s), profile={self.profile or 'default'}, "
            f"target={self.values['target_language']}"
        )
        choice = self._choose(
            "Translate this folder?",
            [
                ("start", "Start translation"),
                ("dry-run", "Dry run only"),
                ("cancel", "Cancel"),
            ],
            default="start",
        )
        if choice == "cancel":
            self.console.print("Cancelled.")
            self._record_event("series_cancelled", original, {"path": str(path)})
            return
        values = dict(self.values)
        if choice == "dry-run":
            values["dry_run"] = True
        result = translate_series(
            root=path,
            values=values,
            backend_factory=lambda: build_backend_from_values(values),
            console=self.console,
            recursive=False,
            overwrite=False,
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

    def _handle_edit(self, *, line: str, explicit: bool) -> None:
        references = self._extract_references(line)
        if not references:
            raise ValueError("Use /edit @translated-file.srt <instruction>.")
        target_path = references[0]
        instruction = self._remove_references(line).strip()
        if explicit:
            instruction = re.sub(r"^/edit\b", "", instruction, flags=re.IGNORECASE).strip()
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
            line,
            {
                "target_path": str(result.target_path),
                "backup_path": str(result.backup_path),
            },
        )

    def _diagnose_path(self, path: Path, *, original: str) -> None:
        report = diagnose_path(path)
        self.console.print(format_diagnostic_report(report))
        self._record_event("diagnose_path", original, {"path": str(path), "diagnosis": report.diagnosis})

    def _handle_diagnostic_text_or_latest(self, line: str) -> None:
        latest = self._latest_failure_log()
        if latest is not None and len(line.splitlines()) == 1:
            self._diagnose_path(latest, original=line)
            return
        report = diagnose_text(line)
        self.console.print(format_diagnostic_report(report))
        self._record_event("diagnose_text", line, {"diagnosis": report.diagnosis})

    def _handle_profile_command(self, rest: str) -> None:
        profile_name = rest.strip()
        if not profile_name:
            self._print_profiles()
            return
        if self.config is None or profile_name not in self.config.profiles:
            raise ValueError(f"Config profile '{profile_name}' was not found.")
        self.profile = profile_name
        self.values = self._values_for_profile(profile_name)
        self.session.profile = profile_name
        self.console.print(
            f"[bold green]Profile switched:[/bold green] {profile_name} "
            f"({self.values['provider']} / {self.values['model']})"
        )

    def _print_profiles(self) -> None:
        if self.config is None or not self.config.profiles:
            self.console.print("No configured profiles were found. Using built-in mock defaults.")
            return
        self.console.print("[bold green]Profiles:[/bold green]")
        for name in sorted(self.config.profiles):
            marker = "*" if name == self.profile else " "
            values = self._values_for_profile(name)
            self.console.print(f" {marker} {name}: {values['provider']} / {values['model']}")

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
            self.console.print(f" {marker} {session.id}  profile={session.profile or 'default'}")

    def _resume_latest_session(self) -> None:
        latest = self.store.latest()
        if latest is None:
            self.console.print("[bold yellow]No previous agent session found.[/bold yellow]")
            return
        self.session = latest
        if latest.profile is not None and self.config is not None and latest.profile in self.config.profiles:
            self.profile = latest.profile
            self.values = self._values_for_profile(latest.profile)
        self.console.print(f"[bold green]Resumed session:[/bold green] {latest.id}")
        self._record_event("resume", "/resume", {"session_id": latest.id})

    def _print_help(self) -> None:
        self.console.print(
            "\n".join(
                [
                    "Commands:",
                    "  @file.srt                         translate one subtitle file",
                    "  @folder                           translate a subtitle folder as a series",
                    "  分析 @.subbake/.../failure.json     diagnose a SubBake failure log",
                    "  /edit @episode.translated.srt ...  edit generated translated subtitles",
                    "  /model or /profile                 list configured profiles",
                    "  /model <profile>                   switch profile",
                    "  /clear                             start a new agent session",
                    "  /sessions                          list recent sessions",
                    "  /resume                            resume the latest session",
                    "  /exit                              quit",
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
        prompt = f"sbake[{self.profile or 'default'}]> "
        prompt_toolkit_prompt = _prompt_toolkit_prompt()
        if self.interactive and prompt_toolkit_prompt is not None:
            return prompt_toolkit_prompt(prompt)
        return self.console.input(prompt, markup=False)

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

    def _is_edit_intent(self, line: str) -> bool:
        lowered = line.casefold()
        return any(word in lowered for word in EDIT_WORDS)

    def _is_diagnostic_intent(self, line: str) -> bool:
        lowered = line.casefold()
        return any(word in lowered for word in DIAGNOSE_WORDS)

    def _latest_failure_log(self) -> Path | None:
        candidates = list((self.project_root / ".subbake").glob("runs/*/failures/*.json"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def start_interactive_agent(*, console: Console, resume: bool = False) -> None:
    SubBakeAgent(console=console, resume=resume).run()


def _prompt_toolkit_prompt():
    try:
        from prompt_toolkit import prompt
    except Exception:
        return None
    return prompt


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
