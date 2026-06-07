"""Agent loop trace visualization, prompt_toolkit widgets, and general utilities.

Extracted from agent.py to reduce module size. All functions keep their
original names (with leading underscores) for backward compatibility.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text

# ---------------------------------------------------------------------------
# Re-export constants used by this module
# ---------------------------------------------------------------------------

PICKER_CANCEL_TOKEN = "__subbake_picker_cancelled__"

REFERENCE_RE = re.compile(r'@(?:\"([^\"]+)\"|\'([^\']+)\'|(\S+))')

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
    ("/undo", "undo the last file operation"),
    ("/resume", "resume the latest session"),
    ("/exit", "quit"),
    ("/quit", "quit"),
)

# ---------------------------------------------------------------------------
# PickerChoice dataclass (needed by picker functions below)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PickerChoice:
    value: str
    label: str
    completion_text: str
    display: str
    meta: str
    search_text: str




class _AgentLoopTrace:
    TRACE_STYLES = {
        "THINK": "dim",
        "TOOL": "bold yellow",
        "OBSERVE": "green",
        "EXECUTE": "bold green",
        "PLAN": "bold cyan",
        "FINAL": "bold green",
    }

    def __init__(self, *, console: Console, interactive: bool) -> None:
        self.console = console
        self.interactive = interactive
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        if self.interactive:
            header = Text()
            header.append("Agent Loop", style="bold cyan")
            header.append("  bounded discovery", style="dim")
            self.console.print(header)
            return
        self.console.print("Agent Loop")

    @contextmanager
    def thinking(self) -> Iterator[None]:
        if self.interactive:
            with self.console.status("THINK model deciding", spinner="dots"):
                yield
            self.think()
            return
        self.think()
        yield

    def think(self) -> None:
        self._emit("THINK model deciding")

    def tool(self, tool_name: str, arguments: dict[str, Any]) -> None:
        suffix = _trace_arguments(arguments)
        self._emit(f"TOOL {tool_name}{suffix}")

    def observe(self, preview: str) -> None:
        self._emit(f"OBSERVE {preview}")

    def final(self, decision: dict[str, Any]) -> None:
        action = str(decision.get("action") or "")
        if action == "tool_call":
            tool_name = str(decision.get("tool_name") or "")
            suffix = _trace_arguments(dict(decision.get("arguments") or {}), include_modes=True)
            self._emit(f"EXECUTE {tool_name}{suffix}")
            return
        if action == "plan":
            tool_calls = [call for call in decision.get("tool_calls") or [] if isinstance(call, dict)]
            if tool_calls:
                first = tool_calls[0]
                tool_name = str(first.get("tool_name") or "")
                suffix = _trace_arguments(dict(first.get("arguments") or {}), include_modes=True)
                self._emit(f"PLAN {tool_name}{suffix}")
                return
        self._emit(f"FINAL {action or 'respond'}")

    def _emit(self, line: str) -> None:
        if not self.interactive:
            self.console.print(line)
            return

        label, _, detail = line.partition(" ")
        row = Text("  ")
        row.append("| ", style="dim")
        row.append(label, style=self.TRACE_STYLES.get(label, "bold"))
        if detail:
            row.append(" ")
            row.append(detail)
        self.console.print(row)


def _trace_arguments(arguments: dict[str, Any], *, include_modes: bool = False) -> str:
    parts: list[str] = []
    for key in ("path", "old_path", "new_path", "query", "pattern"):
        value = arguments.get(key)
        if value is None or value == "":
            continue
        parts.append(_trace_value(str(value)))
    if include_modes:
        if arguments.get("bilingual") is True:
            parts.append("bilingual")
        output_format = arguments.get("output_format")
        if output_format:
            parts.append(f"output={output_format}")
    return "" if not parts else " " + " ".join(parts)


def _trace_value(value: str) -> str:
    if re.search(r"\s", value):
        return json.dumps(value, ensure_ascii=False)
    return value


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


def _language_phrases() -> tuple[tuple[str, str], ...]:
    return (
        ("简体中文", "Chinese"),
        ("中文", "Chinese"),
        ("汉语", "Chinese"),
        ("chinese", "Chinese"),
        ("zh-cn", "Chinese"),
        ("zh", "Chinese"),
        ("繁体中文", "Traditional Chinese"),
        ("traditional chinese", "Traditional Chinese"),
        ("英文", "English"),
        ("英语", "English"),
        ("english", "English"),
        ("en", "English"),
        ("日文", "Japanese"),
        ("日语", "Japanese"),
        ("japanese", "Japanese"),
        ("ja", "Japanese"),
        ("韩文", "Korean"),
        ("韩语", "Korean"),
        ("korean", "Korean"),
        ("ko", "Korean"),
        ("法文", "French"),
        ("法语", "French"),
        ("french", "French"),
        ("fr", "French"),
        ("德文", "German"),
        ("德语", "German"),
        ("german", "German"),
        ("de", "German"),
        ("西班牙文", "Spanish"),
        ("西班牙语", "Spanish"),
        ("spanish", "Spanish"),
        ("es", "Spanish"),
        ("葡萄牙文", "Portuguese"),
        ("葡萄牙语", "Portuguese"),
        ("portuguese", "Portuguese"),
        ("pt", "Portuguese"),
        ("俄文", "Russian"),
        ("俄语", "Russian"),
        ("russian", "Russian"),
        ("ru", "Russian"),
        ("意大利文", "Italian"),
        ("意大利语", "Italian"),
        ("italian", "Italian"),
        ("it", "Italian"),
    )


def _output_format_patterns() -> tuple[str, ...]:
    return (
        r"(?:输出|生成|保存|导出|产出).{0,12}\.?(srt|vtt|txt)(?:\s*格式|\s*文件|\s*字幕)?",
        r"\.?(srt|vtt|txt).{0,8}(?:格式|输出)",
        r"(?:output|export|save).{0,12}\.?(srt|vtt|txt)\b",
    )


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


def _verify_write_text(path: Path, expected: str) -> None:
    """Read back a just-written file and verify its content matches exactly."""
    try:
        actual = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Write verification failed: cannot read back {path}: {exc}") from exc
    if actual != expected:
        raise OSError(
            f"Write verification failed for {path}: "
            f"content mismatch (expected {len(expected)} bytes, "
            f"got {len(actual)} bytes)"
        )


# ---------------------------------------------------------------------------
# Backward-compatibility: import all names from this module into agent.py
# ---------------------------------------------------------------------------

__all__ = [
    "_AgentLoopTrace",
    "_trace_arguments",
    "_trace_value",
    "PICKER_CANCEL_TOKEN",
    "REFERENCE_RE",
    "AGENT_COMMANDS",
    "PickerChoice",
]
