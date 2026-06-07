"""Agent UI helpers: print, display, plan mode.

Extracted from agent.py. All functions take explicit parameters
rather than depending on an agent instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console


def print_translation_start(
    console: Console,
    *,
    values: dict[str, Any],
    file_count: int,
    suffixes: set[str],
    series: bool,
    path: Path | None = None,
    original: str = "",
) -> None:
    """Print the translation-start status line."""
    action = "规划" if bool(values["dry_run"]) else "翻译"
    subject = path.name if path is not None else _file_count_label(file_count, suffixes)
    render_label = _render_mode_label(original, values)
    scope = "同一系列" if series else "这个文件"
    console.print(f"[bold green]Preparing:[/bold green] 现在要按{scope}{action} {subject}，{render_label}。")


def print_file_completion(console: Console, *, output_path: Path | None, dry_run: bool) -> None:
    """Print the translation-completion status line."""
    if dry_run:
        console.print("[bold green]Completed:[/bold green] 已完成翻译规划。")
        return
    console.print(f"[bold green]Completed:[/bold green] 已完成翻译，输出 {output_path}。")


def print_series_completion(console: Console, result) -> None:
    """Print the series-completion status line."""
    if result.failure_count:
        console.print(
            "[bold yellow]Completed:[/bold yellow] "
            f"已完成 {result.processed_count} 个，跳过 {result.skipped_count} 个，失败 {result.failure_count} 个。"
        )
        return
    console.print(
        "[bold green]Completed:[/bold green] "
        f"已完成 {result.processed_count} 个文件翻译，跳过 {result.skipped_count} 个。"
    )


def print_file_op_result(console: Console, result) -> None:
    """Print a file-operation result."""
    if result.action == "renamed" and result.new_path is not None:
        console.print(f"[bold green]Renamed:[/bold green] {result.path} -> {result.new_path}")
    else:
        console.print(f"[bold green]{result.action.title()}:[/bold green] {result.path}")
    if result.backup_path is not None:
        console.print(f"[bold green]Backup:[/bold green] {result.backup_path}")


def print_series_summary(console: Console, result) -> None:
    """Print a one-line series translation summary."""
    console.print(
        "[bold green]Series result:[/bold green] "
        f"{result.processed_count} processed, {result.skipped_count} skipped, {result.failure_count} failed"
    )


def print_tool_call_preview(console: Console, call: dict[str, Any]) -> None:
    """Print a preview of a tool call (used in plan mode)."""
    tool_name = str(call.get("tool_name") or "unknown")
    arguments = dict(call.get("arguments") or {})

    console.print(f"[bold]{tool_name}[/bold]")

    path = str(arguments.get("path") or arguments.get("old_path") or "")
    if path:
        console.print(f"  path: {path}")
    new_path = str(arguments.get("new_path") or "")
    if new_path:
        console.print(f"  \u2192 {new_path}")

    if tool_name in {"create_file", "append_file"}:
        _print_content_preview(console, str(arguments.get("content") or ""))
    elif tool_name == "replace_in_file":
        _print_replace_preview(console, str(arguments.get("old") or ""), str(arguments.get("new") or ""))
    elif tool_name == "edit_subtitle":
        instruction = str(arguments.get("instruction") or "")
        if instruction:
            console.print(f"  instruction: {instruction}")
    elif tool_name in {"translate_file", "translate_series"}:
        _print_translation_options(console, arguments)


def print_help(console: Console) -> None:
    """Print the agent help text."""
    console.print(
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
                "  Tab                    complete slash commands & autocomplete",
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


def render_mode_label(original: str, values: dict[str, Any]) -> str:
    """Build a descriptive label showing translation mode and settings."""
    if bool(values["bilingual"]):
        base_label = "生成中英双语字幕" if "中英" in original else "生成双语字幕"
        label = f"{base_label}，目标语言 {values['target_language']}"
    else:
        label = f"目标语言 {values['target_language']}"
    output_format = values.get("output_format")
    if output_format is not None:
        label = f"{label}，输出 {str(output_format).upper()} 格式"
    return label


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _file_count_label(file_count: int, suffixes: set[str]) -> str:
    """Build a human-readable file count label."""
    suffix_label = ", ".join(sorted(suffixes))
    if suffix_label:
        return f"{file_count} 个 {suffix_label} 文件"
    return f"{file_count} 个字幕文件"


def _render_mode_label(original: str, values: dict[str, Any]) -> str:
    """Build a descriptive label showing translation mode and settings."""
    if bool(values["bilingual"]):
        base_label = "生成中英双语字幕" if "中英" in original else "生成双语字幕"
        label = f"{base_label}，目标语言 {values['target_language']}"
    else:
        label = f"目标语言 {values['target_language']}"
    output_format = values.get("output_format")
    if output_format is not None:
        label = f"{label}，输出 {str(output_format).upper()} 格式"
    return label


def _print_content_preview(console: Console, content: str) -> None:
    """Print a preview of file content."""
    if not content:
        console.print("  [dim](empty content)[/dim]")
        return
    preview = content if len(content) <= 300 else f"{content[:300]}..."
    console.print(f"  content ({len(content)} chars):")
    for line in preview.splitlines()[:12]:
        console.print(f"  \u2502 {line}")
    if len(content) > 300 or len(preview.splitlines()) > 12:
        console.print("  \u2502 [dim]...[/dim]")


def _print_replace_preview(console: Console, old: str, new: str) -> None:
    """Print a preview of a text replacement."""
    old_preview = old if len(old) <= 120 else f"{old[:120]}..."
    new_preview = new if len(new) <= 120 else f"{new[:120]}..."
    console.print(f"  old: {old_preview}")
    console.print(f"  new: {new_preview}")


def _print_translation_options(console: Console, arguments: dict[str, Any]) -> None:
    """Print translation options from tool arguments."""
    for key in ("target_language", "source_language", "output_format"):
        value = arguments.get(key)
        if value:
            console.print(f"  {key}: {value}")
    for key in ("bilingual", "dry_run", "fast", "recursive", "overwrite"):
        value = arguments.get(key)
        if value:
            console.print(f"  {key}: {value}")
