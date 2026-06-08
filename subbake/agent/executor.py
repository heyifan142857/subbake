"""Translation execution functions for SubBakeAgent.

These functions handle the actual translation workflow: single-file translation,
series translation, and subtitle editing. They are called by the agent's dispatch
logic in ``_core.py``.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._core import SubBakeAgent

from subbake.editing import edit_generated_subtitle as _edit_generated_subtitle, is_generated_subtitle
from subbake.pipeline import SubtitlePipeline
from subbake.runtime_options import build_pipeline_options
from subbake.series import (
    discover_series_files,
    resolve_series_output_path,
    translate_series,
)
from subbake.ui import Dashboard
from subbake import runtime_options as _runtime_options


def translate_file(
    agent: SubBakeAgent,
    path: Path,
    *,
    original: str,
    arguments: dict[str, Any] | None = None,
) -> None:
    """Translate a single subtitle file and record the operation."""
    if not path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {path}")
    values = agent._translation_values_for_tool(arguments or {})
    agent._print_translation_start(
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
        dashboard=Dashboard(console=agent.console),
        cancel_requested=_agent_cancel_requested(agent),
    )
    existed_before, backup_path = (
        _translation_output_undo_snapshot(agent, pipeline.output_path)
        if not options.dry_run
        else (False, None)
    )
    result = pipeline.run()
    if result.dry_run:
        agent.console.print(
            f"[bold yellow]Dry run:[/bold yellow] {len(result.planned_batches)} batch(es) planned."
        )
        agent._print_file_completion(output_path=None, dry_run=True)
    else:
        agent.console.print(f"[bold green]Output:[/bold green] {result.output_path}")
        agent._print_file_completion(output_path=result.output_path, dry_run=False)
    agent._record_event(
        "translate_file",
        original,
        {
            "input_path": str(path),
            "output_path": str(result.output_path) if result.output_path else None,
            "bilingual": bool(values["bilingual"]),
            "source_language": str(values["source_language"]),
            "target_language": str(values["target_language"]),
            "output_format": values.get("output_format"),
            "dry_run": bool(values["dry_run"]),
            "summary": _translate_file_summary(path, result, values),
            "batches_translated": result.batches_translated,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
            "fast": bool(values.get("fast", False)),
        },
    )
    # Record assistant event so translation results appear in conversation history
    if result.dry_run:
        agent._record_event(
            "assistant", f"已规划 {path.name}，共 {result.batches_translated} 批次。",
            {"decision": "tool_call", "tool": "translate_file"},
        )
    else:
        agent._record_event(
            "assistant", f"已完成翻译，输出 {result.output_path}。",
            {"decision": "tool_call", "tool": "translate_file"},
        )
    if result.output_path is not None and not result.dry_run:
        _record_translation_output_file_operation(
            agent,
            output_path=result.output_path,
            original=original,
            existed_before=existed_before,
            backup_path=backup_path,
        )


def translate_series_tool(
    agent: SubBakeAgent,
    path: Path,
    *,
    original: str,
    recursive: bool,
    overwrite: bool,
    dry_run: bool,
    suffixes: set[str] | None,
    arguments: dict[str, Any],
) -> None:
    """Translate a series of subtitle files in a directory."""
    files = discover_series_files(path, recursive=recursive, suffixes=suffixes)
    if not files:
        suffix_label = ", ".join(sorted(suffixes)) if suffixes else "subtitle"
        agent.console.print(
            f"[bold yellow]No {suffix_label} files found.[/bold yellow]"
        )
        agent._record_event(
            "series_empty",
            original,
            {
                "path": str(path),
                "suffixes": sorted(suffixes) if suffixes else None,
            },
        )
        agent._record_event(
            "assistant", "未找到需要翻译的字幕文件。",
            {"decision": "tool_call", "tool": "translate_series"},
        )
        return
    values = agent._translation_values_for_tool(arguments)
    if dry_run:
        values["dry_run"] = True
    agent._print_translation_start(
        original=original,
        values=values,
        file_count=len(files),
        suffixes={file_path.suffix.lower() for file_path in files},
        series=True,
    )
    agent.console.print(
        "[bold green]Series:[/bold green] "
        f"{len(files)} file(s), profile={agent.profile or 'default'}, "
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
            undo_snapshots[output_path] = _translation_output_undo_snapshot(agent, output_path)
    result = translate_series(
        root=path,
        values=values,
        backend_factory=lambda: _runtime_options.build_backend_from_values(values),
        console=agent.console,
        recursive=recursive,
        overwrite=overwrite,
        suffixes=suffixes,
        cancel_requested=_agent_cancel_requested(agent),
    )
    agent._print_series_summary(result)
    agent._print_series_completion(result)
    agent._record_event(
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
            "summary": _series_summary(result, values),
        },
    )
    # Record assistant event so series translation results appear in conversation history
    if result.failure_count:
        agent._record_event(
            "assistant", f"已完成 {result.processed_count} 个，跳过 {result.skipped_count} 个，失败 {result.failure_count} 个。",
            {"decision": "tool_call", "tool": "translate_series"},
        )
    else:
        agent._record_event(
            "assistant", f"已完成 {result.processed_count} 个文件翻译。",
            {"decision": "tool_call", "tool": "translate_series"},
        )
    operation_group_id = uuid.uuid4().hex
    for item in result.processed:
        if item.output_path is None:
            continue
        existed_before, backup_path = undo_snapshots.get(item.output_path, (False, None))
        _record_translation_output_file_operation(
            agent,
            output_path=item.output_path,
            original=original,
            existed_before=existed_before,
            backup_path=backup_path,
            group_id=operation_group_id,
        )


def edit_generated_subtitle(
    agent: SubBakeAgent,
    *,
    target_path: Path,
    instruction: str,
    original: str,
) -> None:
    """Edit a previously generated subtitle file with an instruction."""
    if not instruction:
        raise ValueError("Subtitle edit needs an instruction.")
    if not is_generated_subtitle(target_path):
        raise ValueError(
            "Agent edits are limited to generated subtitles such as *.translated.* or *.bilingual.*."
        )
    values = dict(agent.values)
    values["dry_run"] = False
    backend = _runtime_options.build_backend_from_values(values)
    if backend is None:
        raise RuntimeError("Subtitle edits require a model backend.")
    result = _edit_generated_subtitle(
        target_path=target_path,
        instruction=instruction,
        backend=backend,
        values=values,
        project_root=agent.project_root,
        cancel_requested=_agent_cancel_requested(agent),
    )
    agent.console.print(f"[bold green]Edited:[/bold green] {result.target_path}")
    agent.console.print(f"[bold green]Backup:[/bold green] {result.backup_path}")
    if result.translation_memory_path is not None:
        agent.console.print(
            f"[bold green]Translation memory:[/bold green] {result.translation_memory_path}"
        )
    if result.edit_notes:
        agent.console.print(f"[bold green]Notes:[/bold green] {result.edit_notes}")
    agent._record_event(
        "edit",
        original,
        {
            "target_path": str(result.target_path),
            "backup_path": str(result.backup_path),
            "instruction": instruction,
            "summary": f"编辑了 {result.target_path.name}：{_short_instruction(instruction)}",
        },
    )
    agent._record_event(
        "assistant", f"已完成编辑，输出 {result.target_path}。",
        {"decision": "tool_call", "tool": "edit_subtitle"},
    )


def _short_instruction(text: str, *, limit: int = 60) -> str:
    """Truncate an edit instruction for display in a summary."""
    cleaned = text.strip().replace("\n", " ")
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit - 3]}..."


def _agent_cancel_requested(agent: SubBakeAgent):
    return getattr(agent, "_cancel_requested", None)


# ---- helpers shared between translate_file and translate_series_tool --------


def _translation_output_undo_snapshot(
    agent: SubBakeAgent,
    output_path: Path,
) -> tuple[bool, Path | None]:
    """Take a pre-translation snapshot of an existing output file for undo."""
    if not output_path.exists():
        return False, None
    return True, _backup_translation_output_for_undo(agent, output_path)


def _backup_translation_output_for_undo(
    agent: SubBakeAgent,
    output_path: Path,
) -> Path:
    """Copy an output file to the agent backup directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    resolved_output = output_path.resolve()
    resolved_root = agent.project_root.resolve()
    try:
        relative = resolved_output.relative_to(resolved_root)
    except ValueError:
        digest = hashlib.sha1(str(resolved_output).encode("utf-8")).hexdigest()[:12]
        relative = Path("__external__") / digest / output_path.name
    backup_path = agent.project_root / ".subbake" / "agent" / "backups" / timestamp / relative
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.exists():
        backup_path = backup_path.with_name(
            f"{backup_path.stem}-{output_path.stat().st_mtime_ns}{backup_path.suffix}"
        )
    shutil.copy2(output_path, backup_path)
    return backup_path


def _record_translation_output_file_operation(
    agent: SubBakeAgent,
    *,
    output_path: Path,
    original: str,
    existed_before: bool,
    backup_path: Path | None,
    group_id: str | None = None,
) -> None:
    """Record a translation output file operation for undo tracking."""
    data: dict[str, Any] = {
        "action": "modified" if existed_before else "created",
        "path": str(output_path),
    }
    if backup_path is not None:
        data["backup_path"] = str(backup_path)
    if group_id is not None:
        data["group_id"] = group_id
    agent._record_event("file_operation", original, data)


def _translate_file_summary(path: Path, result, values: dict[str, Any]) -> str:
    """Build a human-readable summary string for a file translation event."""
    if bool(values["dry_run"]):
        return f"规划了 {path.name}（{values['target_language']}，{result.batches_translated} 批次）"
    output_name = result.output_path.name if result.output_path else path.name
    tokens_str = _format_tokens(result.usage.total_tokens)
    return (
        f"翻译了 {path.name} → {output_name}"
        f"（{values['target_language']}，{result.batches_translated} 批次，{tokens_str}）"
    )


def _series_summary(result, values: dict[str, Any]) -> str:
    """Build a human-readable summary string for a series translation event."""
    failures_str = f"，{result.failure_count} 个失败" if result.failure_count else ""
    skipped_str = f"，{result.skipped_count} 个跳过" if result.skipped_count else ""
    return (
        f"翻译了 {result.processed_count} 个文件"
        f"（{values['target_language']}{skipped_str}{failures_str}）"
    )


def _format_tokens(count: int) -> str:
    """Format a token count for display."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}K"
    return str(count)
