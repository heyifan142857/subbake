from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from subbake.checker import validate_translation_batch
from subbake.entities import SubtitleSegment
from subbake.models.base_model import LLMBackend, parse_translation_lines
from subbake.parsers import load_document, render_document
from subbake.prompts import build_subtitle_edit_messages
from subbake.storage import TranslationMemoryStore, build_runtime_paths

SUPPORTED_EDIT_SUFFIXES = {".srt", ".vtt", ".txt"}
GENERATED_MARKERS = (".translated.", ".bilingual.")


@dataclass(slots=True)
class SubtitleEditResult:
    target_path: Path
    backup_path: Path
    edit_notes: str = ""
    translation_memory_path: Path | None = None


def is_generated_subtitle(path: Path) -> bool:
    return any(marker in path.name for marker in GENERATED_MARKERS)


def edit_generated_subtitle(
    *,
    target_path: Path,
    instruction: str,
    backend: LLMBackend,
    values: dict,
    project_root: Path,
    allow_non_generated: bool = False,
) -> SubtitleEditResult:
    if not target_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {target_path}")
    if target_path.suffix.lower() not in SUPPORTED_EDIT_SUFFIXES:
        raise ValueError("Agent subtitle edits only support .srt, .vtt, and .txt files.")
    if not is_generated_subtitle(target_path) and not allow_non_generated:
        raise ValueError(
            "Refusing to edit a source-looking subtitle file. "
            "Ask the agent to edit a *.translated.* or *.bilingual.* output file."
        )

    target_document = load_document(target_path)
    source_document = _load_inferred_source_document(target_path, target_document.segments)
    messages = build_subtitle_edit_messages(
        target_segments=target_document.segments,
        source_segments=source_document.segments if source_document is not None else None,
        instruction=instruction,
        target_language=str(values["target_language"]),
    )
    payload, _ = backend.generate_json(messages)
    lines = parse_translation_lines(payload.get("lines", []))
    validate_translation_batch(target_document.segments, lines)
    edited_segments = [
        SubtitleSegment(
            id=source.id,
            text=line.translation,
            start=source.start,
            end=source.end,
            identifier=source.identifier,
            settings=source.settings,
        )
        for source, line in zip(target_document.segments, lines, strict=True)
    ]

    backup_path = _backup_target(target_path, project_root=project_root)
    rendered = render_document(
        target_document,
        edited_segments,
        bilingual=False,
        output_format=target_document.format,
    )
    target_path.write_text(rendered, encoding="utf-8")
    _verify_write_text(target_path, rendered)

    translation_memory_path = None
    if source_document is not None:
        translation_memory_path = _sync_translation_memory(
            source_path=source_document.path,
            source_segments=source_document.segments,
            edited_segments=edited_segments,
            values=values,
        )

    return SubtitleEditResult(
        target_path=target_path,
        backup_path=backup_path,
        edit_notes=str(payload.get("edit_notes", "")).strip(),
        translation_memory_path=translation_memory_path,
    )


def _load_inferred_source_document(
    target_path: Path,
    target_segments: list[SubtitleSegment],
):
    source_path = _infer_source_path(target_path)
    if source_path is None or not source_path.exists():
        return None
    try:
        source_document = load_document(source_path)
    except Exception:
        return None
    if len(source_document.segments) != len(target_segments):
        return None
    return source_document


def _infer_source_path(target_path: Path) -> Path | None:
    name = target_path.name
    for marker in GENERATED_MARKERS:
        if marker not in name:
            continue
        source_name = name.replace(marker, ".", 1)
        return target_path.with_name(source_name)
    return None


def _backup_target(target_path: Path, *, project_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = project_root / ".subbake" / "agent" / "backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / target_path.name
    if backup_path.exists():
        backup_path = backup_dir / f"{target_path.stem}-{target_path.stat().st_mtime_ns}{target_path.suffix}"
    shutil.copy2(target_path, backup_path)
    return backup_path


def _sync_translation_memory(
    *,
    source_path: Path,
    source_segments: list[SubtitleSegment],
    edited_segments: list[SubtitleSegment],
    values: dict,
) -> Path | None:
    runtime_paths = build_runtime_paths(
        input_path=source_path,
        work_dir=values.get("work_dir"),
        glossary_path=values.get("glossary_path"),
        source_language=str(values["source_language"]),
        target_language=str(values["target_language"]),
        fast_mode=bool(values["fast"]),
    )
    store = TranslationMemoryStore(runtime_paths.translation_memory_path)
    memory = store.load()
    changed = False
    for source, edited in zip(source_segments, edited_segments, strict=True):
        key = _translation_memory_key(source.text)
        if key is None or not edited.text.strip():
            continue
        if memory.get(key) == edited.text:
            continue
        memory[key] = edited.text
        changed = True
    if not changed:
        return runtime_paths.translation_memory_path
    store.save(memory)
    _write_edit_note(runtime_paths.translation_memory_path, source_path)
    return runtime_paths.translation_memory_path


def _translation_memory_key(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    normalized = re.sub(r"\s+", " ", stripped.casefold())
    return re.sub(r"\s+([,.!?;:])", r"\1", normalized)


def _write_edit_note(translation_memory_path: Path, source_path: Path) -> None:
    note_path = translation_memory_path.with_suffix(".edits.json")
    entries = []
    if note_path.exists():
        try:
            data = json.loads(note_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                entries = data
        except json.JSONDecodeError:
            entries = []
    entries.append(
        {
            "source_path": str(source_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    note_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(entries, ensure_ascii=False, indent=2)
    note_path.write_text(serialized, encoding="utf-8")
    _verify_json_write(note_path, serialized)


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


def _verify_json_write(path: Path, expected_serialized: str) -> None:
    """Read back a just-written JSON file and verify it contains valid, matching JSON."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Write verification failed: cannot read back {path}: {exc}") from exc
    if not raw.strip():
        raise OSError(f"Write verification failed: {path} is empty after write")
    if raw != expected_serialized:
        raise OSError(
            f"Write verification failed for {path}: "
            f"content mismatch (expected {len(expected_serialized)} bytes, "
            f"got {len(raw)} bytes)"
        )
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OSError(f"Write verification failed: {path} contains invalid JSON: {exc}") from exc
