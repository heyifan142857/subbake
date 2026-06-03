from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rich.console import Console

from subbake.entities import PipelineResult
from subbake.models.base_model import LLMBackend
from subbake.pipeline import SubtitlePipeline
from subbake.runtime_options import build_pipeline_options
from subbake.ui import Dashboard

SUPPORTED_SUBTITLE_SUFFIXES = {".srt", ".vtt", ".txt"}
GENERATED_MARKERS = (".translated.", ".bilingual.")


@dataclass(slots=True)
class SeriesFileResult:
    input_path: Path
    output_path: Path | None = None
    skipped: bool = False
    reason: str = ""
    result: PipelineResult | None = None


@dataclass(slots=True)
class SeriesResult:
    root: Path
    files: list[Path]
    processed: list[SeriesFileResult] = field(default_factory=list)
    skipped: list[SeriesFileResult] = field(default_factory=list)
    failures: list[SeriesFileResult] = field(default_factory=list)

    @property
    def processed_count(self) -> int:
        return len(self.processed)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def failure_count(self) -> int:
        return len(self.failures)


def discover_series_files(
    root: Path,
    *,
    recursive: bool = False,
    suffixes: set[str] | None = None,
) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Series folder not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Series target must be a directory: {root}")

    supported_suffixes = _normalize_suffixes(suffixes)
    candidates = root.rglob("*") if recursive else root.glob("*")
    files = [
        path
        for path in candidates
        if _is_supported_source(path, root=root, suffixes=supported_suffixes)
    ]
    return sorted(files, key=_natural_sort_key)


def translate_series(
    *,
    root: Path,
    values: dict,
    backend_factory: Callable[[], LLMBackend | None],
    console: Console,
    recursive: bool = False,
    overwrite: bool = False,
    suffixes: set[str] | None = None,
) -> SeriesResult:
    files = discover_series_files(root, recursive=recursive, suffixes=suffixes)
    result = SeriesResult(root=root, files=files)
    shared_values = dict(values)
    if shared_values.get("work_dir") is None:
        shared_values["work_dir"] = root / ".subbake"

    backend: LLMBackend | None = None
    if not bool(shared_values["dry_run"]):
        backend = backend_factory()

    for input_path in files:
        output_path = resolve_series_output_path(
            input_path,
            output_format=shared_values.get("output_format"),
            bilingual=bool(shared_values["bilingual"]),
        )
        if output_path.exists() and not overwrite:
            result.skipped.append(
                SeriesFileResult(
                    input_path=input_path,
                    output_path=output_path,
                    skipped=True,
                    reason="output exists",
                )
            )
            continue

        file_values = dict(shared_values)
        options = build_pipeline_options(
            input_path=input_path,
            output_path=None,
            values=file_values,
        )
        try:
            pipeline = SubtitlePipeline(
                backend=backend,
                options=options,
                dashboard=Dashboard(console=console),
            )
            pipeline_result = pipeline.run()
        except Exception as exc:
            result.failures.append(
                SeriesFileResult(
                    input_path=input_path,
                    output_path=output_path,
                    reason=str(exc),
                )
            )
            continue

        result.processed.append(
            SeriesFileResult(
                input_path=input_path,
                output_path=pipeline_result.output_path,
                result=pipeline_result,
            )
        )
    return result


def resolve_series_output_path(
    input_path: Path,
    *,
    output_format: object,
    bilingual: bool,
) -> Path:
    if output_format is None:
        suffix = input_path.suffix.lower()
    else:
        normalized = str(output_format).strip().lower().lstrip(".")
        suffix = f".{normalized}"
    if suffix not in SUPPORTED_SUBTITLE_SUFFIXES:
        raise ValueError("Supported output formats are srt, vtt, and txt.")
    flavor = "bilingual" if bilingual else "translated"
    return input_path.with_name(f"{input_path.stem}.{flavor}{suffix}")


def _is_supported_source(path: Path, *, root: Path, suffixes: set[str]) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in suffixes:
        return False
    if any(marker in path.name for marker in GENERATED_MARKERS):
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return ".subbake" not in relative.parts


def _normalize_suffixes(suffixes: set[str] | None) -> set[str]:
    if suffixes is None:
        return SUPPORTED_SUBTITLE_SUFFIXES
    normalized = {f".{suffix.lower().lstrip('.')}" for suffix in suffixes}
    unsupported = normalized - SUPPORTED_SUBTITLE_SUFFIXES
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported series input suffix: {names}")
    return normalized


def _natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", str(path).casefold())
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key
