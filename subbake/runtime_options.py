from __future__ import annotations

from pathlib import Path
from typing import Any

from subbake.entities import DEFAULT_BATCH_SIZE, PipelineOptions
from subbake.languages import normalize_language_name
from subbake.models import build_backend
from subbake.models.base_model import LLMBackend


DEFAULT_TRANSLATION_VALUES: dict[str, Any] = {
    "provider": "mock",
    "model": "mock-zh",
    "api_key": None,
    "base_url": None,
    "output_format": None,
    "batch_size": DEFAULT_BATCH_SIZE,
    "fast": False,
    "bilingual": False,
    "source_language": "Auto",
    "target_language": "Chinese",
    "retries": 2,
    "final_review": True,
    "timeout": 120.0,
    "dry_run": False,
    "resume": True,
    "cache": True,
    "agent": True,
    "agent_repair_attempts": 2,
    "work_dir": None,
    "glossary_path": None,
    # Transcription defaults
    "transcriber": "whisper_api",
    "whisper_model": "small",
    "whisper_version": "latest",
    "whisper_bin_dir": None,
    "whisper_api_model": "whisper-1",
}


def merge_translation_values(*groups: dict[str, Any]) -> dict[str, Any]:
    values = dict(DEFAULT_TRANSLATION_VALUES)
    for group in groups:
        for key, value in group.items():
            if key in values:
                values[key] = value
    return values


def build_pipeline_options(
    *,
    input_path: Path,
    output_path: Path | None,
    values: dict[str, Any],
) -> PipelineOptions:
    normalized_source_language = normalize_language_name(
        str(values["source_language"]),
        allow_auto=True,
    )
    normalized_target_language = normalize_language_name(str(values["target_language"]))
    fast_mode = bool(values["fast"])

    return PipelineOptions(
        input_path=input_path,
        output_path=output_path,
        output_format=_optional_string(values.get("output_format")),
        provider=str(values["provider"]),
        model=str(values["model"]),
        batch_size=int(values["batch_size"]),
        fast_mode=fast_mode,
        bilingual=bool(values["bilingual"]),
        source_language=normalized_source_language,
        target_language=normalized_target_language,
        retries=int(values["retries"]),
        final_review=bool(values["final_review"]) and not fast_mode,
        timeout_seconds=float(values["timeout"]),
        api_key=_optional_string(values.get("api_key")),
        base_url=_optional_string(values.get("base_url")),
        dry_run=bool(values["dry_run"]),
        resume=bool(values["resume"]),
        use_cache=bool(values["cache"]),
        agent=bool(values["agent"]),
        agent_repair_attempts=int(values["agent_repair_attempts"]),
        work_dir=_optional_path(values.get("work_dir")),
        glossary_path=_optional_path(values.get("glossary_path")),
    )


def build_backend_from_values(values: dict[str, Any]) -> LLMBackend | None:
    if bool(values["dry_run"]):
        return None
    return build_backend(
        provider=str(values["provider"]),
        model=str(values["model"]),
        api_key=_optional_string(values.get("api_key")),
        base_url=_optional_string(values.get("base_url")),
        timeout_seconds=float(values["timeout"]),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def build_transcriber_from_values(values: dict[str, Any], *, project_root: Path) -> "TranscriberBackend":
    """Build a TranscriberBackend from translation/transcription values."""
    from subbake.transcriber import build_transcriber

    return build_transcriber(
        provider=str(values.get("transcriber", "whisper_api")),
        project_root=project_root,
        api_key=_optional_string(values.get("api_key")),
        base_url=_optional_string(values.get("base_url")),
        model=_optional_string(values.get("whisper_api_model")),
        whisper_model=str(values.get("whisper_model", "small")),
        timeout_seconds=float(values.get("timeout", 300.0)),
    )
