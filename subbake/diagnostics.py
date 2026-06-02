from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DiagnosticReport:
    source: str
    diagnosis: str
    suggestions: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)


def diagnose_path(path: Path) -> DiagnosticReport:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return diagnose_text(text, source=str(path))
        if isinstance(payload, dict):
            return diagnose_failure_payload(payload, source=str(path))
    return diagnose_text(text, source=str(path))


def diagnose_failure_payload(payload: dict[str, Any], *, source: str = "failure log") -> DiagnosticReport:
    attempts = _collect_attempts(payload)
    errors = _collect_errors(attempts)
    metadata = _collect_metadata(attempts)
    details = _payload_details(payload, errors, metadata)
    diagnosis, suggestions = _diagnose_errors(errors, metadata)
    return DiagnosticReport(
        source=source,
        diagnosis=diagnosis,
        suggestions=suggestions,
        details=details,
    )


def diagnose_text(text: str, *, source: str = "pasted log") -> DiagnosticReport:
    lowered = text.casefold()
    errors = [line.strip() for line in text.splitlines() if line.strip()]
    metadata: list[dict[str, Any]] = []
    if "status=429" in lowered or "status_code" in lowered and "429" in lowered:
        metadata.append({"status_code": 429})
    if "status=500" in lowered or "status_code" in lowered and "500" in lowered:
        metadata.append({"status_code": 500})
    if "timeout" in lowered or "timed out" in lowered:
        metadata.append({"reason": "timeout"})
    diagnosis, suggestions = _diagnose_errors(errors, metadata)
    return DiagnosticReport(
        source=source,
        diagnosis=diagnosis,
        suggestions=suggestions,
        details=errors[-4:],
    )


def format_diagnostic_report(report: DiagnosticReport) -> str:
    lines = [
        f"Source: {report.source}",
        f"Diagnosis: {report.diagnosis}",
    ]
    if report.details:
        lines.append("Details:")
        lines.extend(f"- {detail}" for detail in report.details)
    if report.suggestions:
        lines.append("Suggestions:")
        lines.extend(f"- {suggestion}" for suggestion in report.suggestions)
    return "\n".join(lines)


def _collect_attempts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for key in ("attempts", "agent_attempts"):
        value = payload.get(key)
        if isinstance(value, list):
            attempts.extend(item for item in value if isinstance(item, dict))
    return attempts


def _collect_errors(attempts: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for attempt in attempts:
        error = str(attempt.get("error") or "").strip()
        if error:
            errors.append(error)
        split_retry = attempt.get("split_retry")
        if isinstance(split_retry, dict):
            split_error = str(split_retry.get("error") or "").strip()
            if split_error:
                errors.append(split_error)
    return errors


def _collect_metadata(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for attempt in attempts:
        value = attempt.get("error_meta")
        if isinstance(value, dict):
            metadata.append(value)
        split_retry = attempt.get("split_retry")
        if isinstance(split_retry, dict):
            split_value = split_retry.get("error_meta")
            if isinstance(split_value, dict):
                metadata.append(split_value)
    return metadata


def _payload_details(
    payload: dict[str, Any],
    errors: list[str],
    metadata: list[dict[str, Any]],
) -> list[str]:
    details: list[str] = []
    stage = payload.get("stage")
    batch_index = payload.get("batch_index")
    if stage is not None and batch_index is not None:
        details.append(f"{stage} batch {batch_index}")
    if errors:
        details.append(f"Last error: {errors[-1]}")
    for item in metadata[-2:]:
        parts: list[str] = []
        if item.get("status_code") is not None:
            parts.append(f"status={item['status_code']}")
        if item.get("request_id"):
            parts.append(f"request_id={item['request_id']}")
        if item.get("reason"):
            parts.append(f"reason={item['reason']}")
        if parts:
            details.append(", ".join(parts))
    return details


def _diagnose_errors(
    errors: list[str],
    metadata: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    for error in reversed(errors):
        if "Line count mismatch:" in error:
            return (
                "Model output dropped, inserted, or merged subtitle entries.",
                [
                    "Rerun with a smaller --batch-size.",
                    "Keep --resume enabled so completed batches are reused.",
                    "If this repeats, try a stronger model profile with /model.",
                ],
            )
        if "Empty translation for subtitle id" in error:
            return (
                "Model returned an empty translation for a non-empty subtitle.",
                [
                    "Rerun with a smaller --batch-size or without --fast.",
                    "Inspect the saved batch for very short fragments or malformed tags.",
                ],
            )
        if "ID mismatch:" in error:
            return (
                "Model changed subtitle ids instead of preserving the source structure.",
                [
                    "Rerun the same command; the retry prompt usually corrects this.",
                    "If it repeats, lower --batch-size or switch profile with /model.",
                ],
            )

    for item in reversed(metadata):
        status_code = item.get("status_code")
        if status_code == 429:
            return (
                "Provider rate limit was hit.",
                [
                    "Wait and rerun with resume enabled.",
                    "Use a smaller --batch-size or a different profile if the provider has strict limits.",
                ],
            )
        if isinstance(status_code, int) and 500 <= status_code < 600:
            return (
                "Provider returned a temporary server-side error.",
                ["Rerun later with resume enabled so finished batches are not repeated."],
            )
        if item.get("reason"):
            return (
                "Request failed before receiving a valid model response.",
                ["Check network, API base URL, and credentials with sbake check-key."],
            )

    combined = "\n".join(errors).casefold()
    if "missing api key" in combined or "credential" in combined:
        return (
            "Provider credentials are missing or invalid.",
            ["Run sbake check-key --profile <name> or set the configured API key environment variable."],
        )
    if "unsupported input format" in combined:
        return (
            "The referenced file is not a supported subtitle format.",
            ["Use .srt, .vtt, or .txt input files."],
        )
    if "timeout" in combined or "timed out" in combined:
        return (
            "The provider request timed out.",
            ["Increase --timeout or use a faster profile."],
        )
    if errors:
        return (
            "The failure is not one of SubBake's known structural cases.",
            ["Open the saved failure sample and inspect the last payload and request metadata."],
        )
    return (
        "No specific failure pattern was found.",
        ["Share a SubBake failure JSON or paste the terminal error text for a better diagnosis."],
    )
