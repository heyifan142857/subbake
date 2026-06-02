from __future__ import annotations

import json

from subbake.entities import SubtitleSegment
from subbake.memory import ContextMemory


def select_relevant_glossary(
    glossary: dict[str, str],
    texts: list[str],
    limit: int = 24,
) -> dict[str, str]:
    if not glossary or not texts:
        return {}

    haystack = "\n".join(texts).casefold()
    matched: dict[str, str] = {}
    for source, target in glossary.items():
        if source.casefold() in haystack or target.casefold() in haystack:
            matched[source] = target
            if len(matched) >= limit:
                break
    return matched


def build_translation_messages(
    batch_segments: list[SubtitleSegment],
    memory: ContextMemory,
    source_language: str,
    target_language: str,
    fast_mode: bool = False,
) -> list[dict[str, str]]:
    batch_texts = [segment.text for segment in batch_segments if segment.text]
    context_payload = {
        "src": source_language,
        "tgt": target_language,
    }
    if not fast_mode:
        context_payload["rules"] = list(memory.style_rules)
    structure_notes = _translation_structure_notes(batch_segments)
    if structure_notes:
        context_payload["structure_notes"] = structure_notes
    if not fast_mode:
        recent = list(memory.recent_summaries[-memory.max_summaries :])
        if recent:
            context_payload["recent"] = recent
        glossary = select_relevant_glossary(memory.glossary, batch_texts)
        if glossary:
            context_payload["glossary"] = glossary

    if fast_mode:
        system_prompt = (
            "You are a fast subtitle translator.\n"
            "Return valid JSON only.\n"
            f"Translate into {target_language}.\n"
            "Prioritize finishing the batch while keeping subtitle order, count, and ids exact.\n"
            "Never merge, drop, or insert subtitle entries even when the spoken sentence spans multiple subtitle lines.\n"
            "Every non-empty source entry must produce one non-empty translated entry with the same id."
        )
    else:
        system_prompt = (
            "You are a professional subtitle translator.\n"
            "Return valid JSON only.\n"
            f"Translate into {target_language}.\n"
            "Keep subtitle order, count, and ids exact.\n"
            "Never merge, drop, or insert subtitle entries even when the spoken sentence spans multiple subtitle lines.\n"
            "Every non-empty source entry must produce one non-empty translated entry with the same id."
        )
    batch_payload = {
        "lines": [
            {
                "id": segment.id,
                "text": segment.text,
            }
            for segment in batch_segments
        ]
    }
    speed_note = (
        "Best-effort speed mode: if a fragment is unclear, still provide a short plausible translation for that id instead of leaving it blank.\n"
        if fast_mode
        else ""
    )
    user_prompt = (
        "TASK_START\n"
        "translate_subtitles\n"
        "TASK_END\n"
        f"Translate each line into {target_language}.\n"
        "Preserve line count, order, and ids exactly.\n"
        "Even if one spoken sentence spans multiple subtitle entries, keep each subtitle entry separate.\n"
        "Never merge neighboring lines to complete a sentence.\n"
        "If a line is only a fragment like 'and ...' or 'that ...', still translate that line alone and keep its id.\n"
        "Do not move words from one subtitle id into another subtitle id.\n"
        "Do not absorb a short fragment into the previous or next entry.\n"
        "Keep blank lines blank. Keep tone, slang, and profanity intact.\n"
        "Favor natural subtitle phrasing over literal wording.\n"
        f"{speed_note}"
        'Return JSON only with keys "lines", "summary", and "glossary_updates".\n'
        "CONTEXT_JSON_START\n"
        f"{_compact_json(context_payload)}\n"
        "CONTEXT_JSON_END\n"
        "BATCH_JSON_START\n"
        f"{_compact_json(batch_payload)}\n"
        "BATCH_JSON_END\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_review_messages(
    source_segments: list[SubtitleSegment],
    translated_segments: list[SubtitleSegment],
    memory: ContextMemory,
    target_language: str,
    reasons: list[str],
) -> list[dict[str, str]]:
    source_texts = [segment.text for segment in source_segments if segment.text]
    translated_texts = [segment.text for segment in translated_segments if segment.text]
    relevant_glossary = select_relevant_glossary(
        memory.glossary,
        source_texts + translated_texts,
    )
    system_prompt = (
        "You are performing a targeted subtitle QA review.\n"
        "Return valid JSON only.\n"
        f"Review {target_language} subtitles.\n"
        "Only fix terminology, consistency, and readability issues without changing the number of entries."
    )
    review_payload = {
        "tgt": target_language,
        "reasons": reasons,
        "expected_count": len(source_segments),
        "expected_ids": [source.id for source in source_segments],
        "lines": [
            {
                "id": source.id,
                "source": source.text,
                "translation": translated.text,
            }
            for source, translated in zip(source_segments, translated_segments, strict=True)
        ],
    }
    recent = list(memory.recent_summaries[-memory.max_summaries :])
    if recent:
        review_payload["recent"] = recent
    if relevant_glossary:
        review_payload["glossary"] = relevant_glossary

    user_prompt = (
        "TASK_START\n"
        "review_translations\n"
        "TASK_END\n"
        "Review only this high-risk batch.\n"
        "Use the input lines array as the complete authoritative list of subtitle entries.\n"
        "Do not remove, reorder, merge, or renumber entries.\n"
        "Return exactly one output object for each input line id, in the same order as expected_ids.\n"
        "Prefer minimal edits; leave good lines untouched.\n"
        'Return JSON only with keys "lines" and "review_notes".\n'
        "REVIEW_JSON_START\n"
        f"{_compact_json(review_payload)}\n"
        "REVIEW_JSON_END\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_agent_repair_messages(
    *,
    stage: str,
    source_segments: list[SubtitleSegment],
    target_language: str,
    last_error: str,
    attempt_logs: list[dict],
    agent_attempt_logs: list[dict] | None = None,
    translated_segments: list[SubtitleSegment] | None = None,
) -> list[dict[str, str]]:
    stage = stage.strip().lower()
    if stage not in {"translate", "review"}:
        raise ValueError(f"Unsupported agent repair stage: {stage}")

    expected_ids = [segment.id for segment in source_segments]
    repair_payload = {
        "stage": stage,
        "target_language": target_language,
        "last_error": last_error,
        "expected_count": len(source_segments),
        "expected_ids": expected_ids,
        "source_lines": [
            {
                "id": segment.id,
                "text": segment.text,
            }
            for segment in source_segments
        ],
        "failed_attempts": _compact_attempt_logs(attempt_logs),
        "agent_attempts": _compact_attempt_logs(agent_attempt_logs or []),
    }
    if translated_segments is not None:
        repair_payload["current_translations"] = [
            {
                "id": segment.id,
                "translation": segment.text,
            }
            for segment in translated_segments
        ]

    task = "agent_repair_translation" if stage == "translate" else "agent_repair_review"
    return_keys = (
        '"lines", "summary", and "glossary_updates"'
        if stage == "translate"
        else '"lines" and "review_notes"'
    )
    system_prompt = (
        "You are SubBake's runtime repair agent.\n"
        "Return valid JSON only.\n"
        "Repair the failed model output without changing source text, subtitle ids, order, count, runtime config, or files.\n"
        f"Every non-empty source entry must produce one non-empty {target_language} translation with the same id."
    )
    user_prompt = (
        "TASK_START\n"
        f"{task}\n"
        "TASK_END\n"
        "Read this failure log and return a corrected response for the same batch.\n"
        "Use expected_ids as the complete authoritative list and preserve that exact order.\n"
        "Do not explain the fix. Do not include markdown.\n"
        f"Return JSON only with keys {return_keys}.\n"
        "AGENT_REPAIR_JSON_START\n"
        f"{_compact_json(repair_payload)}\n"
        "AGENT_REPAIR_JSON_END\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_subtitle_edit_messages(
    *,
    target_segments: list[SubtitleSegment],
    instruction: str,
    target_language: str,
    source_segments: list[SubtitleSegment] | None = None,
) -> list[dict[str, str]]:
    edit_payload = {
        "target_language": target_language,
        "instruction": instruction,
        "expected_count": len(target_segments),
        "expected_ids": [segment.id for segment in target_segments],
        "lines": [
            {
                "id": segment.id,
                "translation": segment.text,
            }
            for segment in target_segments
        ],
    }
    if source_segments is not None:
        edit_payload["source_lines"] = [
            {
                "id": segment.id,
                "text": segment.text,
            }
            for segment in source_segments
        ]

    system_prompt = (
        "You are SubBake's subtitle editing agent.\n"
        "Return valid JSON only.\n"
        f"Edit {target_language} subtitles according to the user's instruction.\n"
        "Do not change subtitle ids, order, count, timings, or file format."
    )
    user_prompt = (
        "TASK_START\n"
        "agent_edit_subtitle\n"
        "TASK_END\n"
        "Apply the requested edit only where needed.\n"
        "Use expected_ids as the complete authoritative list and preserve that exact order.\n"
        "Keep good lines unchanged. Keep blank lines blank.\n"
        'Return JSON only with keys "lines" and "edit_notes".\n'
        "EDIT_JSON_START\n"
        f"{_compact_json(edit_payload)}\n"
        "EDIT_JSON_END\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _compact_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _compact_attempt_logs(attempt_logs: list[dict], limit: int = 4) -> list[dict]:
    compacted: list[dict] = []
    for attempt_log in attempt_logs[-limit:]:
        payload = attempt_log.get("payload")
        compacted.append(
            {
                "attempt": attempt_log.get("attempt"),
                "cached": attempt_log.get("cached"),
                "error": _truncate_text(str(attempt_log.get("error", "")).strip(), 1200),
                "payload": payload if isinstance(payload, dict) else payload,
                "split_retry": _compact_split_retry(attempt_log.get("split_retry")),
            }
        )
    return compacted


def _compact_split_retry(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    return {
        "triggered": value.get("triggered"),
        "sizes": value.get("sizes"),
        "resolved": value.get("resolved"),
        "error": _truncate_text(str(value.get("error", "")).strip(), 1200),
    }


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _translation_structure_notes(batch_segments: list[SubtitleSegment]) -> list[str]:
    notes: list[str] = []
    if any(
        _is_continuation_line(next_segment.text) and not _ends_sentence(current_segment.text)
        for current_segment, next_segment in zip(batch_segments, batch_segments[1:], strict=False)
    ):
        notes.append(
            "Some neighboring subtitle entries are fragments of the same spoken sentence. Translate each entry separately and keep every original id."
        )
        notes.append(
            "Do not combine neighboring fragments into one fluent sentence. Each original subtitle entry still needs its own non-empty translation."
        )
    if any(not segment.text.strip() for segment in batch_segments):
        notes.append("Blank subtitle entries must stay blank in the output.")
    return notes


def _is_continuation_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0].islower():
        return True
    lowered = stripped.casefold()
    return lowered.startswith(
        (
            "and ",
            "but ",
            "or ",
            "so ",
            "that ",
            "which ",
            "who ",
            "because ",
            "if ",
            "when ",
            "then ",
            "to ",
        )
    )


def _ends_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.endswith((".", "!", "?", "。", "！", "？", "…"))
