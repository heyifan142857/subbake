from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable

from subbake.checker import ValidationError, validate_full_alignment, validate_translation_batch
from subbake.entities import (
    AgentRepairRecord,
    BatchPlanEntry,
    BatchTranslationResult,
    GlossaryEntry,
    PipelineOptions,
    PipelineResult,
    ReviewResult,
    SubtitleDocument,
    SubtitleSegment,
    TranslationLine,
    Usage,
)
from subbake.languages import normalize_language_name
from subbake.memory import ContextMemory
from subbake.models.base_model import (
    BackendRequestError,
    LLMBackend,
    parse_glossary_entries,
    parse_translation_lines,
)
from subbake.parsers import load_document, render_document
from subbake.prompts import (
    build_agent_repair_messages,
    build_review_messages,
    build_translation_messages,
    select_relevant_glossary,
)
from subbake.cancellation import OperationCancelledError, run_interruptibly
from subbake.storage import (
    AgentLogStore,
    BatchShardStore,
    CacheStore,
    FailureStore,
    GlossaryStore,
    ResumeSnapshot,
    RunStateStore,
    TranslationMemoryStore,
    build_render_fingerprint,
    build_request_hash,
    build_translation_fingerprint,
    build_runtime_paths,
    compute_input_signature,
)
from subbake.ui import Dashboard


@dataclass(slots=True)
class BatchSlices:
    index: int
    start_offset: int
    source: list[SubtitleSegment]
    translated: list[SubtitleSegment]


@dataclass(slots=True)
class ReviewBatchPlan:
    index: int
    start_offset: int
    source: list[SubtitleSegment]
    translated: list[SubtitleSegment]
    reasons: list[str]


@dataclass(slots=True)
class AgentRepairOutcome:
    success: bool
    usage: Usage
    attempts: list[dict[str, Any]]
    log_path: Path
    error: str = ""
    translation_result: BatchTranslationResult | None = None
    review_result: ReviewResult | None = None


class SubtitlePipeline:
    def __init__(
        self,
        backend: LLMBackend | None,
        options: PipelineOptions,
        dashboard: Dashboard | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        self.backend = backend
        self.options = replace(
            options,
            source_language=normalize_language_name(options.source_language, allow_auto=True),
            target_language=normalize_language_name(options.target_language),
        )
        self.memory = ContextMemory()
        self.dashboard = dashboard or Dashboard()
        self.cancel_requested = cancel_requested
        self.cache_hits = 0
        self.translation_memory_hits = 0
        self.resumed_translation_batches = 0
        self.resumed_review_batches = 0
        self.runtime_paths = build_runtime_paths(
            input_path=self.options.input_path,
            work_dir=self.options.work_dir,
            glossary_path=self.options.glossary_path,
            source_language=self.options.source_language,
            target_language=self.options.target_language,
            fast_mode=self.options.fast_mode,
        )
        self.cache_store = CacheStore(self.runtime_paths.cache_dir)
        self.glossary_store = GlossaryStore(self.runtime_paths.glossary_path)
        self.translation_memory_store = TranslationMemoryStore(self.runtime_paths.translation_memory_path)
        self.failure_store = FailureStore(self.runtime_paths.failures_dir)
        self.translation_store = BatchShardStore(self.runtime_paths.translated_batches_dir)
        self.review_store = BatchShardStore(self.runtime_paths.reviewed_batches_dir)
        self.agent_log_store = AgentLogStore(self.runtime_paths.agent_logs_dir)
        self.translation_memory: dict[str, str] = {}
        self.agent_repairs: list[AgentRepairRecord] = []
        self.state_store: RunStateStore | None = None
        self.input_signature: dict | None = None
        self.output_format = self._resolve_output_format(options.input_path)
        self.output_path = self._resolve_output_path(options.input_path)

    def _check_cancelled(self) -> None:
        """Raise OperationCancelledError if the user has requested cancellation."""
        if self.cancel_requested is not None and self.cancel_requested():
            raise OperationCancelledError("Operation cancelled by user.")

    def _generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        self._check_cancelled()
        return run_interruptibly(
            lambda: self._require_backend().generate_json(messages),
            cancel_requested=self.cancel_requested,
        )

    def run(self) -> PipelineResult:
        input_path = self.options.input_path
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if self.options.batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")
        if self.options.agent_repair_attempts < 0:
            raise ValueError("Agent repair attempts must be zero or greater.")

        with self.dashboard.running():
            self.dashboard.mark_running("LOAD_FILE")
            self._validate_input_path(input_path)
            self.input_signature = compute_input_signature(input_path)
            self.state_store = RunStateStore(
                path=self.runtime_paths.state_path,
                translation_fingerprint=build_translation_fingerprint(self.options, self.input_signature),
                render_fingerprint=build_render_fingerprint(self.options),
            )
            self.dashboard.mark_done("LOAD_FILE")
            self._check_cancelled()

            self.dashboard.mark_running("PARSE")
            document = load_document(input_path)
            translation_batches = self._chunk_segments(document.segments)
            if self.options.dry_run:
                self.dashboard.set_total_steps(2)
            else:
                self.dashboard.set_total_steps(2 + len(translation_batches) + 2)
            self.dashboard.mark_done("PARSE")
            self._check_cancelled()

            if self.options.dry_run:
                return PipelineResult(
                    output_path=None,
                    batches_translated=0,
                    review_batches=0,
                    usage=self.dashboard.usage,
                    dry_run=True,
                    planned_batches=self._build_batch_plan(translation_batches),
                    cache_hits=0,
                    resumed_translation_batches=0,
                    resumed_review_batches=0,
                    translation_memory_hits=0,
                    state_path=self.runtime_paths.state_path,
                    glossary_path=self.runtime_paths.glossary_path,
                    agent_repairs=list(self.agent_repairs),
                )

            resume = self._load_resume_state(total_translation_batches=len(translation_batches))
            translated_segments = self._translate_document(translation_batches, resume)

            review_plan: list[ReviewBatchPlan] = []
            if self.options.final_review and not self.options.fast_mode and translated_segments:
                review_plan = self._build_review_plan(translation_batches, translated_segments)
            review_progress_steps = len(review_plan) if review_plan else 1
            self.dashboard.set_total_steps(
                2 + len(translation_batches) + 1 + review_progress_steps + 1
            )
            if self._has_resume_progress(resume):
                self._restore_resume_progress(
                    resume=resume,
                    total_translation_batches=len(translation_batches),
                    review_batches=len(review_plan),
                )

            self.dashboard.mark_running("VALIDATE")
            validate_full_alignment(document.segments, translated_segments)
            self.dashboard.mark_done("VALIDATE")
            self._check_cancelled()
            self._save_run_state(
                translation_batches_completed=len(translation_batches),
                review_batches_completed=resume.review_batches_completed,
                validation_completed=True,
            )

            reviewed_segments = translated_segments
            if review_plan:
                reviewed_segments = self._review_document(
                    review_plan,
                    translated_segments=translated_segments,
                    total_translation_batches=len(translation_batches),
                    resume=resume,
                )
                validate_full_alignment(document.segments, reviewed_segments)
            else:
                self.dashboard.mark_skipped("FINAL_REVIEW")

            self._check_cancelled()
            output_segments = self._build_output_segments(document, reviewed_segments)
            self.dashboard.mark_running("WRITE_OUTPUT")
            rendered = render_document(
                document,
                output_segments,
                bilingual=self.options.bilingual,
                output_format=self.output_format,
            )
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(rendered, encoding="utf-8")
            _verify_write_text(self.output_path, rendered)
            self.dashboard.mark_done("WRITE_OUTPUT")
            self.dashboard.clear_batch()
            self._save_run_state(
                translation_batches_completed=len(translation_batches),
                review_batches_completed=len(review_plan),
                validation_completed=True,
            )

        return PipelineResult(
            output_path=self.output_path,
            batches_translated=len(translation_batches),
            review_batches=len(review_plan),
            usage=self.dashboard.usage,
            cache_hits=self.cache_hits,
            resumed_translation_batches=self.resumed_translation_batches,
            resumed_review_batches=self.resumed_review_batches,
            translation_memory_hits=self.translation_memory_hits,
            state_path=self.runtime_paths.state_path,
            glossary_path=self.runtime_paths.glossary_path,
            agent_repairs=list(self.agent_repairs),
        )

    def _translate_document(
        self,
        translation_batches: list[list[SubtitleSegment]],
        resume: ResumeSnapshot,
    ) -> list[SubtitleSegment]:
        translated_segments: list[SubtitleSegment] = list(resume.translated_segments)
        total_batches = len(translation_batches)
        for batch_index, batch_segments in enumerate(
            translation_batches[resume.translation_batches_completed :],
            start=resume.translation_batches_completed + 1,
        ):
            self._check_cancelled()
            label = f"TRANSLATE_BATCH {batch_index}/{total_batches}"
            self.dashboard.mark_running("TRANSLATE_BATCH", label=label)
            started_at = perf_counter()
            batch_result, usage = self._translate_batch_with_retry(batch_segments, batch_index)
            latency = perf_counter() - started_at
            self.dashboard.set_batch(batch_index, total_batches, latency, label)
            self.dashboard.add_usage(usage)
            self.memory.update(batch_result.summary, batch_result.glossary_updates)
            self.glossary_store.save(self.memory.glossary)
            translated_batch = self._materialize_translations(batch_segments, batch_result.lines)
            self._update_translation_memory(batch_segments, translated_batch)
            self.translation_store.save_segments(batch_index, translated_batch)
            translated_segments.extend(translated_batch)
            self._save_run_state(
                translation_batches_completed=batch_index,
                review_batches_completed=resume.review_batches_completed,
                validation_completed=False,
            )
            self.dashboard.mark_done("TRANSLATE_BATCH")
        return translated_segments

    def _review_document(
        self,
        review_plan: list[ReviewBatchPlan],
        translated_segments: list[SubtitleSegment],
        total_translation_batches: int,
        resume: ResumeSnapshot,
    ) -> list[SubtitleSegment]:
        reviewed_segments: list[SubtitleSegment] = list(translated_segments)
        resume_offset = 0
        for batch in review_plan[: resume.review_batches_completed]:
            batch_size = len(batch.source)
            restored_batch = resume.reviewed_segments[resume_offset : resume_offset + batch_size]
            if len(restored_batch) != batch_size:
                raise RuntimeError(
                    "Resume state is missing reviewed batch data. Re-run with --no-resume or clean the run directory."
                )
            reviewed_segments[batch.start_offset : batch.start_offset + batch_size] = restored_batch
            resume_offset += batch_size

        total_batches = len(review_plan)
        for review_position, batch in enumerate(
            review_plan[resume.review_batches_completed :],
            start=resume.review_batches_completed + 1,
        ):
            self._check_cancelled()
            label = f"FINAL_REVIEW {review_position}/{total_batches}"
            self.dashboard.mark_running("FINAL_REVIEW", label=label)
            started_at = perf_counter()
            review_result, usage = self._review_batch_with_retry(
                batch.source,
                batch.translated,
                batch.index,
                batch.reasons,
            )
            latency = perf_counter() - started_at
            self.dashboard.set_batch(review_position, total_batches, latency, label)
            self.dashboard.add_usage(usage)
            reviewed_batch = self._materialize_translations(batch.source, review_result.lines)
            self.review_store.save_segments(review_position, reviewed_batch)
            reviewed_segments[
                batch.start_offset : batch.start_offset + len(reviewed_batch)
            ] = reviewed_batch
            self._save_run_state(
                translation_batches_completed=total_translation_batches,
                review_batches_completed=review_position,
                validation_completed=True,
            )
            self.dashboard.mark_done("FINAL_REVIEW")
        return reviewed_segments

    def _translate_batch_with_retry(
        self,
        batch_segments: list[SubtitleSegment],
        batch_index: int,
    ) -> tuple[BatchTranslationResult, Usage]:
        return self._translate_batch_with_retry_impl(
            batch_segments=batch_segments,
            batch_index=batch_index,
            record_failure=True,
        )

    def _translate_batch_with_retry_impl(
        self,
        *,
        batch_segments: list[SubtitleSegment],
        batch_index: int,
        record_failure: bool,
    ) -> tuple[BatchTranslationResult, Usage]:
        attempts = self.options.retries + 1
        last_error: Exception | None = None
        attempt_logs: list[dict] = []
        last_request_hash = ""
        tm_matches = self._lookup_translation_memory(batch_segments)
        pending_segments = [
            segment for segment in batch_segments if segment.id not in tm_matches
        ]
        if not pending_segments:
            return (
                BatchTranslationResult(
                    lines=[tm_matches[segment.id] for segment in batch_segments],
                    summary="",
                ),
                Usage(),
            )

        for attempt in range(1, attempts + 1):
            payload: dict | None = None
            usage = Usage()
            cached = False
            messages: list[dict[str, str]] = []
            try:
                messages = build_translation_messages(
                    batch_segments=pending_segments,
                    memory=self.memory,
                    source_language=self.options.source_language,
                    target_language=self.options.target_language,
                    fast_mode=self.options.fast_mode,
                )
                if last_error is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous response failed validation.\n"
                                f"Validation error: {last_error}\n"
                                "Re-send corrected JSON only."
                            ),
                        }
                    )
                request_hash = build_request_hash(
                    provider=self.options.provider,
                    model=self.options.model,
                    stage="translate",
                    messages=messages,
                )
                last_request_hash = request_hash
                if self.options.use_cache:
                    cached_entry = self.cache_store.load("translate", request_hash)
                    if cached_entry is not None:
                        payload, _ = cached_entry
                        usage = Usage()
                        cached = True
                        self.cache_hits += 1
                if payload is None:
                    payload, usage = self._generate_json(messages)
                lines = parse_translation_lines(payload.get("lines", []))
                validate_translation_batch(pending_segments, lines)
                glossary_updates = parse_glossary_entries(payload.get("glossary_updates", []))
                result = BatchTranslationResult(
                    lines=self._merge_translation_lines(batch_segments, tm_matches, lines),
                    summary=str(payload.get("summary", "")).strip(),
                    glossary_updates=glossary_updates,
                )
                if self.options.use_cache and not cached:
                    self.cache_store.save("translate", request_hash, payload, usage)
                return result, usage
            except OperationCancelledError:
                raise
            except Exception as exc:
                last_error = exc
                fast_mode_result = self._fast_mode_translation_result(
                    batch_segments=pending_segments,
                    payload=payload,
                )
                if fast_mode_result is not None:
                    return (
                        BatchTranslationResult(
                            lines=self._merge_translation_lines(
                                batch_segments,
                                tm_matches,
                                fast_mode_result.lines,
                            ),
                            summary=fast_mode_result.summary,
                            glossary_updates=fast_mode_result.glossary_updates,
                        ),
                        usage,
                    )
                attempt_log = {
                    "attempt": attempt,
                    "cached": cached,
                    "error": str(exc),
                    "error_meta": self._error_metadata(exc),
                    "payload": payload,
                    "messages": messages,
                }
                if self._should_split_translation_batch(exc, pending_segments):
                    attempt_log["split_retry"] = {
                        "triggered": True,
                        "sizes": self._planned_split_sizes(pending_segments),
                    }
                    try:
                        split_result, split_usage = self._split_translation_batch(
                            batch_segments=pending_segments,
                            batch_index=batch_index,
                        )
                    except OperationCancelledError:
                        raise
                    except Exception as split_exc:
                        attempt_log["split_retry"]["error"] = str(split_exc)
                        attempt_log["split_retry"]["error_meta"] = self._error_metadata(split_exc)
                        attempt_logs.append(attempt_log)
                        if record_failure:
                            agent_outcome = self._repair_translation_with_agent(
                                batch_segments=pending_segments,
                                batch_index=batch_index,
                                last_error=split_exc,
                                attempt_logs=attempt_logs,
                                last_payload=payload,
                            )
                            if agent_outcome is not None and agent_outcome.success:
                                if agent_outcome.translation_result is None:
                                    raise RuntimeError("Agent repair succeeded without a translation result.")
                                return (
                                    BatchTranslationResult(
                                        lines=self._merge_translation_lines(
                                            batch_segments,
                                            tm_matches,
                                            agent_outcome.translation_result.lines,
                                        ),
                                        summary=agent_outcome.translation_result.summary,
                                        glossary_updates=agent_outcome.translation_result.glossary_updates,
                                    ),
                                    agent_outcome.usage,
                                )
                            failure_path = self.failure_store.write(
                                stage="translate",
                                batch_index=batch_index,
                                request_hash=last_request_hash,
                                batch_segments=batch_segments,
                                messages=messages,
                                attempts=attempt_logs,
                                agent_attempts=agent_outcome.attempts if agent_outcome is not None else None,
                            )
                            raise RuntimeError(
                                self._build_translation_failure_message(
                                    batch_index=batch_index,
                                    attempts=attempt,
                                    failure_path=failure_path,
                                    attempt_logs=attempt_logs,
                                    split_fallback=True,
                                    agent_outcome=agent_outcome,
                                )
                            ) from split_exc
                        raise split_exc

                    total_usage = Usage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        total_tokens=usage.total_tokens,
                    )
                    total_usage.add(split_usage)
                    attempt_log["split_retry"]["resolved"] = True
                    attempt_logs.append(attempt_log)
                    return (
                        BatchTranslationResult(
                            lines=self._merge_translation_lines(
                                batch_segments,
                                tm_matches,
                                split_result.lines,
                            ),
                            summary=split_result.summary,
                            glossary_updates=split_result.glossary_updates,
                        ),
                        total_usage,
                    )

                attempt_logs.append(attempt_log)
                if attempt == attempts:
                    if record_failure:
                        agent_outcome = self._repair_translation_with_agent(
                            batch_segments=pending_segments,
                            batch_index=batch_index,
                            last_error=exc,
                            attempt_logs=attempt_logs,
                            last_payload=payload,
                        )
                        if agent_outcome is not None and agent_outcome.success:
                            if agent_outcome.translation_result is None:
                                raise RuntimeError("Agent repair succeeded without a translation result.")
                            return (
                                BatchTranslationResult(
                                    lines=self._merge_translation_lines(
                                        batch_segments,
                                        tm_matches,
                                        agent_outcome.translation_result.lines,
                                    ),
                                    summary=agent_outcome.translation_result.summary,
                                    glossary_updates=agent_outcome.translation_result.glossary_updates,
                                ),
                                agent_outcome.usage,
                            )
                        failure_path = self.failure_store.write(
                            stage="translate",
                            batch_index=batch_index,
                            request_hash=last_request_hash,
                            batch_segments=batch_segments,
                            messages=messages,
                            attempts=attempt_logs,
                            agent_attempts=agent_outcome.attempts if agent_outcome is not None else None,
                        )
                        raise RuntimeError(
                            self._build_translation_failure_message(
                                batch_index=batch_index,
                                attempts=attempts,
                                failure_path=failure_path,
                                attempt_logs=attempt_logs,
                                split_fallback=False,
                                agent_outcome=agent_outcome,
                            )
                        ) from exc
                    raise exc
        raise RuntimeError("Translation batch retry loop ended unexpectedly.")

    def _should_split_translation_batch(
        self,
        exc: Exception,
        batch_segments: list[SubtitleSegment],
    ) -> bool:
        if self.options.fast_mode:
            return False
        return isinstance(exc, ValidationError) and len(batch_segments) > 1

    def _fast_mode_translation_result(
        self,
        *,
        batch_segments: list[SubtitleSegment],
        payload: dict | None,
    ) -> BatchTranslationResult | None:
        if not self.options.fast_mode or not isinstance(payload, dict):
            return None
        raw_lines = payload.get("lines")
        if not isinstance(raw_lines, list):
            return None

        glossary_updates: list[GlossaryEntry] = []
        try:
            glossary_updates = parse_glossary_entries(payload.get("glossary_updates", []))
        except Exception:
            glossary_updates = []

        return BatchTranslationResult(
            lines=self._best_effort_translation_lines(batch_segments, raw_lines),
            summary=str(payload.get("summary", "")).strip(),
            glossary_updates=glossary_updates,
        )

    def _best_effort_translation_lines(
        self,
        source_segments: list[SubtitleSegment],
        raw_lines: list[object],
    ) -> list[TranslationLine]:
        candidates = [
            candidate
            for candidate in (
                self._coerce_best_effort_translation_candidate(item)
                for item in raw_lines
            )
            if candidate is not None
        ]
        used_indexes: set[int] = set()
        resolved: list[TranslationLine] = []

        for source_segment in source_segments:
            translation_text = self._pop_best_effort_translation(candidates, used_indexes, source_segment.id)
            if not source_segment.text.strip():
                translation_text = ""
            elif not translation_text:
                translation_text = source_segment.text
            resolved.append(
                TranslationLine(
                    id=source_segment.id,
                    translation=translation_text,
                )
            )
        return resolved

    def _coerce_best_effort_translation_candidate(
        self,
        item: object,
    ) -> tuple[str | None, str] | None:
        if isinstance(item, dict):
            candidate_id = item.get("id")
            translation = item.get("translation")
            if translation is None:
                translation = item.get("text")
            if translation is None:
                translation = item.get("target")
            if translation is None:
                return None
            return (
                str(candidate_id).strip() if candidate_id is not None else None,
                str(translation).strip(),
            )
        if isinstance(item, str):
            return None, item.strip()
        return None

    def _pop_best_effort_translation(
        self,
        candidates: list[tuple[str | None, str]],
        used_indexes: set[int],
        source_id: str,
    ) -> str:
        for index, (candidate_id, translation) in enumerate(candidates):
            if index in used_indexes or candidate_id != source_id:
                continue
            used_indexes.add(index)
            return translation
        for index, (_, translation) in enumerate(candidates):
            if index in used_indexes:
                continue
            used_indexes.add(index)
            return translation
        return ""

    def _split_translation_batch(
        self,
        *,
        batch_segments: list[SubtitleSegment],
        batch_index: int,
    ) -> tuple[BatchTranslationResult, Usage]:
        split_index = self._translation_split_index(batch_segments)
        left_segments = batch_segments[:split_index]
        right_segments = batch_segments[split_index:]
        left_result, left_usage = self._translate_batch_with_retry_impl(
            batch_segments=left_segments,
            batch_index=batch_index,
            record_failure=False,
        )
        self._check_cancelled()
        right_result, right_usage = self._translate_batch_with_retry_impl(
            batch_segments=right_segments,
            batch_index=batch_index,
            record_failure=False,
        )
        combined_usage = Usage(
            input_tokens=left_usage.input_tokens,
            output_tokens=left_usage.output_tokens,
            total_tokens=left_usage.total_tokens,
        )
        combined_usage.add(right_usage)
        return (
            BatchTranslationResult(
                lines=left_result.lines + right_result.lines,
                summary=self._combine_batch_summaries(
                    left_result.summary,
                    right_result.summary,
                ),
                glossary_updates=self._combine_glossary_updates(
                    left_result.glossary_updates,
                    right_result.glossary_updates,
                ),
            ),
            combined_usage,
        )

    def _planned_split_sizes(self, batch_segments: list[SubtitleSegment]) -> list[int]:
        split_index = self._translation_split_index(batch_segments)
        return [split_index, len(batch_segments) - split_index]

    def _translation_split_index(self, batch_segments: list[SubtitleSegment]) -> int:
        midpoint = len(batch_segments) // 2
        candidates = [
            index
            for index in range(1, len(batch_segments))
            if self._is_semantic_boundary(batch_segments[index - 1], batch_segments[index])
        ]
        if candidates:
            return min(candidates, key=lambda index: abs(index - midpoint))
        return midpoint

    def _combine_batch_summaries(self, *summaries: str) -> str:
        parts: list[str] = []
        for summary in summaries:
            clean = summary.strip()
            if clean and clean not in parts:
                parts.append(clean)
        return " | ".join(parts[: self.memory.max_summaries])

    def _combine_glossary_updates(self, *groups: list[GlossaryEntry]) -> list[GlossaryEntry]:
        merged: dict[str, str] = {}
        for group in groups:
            for entry in group:
                merged[entry.source] = entry.target
        return [
            GlossaryEntry(source=source, target=target)
            for source, target in merged.items()
        ]

    def _build_translation_failure_message(
        self,
        *,
        batch_index: int,
        attempts: int,
        failure_path: Path,
        attempt_logs: list[dict],
        split_fallback: bool,
        agent_outcome: AgentRepairOutcome | None = None,
    ) -> str:
        headline = f"Translation batch {batch_index} failed after {attempts} attempt"
        if attempts != 1:
            headline += "s"
        if split_fallback:
            headline += " and automatic split retry"

        diagnosis, detail, suggestion = self._diagnose_translation_failure(attempt_logs)
        details: list[str] = []
        if diagnosis:
            details.append(diagnosis)
        if detail:
            details.append(f"Last error: {detail}")
        if suggestion:
            details.append(suggestion)
        if agent_outcome is not None:
            details.append(self._format_agent_failure_detail(agent_outcome))
        return self._format_failure_message(
            headline=headline,
            details=details,
            failure_path=failure_path,
        )

    def _diagnose_translation_failure(self, attempt_logs: list[dict]) -> tuple[str | None, str | None, str | None]:
        error_messages = self._collect_attempt_errors(attempt_logs)
        for error_message in reversed(error_messages):
            if "Line count mismatch:" in error_message:
                return (
                    "Model output is missing subtitle entries or merged neighboring lines.",
                    error_message,
                    self._smaller_batch_retry_suggestion(),
                )
            if "Empty translation for subtitle id" in error_message:
                return (
                    "Model output contains one or more empty subtitle translations.",
                    error_message,
                    self._smaller_batch_retry_suggestion(),
                )
            if "ID mismatch:" in error_message:
                return (
                    "Model output changed subtitle ids instead of preserving the original structure.",
                    error_message,
                    self._smaller_batch_retry_suggestion(),
                )

        error_meta = self._collect_attempt_error_meta(attempt_logs)
        for meta in reversed(error_meta):
            status_code = meta.get("status_code")
            if status_code == 429:
                return (
                    "The provider rate-limited this request.",
                    self._backend_error_brief(meta),
                    "Wait a moment and rerun. If the model is also producing unstable structure, a smaller --batch-size may help.",
                )
            if status_code is not None and 500 <= status_code < 600:
                return (
                    "The provider returned a temporary server-side error.",
                    self._backend_error_brief(meta),
                    "This usually succeeds on a later retry. You can rerun the same command, ideally with resume enabled.",
                )
            if meta.get("reason"):
                return (
                    "The request failed before a valid model response was received.",
                    self._backend_error_brief(meta),
                    "Please rerun the command. If this keeps happening, check your network or provider endpoint settings.",
                )

        if error_messages:
            return (None, error_messages[-1], None)
        return (None, None, None)

    def _collect_attempt_errors(self, attempt_logs: list[dict]) -> list[str]:
        collected: list[str] = []
        for attempt_log in attempt_logs:
            error = str(attempt_log.get("error", "")).strip()
            if error:
                collected.append(error)
            split_retry = attempt_log.get("split_retry")
            if isinstance(split_retry, dict):
                split_error = str(split_retry.get("error", "")).strip()
                if split_error:
                    collected.append(split_error)
        return collected

    def _collect_attempt_error_meta(self, attempt_logs: list[dict]) -> list[dict]:
        collected: list[dict] = []
        for attempt_log in attempt_logs:
            error_meta = attempt_log.get("error_meta")
            if isinstance(error_meta, dict):
                collected.append(error_meta)
            split_retry = attempt_log.get("split_retry")
            if isinstance(split_retry, dict):
                split_error_meta = split_retry.get("error_meta")
                if isinstance(split_error_meta, dict):
                    collected.append(split_error_meta)
        return collected

    def _backend_error_brief(self, metadata: dict) -> str | None:
        details: list[str] = []
        if metadata.get("status_code") is not None:
            details.append(f"status={metadata['status_code']}")
        if metadata.get("request_id"):
            details.append(f"request_id={metadata['request_id']}")
        if metadata.get("reason"):
            details.append(f"reason={metadata['reason']}")
        if not details:
            return None
        return ", ".join(details)

    def _smaller_batch_retry_suggestion(self) -> str:
        suggestions = self._suggested_batch_sizes()
        if not suggestions:
            return "This batch is already at the minimum size, so please inspect the saved failure sample or try a different model."
        if len(suggestions) == 1:
            return f"Try rerunning with a smaller --batch-size, for example --batch-size {suggestions[0]}."
        return (
            "Try rerunning with a smaller --batch-size, "
            f"for example --batch-size {suggestions[0]} or --batch-size {suggestions[1]}."
        )

    def _suggested_batch_sizes(self) -> list[int]:
        current = self.options.batch_size
        if current <= 1:
            return []
        suggestions: list[int] = []
        for raw_value in (current // 2, current // 3):
            suggested = self._normalize_batch_size_suggestion(raw_value)
            if 0 < suggested < current and suggested not in suggestions:
                suggestions.append(suggested)
        return suggestions

    def _normalize_batch_size_suggestion(self, value: int) -> int:
        normalized = max(1, value)
        if normalized <= 10:
            return normalized
        return max(5, (normalized // 5) * 5)

    def _format_failure_message(
        self,
        *,
        headline: str,
        details: list[str],
        failure_path: Path,
    ) -> str:
        lines = [self._ensure_sentence(headline)]
        lines.extend(
            self._ensure_sentence(detail)
            for detail in details
            if detail.strip()
        )
        lines.append("Failure sample saved to:")
        lines.append(str(failure_path))
        return "\n".join(lines)

    def _ensure_sentence(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned
        if cleaned.endswith((".", "!", "?", "。", "！", "？")):
            return cleaned
        return f"{cleaned}."

    def _review_batch_with_retry(
        self,
        source_segments: list[SubtitleSegment],
        translated_segments: list[SubtitleSegment],
        batch_index: int,
        reasons: list[str],
    ) -> tuple[ReviewResult, Usage]:
        attempts = self.options.retries + 1
        last_error: Exception | None = None
        attempt_logs: list[dict] = []
        last_request_hash = ""
        for attempt in range(1, attempts + 1):
            payload: dict | None = None
            usage = Usage()
            cached = False
            messages: list[dict[str, str]] = []
            try:
                messages = build_review_messages(
                    source_segments=source_segments,
                    translated_segments=translated_segments,
                    memory=self.memory,
                    target_language=self.options.target_language,
                    reasons=reasons,
                )
                if last_error is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous review response failed validation.\n"
                                f"Validation error: {last_error}\n"
                                "Re-send corrected JSON only."
                            ),
                        }
                    )
                request_hash = build_request_hash(
                    provider=self.options.provider,
                    model=self.options.model,
                    stage="review",
                    messages=messages,
                )
                last_request_hash = request_hash
                if self.options.use_cache:
                    cached_entry = self.cache_store.load("review", request_hash)
                    if cached_entry is not None:
                        payload, _ = cached_entry
                        usage = Usage()
                        cached = True
                        self.cache_hits += 1
                if payload is None:
                    payload, usage = self._generate_json(messages)
                lines = parse_translation_lines(payload.get("lines", []))
                validate_translation_batch(source_segments, lines)
                result = ReviewResult(
                    lines=lines,
                    review_notes=str(payload.get("review_notes", "")).strip(),
                )
                if self.options.use_cache and not cached:
                    self.cache_store.save("review", request_hash, payload, usage)
                return result, usage
            except OperationCancelledError:
                raise
            except Exception as exc:
                last_error = exc
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "cached": cached,
                        "error": str(exc),
                        "error_meta": self._error_metadata(exc),
                        "payload": payload,
                        "messages": messages,
                    }
                )
                if attempt == attempts:
                    agent_outcome = self._repair_review_with_agent(
                        source_segments=source_segments,
                        translated_segments=translated_segments,
                        batch_index=batch_index,
                        last_error=exc,
                        attempt_logs=attempt_logs,
                        last_payload=payload,
                    )
                    if agent_outcome is not None and agent_outcome.success:
                        if agent_outcome.review_result is None:
                            raise RuntimeError("Agent repair succeeded without a review result.")
                        return agent_outcome.review_result, agent_outcome.usage
                    failure_path = self.failure_store.write(
                        stage="review",
                        batch_index=batch_index,
                        request_hash=last_request_hash,
                        batch_segments=source_segments,
                        translated_segments=translated_segments,
                        messages=messages,
                        attempts=attempt_logs,
                        agent_attempts=agent_outcome.attempts if agent_outcome is not None else None,
                    )
                    details = [f"Last error: {exc}"]
                    if agent_outcome is not None:
                        details.append(self._format_agent_failure_detail(agent_outcome))
                    raise RuntimeError(
                        self._format_failure_message(
                            headline=f"Final review batch {batch_index} failed after {attempts} attempts",
                            details=details,
                            failure_path=failure_path,
                        )
                    ) from exc
        raise RuntimeError("Review batch retry loop ended unexpectedly.")

    def _repair_translation_with_agent(
        self,
        *,
        batch_segments: list[SubtitleSegment],
        batch_index: int,
        last_error: Exception,
        attempt_logs: list[dict],
        last_payload: object,
    ) -> AgentRepairOutcome | None:
        if not self._should_run_agent_repair(last_error, last_payload):
            return None

        max_attempts = self.options.agent_repair_attempts
        agent_attempt_logs: list[dict[str, Any]] = []
        total_usage = Usage()
        log_path = self.agent_log_store.path_for("translate", batch_index)
        repair_error: Exception = last_error

        for attempt in range(1, max_attempts + 1):
            self._record_agent_repair(
                stage="translate",
                batch_index=batch_index,
                attempt=attempt,
                max_attempts=max_attempts,
                status="running",
                error=str(repair_error),
                log_path=log_path,
            )
            payload: object = None
            usage = Usage()
            cached = False
            messages = build_agent_repair_messages(
                stage="translate",
                source_segments=batch_segments,
                target_language=self.options.target_language,
                last_error=str(repair_error),
                attempt_logs=attempt_logs,
                agent_attempt_logs=agent_attempt_logs,
            )
            try:
                request_hash = build_request_hash(
                    provider=self.options.provider,
                    model=self.options.model,
                    stage="agent_translate_repair",
                    messages=messages,
                )
                if self.options.use_cache:
                    cached_entry = self.cache_store.load("agent_translate_repair", request_hash)
                    if cached_entry is not None:
                        payload, _ = cached_entry
                        cached = True
                        self.cache_hits += 1
                if payload is None:
                    payload, usage = self._generate_json(messages)
                    total_usage.add(usage)
                lines = parse_translation_lines(payload.get("lines", []))  # type: ignore[union-attr]
                validate_translation_batch(batch_segments, lines)
                glossary_updates = parse_glossary_entries(payload.get("glossary_updates", []))  # type: ignore[union-attr]
                result = BatchTranslationResult(
                    lines=lines,
                    summary=str(payload.get("summary", "")).strip(),  # type: ignore[union-attr]
                    glossary_updates=glossary_updates,
                )
                if self.options.use_cache and not cached:
                    self.cache_store.save("agent_translate_repair", request_hash, payload, usage)  # type: ignore[arg-type]
                agent_attempt_logs.append(
                    {
                        "attempt": attempt,
                        "cached": cached,
                        "error": None,
                        "error_meta": None,
                        "payload": payload,
                        "messages": messages,
                    }
                )
                log_path = self.agent_log_store.write(
                    stage="translate",
                    batch_index=batch_index,
                    success=True,
                    attempts=agent_attempt_logs,
                )
                self.agent_repairs.append(
                    AgentRepairRecord(
                        stage="translate",
                        batch_index=batch_index,
                        attempts=attempt,
                        success=True,
                        log_path=log_path,
                    )
                )
                self._record_agent_repair(
                    stage="translate",
                    batch_index=batch_index,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status="repaired",
                    error="",
                    log_path=log_path,
                )
                return AgentRepairOutcome(
                    success=True,
                    usage=total_usage,
                    attempts=agent_attempt_logs,
                    log_path=log_path,
                    translation_result=result,
                )
            except OperationCancelledError:
                raise
            except Exception as exc:
                repair_error = exc
                agent_attempt_logs.append(
                    {
                        "attempt": attempt,
                        "cached": cached,
                        "error": str(exc),
                        "error_meta": self._error_metadata(exc),
                        "payload": payload,
                        "messages": messages,
                    }
                )
                log_path = self.agent_log_store.write(
                    stage="translate",
                    batch_index=batch_index,
                    success=False,
                    attempts=agent_attempt_logs,
                    final_error=str(exc),
                )
                self._record_agent_repair(
                    stage="translate",
                    batch_index=batch_index,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status="failed" if attempt == max_attempts else "retrying",
                    error=str(exc),
                    log_path=log_path,
                )

        self.agent_repairs.append(
            AgentRepairRecord(
                stage="translate",
                batch_index=batch_index,
                attempts=max_attempts,
                success=False,
                log_path=log_path,
                error=str(repair_error),
            )
        )
        return AgentRepairOutcome(
            success=False,
            usage=total_usage,
            attempts=agent_attempt_logs,
            log_path=log_path,
            error=str(repair_error),
        )

    def _repair_review_with_agent(
        self,
        *,
        source_segments: list[SubtitleSegment],
        translated_segments: list[SubtitleSegment],
        batch_index: int,
        last_error: Exception,
        attempt_logs: list[dict],
        last_payload: object,
    ) -> AgentRepairOutcome | None:
        if not self._should_run_agent_repair(last_error, last_payload):
            return None

        max_attempts = self.options.agent_repair_attempts
        agent_attempt_logs: list[dict[str, Any]] = []
        total_usage = Usage()
        log_path = self.agent_log_store.path_for("review", batch_index)
        repair_error: Exception = last_error

        for attempt in range(1, max_attempts + 1):
            self._record_agent_repair(
                stage="review",
                batch_index=batch_index,
                attempt=attempt,
                max_attempts=max_attempts,
                status="running",
                error=str(repair_error),
                log_path=log_path,
            )
            payload: object = None
            usage = Usage()
            cached = False
            messages = build_agent_repair_messages(
                stage="review",
                source_segments=source_segments,
                translated_segments=translated_segments,
                target_language=self.options.target_language,
                last_error=str(repair_error),
                attempt_logs=attempt_logs,
                agent_attempt_logs=agent_attempt_logs,
            )
            try:
                request_hash = build_request_hash(
                    provider=self.options.provider,
                    model=self.options.model,
                    stage="agent_review_repair",
                    messages=messages,
                )
                if self.options.use_cache:
                    cached_entry = self.cache_store.load("agent_review_repair", request_hash)
                    if cached_entry is not None:
                        payload, _ = cached_entry
                        cached = True
                        self.cache_hits += 1
                if payload is None:
                    payload, usage = self._generate_json(messages)
                    total_usage.add(usage)
                lines = parse_translation_lines(payload.get("lines", []))  # type: ignore[union-attr]
                validate_translation_batch(source_segments, lines)
                result = ReviewResult(
                    lines=lines,
                    review_notes=str(payload.get("review_notes", "")).strip(),  # type: ignore[union-attr]
                )
                if self.options.use_cache and not cached:
                    self.cache_store.save("agent_review_repair", request_hash, payload, usage)  # type: ignore[arg-type]
                agent_attempt_logs.append(
                    {
                        "attempt": attempt,
                        "cached": cached,
                        "error": None,
                        "error_meta": None,
                        "payload": payload,
                        "messages": messages,
                    }
                )
                log_path = self.agent_log_store.write(
                    stage="review",
                    batch_index=batch_index,
                    success=True,
                    attempts=agent_attempt_logs,
                )
                self.agent_repairs.append(
                    AgentRepairRecord(
                        stage="review",
                        batch_index=batch_index,
                        attempts=attempt,
                        success=True,
                        log_path=log_path,
                    )
                )
                self._record_agent_repair(
                    stage="review",
                    batch_index=batch_index,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status="repaired",
                    error="",
                    log_path=log_path,
                )
                return AgentRepairOutcome(
                    success=True,
                    usage=total_usage,
                    attempts=agent_attempt_logs,
                    log_path=log_path,
                    review_result=result,
                )
            except OperationCancelledError:
                raise
            except Exception as exc:
                repair_error = exc
                agent_attempt_logs.append(
                    {
                        "attempt": attempt,
                        "cached": cached,
                        "error": str(exc),
                        "error_meta": self._error_metadata(exc),
                        "payload": payload,
                        "messages": messages,
                    }
                )
                log_path = self.agent_log_store.write(
                    stage="review",
                    batch_index=batch_index,
                    success=False,
                    attempts=agent_attempt_logs,
                    final_error=str(exc),
                )
                self._record_agent_repair(
                    stage="review",
                    batch_index=batch_index,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status="failed" if attempt == max_attempts else "retrying",
                    error=str(exc),
                    log_path=log_path,
                )

        self.agent_repairs.append(
            AgentRepairRecord(
                stage="review",
                batch_index=batch_index,
                attempts=max_attempts,
                success=False,
                log_path=log_path,
                error=str(repair_error),
            )
        )
        return AgentRepairOutcome(
            success=False,
            usage=total_usage,
            attempts=agent_attempt_logs,
            log_path=log_path,
            error=str(repair_error),
        )

    def _should_run_agent_repair(self, exc: Exception, payload: object) -> bool:
        if not self.options.agent or self.options.agent_repair_attempts <= 0:
            return False
        if isinstance(exc, BackendRequestError):
            return False
        if isinstance(exc, ValidationError):
            return True
        if isinstance(exc, KeyError):
            return True
        if isinstance(exc, ValueError) and "Failed to parse JSON object" in str(exc):
            return True
        if payload is not None and isinstance(exc, (AttributeError, TypeError, ValueError)):
            return True
        return False

    def _record_agent_repair(
        self,
        *,
        stage: str,
        batch_index: int,
        attempt: int,
        max_attempts: int,
        status: str,
        error: str,
        log_path: Path,
    ) -> None:
        recorder = getattr(self.dashboard, "record_agent_repair", None)
        if not callable(recorder):
            return
        recorder(
            stage=stage,
            batch_index=batch_index,
            attempt=attempt,
            max_attempts=max_attempts,
            status=status,
            error=error,
            log_path=str(log_path),
        )

    def _format_agent_failure_detail(self, outcome: AgentRepairOutcome) -> str:
        return (
            f"Agent repair failed after {len(outcome.attempts)} attempts. "
            f"Agent log saved to: {outcome.log_path}"
        )

    def _build_output_segments(
        self,
        document: SubtitleDocument,
        translated_segments: list[SubtitleSegment],
    ) -> list[SubtitleSegment]:
        if document.format not in {"srt", "vtt"} or not self.options.bilingual:
            return translated_segments

        bilingual_segments: list[SubtitleSegment] = []
        for source, translated in zip(document.segments, translated_segments, strict=True):
            bilingual_segments.append(
                SubtitleSegment(
                    id=translated.id,
                    start=translated.start,
                    end=translated.end,
                    identifier=translated.identifier,
                    settings=translated.settings,
                    text="\n".join(part for part in [source.text, translated.text] if part != ""),
                )
            )
        return bilingual_segments

    def _materialize_translations(
        self,
        source_segments: list[SubtitleSegment],
        lines: list[TranslationLine],
    ) -> list[SubtitleSegment]:
        rendered: list[SubtitleSegment] = []
        for source, line in zip(source_segments, lines, strict=True):
            rendered.append(
                SubtitleSegment(
                    id=source.id,
                    start=source.start,
                    end=source.end,
                    identifier=source.identifier,
                    settings=source.settings,
                    text=line.translation,
                )
            )
        return rendered

    def _lookup_translation_memory(
        self,
        batch_segments: list[SubtitleSegment],
    ) -> dict[str, TranslationLine]:
        if not self.options.use_cache:
            return {}

        matches: dict[str, TranslationLine] = {}
        for segment in batch_segments:
            key = self._translation_memory_key(segment.text)
            if key is None:
                continue
            cached_translation = self.translation_memory.get(key)
            if cached_translation is None:
                continue
            self.translation_memory_hits += 1
            matches[segment.id] = TranslationLine(
                id=segment.id,
                translation=cached_translation,
            )
        return matches

    def _update_translation_memory(
        self,
        source_segments: list[SubtitleSegment],
        translated_segments: list[SubtitleSegment],
    ) -> None:
        if not self.options.use_cache:
            return

        changed = False
        for source, translated in zip(source_segments, translated_segments, strict=True):
            key = self._translation_memory_key(source.text)
            if key is None or not translated.text.strip():
                continue
            if self.translation_memory.get(key) == translated.text:
                continue
            self.translation_memory[key] = translated.text
            changed = True

        if changed:
            self.translation_memory_store.save(self.translation_memory)

    def _translation_memory_key(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        normalized = re.sub(r"\s+", " ", stripped.casefold())
        normalized = re.sub(r"\s+([,.!?;:])", r"\1", normalized)
        return normalized

    def _merge_translation_lines(
        self,
        source_segments: list[SubtitleSegment],
        tm_matches: dict[str, TranslationLine],
        generated_lines: list[TranslationLine],
    ) -> list[TranslationLine]:
        generated_by_id = {line.id: line for line in generated_lines}
        merged: list[TranslationLine] = []
        for segment in source_segments:
            line = tm_matches.get(segment.id) or generated_by_id.get(segment.id)
            if line is None:
                raise KeyError(f"Missing translation for subtitle id {segment.id}.")
            merged.append(line)
        return merged

    def _chunk_segments(self, segments: list[SubtitleSegment]) -> list[list[SubtitleSegment]]:
        if not segments:
            return []
        batches: list[list[SubtitleSegment]] = []
        batch: list[SubtitleSegment] = []
        batch_chars = 0
        batch_tokens = 0

        for index, segment in enumerate(segments):
            segment_chars = len(segment.text.strip())
            segment_tokens = self._estimate_text_tokens(segment.text)
            if batch:
                segment_limit, _, hard_chars, _, hard_tokens = self._effective_batch_limits(batch)
                if (
                    len(batch) >= self.options.batch_size
                    or batch_chars + segment_chars > hard_chars
                    or batch_tokens + segment_tokens > hard_tokens
                    or (
                        len(batch) >= segment_limit
                        and self._is_semantic_boundary(batch[-1], segment)
                    )
                ):
                    batches.append(batch)
                    batch = []
                    batch_chars = 0
                    batch_tokens = 0

            batch.append(segment)
            batch_chars += segment_chars
            batch_tokens += segment_tokens

            next_segment = segments[index + 1] if index + 1 < len(segments) else None
            if next_segment is None:
                continue
            segment_limit, target_chars, hard_chars, target_tokens, hard_tokens = self._effective_batch_limits(batch)
            if len(batch) >= self.options.batch_size:
                batches.append(batch)
                batch = []
                batch_chars = 0
                batch_tokens = 0
                continue
            adaptive_grace = max(2, segment_limit // 4)
            if len(batch) >= segment_limit and (
                self._is_semantic_boundary(segment, next_segment)
                or len(batch) >= min(self.options.batch_size, segment_limit + adaptive_grace)
            ):
                batches.append(batch)
                batch = []
                batch_chars = 0
                batch_tokens = 0
                continue
            if (
                (batch_chars >= target_chars or batch_tokens >= target_tokens)
                and self._is_semantic_boundary(segment, next_segment)
            ) or batch_chars >= hard_chars or batch_tokens >= hard_tokens:
                batches.append(batch)
                batch = []
                batch_chars = 0
                batch_tokens = 0

        if batch:
            batches.append(batch)
        return batches

    def _effective_batch_limits(self, batch: list[SubtitleSegment]) -> tuple[int, int, int, int, int]:
        max_segments = self._adaptive_batch_segment_limit(batch)
        target_chars, hard_chars, target_tokens, hard_tokens = self._smart_batch_limits(max_segments)
        return max_segments, target_chars, hard_chars, target_tokens, hard_tokens

    def _smart_batch_limits(self, max_segments: int | None = None) -> tuple[int, int, int, int]:
        max_segments = max_segments or self.options.batch_size
        if self.options.fast_mode:
            target_chars = max(480, min(2600, max_segments * 48))
            hard_chars = max(target_chars + max(180, target_chars // 2), target_chars)
            target_tokens = max(160, min(960, max_segments * 12))
            hard_tokens = max(target_tokens + max(60, target_tokens // 2), target_tokens)
            return target_chars, hard_chars, target_tokens, hard_tokens
        target_chars = max(320, min(1800, max_segments * 36))
        hard_chars = max(target_chars + max(120, target_chars // 3), target_chars)
        target_tokens = max(120, min(720, max_segments * 10))
        hard_tokens = max(target_tokens + max(40, target_tokens // 3), target_tokens)
        return target_chars, hard_chars, target_tokens, hard_tokens

    def _adaptive_batch_segment_limit(self, batch: list[SubtitleSegment]) -> int:
        base_limit = self.options.batch_size
        if self.options.fast_mode:
            return base_limit
        if base_limit <= 8 or not batch:
            return base_limit

        risk_score = self._source_batch_risk_score(batch)
        if risk_score >= 6:
            return min(base_limit, max(6, base_limit // 4))
        if risk_score >= 3:
            return min(base_limit, max(8, base_limit // 2))
        if risk_score >= 1:
            return min(base_limit, max(10, (base_limit * 2) // 3))
        return base_limit

    def _source_batch_risk_score(self, batch: list[SubtitleSegment]) -> int:
        score = 0
        for segment in batch:
            text = segment.text.strip()
            if not text:
                score += 1
                continue
            if self._is_fragment_line(text):
                score += 1
            if self._has_speaker_marker(text):
                score += 1
            if self._contains_formatting(text):
                score += 1

        for current_segment, next_segment in zip(batch, batch[1:], strict=False):
            if self._is_split_sentence_pair(current_segment.text, next_segment.text):
                score += 2
        return score

    def _is_fragment_line(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if self._starts_like_continuation(stripped):
            return True
        if stripped.endswith((",", "，", ";", "；", ":", "：", "-", "–", "—", "...", "…")):
            return True
        return not self._ends_sentence(stripped) and len(stripped) <= 20

    def _starts_like_continuation(self, text: str) -> bool:
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
                "for ",
                "with ",
                "without ",
                "from ",
                "into ",
                "about ",
                "as ",
            )
        )

    def _is_split_sentence_pair(self, current_text: str, next_text: str) -> bool:
        current = current_text.strip()
        nxt = next_text.strip()
        if not current or not nxt:
            return False
        if self._has_speaker_marker(current) != self._has_speaker_marker(nxt):
            return False
        if self._ends_sentence(current):
            return False
        if self._starts_like_continuation(nxt):
            return True
        if current.endswith((",", "，", ";", "；", ":", "：", "-", "–", "—", "...", "…")):
            return True
        return len(current) <= 28 and len(nxt) <= 28

    def _estimate_text_tokens(self, text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        return max(1, len(stripped) // 4)

    def _is_semantic_boundary(
        self,
        current_segment: SubtitleSegment,
        next_segment: SubtitleSegment,
    ) -> bool:
        current_text = current_segment.text.strip()
        next_text = next_segment.text.strip()
        if not current_text or not next_text:
            return True
        if self._has_speaker_marker(current_text) != self._has_speaker_marker(next_text):
            return True
        return self._ends_sentence(current_text)

    def _ends_sentence(self, text: str) -> bool:
        return re.search(r'[.!?。！？…]["\')\]]*$', text.strip()) is not None

    def _has_speaker_marker(self, text: str) -> bool:
        stripped = text.lstrip()
        if stripped.startswith(("-", "–", "—", ">>")):
            return True
        return re.match(r"^[A-Z][A-Za-z0-9 _-]{0,20}:\s", stripped) is not None

    def _contains_formatting(self, text: str) -> bool:
        return re.search(r"</?[^>]+>|{[^}]+}", text) is not None

    def _has_named_terms(self, text: str) -> bool:
        matches = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        if not matches:
            return False
        leading_match = re.match(r"^\W*([A-Z][a-z]{2,})\b", text.strip())
        if len(matches) == 1 and leading_match and matches[0] == leading_match.group(1):
            return False
        return True

    def _is_dense_segment(self, text: str) -> bool:
        stripped = text.strip()
        return len(stripped) >= 84 or stripped.count("\n") >= 1

    def _is_glossary_term_risky(
        self,
        term: str,
        source_segments: list[SubtitleSegment],
    ) -> bool:
        normalized = term.strip()
        if not normalized:
            return False
        if " " in normalized or "-" in normalized or any(character.isdigit() for character in normalized):
            return True
        if re.search(r"[A-Z].*[A-Z]", normalized[1:]):
            return True

        pattern = re.compile(rf"\b{re.escape(normalized)}\b")
        for segment in source_segments:
            for match in pattern.finditer(segment.text.strip()):
                if match.start() > 0:
                    return True
        return False

    def _build_batch_plan(self, batches: list[list[SubtitleSegment]]) -> list[BatchPlanEntry]:
        plans: list[BatchPlanEntry] = []
        for index, batch in enumerate(batches, start=1):
            if not batch:
                continue
            plans.append(
                BatchPlanEntry(
                    index=index,
                    size=len(batch),
                    first_id=batch[0].id,
                    last_id=batch[-1].id,
                )
            )
        return plans

    def _pair_batches(
        self,
        translation_batches: list[list[SubtitleSegment]],
        translated_segments: list[SubtitleSegment],
    ) -> list[BatchSlices]:
        batches: list[BatchSlices] = []
        offset = 0
        for index, source_batch in enumerate(translation_batches, start=1):
            start_offset = offset
            translated_batch = translated_segments[offset : offset + len(source_batch)]
            offset += len(source_batch)
            batches.append(
                BatchSlices(
                    index=index,
                    start_offset=start_offset,
                    source=source_batch,
                    translated=translated_batch,
                )
            )
        return batches

    def _build_review_plan(
        self,
        translation_batches: list[list[SubtitleSegment]],
        translated_segments: list[SubtitleSegment],
    ) -> list[ReviewBatchPlan]:
        plan: list[ReviewBatchPlan] = []
        for batch in self._pair_batches(translation_batches, translated_segments):
            reasons = self._review_reasons(batch.source, batch.translated)
            if not reasons:
                continue
            plan.append(
                ReviewBatchPlan(
                    index=batch.index,
                    start_offset=batch.start_offset,
                    source=batch.source,
                    translated=batch.translated,
                    reasons=reasons,
                )
            )
        return plan

    def _review_reasons(
        self,
        source_segments: list[SubtitleSegment],
        translated_segments: list[SubtitleSegment],
    ) -> list[str]:
        reasons: list[str] = []
        score = 0
        glossary_hits = {
            source: target
            for source, target in select_relevant_glossary(
                self.memory.glossary,
                [segment.text for segment in source_segments + translated_segments if segment.text],
                limit=8,
            ).items()
            if self._is_glossary_term_risky(source, source_segments)
        }
        if glossary_hits:
            reasons.append("glossary consistency")
            score += 2
        elif any(self._has_named_terms(segment.text) for segment in source_segments):
            reasons.append("names and terms")
            score += 2
        if any(self._has_speaker_marker(segment.text) for segment in source_segments):
            reasons.append("speaker changes")
            score += 2
        if any(self._contains_formatting(segment.text) for segment in source_segments):
            reasons.append("formatting and tags")
            score += 2
        if any(
            self._is_dense_segment(source.text) or self._is_dense_segment(translated.text)
            for source, translated in zip(source_segments, translated_segments, strict=True)
        ):
            reasons.append("readability")
            score += 1
        return reasons if score >= 2 else []

    def _resolve_output_format(self, input_path: Path) -> str:
        supported_suffixes = {".srt", ".vtt", ".txt"}
        configured_format = self.options.output_format
        if configured_format is not None:
            configured_format = configured_format.strip().lower().lstrip(".")
            if f".{configured_format}" not in supported_suffixes:
                raise ValueError("Supported output formats are srt, vtt, and txt.")

        output_suffix_format: str | None = None
        if self.options.output_path is not None:
            output_suffix = self.options.output_path.suffix.lower()
            if output_suffix:
                if output_suffix not in supported_suffixes:
                    raise ValueError(
                        f"Unsupported output file extension: {output_suffix}. "
                        "Use .srt, .vtt, .txt, or pass --output-format."
                    )
                output_suffix_format = output_suffix.lstrip(".")

        if configured_format is not None and output_suffix_format is not None and configured_format != output_suffix_format:
            raise ValueError(
                "Output format conflict: --output-format does not match the suffix in --output."
            )

        effective_output_format = (
            configured_format
            or output_suffix_format
            or input_path.suffix.lower().lstrip(".")
        )

        if input_path.suffix.lower() == ".txt" and effective_output_format in {"srt", "vtt"}:
            raise ValueError(
                "Cannot render .txt input as .srt or .vtt because plain text input has no timing information."
            )
        return effective_output_format

    def _resolve_output_path(self, input_path: Path) -> Path:
        if self.options.output_path is not None:
            return self.options.output_path
        suffix = f".{self.output_format}"
        flavor = "bilingual" if self.options.bilingual else "translated"
        return input_path.with_name(f"{input_path.stem}.{flavor}{suffix}")

    def _validate_input_path(self, input_path: Path) -> None:
        supported = {".srt", ".vtt", ".txt"}
        if input_path.suffix.lower() not in supported:
            raise ValueError("Supported input formats are .srt, .vtt, and .txt.")

    def _load_resume_state(self, total_translation_batches: int) -> ResumeSnapshot:
        if not self.options.resume or self.state_store is None:
            self._load_persistent_glossary()
            return ResumeSnapshot()

        snapshot = self.state_store.load()
        if snapshot is None:
            self._load_persistent_glossary()
            return ResumeSnapshot()

        self.resumed_translation_batches = snapshot.translation_batches_completed
        self.resumed_review_batches = snapshot.review_batches_completed
        self.translation_memory = self.translation_memory_store.load()
        try:
            if snapshot.translation_batches_completed and not snapshot.translated_segments:
                snapshot.translated_segments = self.translation_store.load_segments(
                    snapshot.translation_batches_completed
                )
            if snapshot.review_batches_completed and not snapshot.reviewed_segments:
                snapshot.reviewed_segments = self.review_store.load_segments(
                    snapshot.review_batches_completed
                )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Resume state is missing batch shards. Re-run with --no-resume or clean the run directory."
            ) from exc

        self.memory = snapshot.memory
        self.dashboard.restore_usage(snapshot.usage)
        self._restore_resume_progress(
            resume=snapshot,
            total_translation_batches=total_translation_batches,
            review_batches=0,
        )
        return snapshot

    def _restore_resume_progress(
        self,
        *,
        resume: ResumeSnapshot,
        total_translation_batches: int,
        review_batches: int,
    ) -> None:
        self.dashboard.restore_progress(
            self._restored_completed_steps(
                translation_batches_completed=resume.translation_batches_completed,
                review_batches_completed=min(resume.review_batches_completed, review_batches),
                total_translation_batches=total_translation_batches,
                validation_completed=resume.validation_completed or resume.review_batches_completed > 0,
                review_batches=review_batches,
            )
        )
        self.dashboard.restore_stage_progress(
            translation_batches_completed=resume.translation_batches_completed,
            total_translation_batches=total_translation_batches,
            review_batches_completed=min(resume.review_batches_completed, review_batches),
            review_batches=review_batches,
            validation_completed=resume.validation_completed or resume.review_batches_completed > 0,
        )

    def _has_resume_progress(self, resume: ResumeSnapshot) -> bool:
        return (
            resume.translation_batches_completed > 0
            or resume.review_batches_completed > 0
            or resume.validation_completed
        )

    def _load_persistent_glossary(self) -> None:
        glossary = self.glossary_store.load()
        if glossary:
            self.memory.load_glossary(glossary)
        self.translation_memory = self.translation_memory_store.load()

    def _restored_completed_steps(
        self,
        translation_batches_completed: int,
        review_batches_completed: int,
        total_translation_batches: int,
        validation_completed: bool,
        review_batches: int,
    ) -> int:
        completed_steps = 2 + translation_batches_completed
        if validation_completed and translation_batches_completed >= total_translation_batches:
            completed_steps += 1 + review_batches_completed
        return completed_steps

    def _save_run_state(
        self,
        *,
        translation_batches_completed: int,
        review_batches_completed: int,
        validation_completed: bool,
    ) -> None:
        if self.options.dry_run or self.state_store is None or self.input_signature is None:
            return
        self.state_store.save(
            options=self.options,
            output_path=self.output_path,
            input_signature=self.input_signature,
            usage=self.dashboard.usage,
            memory=self.memory,
            translation_batches_completed=translation_batches_completed,
            review_batches_completed=review_batches_completed,
            validation_completed=validation_completed,
        )

    def _error_metadata(self, exc: Exception) -> dict | None:
        if isinstance(exc, BackendRequestError):
            return exc.metadata.to_dict()
        return None

    def _require_backend(self) -> LLMBackend:
        if self.backend is None:
            raise RuntimeError("No backend configured. Disable --dry-run or provide a backend.")
        return self.backend


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
