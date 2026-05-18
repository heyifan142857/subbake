from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BATCH_SIZE = 30


@dataclass(slots=True)
class SubtitleSegment:
    id: str
    text: str
    start: str | None = None
    end: str | None = None
    identifier: str | None = None
    settings: str | None = None


@dataclass(slots=True)
class PassthroughBlock:
    insert_before: int
    content: str


@dataclass(slots=True)
class SubtitleDocument:
    path: Path
    format: str
    segments: list[SubtitleSegment]
    header: str | None = None
    passthrough_blocks: list[PassthroughBlock] = field(default_factory=list)


@dataclass(slots=True)
class GlossaryEntry:
    source: str
    target: str


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens


@dataclass(slots=True)
class TranslationLine:
    id: str
    translation: str


@dataclass(slots=True)
class BatchTranslationResult:
    lines: list[TranslationLine]
    summary: str = ""
    glossary_updates: list[GlossaryEntry] = field(default_factory=list)


@dataclass(slots=True)
class ReviewResult:
    lines: list[TranslationLine]
    review_notes: str = ""


@dataclass(slots=True)
class BatchPlanEntry:
    index: int
    size: int
    first_id: str
    last_id: str


@dataclass(slots=True)
class AgentRepairRecord:
    stage: str
    batch_index: int
    attempts: int
    success: bool
    log_path: Path
    error: str = ""


@dataclass(slots=True)
class PipelineOptions:
    input_path: Path
    output_path: Path | None = None
    output_format: str | None = None
    provider: str = "mock"
    model: str = "mock-zh"
    batch_size: int = DEFAULT_BATCH_SIZE
    fast_mode: bool = False
    bilingual: bool = False
    target_language: str = "Chinese"
    source_language: str = "Auto"
    retries: int = 2
    final_review: bool = True
    timeout_seconds: float = 120.0
    api_key: str | None = None
    base_url: str | None = None
    dry_run: bool = False
    resume: bool = True
    use_cache: bool = True
    agent: bool = True
    agent_repair_attempts: int = 2
    work_dir: Path | None = None
    glossary_path: Path | None = None


@dataclass(slots=True)
class PipelineResult:
    output_path: Path | None
    batches_translated: int
    review_batches: int
    usage: Usage
    dry_run: bool = False
    planned_batches: list[BatchPlanEntry] = field(default_factory=list)
    cache_hits: int = 0
    resumed_translation_batches: int = 0
    resumed_review_batches: int = 0
    translation_memory_hits: int = 0
    state_path: Path | None = None
    glossary_path: Path | None = None
    agent_repairs: list[AgentRepairRecord] = field(default_factory=list)
