from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import re
from time import monotonic

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from subbake.entities import Usage


@dataclass(slots=True)
class BatchSnapshot:
    index: int = 0
    total: int = 0
    latency_seconds: float = 0.0
    stage_label: str = "IDLE"


@dataclass(slots=True)
class AgentRepairSnapshot:
    stage: str
    batch_index: int
    attempt: int
    max_attempts: int
    status: str
    error: str
    log_path: str | None = None


class Dashboard:
    BATCH_STAGES = ("TRANSLATE_BATCH", "FINAL_REVIEW")

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.usage = Usage()
        self.batch = BatchSnapshot()
        self.total_steps = 1
        self.completed_steps = 0
        self.stage_order = [
            "LOAD_FILE",
            "PARSE",
            "TRANSLATE_BATCH",
            "VALIDATE",
            "FINAL_REVIEW",
            "WRITE_OUTPUT",
        ]
        self.stage_states = {stage: "pending" for stage in self.stage_order}
        self.fixed_stage_durations = {
            "LOAD_FILE": [],
            "PARSE": [],
            "VALIDATE": [],
            "WRITE_OUTPUT": [],
        }
        self.fixed_stage_defaults = {
            "LOAD_FILE": 0.1,
            "PARSE": 0.1,
            "VALIDATE": 0.2,
            "WRITE_OUTPUT": 0.1,
        }
        self.batch_stage_durations = {stage: [] for stage in self.BATCH_STAGES}
        self.batch_stage_totals = {stage: 0 for stage in self.BATCH_STAGES}
        self.batch_stage_current = {stage: 0 for stage in self.BATCH_STAGES}
        self.agent_repairs: list[AgentRepairSnapshot] = []
        self.current_stage: str | None = None
        self.current_stage_started_at: float | None = None
        self.eta_display_seconds: int | None = None
        self.eta_last_stage: str | None = None
        self.eta_last_updated_at: float | None = None
        self.spinner_frames = ["·  ", "·· ", "···", " ··", "  ·", " ··"]
        self.live = Live(self, console=self.console, refresh_per_second=8)

    @contextmanager
    def running(self):
        with self.live:
            self.refresh()
            yield self

    def set_total_steps(self, total_steps: int) -> None:
        self.total_steps = max(1, total_steps)
        translate_total = self.batch_stage_totals["TRANSLATE_BATCH"]
        if translate_total:
            inferred_review_total = max(0, self.total_steps - translate_total - 4)
            self.batch_stage_totals["FINAL_REVIEW"] = inferred_review_total
        self.refresh()

    def mark_running(self, stage: str, label: str | None = None) -> None:
        for key, value in list(self.stage_states.items()):
            if value == "running":
                self.stage_states[key] = "pending"
        self.stage_states[stage] = "running"
        self.current_stage = stage
        self.current_stage_started_at = monotonic()
        if label:
            self.batch.stage_label = label
            parsed = self._parse_batch_label(label)
            if parsed is not None:
                index, total = parsed
                self.batch = BatchSnapshot(
                    index=index,
                    total=total,
                    latency_seconds=0.0,
                    stage_label=label,
                )
                self.batch_stage_current[stage] = index
                self.batch_stage_totals[stage] = total
        self.refresh()

    def mark_done(self, stage: str, advance: bool = True) -> None:
        if stage in self.fixed_stage_durations and self.current_stage == stage and self.current_stage_started_at is not None:
            duration = max(0.0, monotonic() - self.current_stage_started_at)
            self.fixed_stage_durations[stage].append(duration)
        self.stage_states[stage] = "done"
        if advance:
            self.completed_steps += 1
        if self.current_stage == stage:
            self.current_stage = None
            self.current_stage_started_at = None
        self.refresh()

    def mark_skipped(self, stage: str, advance: bool = True) -> None:
        if self.current_stage == stage:
            self.current_stage = None
            self.current_stage_started_at = None
        self.stage_states[stage] = "skipped"
        if advance:
            self.completed_steps += 1
        self.refresh()

    def add_usage(self, usage: Usage) -> None:
        self.usage.add(usage)
        self.refresh()

    def restore_usage(self, usage: Usage) -> None:
        self.usage = Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )
        self.refresh()

    def restore_progress(self, completed_steps: int) -> None:
        self.completed_steps = max(0, completed_steps)
        self.refresh()

    def restore_stage_progress(
        self,
        *,
        translation_batches_completed: int,
        total_translation_batches: int,
        review_batches_completed: int,
        review_batches: int,
        validation_completed: bool,
    ) -> None:
        if total_translation_batches > 0:
            self.batch_stage_totals["TRANSLATE_BATCH"] = total_translation_batches
            self.batch_stage_current["TRANSLATE_BATCH"] = min(
                translation_batches_completed,
                total_translation_batches,
            )
            if translation_batches_completed >= total_translation_batches:
                self.stage_states["TRANSLATE_BATCH"] = "done"

        if validation_completed and translation_batches_completed >= total_translation_batches:
            self.stage_states["VALIDATE"] = "done"

        if review_batches > 0:
            self.batch_stage_totals["FINAL_REVIEW"] = review_batches
            self.batch_stage_current["FINAL_REVIEW"] = min(
                review_batches_completed,
                review_batches,
            )
            if review_batches_completed >= review_batches:
                self.stage_states["FINAL_REVIEW"] = "done"

        self.refresh()

    def set_batch(self, index: int, total: int, latency_seconds: float, stage_label: str) -> None:
        self.batch = BatchSnapshot(
            index=index,
            total=total,
            latency_seconds=latency_seconds,
            stage_label=stage_label,
        )
        current_stage = self.current_stage or self._stage_from_label(stage_label)
        if current_stage in self.batch_stage_durations:
            self.batch_stage_current[current_stage] = index
            self.batch_stage_totals[current_stage] = total
            self.batch_stage_durations[current_stage].append(latency_seconds)
        self.refresh()

    def clear_batch(self) -> None:
        self.batch = BatchSnapshot()
        self._reset_eta_state()
        self.refresh()

    def record_agent_repair(
        self,
        *,
        stage: str,
        batch_index: int,
        attempt: int,
        max_attempts: int,
        status: str,
        error: str,
        log_path: str | None = None,
    ) -> None:
        self.agent_repairs.append(
            AgentRepairSnapshot(
                stage=stage,
                batch_index=batch_index,
                attempt=attempt,
                max_attempts=max_attempts,
                status=status,
                error=error,
                log_path=log_path,
            )
        )
        self.agent_repairs = self.agent_repairs[-6:]
        self.refresh()

    def refresh(self) -> None:
        self.live.refresh()

    def __rich__(self) -> Panel:
        return self.render()

    def render(self) -> Panel:
        timeline_rows: list[Text] = []
        for stage in self.stage_order[:-1]:
            state = self.stage_states[stage]
            label = self._timeline_stage_label(stage)
            style, icon = self._timeline_indicator(state)
            row = Text()
            row.append("[", style=style)
            row.append(icon, style=style)
            row.append("] ", style=style)
            row.append(label)
            timeline_rows.append(row)

        stats = Table.grid(padding=(0, 2))
        stats.add_column(justify="left")
        stats.add_column(justify="right")
        stats.add_row("Progress", self._progress_bar())
        stats.add_row("ETA", self._eta_display())
        stats.add_row("Input tokens", f"{self.usage.input_tokens:,}")
        stats.add_row("Output tokens", f"{self.usage.output_tokens:,}")
        stats.add_row("Total tokens", f"{self.usage.total_tokens:,}")

        batch_table = Table.grid(padding=(0, 2))
        batch_table.add_column(justify="left")
        batch_table.add_column(justify="right")
        batch_label = (
            f"{self.batch.index}/{self.batch.total}" if self.batch.total else "-"
        )
        batch_table.add_row("Current batch", batch_label)
        batch_table.add_row("Latency", self._batch_latency_display())

        sections: list[object] = [
            Text("subbake", style="bold cyan"),
            Text(""),
            Text("Timeline", style="bold"),
            *timeline_rows,
            Text("Usage", style="bold"),
            stats,
            Text("Current batch", style="bold"),
            batch_table,
        ]
        agent_panel = self._agent_repair_panel()
        if agent_panel is not None:
            sections.extend(
                [
                    Text("Agent repair", style="bold"),
                    agent_panel,
                ]
            )

        group = Group(*sections)
        return Panel(group, border_style="cyan", title="Subtitle Translation")

    def _agent_repair_panel(self) -> Table | None:
        if not self.agent_repairs:
            return None
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_row("Batch", "Attempt", "Status", "Error", "Log")
        for repair in self.agent_repairs:
            table.add_row(
                f"{repair.stage} {repair.batch_index}",
                f"{repair.attempt}/{repair.max_attempts}",
                repair.status,
                self._truncate_display(repair.error, 48),
                repair.log_path or "-",
            )
        return table

    def _timeline_stage_label(self, stage: str) -> str:
        if self.stage_states.get(stage) == "skipped":
            return f"{stage} SKIPPED"
        if stage in self.BATCH_STAGES:
            total = self.batch_stage_totals.get(stage, 0)
            if total:
                current = self.batch_stage_current.get(stage, 0)
                return f"{stage} {current}/{total}"
        return stage

    def _progress_bar(self, width: int = 20) -> str:
        ratio = min(1.0, self.completed_steps / self.total_steps)
        filled = int(ratio * width)
        return f"[{'█' * filled}{'-' * (width - filled)}] {ratio * 100:>5.1f}%"

    def _batch_latency_display(self) -> str:
        if not self.batch.total:
            return "-"
        return f"{self._current_batch_elapsed_seconds():.2f}s"

    def _current_batch_elapsed_seconds(self) -> float:
        if self.current_stage in self.BATCH_STAGES and self.current_stage_started_at is not None:
            return max(0.0, monotonic() - self.current_stage_started_at)
        return self.batch.latency_seconds

    def _eta_display(self) -> str:
        if self.current_stage not in self.BATCH_STAGES:
            self._reset_eta_state()
            return "-"
        if not self._has_eta_confidence(self.current_stage):
            return "-"
        seconds = self._estimated_remaining_seconds()
        if seconds is None:
            return "-"
        return self._format_duration(self._smoothed_eta_seconds(seconds))

    def _has_eta_confidence(self, stage: str) -> bool:
        total_batches = self.batch_stage_totals.get(stage, 0)
        if total_batches <= 1:
            return False
        completed_samples = len(self.batch_stage_durations.get(stage, []))
        required_samples = 2 if total_batches >= 4 else 1
        return completed_samples >= required_samples

    def _smoothed_eta_seconds(self, raw_seconds: float) -> int:
        now = monotonic()
        stage = self.current_stage
        quantized = self._quantize_eta_seconds(raw_seconds)
        displayed = self._countdown_eta_seconds(now)
        if (
            self.eta_display_seconds is None
            or self.eta_last_stage != stage
            or self.eta_last_updated_at is None
            or displayed is None
        ):
            self._set_eta_anchor(stage=stage, seconds=quantized, now=now)
            return quantized

        current = displayed
        if quantized == current:
            return current

        if abs(quantized - current) < self._eta_recalibration_threshold_seconds(current):
            return current

        interval = self._eta_update_interval_seconds(current=current, target=quantized)
        if now - self.eta_last_updated_at < interval:
            return current

        recalibrated = self._recalibrated_eta_seconds(current=current, target=quantized)
        self._set_eta_anchor(stage=stage, seconds=recalibrated, now=now)
        return recalibrated

    def _countdown_eta_seconds(self, now: float) -> int | None:
        if self.eta_display_seconds is None or self.eta_last_updated_at is None:
            return None
        elapsed_seconds = max(0, int(now - self.eta_last_updated_at))
        return max(0, self.eta_display_seconds - elapsed_seconds)

    def _set_eta_anchor(self, *, stage: str | None, seconds: int, now: float) -> None:
        self.eta_display_seconds = seconds
        self.eta_last_stage = stage
        self.eta_last_updated_at = now

    def _recalibrated_eta_seconds(self, *, current: int, target: int) -> int:
        if target <= current:
            return target
        step = self._eta_recalibration_step_seconds(max(current, target))
        return min(target, current + step)

    def _quantize_eta_seconds(self, seconds: float) -> int:
        bounded = max(1.0, seconds)
        step = self._eta_step_seconds(bounded)
        return int(math.ceil(bounded / step) * step)

    def _eta_step_seconds(self, seconds: float) -> int:
        if seconds < 45:
            return 1
        if seconds < 60:
            return 2
        if seconds < 180:
            return 5
        if seconds < 600:
            return 10
        if seconds < 1800:
            return 15
        if seconds < 3600:
            return 30
        return 60

    def _eta_update_interval_seconds(self, *, current: int, target: int) -> float:
        remaining = max(1, min(current, target))
        increasing = target > current
        if remaining <= 30:
            return 1.0 if increasing else 0.5
        if remaining <= 60:
            return 2.0 if increasing else 1.0
        if remaining <= 180:
            return 3.0 if increasing else 2.0
        if remaining <= 600:
            return 5.0 if increasing else 3.0
        return 8.0 if increasing else 4.0

    def _eta_recalibration_threshold_seconds(self, current: int) -> int:
        if current <= 30:
            return 1
        if current <= 60:
            return 2
        if current <= 180:
            return 5
        return max(5, self._eta_step_seconds(current))

    def _eta_recalibration_step_seconds(self, seconds: int) -> int:
        if seconds <= 30:
            return 3
        if seconds <= 60:
            return 5
        if seconds <= 180:
            return 10
        return max(10, self._eta_step_seconds(seconds))

    def _reset_eta_state(self) -> None:
        self.eta_display_seconds = None
        self.eta_last_stage = None
        self.eta_last_updated_at = None

    def _estimated_remaining_seconds(self) -> float | None:
        remaining_seconds = 0.0
        has_estimate = False
        running_stage = self.current_stage
        running_elapsed = 0.0
        if self.current_stage_started_at is not None:
            running_elapsed = max(0.0, monotonic() - self.current_stage_started_at)

        for stage in self.fixed_stage_durations:
            state = self.stage_states[stage]
            if state == "done":
                continue
            estimate = self._estimate_fixed_stage_seconds(
                stage,
                running_elapsed if running_stage == stage else None,
            )
            if estimate is None:
                continue
            has_estimate = True
            if running_stage == stage:
                remaining_seconds += max(estimate - running_elapsed, 0.0)
            else:
                remaining_seconds += estimate

        for stage in self.BATCH_STAGES:
            total_batches = self.batch_stage_totals[stage]
            if total_batches <= 0:
                continue
            state = self.stage_states[stage]
            if state == "done":
                continue
            average_seconds = self._estimate_batch_average_seconds(
                stage,
                running_elapsed if running_stage == stage else None,
            )
            if average_seconds is None:
                continue
            has_estimate = True
            if running_stage == stage:
                current_index = max(1, self.batch_stage_current[stage])
                batches_remaining = total_batches - current_index + 1
                remaining_seconds += max((average_seconds * batches_remaining) - running_elapsed, 0.0)
            else:
                remaining_seconds += average_seconds * total_batches

        if not has_estimate:
            return None
        return max(0.0, remaining_seconds)

    def _estimate_fixed_stage_seconds(
        self,
        stage: str,
        running_elapsed: float | None,
    ) -> float | None:
        durations = self.fixed_stage_durations[stage]
        if durations:
            return sum(durations) / len(durations)
        if running_elapsed is not None:
            return max(self.fixed_stage_defaults.get(stage, 0.1), running_elapsed)
        return self.fixed_stage_defaults.get(stage)

    def _estimate_batch_average_seconds(
        self,
        stage: str,
        running_elapsed: float | None,
    ) -> float | None:
        samples = list(self.batch_stage_durations[stage])
        if running_elapsed is not None:
            samples.append(running_elapsed)
        if samples:
            return sum(samples) / len(samples)

        for other_stage in self.BATCH_STAGES:
            other_samples = self.batch_stage_durations[other_stage]
            if other_samples:
                return sum(other_samples) / len(other_samples)
        return None

    def _parse_batch_label(self, label: str) -> tuple[int, int] | None:
        match = re.match(r"^(?:TRANSLATE_BATCH|FINAL_REVIEW) (\d+)/(\d+)$", label)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    def _stage_from_label(self, label: str) -> str | None:
        if label.startswith("TRANSLATE_BATCH"):
            return "TRANSLATE_BATCH"
        if label.startswith("FINAL_REVIEW"):
            return "FINAL_REVIEW"
        return None

    def _truncate_display(self, value: str, limit: int) -> str:
        cleaned = value.strip().replace("\n", " ")
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[:limit]}..."

    def _format_duration(self, seconds: float) -> str:
        rounded = max(0, int(round(seconds)))
        minutes, secs = divmod(rounded, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def _timeline_indicator(self, state: str) -> tuple[str, str]:
        if state == "done":
            return "green", " ✓ "
        if state == "skipped":
            return "bright_black", " - "
        if state == "running":
            frame_index = int(monotonic() * 8) % len(self.spinner_frames)
            return "yellow", self.spinner_frames[frame_index]
        return "white", "   "
