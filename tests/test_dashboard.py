from __future__ import annotations

import unittest
from unittest.mock import patch

from rich.console import Console

from subbake.ui.dashboard import Dashboard


class DashboardTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dashboard = Dashboard()
        self.dashboard.live.refresh = lambda: None

    def test_running_batch_shows_current_batch_and_elapsed_latency(self) -> None:
        self.dashboard.stage_states["LOAD_FILE"] = "done"
        self.dashboard.stage_states["PARSE"] = "done"
        self.dashboard.set_total_steps(8)

        with patch("subbake.ui.dashboard.monotonic", return_value=10.0):
            self.dashboard.mark_running("TRANSLATE_BATCH", label="TRANSLATE_BATCH 1/4")

        self.assertEqual(self.dashboard.batch.index, 1)
        self.assertEqual(self.dashboard.batch.total, 4)

        with patch("subbake.ui.dashboard.monotonic", return_value=13.25):
            self.assertEqual(self.dashboard._batch_latency_display(), "3.25s")
            self.assertEqual(self.dashboard._eta_display(), "-")

    def test_eta_shows_after_multiple_completed_batches_and_counts_down_each_second(self) -> None:
        with patch("subbake.ui.dashboard.monotonic", side_effect=[0.0, 0.1]):
            self.dashboard.mark_running("LOAD_FILE")
            self.dashboard.mark_done("LOAD_FILE")
        with patch("subbake.ui.dashboard.monotonic", side_effect=[0.1, 0.2]):
            self.dashboard.mark_running("PARSE")
            self.dashboard.mark_done("PARSE")

        self.dashboard.set_total_steps(8)

        with patch("subbake.ui.dashboard.monotonic", return_value=1.0):
            self.dashboard.mark_running("TRANSLATE_BATCH", label="TRANSLATE_BATCH 1/4")
        self.dashboard.set_batch(1, 4, 2.0, "TRANSLATE_BATCH 1/4")
        with patch("subbake.ui.dashboard.monotonic", return_value=4.0):
            self.dashboard.mark_done("TRANSLATE_BATCH")
        with patch("subbake.ui.dashboard.monotonic", return_value=4.0):
            self.dashboard.mark_running("TRANSLATE_BATCH", label="TRANSLATE_BATCH 2/4")
        self.dashboard.set_batch(2, 4, 2.5, "TRANSLATE_BATCH 2/4")
        with patch("subbake.ui.dashboard.monotonic", return_value=6.5):
            self.dashboard.mark_done("TRANSLATE_BATCH")
        with patch("subbake.ui.dashboard.monotonic", return_value=6.5):
            self.dashboard.mark_running("TRANSLATE_BATCH", label="TRANSLATE_BATCH 3/4")

        with patch("subbake.ui.dashboard.monotonic", return_value=7.0):
            first_eta = self.dashboard._eta_display()
        with patch("subbake.ui.dashboard.monotonic", return_value=8.0):
            second_eta = self.dashboard._eta_display()

        self.assertNotEqual(first_eta, "-")
        self.assertEqual(self._duration_to_seconds(first_eta) - 1, self._duration_to_seconds(second_eta))

    def test_eta_recalibrates_faster_when_near_completion(self) -> None:
        self.dashboard.batch_stage_totals["TRANSLATE_BATCH"] = 4
        self.dashboard.batch_stage_durations["TRANSLATE_BATCH"] = [10.0, 10.0]
        self.dashboard.batch_stage_current["TRANSLATE_BATCH"] = 3
        self.dashboard.current_stage = "TRANSLATE_BATCH"
        self.dashboard.current_stage_started_at = 100.0

        with patch("subbake.ui.dashboard.monotonic", return_value=102.0):
            first_eta = self.dashboard._eta_display()

        self.dashboard.batch_stage_durations["TRANSLATE_BATCH"].append(40.0)
        with patch("subbake.ui.dashboard.monotonic", return_value=104.0):
            second_eta = self.dashboard._eta_display()

        self.assertNotEqual(first_eta, "-")
        self.assertGreater(self._duration_to_seconds(second_eta), self._duration_to_seconds(first_eta) - 2)

    def test_timeline_keeps_translate_and_review_progress_separate(self) -> None:
        console = Console(record=True, width=120)
        dashboard = Dashboard(console=console)
        dashboard.live.refresh = lambda: None

        dashboard.stage_states["LOAD_FILE"] = "done"
        dashboard.stage_states["PARSE"] = "done"
        dashboard.stage_states["TRANSLATE_BATCH"] = "done"
        dashboard.stage_states["VALIDATE"] = "done"
        dashboard.stage_states["FINAL_REVIEW"] = "running"
        dashboard.batch_stage_current["TRANSLATE_BATCH"] = 36
        dashboard.batch_stage_totals["TRANSLATE_BATCH"] = 87
        dashboard.batch_stage_current["FINAL_REVIEW"] = 12
        dashboard.batch_stage_totals["FINAL_REVIEW"] = 24

        with patch("subbake.ui.dashboard.monotonic", return_value=10.0):
            console.print(dashboard.render())

        rendered = console.export_text()
        self.assertIn("TRANSLATE_BATCH 36/87", rendered)
        self.assertIn("FINAL_REVIEW 12/24", rendered)

    def test_timeline_shows_skipped_stage_label(self) -> None:
        console = Console(record=True, width=120)
        dashboard = Dashboard(console=console)
        dashboard.live.refresh = lambda: None

        dashboard.stage_states["LOAD_FILE"] = "done"
        dashboard.stage_states["PARSE"] = "done"
        dashboard.stage_states["TRANSLATE_BATCH"] = "done"
        dashboard.stage_states["VALIDATE"] = "done"
        dashboard.mark_skipped("FINAL_REVIEW")

        with patch("subbake.ui.dashboard.monotonic", return_value=10.0):
            console.print(dashboard.render())

        rendered = console.export_text()
        self.assertIn("FINAL_REVIEW SKIPPED", rendered)

    def test_skipped_stage_advances_progress(self) -> None:
        self.dashboard.set_total_steps(6)
        self.dashboard.restore_progress(4)

        self.dashboard.mark_skipped("FINAL_REVIEW")

        self.assertEqual(self.dashboard.completed_steps, 5)
        self.assertIn(" 83.3%", self.dashboard._progress_bar())

    def test_resume_restore_marks_fully_reused_stages_done(self) -> None:
        console = Console(record=True, width=120)
        dashboard = Dashboard(console=console)
        dashboard.live.refresh = lambda: None

        dashboard.stage_states["LOAD_FILE"] = "done"
        dashboard.stage_states["PARSE"] = "done"
        dashboard.restore_stage_progress(
            translation_batches_completed=36,
            total_translation_batches=36,
            review_batches_completed=12,
            review_batches=12,
            validation_completed=True,
        )

        with patch("subbake.ui.dashboard.monotonic", return_value=10.0):
            console.print(dashboard.render())

        rendered = console.export_text()
        self.assertIn("[ ✓ ] TRANSLATE_BATCH 36/36", rendered)
        self.assertIn("[ ✓ ] VALIDATE", rendered)
        self.assertIn("[ ✓ ] FINAL_REVIEW 12/12", rendered)

    def test_agent_repair_panel_shows_attempt_status_and_log_path(self) -> None:
        console = Console(record=True, width=160)
        dashboard = Dashboard(console=console)
        dashboard.live.refresh = lambda: None

        dashboard.record_agent_repair(
            stage="translate",
            batch_index=2,
            attempt=1,
            max_attempts=2,
            status="running",
            error="Line count mismatch: expected 1 lines, received 0.",
            log_path="/tmp/agent_logs/translate_batch_0002.json",
        )

        console.print(dashboard.render())
        rendered = console.export_text()

        self.assertIn("Agent repair", rendered)
        self.assertIn("translate 2", rendered)
        self.assertIn("1/2", rendered)
        self.assertIn("running", rendered)
        self.assertIn("translate_batch_0002.json", rendered)

    def _duration_to_seconds(self, value: str) -> int:
        if value.endswith("s") and "m" not in value and "h" not in value:
            return int(value[:-1])
        minutes = 0
        seconds = 0
        hours = 0
        for part in value.split():
            if part.endswith("h"):
                hours = int(part[:-1])
            elif part.endswith("m"):
                minutes = int(part[:-1])
            elif part.endswith("s"):
                seconds = int(part[:-1])
        return (hours * 3600) + (minutes * 60) + seconds
