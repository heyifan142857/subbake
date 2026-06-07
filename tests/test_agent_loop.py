from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from pathlib import Path

from subbake.agent.loop import rank_file_candidates


class AgentLoopScoringTestCase(unittest.TestCase):
    @contextlib.contextmanager
    def _isolated_filesystem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                yield Path(tmpdir)
            finally:
                os.chdir(old_cwd)

    def test_generated_subtitle_candidate_maps_back_to_source(self) -> None:
        with self._isolated_filesystem() as root:
            source = root / "episode.srt"
            translated = root / "episode.translated.srt"
            source.write_text("hello\n", encoding="utf-8")
            translated.write_text("[MOCK-ZH] hello\n", encoding="utf-8")

            candidates = rank_file_candidates([translated, source], "episode", project_root=root)

            translated_candidate = next(candidate for candidate in candidates if candidate.kind == "translated")
            self.assertEqual(translated_candidate.inferred_source_path, "episode.srt")

    def test_media_file_scores_below_subtitle_source(self) -> None:
        with self._isolated_filesystem() as root:
            subtitle = root / "The Matrix.srt"
            media = root / "The Matrix.mkv"
            subtitle.write_text("hello\n", encoding="utf-8")
            media.write_text("video placeholder\n", encoding="utf-8")

            candidates = rank_file_candidates([media, subtitle], "The Matrix", project_root=root)

            self.assertEqual(candidates[0].path, "The Matrix.srt")
            self.assertGreater(candidates[0].score, candidates[1].score)

    def test_source_subtitle_beats_generated_output_as_execution_input(self) -> None:
        with self._isolated_filesystem() as root:
            source = root / "The Matrix.srt"
            bilingual = root / "The Matrix.bilingual.srt"
            source.write_text("hello\n", encoding="utf-8")
            bilingual.write_text("hello\n[MOCK-ZH] hello\n", encoding="utf-8")

            candidates = rank_file_candidates([bilingual, source], "黑客帝国", project_root=root)

            self.assertEqual(candidates[0].path, "The Matrix.srt")
            self.assertEqual(candidates[0].kind, "source")


if __name__ == "__main__":
    unittest.main()
