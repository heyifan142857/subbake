from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from pathlib import Path

from subbake.agent.loop import rank_file_candidates
from subbake.agent.target import infer_target_from_user_language


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


class InferTargetFromUserLanguageTestCase(unittest.TestCase):
    def test_chinese_query_returns_chinese(self) -> None:
        self.assertEqual(infer_target_from_user_language("翻译这个文件"), "Chinese")

    def test_chinese_with_mixed_punctuation_returns_chinese(self) -> None:
        self.assertEqual(infer_target_from_user_language("请翻译 @file.srt 为中文"), "Chinese")

    def test_english_query_returns_english(self) -> None:
        self.assertEqual(infer_target_from_user_language("translate this file"), "English")

    def test_english_with_file_reference_returns_english(self) -> None:
        self.assertEqual(infer_target_from_user_language("translate @file.srt"), "English")

    def test_empty_query_returns_none(self) -> None:
        self.assertIsNone(infer_target_from_user_language(""))
        self.assertIsNone(infer_target_from_user_language("   "))

    def test_japanese_query_returns_japanese(self) -> None:
        self.assertEqual(infer_target_from_user_language("このファイルを翻訳してください"), "Japanese")

    def test_korean_query_returns_korean(self) -> None:
        self.assertEqual(infer_target_from_user_language("이 파일을 번역해 주세요"), "Korean")

    def test_no_script_returns_none(self) -> None:
        self.assertIsNone(infer_target_from_user_language("1234567890"))
        self.assertIsNone(infer_target_from_user_language("!@#$%^&*()"))

    def test_chinese_with_english_word_still_chinese(self) -> None:
        # Chinese script detection takes priority over English
        self.assertEqual(infer_target_from_user_language("翻译这个 subtitle 文件"), "Chinese")

    def test_english_with_chinese_character_returns_chinese(self) -> None:
        # Any Chinese character triggers Chinese detection
        self.assertEqual(infer_target_from_user_language("translate this 字幕 file"), "Chinese")


if __name__ == "__main__":
    unittest.main()
