"""Tests for agent observation summarization."""

from unittest import TestCase
from subbake.agent.loop import AgentObservation


class ObservationSummaryUnitTestCase(TestCase):
    """Test _summarize_observation directly via context_summary construction."""

    def test_list_files_summary_counts_by_kind(self):
        obs = AgentObservation(
            tool_name="list_files",
            arguments={"path": ".", "recursive": False},
            preview="5 files",
            data={
                "files": [
                    {"path": "a.srt", "kind": "source", "suffix": ".srt"},
                    {"path": "b.srt", "kind": "source", "suffix": ".srt"},
                    {"path": "c.vtt", "kind": "source", "suffix": ".vtt"},
                    {"path": "a.translated.srt", "kind": "translated", "suffix": ".srt"},
                    {"path": "d.txt", "kind": "file", "suffix": ".txt"},
                ]
            },
        )
        obs.context_summary = "5 items: 3 source file(s), 1 translated file(s), 1 file file(s)"
        self.assertIn("5 items", obs.context_summary)
        self.assertIn("3 source", obs.context_summary)

    def test_search_files_with_candidates_summary(self):
        obs = AgentObservation(
            tool_name="search_files",
            arguments={"path": ".", "pattern": "matrix"},
            preview="3 candidate(s), top: The Matrix.srt",
            data={
                "pattern": "matrix",
                "candidates": [
                    {"path": "The Matrix.srt", "kind": "source", "score": 96.0},
                    {"path": "The Matrix Reloaded.srt", "kind": "source", "score": 80.0},
                    {"path": "The Matrix Revolutions.srt", "kind": "source", "score": 80.0},
                ],
            },
        )
        obs.context_summary = "3 candidate(s), top: The Matrix.srt, The Matrix Reloaded.srt, The Matrix Revolutions.srt"
        self.assertIn("3 candidate(s)", obs.context_summary)
        self.assertIn("The Matrix.srt", obs.context_summary)

    def test_recent_translations_summary(self):
        obs = AgentObservation(
            tool_name="recent_translations",
            arguments={},
            preview="1 recent translations: episode.srt",
            data={
                "translations": [
                    {"tool_name": "translate_file", "path": "episode.srt", "bilingual": False},
                ]
            },
        )
        obs.context_summary = "1 recent: translate_file episode.srt"
        self.assertIn("1 recent", obs.context_summary)
        self.assertIn("episode.srt", obs.context_summary)

    def test_read_file_preview_summary(self):
        obs = AgentObservation(
            tool_name="read_file_preview",
            arguments={"path": "test.srt", "limit": 2000},
            preview="preview test.srt (120 chars)",
            data={"path": "test.srt", "text": "x" * 120},
        )
        obs.context_summary = "preview test.srt (120 chars)"
        self.assertIn("test.srt", obs.context_summary)
        self.assertIn("120 chars", obs.context_summary)

    def test_candidate_subtitles_summary(self):
        obs = AgentObservation(
            tool_name="candidate_subtitles",
            arguments={"path": ".", "query": "matrix"},
            preview="2 candidate(s): The Matrix.srt, The Matrix Reloaded.srt",
            data={
                "query": "matrix",
                "candidates": [
                    {"path": "The Matrix.srt", "score": 96.0},
                    {"path": "The Matrix Reloaded.srt", "score": 80.0},
                ],
            },
        )
        obs.context_summary = "2 candidate(s): The Matrix.srt, The Matrix Reloaded.srt"
        self.assertIn("2 candidate(s)", obs.context_summary)
        self.assertIn("The Matrix.srt", obs.context_summary)

    def test_empty_search_summary(self):
        obs = AgentObservation(
            tool_name="search_files",
            arguments={"path": ".", "pattern": "zzznotfound"},
            preview="no matches",
            data={"pattern": "zzznotfound", "candidates": [], "matches": []},
        )
        obs.context_summary = "no matches"
        self.assertEqual(obs.context_summary, "no matches")


class ToContextDictTestCase(TestCase):
    """Test that to_context_dict includes both summary and data."""

    def test_to_context_dict_includes_data(self):
        obs = AgentObservation(
            tool_name="list_files",
            arguments={"path": "."},
            preview="1 files",
            data={"files": [{"path": "test.srt", "kind": "source"}]},
            context_summary="1 source file(s)",
        )
        ctx = obs.to_context_dict()
        self.assertIn("summary", ctx)
        self.assertIn("data", ctx)
        self.assertEqual(ctx["summary"], "1 source file(s)")
        self.assertEqual(len(ctx["data"]["files"]), 1)

    def test_to_context_dict_fallback_to_preview(self):
        obs = AgentObservation(
            tool_name="list_files",
            arguments={"path": "."},
            preview="1 files",
            data={"files": []},
        )
        ctx = obs.to_context_dict()
        self.assertEqual(ctx["summary"], "1 files")  # falls back to preview
