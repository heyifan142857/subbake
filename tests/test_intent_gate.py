"""Tests for the agent intent gate classification."""

import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from subbake.agent import (
    SubBakeAgent,
    TOOL_CATEGORIES,
    ALWAYS_AVAILABLE_TOOLS,
)
from subbake.agent.intent import (
    CONFIDENCE_LOW_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    CONFIDENCE_MIN_OBSERVATIONS,
    _fallback_intent_classification,
    _mock_classify_intent,
    apply_confidence_gate,
    intent_to_decision,
)
from rich.console import Console


class IntentGateMockClassificationTestCase(TestCase):
    """Test that _mock_classify_intent returns correct categories."""

    def setUp(self):
        self._orig_cwd = Path.cwd()
        self._tmpdir = tempfile.mkdtemp()
        os.chdir(self._tmpdir)
        Path("subbake.toml").write_text(
            "[defaults]\nprovider = \"mock\"\n", encoding="utf-8"
        )
        Path("episode.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")
        Path("episode.translated.srt").write_text(
            "1\n00:00:01,000 --> 00:00:02,000\n[MOCK-ZH] Hello\n", encoding="utf-8"
        )
        self.console = Console()
        self.agent = SubBakeAgent(console=self.console, resume=False)

    def tearDown(self):
        os.chdir(str(self._orig_cwd))

    def test_translate_file_with_reference(self):
        line = "翻译 @episode.srt 成英文"
        intent = _mock_classify_intent(self.agent, line)
        self.assertIsNotNone(intent)
        self.assertEqual(intent["category"], "translate_file")
        self.assertIn("episode.srt", intent["parameters"].get("path", ""))

    def test_translate_series_with_directory(self):
        dir_path = Path("Season01")
        dir_path.mkdir()
        line = "翻译 @Season01"
        intent = _mock_classify_intent(self.agent, line)
        self.assertIsNotNone(intent)
        self.assertEqual(intent["category"], "translate_series")

    def test_edit_subtitle_with_generated_file(self):
        line = "请修改 @episode.translated.srt 把角色名统一一下"
        intent = _mock_classify_intent(self.agent, line)
        self.assertIsNotNone(intent)
        self.assertEqual(intent["category"], "edit_subtitle")
        self.assertIn("episode.translated.srt", intent["parameters"].get("path", ""))

    def test_ambiguous_request_returns_none(self):
        line = "你好，有什么功能？"
        intent = _mock_classify_intent(self.agent, line)
        self.assertIsNone(intent)

    def test_matrix_request_returns_none(self):
        line = "The Matrix"
        intent = _mock_classify_intent(self.agent, line)
        self.assertIsNone(intent)


class IntentGateDecisionTestCase(TestCase):
    """Test that _intent_to_decision routes correctly."""

    def setUp(self):
        self._orig_cwd = Path.cwd()
        self._tmpdir = tempfile.mkdtemp()
        os.chdir(self._tmpdir)
        Path("test.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n", encoding="utf-8")
        from subbake.agent import SubBakeAgent
        from rich.console import Console
        self.agent = SubBakeAgent(console=Console(), resume=False)

    def tearDown(self):
        os.chdir(str(self._orig_cwd))

    def test_chat_returns_respond(self):
        intent = {"category": "chat", "parameters": {}, "confidence": 0.5, "reason": "test"}
        decision = intent_to_decision(self.agent, intent, "hello", run_agent_loop=lambda *a, **kw: {}, agent_loop_max_steps=5)
        self.assertEqual(decision["action"], "respond")

    def test_high_confidence_with_path_skips_agent_loop(self):
        intent = {
            "category": "translate_file",
            "parameters": {"path": str(Path.cwd() / "test.srt")},
            "confidence": 0.9,
            "reason": "test",
        }
        decision = intent_to_decision(self.agent, intent, "translate test.srt", run_agent_loop=lambda *a, **kw: {}, agent_loop_max_steps=5)
        self.assertEqual(decision["action"], "final_tool_call")
        self.assertEqual(decision["tool_name"], "translate_file")

    def test_edit_without_instruction_goes_to_agent_loop(self):
        """edit_subtitle needs instruction, so without it should not skip to direct execution."""
        intent = {
            "category": "edit_subtitle",
            "parameters": {"path": str(Path.cwd() / "test.srt")},
            "confidence": 0.9,
            "reason": "test",
        }
        decision = intent_to_decision(self.agent, intent, "edit test.srt", run_agent_loop=lambda *a, **kw: {}, agent_loop_max_steps=5)
        # Should NOT be final_tool_call (direct execution) since instruction is missing
        self.assertNotEqual(decision.get("action"), "final_tool_call")

    def test_low_confidence_asks_user(self):
        intent = {
            "category": "translate_file",
            "parameters": {},
            "confidence": 0.3,
            "reason": "uncertain",
        }
        decision = intent_to_decision(self.agent, intent, "do something", run_agent_loop=lambda *a, **kw: {}, agent_loop_max_steps=5)
        self.assertEqual(decision["action"], "ask_user")


class FallbackClassificationTestCase(TestCase):
    """Test fallback intent classification."""

    def setUp(self):
        self._orig_cwd = Path.cwd()
        self._tmpdir = tempfile.mkdtemp()
        os.chdir(self._tmpdir)
        Path("test.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n", encoding="utf-8")
        from subbake.agent import SubBakeAgent
        from rich.console import Console
        self.agent = SubBakeAgent(console=Console(), resume=False)

    def tearDown(self):
        os.chdir(str(self._orig_cwd))

    def test_translate_with_ref_in_fallback(self):
        line = "翻译 @test.srt"
        result = _fallback_intent_classification(self.agent, line)
        self.assertIsNotNone(result)
        self.assertEqual(result["category"], "translate_file")

    def test_ambiguous_fallback_returns_none(self):
        line = "hello world"
        result = _fallback_intent_classification(self.agent, line)
        self.assertIsNone(result)


class ConfidenceGateTestCase(TestCase):
    """Test confidence gating logic."""

    def setUp(self):
        self._orig_cwd = Path.cwd()
        self._tmpdir = tempfile.mkdtemp()
        os.chdir(self._tmpdir)
        from subbake.agent import SubBakeAgent
        from rich.console import Console
        from subbake.agent.loop import AgentLoopState
        self.agent = SubBakeAgent(console=Console(), resume=False)
        self.state = AgentLoopState(original_user_message="test", allowed_tools=("list_files",))

    def tearDown(self):
        os.chdir(str(self._orig_cwd))

    def test_low_confidence_responds(self):
        decision = {"action": "final_tool_call", "tool_name": "translate_file", "confidence": 0.3}
        gated = apply_confidence_gate(decision, self.state)
        self.assertIsNotNone(gated)
        self.assertEqual(gated["action"], "respond")

    def test_medium_confidence_without_observations_asks_user(self):
        decision = {"action": "final_tool_call", "tool_name": "translate_file", "confidence": 0.5}
        self.state.observations = []  # no observations
        gated = apply_confidence_gate(decision, self.state)
        self.assertIsNotNone(gated)
        self.assertEqual(gated["action"], "ask_user")

    def test_medium_confidence_with_observations_passes(self):
        decision = {"action": "final_tool_call", "tool_name": "translate_file", "confidence": 0.5}
        # Add enough observations
        from subbake.agent.loop import AgentObservation
        self.state.observations = [
            AgentObservation(tool_name="list_files", arguments={}, preview="test")
            for _ in range(CONFIDENCE_MIN_OBSERVATIONS)
        ]
        gated = apply_confidence_gate(decision, self.state)
        self.assertIsNone(gated)  # passes through

    def test_high_confidence_passes(self):
        decision = {"action": "final_tool_call", "tool_name": "translate_file", "confidence": 0.85}
        gated = apply_confidence_gate(decision, self.state)
        self.assertIsNone(gated)

    def test_tool_call_not_gated(self):
        """Discovery tool_call should not be gated."""
        decision = {"action": "tool_call", "tool_name": "list_files", "confidence": 0.3}
        gated = apply_confidence_gate(decision, self.state)
        self.assertIsNone(gated)

    def test_no_confidence_data_passes(self):
        decision = {"action": "final_tool_call", "tool_name": "translate_file"}
        gated = apply_confidence_gate(decision, self.state)
        self.assertIsNone(gated)


class ToolCategoriesTestCase(TestCase):
    """Test tool category filtering."""

    def test_translate_file_filters_correctly(self):
        tools = ALWAYS_AVAILABLE_TOOLS + tuple(TOOL_CATEGORIES["translate_file"])
        self.assertIn("translate_file", tools)
        self.assertNotIn("edit_subtitle", tools)

    def test_edit_subtitle_filters_correctly(self):
        tools = ALWAYS_AVAILABLE_TOOLS + tuple(TOOL_CATEGORIES["edit_subtitle"])
        self.assertIn("edit_subtitle", tools)
        self.assertNotIn("translate_file", tools)

    def test_all_categories_have_valid_tools(self):
        for category, tool_names in TOOL_CATEGORIES.items():
            for name in tool_names:
                self.assertIsInstance(name, str)
