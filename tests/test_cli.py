from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.core import ParameterSource
from prompt_toolkit.document import Document
from rich.console import Console
from typer.testing import CliRunner

from subbake import __version__
from subbake.agent import (
    CONFIG_BOOTSTRAP_CREATE,
    NEW_PROFILE_VALUE,
    SubBakeAgent,
)
from subbake.agent.trace import (
    AGENT_COMMANDS,
    _AgentLoopTrace,
    _default_api_key_env,
    _matching_picker_choices,
    _picker_choices,
    _resolve_picker_selection,
    _resolve_text_prompt_value,
    _slash_command_completer,
    _text_prompt_matches,
    _unique_slash_command_match,
)

from subbake.app import _configured_value, app
from subbake.config import load_app_config
from subbake.entities import Usage
from subbake.models.base_model import LLMBackend
from subbake.storage import build_runtime_paths


class CliAgentRepairBackend(LLMBackend):
    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        prompt = "\n".join(message["content"] for message in messages)
        task = self._extract_between(prompt, "TASK_START", "TASK_END")
        if task == "translate_subtitles":
            payload = json.loads(self._extract_between(prompt, "BATCH_JSON_START", "BATCH_JSON_END"))
            return (
                {
                    "lines": [
                        {"id": item["id"], "translation": ""}
                        for item in payload["lines"]
                    ],
                    "summary": "broken",
                    "glossary_updates": [],
                },
                Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        if task == "agent_repair_translation":
            payload = json.loads(self._extract_between(prompt, "AGENT_REPAIR_JSON_START", "AGENT_REPAIR_JSON_END"))
            return (
                {
                    "lines": [
                        {"id": item["id"], "translation": f"[AGENT] {item['text']}"}
                        for item in payload["source_lines"]
                    ],
                    "summary": "repaired",
                    "glossary_updates": [],
                },
                Usage(input_tokens=2, output_tokens=2, total_tokens=4),
            )
        raise RuntimeError(f"Unexpected task: {task}")

    def check_credentials(self) -> tuple[bool, str]:
        return True, "ok"

    def _extract_between(self, text: str, start_marker: str, end_marker: str) -> str:
        start_index = text.index(start_marker) + len(start_marker)
        end_index = text.index(end_marker, start_index)
        return text[start_index:end_index].strip()


class LoopingAgentBackend(LLMBackend):
    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        return (
            {
                "action": "tool_call",
                "message": "Still looking.",
                "tool_name": "list_files",
                "arguments": {"path": ".", "recursive": False},
                "reason": "Keep discovering.",
                "confidence": 0.5,
            },
            Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    def check_credentials(self) -> tuple[bool, str]:
        return True, "ok"


class CLITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    @contextlib.contextmanager
    def _isolated_filesystem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                yield tmpdir
            finally:
                os.chdir(old_cwd)

    def test_configured_value_keeps_changed_cli_value_when_parameter_source_is_unreliable(self) -> None:
        class DummyParameter:
            name = "target_language"
            default = "Chinese"

        class DummyCommand:
            params = [DummyParameter()]

        class DummyContext:
            command = DummyCommand()

            def get_parameter_source(self, parameter_name: str):
                return None

        value = _configured_value(
            DummyContext(),
            "target_language",
            "zh",
            {"target_language": "en"},
        )

        self.assertEqual(value, "zh")

    def test_configured_value_accepts_dashed_parameter_source_names(self) -> None:
        class DummyParameter:
            name = "target_language"
            default = "Chinese"

        class DummyCommand:
            params = [DummyParameter()]

        class DummyContext:
            command = DummyCommand()

            def get_parameter_source(self, parameter_name: str):
                if parameter_name == "target-language":
                    return ParameterSource.COMMANDLINE
                return None

        value = _configured_value(
            DummyContext(),
            "target_language",
            "Chinese",
            {"target_language": "en"},
        )

        self.assertEqual(value, "Chinese")

    def test_agent_slash_command_completer_filters_commands(self) -> None:
        completer = _slash_command_completer()

        all_commands = list(completer.get_completions(Document("/"), None))
        model_commands = list(completer.get_completions(Document("/mo"), None))
        no_commands = list(completer.get_completions(Document("hello /mo"), None))

        self.assertEqual([completion.text for completion in all_commands], [command for command, _ in AGENT_COMMANDS])
        self.assertEqual([completion.text for completion in model_commands], ["/model"])
        self.assertEqual(no_commands, [])

    def test_agent_tab_completion_resolves_unique_slash_command(self) -> None:
        self.assertEqual(_unique_slash_command_match("/clea"), "/clear")
        self.assertEqual(_unique_slash_command_match("/mo"), "/model")
        self.assertIsNone(_unique_slash_command_match("/p"))

    def test_agent_inline_picker_filters_by_label_and_metadata(self) -> None:
        choices = _picker_choices(
            [
                ("alpha", "alpha: mock / mock-alpha"),
                ("beta", "beta: openai / gpt-4o-mini"),
                (NEW_PROFILE_VALUE, "new"),
            ],
            default="beta",
        )

        self.assertEqual([choice.value for choice in choices], ["beta", "alpha", NEW_PROFILE_VALUE])
        self.assertEqual([choice.value for choice in _matching_picker_choices("", choices)], ["beta", "alpha", NEW_PROFILE_VALUE])
        self.assertEqual([choice.value for choice in _matching_picker_choices("openai", choices)], ["beta"])
        self.assertEqual([choice.value for choice in _matching_picker_choices("mock-alpha", choices)], ["alpha"])

    def test_agent_inline_picker_resolves_visible_or_typed_selection(self) -> None:
        choices = _picker_choices(
            [
                ("alpha", "alpha: mock / mock-alpha"),
                ("beta", "beta: openai / gpt-4o-mini"),
                (NEW_PROFILE_VALUE, "new"),
            ],
            default="beta",
        )

        self.assertEqual(_resolve_picker_selection("", choices, default="beta"), "beta")
        self.assertEqual(_resolve_picker_selection("beta", choices, default="beta"), "beta")
        self.assertEqual(_resolve_picker_selection("openai", choices, default="beta"), "beta")
        self.assertEqual(_resolve_picker_selection("new", choices, default="beta"), NEW_PROFILE_VALUE)
        self.assertIsNone(_resolve_picker_selection("a", choices, default="beta"))

    def test_agent_inline_text_prompt_helpers(self) -> None:
        self.assertEqual(_resolve_text_prompt_value("", default="Chinese"), "Chinese")
        self.assertEqual(_resolve_text_prompt_value("ja", default="Chinese"), "ja")
        self.assertEqual(
            _text_prompt_matches("op", ("mock", "openai", "anthropic", "gemini", "openai-compatible")),
            ["openai", "openai-compatible"],
        )
        self.assertEqual(_text_prompt_matches("", ("zh", "en")), ["zh", "en"])

    def test_agent_new_profile_default_api_key_env_depends_on_provider(self) -> None:
        self.assertEqual(_default_api_key_env("openai"), "OPENAI_API_KEY")
        self.assertEqual(_default_api_key_env("openai-compatible"), "OPENAI_API_KEY")
        self.assertEqual(_default_api_key_env("compatible"), "OPENAI_API_KEY")
        self.assertEqual(_default_api_key_env("anthropic"), "ANTHROPIC_API_KEY")
        self.assertEqual(_default_api_key_env("gemini"), "GEMINI_API_KEY")
        self.assertEqual(_default_api_key_env("mock"), "")

    def test_root_help_mentions_main_commands(self) -> None:
        result = self.runner.invoke(app, ["--help"])
        output = self._strip_ansi(result.stdout)

        self.assertEqual(result.exit_code, 0)
        self.assertIn("LLM subtitle translation CLI with Chinese as the default target language", output)
        self.assertIn("another target such as en / ja / fr.", output)
        self.assertIn("Common commands:", output)
        self.assertIn("sbake translate input.srt", output)
        self.assertIn("--output-format", output)
        self.assertIn("--provider", output)
        self.assertIn("--fast", output)
        self.assertIn("--no-agent", output)
        self.assertIn("--target-language", output)
        self.assertIn("--config", output)
        self.assertIn("--profile", output)
        self.assertIn("sbake series", output)
        self.assertIn("sbake resume", output)
        self.assertIn("sbake check-key", output)
        self.assertIn("sbake clean input.srt", output)

    def test_bare_sbake_starts_agent_and_can_exit(self) -> None:
        with self._isolated_filesystem():
            result = self.runner.invoke(app, [], input="/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("SubBake agent", output)
            self.assertIn(__version__, output)
            self.assertIn("sbake[", output)

    def test_agent_model_command_switches_config_profile(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[profiles.alpha]\n"
                'provider = "mock"\n'
                'model = "mock-alpha"\n\n'
                "[profiles.beta]\n"
                'provider = "mock"\n'
                'model = "mock-beta"\n',
                encoding="utf-8",
            )

            result = self.runner.invoke(app, [], input="/model beta\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Profile switched: beta", output)
            self.assertIn("mock / mock-beta", output)

    def test_agent_model_command_lists_profiles_in_non_interactive_mode(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[profiles.alpha]\n"
                'provider = "mock"\n'
                'model = "mock-alpha"\n\n'
                "[profiles.beta]\n"
                'provider = "mock"\n'
                'model = "mock-beta"\n',
                encoding="utf-8",
            )

            result = self.runner.invoke(app, [], input="/model\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Profiles:", output)
            self.assertIn("alpha: mock / mock-alpha", output)
            self.assertIn("beta: mock / mock-beta", output)
            self.assertIn("new: create a new model profile", output)

    def test_agent_session_command_lists_session_titles(self) -> None:
        with self._isolated_filesystem():
            result = self.runner.invoke(app, [], input="hello session\n/session\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Recent sessions:", output)
            self.assertIn("hello session", output)

    def test_agent_session_command_switches_by_id(self) -> None:
        with self._isolated_filesystem():
            first = self.runner.invoke(app, [], input="first session\n/exit\n")
            self.assertEqual(first.exit_code, 0)
            session_files = sorted(Path(".subbake/agent/sessions").glob("*.json"))
            self.assertEqual(len(session_files), 1)

            result = self.runner.invoke(app, [], input=f"/session {session_files[0].stem}\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Session switched:", output)
            self.assertIn("first session", output)

    def test_agent_profile_new_is_interactive_only_from_scripted_input(self) -> None:
        with self._isolated_filesystem():
            result = self.runner.invoke(app, [], input="/profile new\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Profile creation is available from the interactive /profile picker.", output)

    def test_agent_offers_config_bootstrap_when_interactive_without_config(self) -> None:
        with self._isolated_filesystem():
            with (
                patch("subbake.config.discover_config_path", return_value=None),
                patch("subbake.config.discover_project_config_path", return_value=None),
            ):
                agent = SubBakeAgent(console=Console(record=True), resume=False)

            agent.interactive = True
            with (
                patch.object(agent, "_select_from_list", return_value=CONFIG_BOOTSTRAP_CREATE) as select,
                patch.object(agent, "_create_profile_interactively") as create_profile,
            ):
                agent._maybe_offer_config_bootstrap()

            select.assert_called_once()
            create_profile.assert_called_once()

    def test_agent_can_create_first_config_profile(self) -> None:
        with self._isolated_filesystem():
            config_path = Path("xdg/subbake/config.toml")
            with (
                patch("subbake.config.discover_config_path", return_value=None),
                patch("subbake.config.discover_project_config_path", return_value=None),
                patch("subbake.config.global_config_candidates", return_value=[config_path]),
            ):
                agent = SubBakeAgent(console=Console(record=True), resume=False)
                agent.interactive = True
                answers = iter(["chatgpt", "openai", "gpt-4o-mini", "OPENAI_API_KEY", "", "Chinese"])
                with patch.object(agent, "_prompt_text", side_effect=lambda *args, **kwargs: next(answers)):
                    agent._create_profile_interactively()

            config = load_app_config(config_path)
            self.assertEqual(agent.profile, "chatgpt")
            self.assertEqual(agent.session.config_path, str(config_path))
            self.assertEqual(config.default_profile, "chatgpt")
            self.assertEqual(config.profiles["chatgpt"]["provider"], "openai")
            self.assertEqual(config.profiles["chatgpt"]["model"], "gpt-4o-mini")
            self.assertEqual(config.profiles["chatgpt"]["api_key_env"], "OPENAI_API_KEY")

    def test_agent_created_profile_uses_current_config_when_present(self) -> None:
        with self._isolated_filesystem():
            config_path = Path("subbake.toml")
            config_path.write_text("[defaults]\nprovider = \"mock\"\n", encoding="utf-8")
            agent = SubBakeAgent(console=Console(record=True), resume=False)
            agent.interactive = True
            answers = iter(["local", "mock", "mock-zh", "", "", "Chinese"])
            with patch.object(agent, "_prompt_text", side_effect=lambda *args, **kwargs: next(answers)):
                agent._create_profile_interactively()

            config = load_app_config(config_path)
            self.assertEqual(agent.profile, "local")
            self.assertEqual(agent.session.config_path, str(config_path.resolve()))
            self.assertEqual(config.default_profile, "local")
            self.assertEqual(config.profiles["local"]["provider"], "mock")

    def test_agent_created_profile_does_not_replace_existing_default(self) -> None:
        with self._isolated_filesystem():
            config_path = Path("subbake.toml")
            config_path.write_text(
                'default_profile = "alpha"\n\n'
                "[profiles.alpha]\n"
                'provider = "mock"\n'
                'model = "mock-alpha"\n',
                encoding="utf-8",
            )
            agent = SubBakeAgent(console=Console(record=True), resume=False)
            agent.interactive = True
            answers = iter(["beta", "mock", "mock-beta", "", "", "Chinese"])
            with patch.object(agent, "_prompt_text", side_effect=lambda *args, **kwargs: next(answers)):
                agent._create_profile_interactively()

            config = load_app_config(config_path)
            self.assertEqual(config.default_profile, "alpha")
            self.assertIn("beta", config.profiles)

    def test_agent_new_profile_cancellation_does_not_write_config(self) -> None:
        with self._isolated_filesystem():
            config_path = Path("xdg/subbake/config.toml")
            with (
                patch("subbake.config.discover_config_path", return_value=None),
                patch("subbake.config.discover_project_config_path", return_value=None),
                patch("subbake.config.global_config_candidates", return_value=[config_path]),
            ):
                agent = SubBakeAgent(console=Console(record=True), resume=False)
                agent.interactive = True
                with patch.object(agent, "_prompt_text", return_value=None):
                    agent._create_profile_interactively()

            self.assertFalse(config_path.exists())
            self.assertIsNone(agent.profile)

    def test_resume_command_starts_agent_when_no_session_exists(self) -> None:
        with self._isolated_filesystem():
            result = self.runner.invoke(app, ["resume"], input="/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("No previous agent session found", output)
            self.assertIn("SubBake agent", output)

    def test_version_flag_prints_package_version(self) -> None:
        result = self.runner.invoke(app, ["-V"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn(f"subbake {__version__}", result.stdout)

    def test_clean_file_target_removes_only_runs_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "episode.srt"
            input_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
            runtime = build_runtime_paths(input_path)
            runtime.run_dir.mkdir(parents=True, exist_ok=True)
            runtime.cache_dir.mkdir(parents=True, exist_ok=True)
            runtime.glossary_path.parent.mkdir(parents=True, exist_ok=True)
            (runtime.run_dir / "run_state.json").write_text("{}", encoding="utf-8")
            (runtime.cache_dir / "sample.json").write_text("{}", encoding="utf-8")
            runtime.glossary_path.write_text("{}", encoding="utf-8")

            result = self.runner.invoke(app, ["clean", str(input_path)])

            self.assertEqual(result.exit_code, 0)
            self.assertFalse(runtime.run_dir.exists())
            self.assertTrue(runtime.cache_dir.exists())
            self.assertTrue(runtime.glossary_path.exists())

    def test_clean_directory_target_removes_all_runtime_artifacts_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_dir = temp_path / "project"
            runtime_root = project_dir / ".subbake"
            (runtime_root / "runs").mkdir(parents=True, exist_ok=True)
            (runtime_root / "cache").mkdir(parents=True, exist_ok=True)
            (runtime_root / "glossary.json").write_text("{}", encoding="utf-8")

            result = self.runner.invoke(app, ["clean", str(project_dir)])

            self.assertEqual(result.exit_code, 0)
            self.assertFalse(runtime_root.exists())

    def test_translate_uses_auto_discovered_config_profile(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                'default_profile = "mock_en"\n\n'
                "[defaults]\n"
                "final_review = false\n"
                "resume = false\n"
                "cache = false\n\n"
                "[profiles.mock_en]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                'target_language = "en"\n',
                encoding="utf-8",
            )
            Path("clip.txt").write_text("hello\n", encoding="utf-8")

            result = self.runner.invoke(app, ["translate", "clip.txt"])

            self.assertEqual(result.exit_code, 0)
            self.assertIn("[MOCK-EN] hello", Path("clip.translated.txt").read_text(encoding="utf-8"))
            output = self._strip_ansi(result.stdout)
            self.assertIn("Config:", output)
            self.assertIn("profile mock_en", output)

    def test_translate_command_line_overrides_config_values(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                'default_profile = "mock_en"\n\n'
                "[profiles.mock_en]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                'target_language = "en"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("clip.txt").write_text("hello\n", encoding="utf-8")

            result = self.runner.invoke(
                app,
                ["translate", "clip.txt", "--target-language", "zh"],
            )

            self.assertEqual(result.exit_code, 0)
            self.assertIn("[MOCK-ZH] hello", Path("clip.translated.txt").read_text(encoding="utf-8"))

    def test_translate_requires_default_profile_when_multiple_profiles_exist(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[profiles.mock_en]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                'target_language = "en"\n\n'
                "[profiles.mock_zh]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                'target_language = "zh"\n',
                encoding="utf-8",
            )
            Path("clip.txt").write_text("hello\n", encoding="utf-8")

            result = self.runner.invoke(app, ["translate", "clip.txt"])
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 1)
            self.assertIn("Multiple config profiles are defined", output)
            self.assertIn("--profile", output)

    def test_translate_profile_option_selects_named_profile(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[profiles.mock_en]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                'target_language = "en"\n'
                "final_review = false\n\n"
                "[profiles.mock_zh]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                'target_language = "zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("clip.txt").write_text("hello\n", encoding="utf-8")

            result = self.runner.invoke(
                app,
                ["translate", "clip.txt", "--profile", "mock_zh"],
            )

            self.assertEqual(result.exit_code, 0)
            self.assertIn("[MOCK-ZH] hello", Path("clip.translated.txt").read_text(encoding="utf-8"))

    def test_translate_can_convert_output_format_from_output_suffix(self) -> None:
        with self._isolated_filesystem():
            Path("clip.srt").write_text(
                "1\n"
                "00:00:01,000 --> 00:00:02,000\n"
                "hello\n",
                encoding="utf-8",
            )

            result = self.runner.invoke(
                app,
                [
                    "translate",
                    "clip.srt",
                    "--provider",
                    "mock",
                    "--model",
                    "mock-zh",
                    "--output",
                    "converted.txt",
                    "--no-final-review",
                ],
            )

            self.assertEqual(result.exit_code, 0)
            self.assertTrue(Path("converted.txt").exists())
            self.assertEqual(Path("converted.txt").read_text(encoding="utf-8"), "[MOCK-ZH] hello\n")

    def test_translate_reports_when_previous_results_are_reused(self) -> None:
        with self._isolated_filesystem():
            Path("clip.txt").write_text("hello\nworld\n", encoding="utf-8")

            first_result = self.runner.invoke(
                app,
                ["translate", "clip.txt", "--provider", "mock", "--model", "mock-zh", "--no-final-review"],
            )
            self.assertEqual(first_result.exit_code, 0)

            second_result = self.runner.invoke(
                app,
                ["translate", "clip.txt", "--provider", "mock", "--model", "mock-zh", "--no-final-review"],
            )

            self.assertEqual(second_result.exit_code, 0)
            output = self._strip_ansi(second_result.stdout)
            self.assertIn("Reused:", output)
            self.assertIn("1 translated batch(es) from resume", output)

    def test_translate_reports_agent_summary_and_log_path_when_triggered(self) -> None:
        with self._isolated_filesystem():
            Path("clip.txt").write_text("hello\n", encoding="utf-8")

            with patch("subbake.app.build_backend", return_value=CliAgentRepairBackend()):
                result = self.runner.invoke(
                    app,
                    [
                        "translate",
                        "clip.txt",
                        "--provider",
                        "mock",
                        "--model",
                        "mock-zh",
                        "--no-final-review",
                        "--retries",
                        "0",
                    ],
                )

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(Path("clip.translated.txt").read_text(encoding="utf-8"), "[AGENT] hello\n")
            output = self._strip_ansi(result.stdout)
            self.assertIn("Agent:", output)
            self.assertIn("1 triggered, 1 repaired", output)
            self.assertIn("translate batch 1", output)
            self.assertIn("Logs:", output)

    def test_series_command_translates_folder_with_shared_runtime_root(self) -> None:
        with self._isolated_filesystem():
            season = Path("season")
            season.mkdir()
            (season / "episode2.txt").write_text("hello Alice\n", encoding="utf-8")
            (season / "episode10.txt").write_text("hello Alice\n", encoding="utf-8")
            (season / "episode2.translated.txt").write_text("existing\n", encoding="utf-8")

            result = self.runner.invoke(
                app,
                [
                    "series",
                    str(season),
                    "--provider",
                    "mock",
                    "--model",
                    "mock-zh",
                    "--no-final-review",
                ],
            )
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertEqual((season / "episode2.translated.txt").read_text(encoding="utf-8"), "existing\n")
            self.assertIn("[MOCK-ZH] hello Alice", (season / "episode10.translated.txt").read_text(encoding="utf-8"))
            self.assertTrue((season / ".subbake").exists())
            self.assertIn("1 processed, 1 skipped, 0 failed", output)

    def test_agent_folder_reference_translates_series(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            season = Path("season")
            season.mkdir()
            (season / "episode1.txt").write_text("hello\n", encoding="utf-8")

            result = self.runner.invoke(app, [], input="翻译 @season\n/exit\n")

            self.assertEqual(result.exit_code, 0)
            self.assertIn("[MOCK-ZH] hello", (season / "episode1.translated.txt").read_text(encoding="utf-8"))

    def test_agent_undo_removes_translated_file_output(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("clip.txt").write_text("hello\n", encoding="utf-8")

            result = self.runner.invoke(app, [], input="翻译 @clip.txt\n/undo\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertFalse(Path("clip.translated.txt").exists())
            self.assertIn("Undo created:", output)

    def test_agent_undo_restores_overwritten_translation_output(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("clip.txt").write_text("hello\n", encoding="utf-8")
            Path("clip.translated.txt").write_text("previous translation\n", encoding="utf-8")

            result = self.runner.invoke(app, [], input="重新翻译 @clip.txt\n/undo\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(Path("clip.translated.txt").read_text(encoding="utf-8"), "previous translation\n")
            self.assertIn("Undo modified:", output)

    def test_agent_undo_removes_series_outputs_as_one_group(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            season = Path("season")
            season.mkdir()
            (season / "episode1.txt").write_text("hello one\n", encoding="utf-8")
            (season / "episode2.txt").write_text("hello two\n", encoding="utf-8")

            result = self.runner.invoke(app, [], input="翻译 @season\n/undo\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertFalse((season / "episode1.translated.txt").exists())
            self.assertFalse((season / "episode2.translated.txt").exists())
            self.assertEqual(output.count("Undo created:"), 2)

    def test_agent_current_directory_srt_request_translates_only_srt_series(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("movie1.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nhello one\n\n",
                encoding="utf-8",
            )
            Path("movie2.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nhello two\n\n",
                encoding="utf-8",
            )
            Path("notes.txt").write_text("do not translate\n", encoding="utf-8")
            Path("movie.mkv").write_text("video placeholder\n", encoding="utf-8")

            result = self.runner.invoke(
                app,
                [],
                input="帮我把目录下.srt文件都翻译了，生成中英双语字幕。注意这是同一系列的作品。\n/exit\n",
            )
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("现在要按同一系列翻译 2 个 .srt 文件", output)
            self.assertIn("生成中英双语字幕", output)
            self.assertLess(output.index("现在要按同一系列翻译"), output.index("Series:"))
            self.assertIn("已完成 2 个文件翻译", output)
            self.assertIn("2 processed, 0 skipped, 0 failed", output)
            self.assertIn("hello one\n[MOCK-ZH] hello one", Path("movie1.bilingual.srt").read_text(encoding="utf-8"))
            self.assertIn("hello two\n[MOCK-ZH] hello two", Path("movie2.bilingual.srt").read_text(encoding="utf-8"))
            self.assertFalse(Path("notes.translated.txt").exists())

    def test_agent_current_directory_series_request_passes_explicit_translation_options(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                'target_language = "zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("movie.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n",
                encoding="utf-8",
            )
            Path("notes.txt").write_text("do not translate\n", encoding="utf-8")

            result = self.runner.invoke(
                app,
                [],
                input="帮我把目录下.srt文件都翻译成英文，生成txt格式字幕。\n/exit\n",
            )
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("目标语言 English", output)
            self.assertIn("输出 TXT", output)
            self.assertIn("[MOCK-EN] hello", Path("movie.translated.txt").read_text(encoding="utf-8"))
            self.assertFalse(Path("movie.translated.srt").exists())
            self.assertFalse(Path("notes.translated.txt").exists())

    def test_agent_can_retarget_latest_chinese_translation_to_chinese_english_bilingual(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                'target_language = "zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("episode.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n",
                encoding="utf-8",
            )

            result = self.runner.invoke(
                app,
                [],
                input=(
                    "翻译 @episode.srt\n"
                    "我误翻译了中文字幕，现在我想变成中英字幕\n"
                    "/exit\n"
                ),
            )
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("目标语言", output)
            self.assertIn("English", output)
            self.assertIn("生成中英双语字幕", output)
            self.assertIn("[MOCK-ZH] 你好", Path("episode.translated.srt").read_text(encoding="utf-8"))
            self.assertIn("你好\n[MOCK-EN] 你好", Path("episode.bilingual.srt").read_text(encoding="utf-8"))

    def test_agent_retargets_referenced_generated_subtitle_to_source_file(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                'target_language = "zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("episode.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n",
                encoding="utf-8",
            )
            Path("episode.translated.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n[MOCK-ZH] 你好\n\n",
                encoding="utf-8",
            )

            result = self.runner.invoke(
                app,
                [],
                input="把 @episode.translated.srt 变成中英字幕\n/exit\n",
            )

            self.assertEqual(result.exit_code, 0)
            self.assertIn("你好\n[MOCK-EN] 你好", Path("episode.bilingual.srt").read_text(encoding="utf-8"))

    def test_agent_retargets_subtitle_by_title_text(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("The Matrix Revolutions 2003 REMASTERED.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello Neo.\n\n",
                encoding="utf-8",
            )
            Path("The Matrix Revolutions 2003 REMASTERED.translated.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n[MOCK-ZH] Hello Neo.\n\n",
                encoding="utf-8",
            )

            result = self.runner.invoke(
                app,
                [],
                input="你之前翻译的The Matrix Revolutions能不能改为中英双语\n/exit\n",
            )

            self.assertEqual(result.exit_code, 0)
            self.assertIn(
                "Hello Neo.\n[MOCK-ZH] Hello Neo.",
                Path("The Matrix Revolutions 2003 REMASTERED.bilingual.srt").read_text(encoding="utf-8"),
            )

    def test_agent_loop_interactive_trace_uses_compact_timeline(self) -> None:
        console = Console(record=True, force_terminal=True, width=100)
        trace = _AgentLoopTrace(console=console, interactive=True)

        trace.start()
        trace.think()
        trace.tool("candidate_subtitles", {"query": "The Matrix"})
        trace.observe("selected The Matrix 1999.srt")
        trace.final(
            {
                "action": "tool_call",
                "tool_name": "translate_file",
                "arguments": {"path": "The Matrix 1999.srt", "bilingual": True},
            }
        )

        output = console.export_text(styles=False)

        self.assertEqual(output.count("Agent Loop"), 1)
        self.assertIn("TOOL candidate_subtitles", output)
        self.assertIn("OBSERVE selected The Matrix 1999.srt", output)
        self.assertIn("EXECUTE translate_file", output)
        self.assertNotIn("Starting bounded discovery.", output)
        self.assertNotRegex(output, r"[╭╮╰╯]")

    def test_agent_loop_discovers_matrix_subtitle_before_bilingual_translation(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("The Matrix 1999.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello Neo.\n\n",
                encoding="utf-8",
            )

            result = self.runner.invoke(app, [], input="把黑客帝国改成中英双语\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Agent Loop", output)
            self.assertIn("TOOL candidate_subtitles", output)
            self.assertIn("OBSERVE selected The Matrix 1999.srt", output)
            self.assertIn("EXECUTE translate_file", output)
            self.assertIn(
                "Hello Neo.\n[MOCK-ZH] Hello Neo.",
                Path("The Matrix 1999.bilingual.srt").read_text(encoding="utf-8"),
            )

    def test_agent_loop_selects_matrix_revolutions_among_matrix_files(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("The Matrix Reloaded.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello Morpheus.\n\n",
                encoding="utf-8",
            )
            Path("The Matrix Revolutions.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello Neo.\n\n",
                encoding="utf-8",
            )

            result = self.runner.invoke(app, [], input="The Matrix Revolutions 改成中英双语\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("TOOL candidate_subtitles", output)
            self.assertIn("OBSERVE selected The Matrix Revolutions.srt", output)
            self.assertTrue(Path("The Matrix Revolutions.bilingual.srt").exists())
            self.assertFalse(Path("The Matrix Reloaded.bilingual.srt").exists())

    def test_agent_loop_asks_user_when_multiple_subtitle_candidates_match(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("The Matrix Reloaded.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello Morpheus.\n\n",
                encoding="utf-8",
            )
            Path("The Matrix Revolutions.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello Neo.\n\n",
                encoding="utf-8",
            )

            result = self.runner.invoke(app, [], input="黑客帝国改成中英双语\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("multiple subtitle candidates", output)
            self.assertIn("The Matrix Reloaded.srt", output)
            self.assertIn("The Matrix Revolutions.srt", output)
            self.assertFalse(Path("The Matrix Reloaded.bilingual.srt").exists())
            self.assertFalse(Path("The Matrix Revolutions.bilingual.srt").exists())

    def test_agent_plan_mode_runs_discovery_but_waits_to_translate(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("The Matrix.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello Neo.\n\n",
                encoding="utf-8",
            )

            result = self.runner.invoke(app, [], input="/plan\n把黑客帝国改成中英双语\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("TOOL candidate_subtitles", output)
            self.assertIn("OBSERVE selected The Matrix.srt", output)
            self.assertIn("PLAN translate_file", output)
            self.assertIn("<proposed_plan>", output)
            self.assertIn("Use /approve", output)
            self.assertFalse(Path("The Matrix.bilingual.srt").exists())

    def test_agent_loop_stops_after_max_steps(self) -> None:
        with self._isolated_filesystem():
            Path("visible.txt").write_text("hello\n", encoding="utf-8")

            with patch("subbake.runtime_options.build_backend_from_values", return_value=LoopingAgentBackend()):
                result = self.runner.invoke(app, [], input="请先想一想怎么处理\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(output.count("TOOL list_files"), 5)
            self.assertIn("Agent loop stopped after 5 steps without a final action.", output)

    def test_agent_retargets_subtitle_by_chinese_title_alias(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("The Matrix Reloaded 2003 REMASTERED.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello Morpheus.\n\n",
                encoding="utf-8",
            )
            Path("The Matrix Revolutions 2003 REMASTERED.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello Neo.\n\n",
                encoding="utf-8",
            )

            result = self.runner.invoke(
                app,
                [],
                input="你之前翻译的黑客帝国3能不能改为中英双语\n/exit\n",
            )

            self.assertEqual(result.exit_code, 0)
            self.assertIn(
                "Hello Neo.\n[MOCK-ZH] Hello Neo.",
                Path("The Matrix Revolutions 2003 REMASTERED.bilingual.srt").read_text(encoding="utf-8"),
            )
            self.assertFalse(Path("The Matrix Reloaded 2003 REMASTERED.bilingual.srt").exists())

    def test_agent_edit_generated_subtitle_creates_backup(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            Path("clip.txt").write_text("hello\n", encoding="utf-8")
            Path("clip.translated.txt").write_text("[MOCK-ZH] hello\n", encoding="utf-8")

            result = self.runner.invoke(
                app,
                [],
                input="请修改 @clip.translated.txt keep the translation unchanged\n/exit\n",
            )
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Edited:", output)
            self.assertEqual(Path("clip.translated.txt").read_text(encoding="utf-8"), "[MOCK-ZH] hello\n")
            backups = list(Path(".subbake/agent/backups").glob("*/clip.translated.txt"))
            self.assertEqual(len(backups), 1)

    def test_agent_file_operations_are_natural_language_tools(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            result = self.runner.invoke(
                app,
                [],
                input=(
                    "创建 @notes.txt first line\n"
                    "追加 @notes.txt second line\n"
                    "替换 @notes.txt second line => updated line\n"
                    "把 @notes.txt 改名为 @renamed.txt\n"
                    "删除 @renamed.txt\n"
                    "/exit\n"
                ),
            )
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertFalse(Path("renamed.txt").exists())
            self.assertIn("Created:", output)
            self.assertIn("Appended:", output)
            self.assertIn("Modified:", output)
            self.assertIn("Renamed:", output)
            self.assertIn("Deleted:", output)
            backups = list(Path(".subbake/agent/backups").glob("**/renamed.txt"))
            self.assertEqual(len(backups), 1)
            self.assertIn("updated line", backups[0].read_text(encoding="utf-8"))

    def test_agent_can_list_current_project_directory(self) -> None:
        with self._isolated_filesystem():
            Path("visible.txt").write_text("hello\n", encoding="utf-8")
            Path("notes").mkdir()
            Path(".git").mkdir()
            Path(".venv").mkdir()

            result = self.runner.invoke(app, [], input="你能看到当前目录有什么吗\n/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Listing files.", output)
            self.assertIn("visible.txt", output)
            self.assertIn("notes", output)
            self.assertNotIn(".git", output)
            self.assertNotIn(".venv", output)
            self.assertNotIn(".subbake", output)

    def test_agent_search_files_matches_file_names(self) -> None:
        with self._isolated_filesystem():
            Path("The Matrix Revolutions.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello Neo.\n\n",
                encoding="utf-8",
            )

            result = self.runner.invoke(
                app,
                [],
                input="在 @. 搜索 黑客帝国\n/exit\n",
            )
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("The Matrix Revolutions.srt", output)

    def test_agent_file_operations_refuse_paths_outside_project(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            outside = Path.cwd().parent / "outside-agent-test.txt"

            result = self.runner.invoke(
                app,
                [],
                input=f"创建 @{outside} nope\n/exit\n",
            )
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertFalse(outside.exists())
            self.assertIn("outside the project root", output)

    def test_agent_plan_mode_requires_approval_before_mutating(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            result = self.runner.invoke(
                app,
                [],
                input="/plan\n创建 @notes.txt first line\n/approve\n/exit\n",
            )
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertTrue(Path("notes.txt").exists())
            self.assertEqual(Path("notes.txt").read_text(encoding="utf-8"), "first line")
            self.assertIn("<proposed_plan>", output)
            self.assertIn("Use /approve", output)

    def test_agent_reject_discards_pending_plan(self) -> None:
        with self._isolated_filesystem():
            Path("subbake.toml").write_text(
                "[defaults]\n"
                'provider = "mock"\n'
                'model = "mock-zh"\n'
                "final_review = false\n",
                encoding="utf-8",
            )
            result = self.runner.invoke(
                app,
                [],
                input="/plan\n创建 @notes.txt first line\n/reject\n/exit\n",
            )
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertFalse(Path("notes.txt").exists())
            self.assertIn("Pending plan discarded", output)

    def _strip_ansi(self, value: str) -> str:
        return re.sub(r"\x1b\[[0-9;]*m", "", value)
