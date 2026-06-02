from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from subbake import __version__
from subbake.app import app
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


class CLITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

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
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(app, [], input="/exit\n")
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("SubBake agent", output)
            self.assertIn("sbake[", output)

    def test_agent_model_command_switches_config_profile(self) -> None:
        with self.runner.isolated_filesystem():
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

    def test_resume_command_starts_agent_when_no_session_exists(self) -> None:
        with self.runner.isolated_filesystem():
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
        with self.runner.isolated_filesystem():
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
        with self.runner.isolated_filesystem():
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
        with self.runner.isolated_filesystem():
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
        with self.runner.isolated_filesystem():
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
        with self.runner.isolated_filesystem():
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
        with self.runner.isolated_filesystem():
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
        with self.runner.isolated_filesystem():
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
        with self.runner.isolated_filesystem():
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
        with self.runner.isolated_filesystem():
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

            result = self.runner.invoke(app, [], input="@season\n/exit\n")

            self.assertEqual(result.exit_code, 0)
            self.assertIn("[MOCK-ZH] hello", (season / "episode1.translated.txt").read_text(encoding="utf-8"))

    def test_agent_edit_generated_subtitle_creates_backup(self) -> None:
        with self.runner.isolated_filesystem():
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
                input="/edit @clip.translated.txt keep the translation unchanged\n/exit\n",
            )
            output = self._strip_ansi(result.stdout)

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Edited:", output)
            self.assertEqual(Path("clip.translated.txt").read_text(encoding="utf-8"), "[MOCK-ZH] hello\n")
            backups = list(Path(".subbake/agent/backups").glob("*/clip.translated.txt"))
            self.assertEqual(len(backups), 1)

    def _strip_ansi(self, value: str) -> str:
        return re.sub(r"\x1b\[[0-9;]*m", "", value)
