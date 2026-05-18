from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from subbake.entities import PipelineOptions, Usage
from subbake.models import build_backend
from subbake.models.base_model import BackendErrorMetadata, BackendRequestError, LLMBackend
from subbake.pipeline import SubtitlePipeline
from subbake.storage import build_runtime_paths


class QuietDashboard:
    def __init__(self) -> None:
        self.usage = Usage()
        self.total_steps = 0
        self.completed_steps = 0

    @contextmanager
    def running(self):
        yield self

    def set_total_steps(self, total_steps: int) -> None:
        self.total_steps = total_steps

    def mark_running(self, stage: str, label: str | None = None) -> None:
        _ = (stage, label)

    def mark_done(self, stage: str, advance: bool = True) -> None:
        _ = stage
        if advance:
            self.completed_steps += 1

    def mark_skipped(self, stage: str) -> None:
        _ = stage
        self.completed_steps += 1

    def add_usage(self, usage: Usage) -> None:
        self.usage.add(usage)

    def restore_usage(self, usage: Usage) -> None:
        self.usage = Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    def restore_progress(self, completed_steps: int) -> None:
        self.completed_steps = completed_steps

    def restore_stage_progress(
        self,
        *,
        translation_batches_completed: int,
        total_translation_batches: int,
        review_batches_completed: int,
        review_batches: int,
        validation_completed: bool,
    ) -> None:
        _ = (
            translation_batches_completed,
            total_translation_batches,
            review_batches_completed,
            review_batches,
            validation_completed,
        )

    def set_batch(self, index: int, total: int, latency_seconds: float, stage_label: str) -> None:
        _ = (index, total, latency_seconds, stage_label)

    def clear_batch(self) -> None:
        return


class RecordingDashboard(QuietDashboard):
    def __init__(self) -> None:
        super().__init__()
        self.agent_events: list[dict] = []

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
        self.agent_events.append(
            {
                "stage": stage,
                "batch_index": batch_index,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "status": status,
                "error": error,
                "log_path": log_path,
            }
        )


class ScriptedBackend(LLMBackend):
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.call_count = 0

    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        self.call_count += 1
        if self.fail_on_call is not None and self.call_count == self.fail_on_call:
            raise RuntimeError("Injected backend failure.")

        prompt = "\n".join(message["content"] for message in messages)
        batch_payload = json.loads(self._extract_between(prompt, "BATCH_JSON_START", "BATCH_JSON_END"))
        return (
            {
                "lines": [
                    {
                        "id": item["id"],
                        "translation": "" if not item["text"].strip() else f"[SCRIPTED] {item['text']}",
                    }
                    for item in batch_payload["lines"]
                ],
                "summary": f"batch-{self.call_count}",
                "glossary_updates": [],
            },
            Usage(input_tokens=10, output_tokens=10, total_tokens=20),
        )

    def check_credentials(self) -> tuple[bool, str]:
        return True, "ok"

    def _extract_between(self, text: str, start_marker: str, end_marker: str) -> str:
        start_index = text.index(start_marker) + len(start_marker)
        end_index = text.index(end_marker, start_index)
        return text[start_index:end_index].strip()


class FailingBackend(LLMBackend):
    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        raise AssertionError("Backend should not be called.")

    def check_credentials(self) -> tuple[bool, str]:
        return True, "ok"


class StructuralFailureBackend(LLMBackend):
    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        prompt = "\n".join(message["content"] for message in messages)
        batch_payload = json.loads(self._extract_between(prompt, "BATCH_JSON_START", "BATCH_JSON_END"))
        lines = batch_payload["lines"]
        batch_size = len(lines)
        self.call_sizes.append(batch_size)

        if batch_size >= 4:
            return (
                {
                    "lines": [
                        {"id": item["id"], "translation": f"[SPLIT] {item['text']}"}
                        for item in lines[:-1]
                    ],
                    "summary": f"invalid-count-{batch_size}",
                    "glossary_updates": [],
                },
                Usage(input_tokens=5, output_tokens=5, total_tokens=10),
            )
        if batch_size >= 2:
            return (
                {
                    "lines": [
                        {
                            "id": item["id"],
                            "translation": "" if index == 0 else f"[SPLIT] {item['text']}",
                        }
                        for index, item in enumerate(lines)
                    ],
                    "summary": f"invalid-empty-{batch_size}",
                    "glossary_updates": [],
                },
                Usage(input_tokens=5, output_tokens=5, total_tokens=10),
            )
        return (
            {
                "lines": [
                    {"id": item["id"], "translation": f"[SPLIT] {item['text']}"}
                    for item in lines
                ],
                "summary": f"ok-{batch_size}",
                "glossary_updates": [],
            },
            Usage(input_tokens=5, output_tokens=5, total_tokens=10),
        )

    def check_credentials(self) -> tuple[bool, str]:
        return True, "ok"

    def _extract_between(self, text: str, start_marker: str, end_marker: str) -> str:
        start_index = text.index(start_marker) + len(start_marker)
        end_index = text.index(end_marker, start_index)
        return text[start_index:end_index].strip()


class AlwaysMissingLineBackend(LLMBackend):
    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        prompt = "\n".join(message["content"] for message in messages)
        batch_payload = json.loads(self._extract_between(prompt, "BATCH_JSON_START", "BATCH_JSON_END"))
        lines = batch_payload["lines"]
        return (
            {
                "lines": [
                    {"id": item["id"], "translation": f"[BROKEN] {item['text']}"}
                    for item in lines[:-1]
                ],
                "summary": "broken",
                "glossary_updates": [],
            },
            Usage(input_tokens=5, output_tokens=5, total_tokens=10),
        )

    def check_credentials(self) -> tuple[bool, str]:
        return True, "ok"

    def _extract_between(self, text: str, start_marker: str, end_marker: str) -> str:
        start_index = text.index(start_marker) + len(start_marker)
        end_index = text.index(end_marker, start_index)
        return text[start_index:end_index].strip()


class AlwaysAttributeErrorBackend(LLMBackend):
    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        _ = messages
        raise AttributeError("'str' object has no attribute 'get'")

    def check_credentials(self) -> tuple[bool, str]:
        return True, "ok"


class FastModeBestEffortBackend(LLMBackend):
    def __init__(self) -> None:
        self.call_count = 0

    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        self.call_count += 1
        prompt = "\n".join(message["content"] for message in messages)
        batch_payload = json.loads(self._extract_between(prompt, "BATCH_JSON_START", "BATCH_JSON_END"))
        first_line = batch_payload["lines"][0]
        return (
            {
                "lines": [f"[FAST] {first_line['text']}"],
                "summary": "fast-mode",
                "glossary_updates": [],
            },
            Usage(input_tokens=4, output_tokens=4, total_tokens=8),
        )

    def check_credentials(self) -> tuple[bool, str]:
        return True, "ok"

    def _extract_between(self, text: str, start_marker: str, end_marker: str) -> str:
        start_index = text.index(start_marker) + len(start_marker)
        end_index = text.index(end_marker, start_index)
        return text[start_index:end_index].strip()


class AgentRepairTranslationBackend(LLMBackend):
    def __init__(self, repair_succeeds: bool = True) -> None:
        self.repair_succeeds = repair_succeeds
        self.tasks: list[str] = []

    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        prompt = "\n".join(message["content"] for message in messages)
        task = self._extract_between(prompt, "TASK_START", "TASK_END")
        self.tasks.append(task)
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
                Usage(input_tokens=3, output_tokens=3, total_tokens=6),
            )
        if task == "agent_repair_translation":
            payload = json.loads(self._extract_between(prompt, "AGENT_REPAIR_JSON_START", "AGENT_REPAIR_JSON_END"))
            return (
                {
                    "lines": [
                        {
                            "id": item["id"],
                            "translation": "" if not self.repair_succeeds else f"[AGENT] {item['text']}",
                        }
                        for item in payload["source_lines"]
                    ],
                    "summary": "agent repaired",
                    "glossary_updates": [],
                },
                Usage(input_tokens=7, output_tokens=7, total_tokens=14),
            )
        raise RuntimeError(f"Unexpected task: {task}")

    def check_credentials(self) -> tuple[bool, str]:
        return True, "ok"

    def _extract_between(self, text: str, start_marker: str, end_marker: str) -> str:
        start_index = text.index(start_marker) + len(start_marker)
        end_index = text.index(end_marker, start_index)
        return text[start_index:end_index].strip()


class AgentRepairReviewBackend(LLMBackend):
    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        prompt = "\n".join(message["content"] for message in messages)
        task = self._extract_between(prompt, "TASK_START", "TASK_END")
        if task == "translate_subtitles":
            payload = json.loads(self._extract_between(prompt, "BATCH_JSON_START", "BATCH_JSON_END"))
            return (
                {
                    "lines": [
                        {"id": item["id"], "translation": f"[SCRIPTED] {item['text']}"}
                        for item in payload["lines"]
                    ],
                    "summary": "translated",
                    "glossary_updates": [],
                },
                Usage(input_tokens=5, output_tokens=5, total_tokens=10),
            )
        if task == "review_translations":
            payload = json.loads(self._extract_between(prompt, "REVIEW_JSON_START", "REVIEW_JSON_END"))
            return (
                {
                    "lines": [
                        {"id": item["id"], "translation": ""}
                        for item in payload["lines"]
                    ],
                    "review_notes": "broken",
                },
                Usage(input_tokens=5, output_tokens=5, total_tokens=10),
            )
        if task == "agent_repair_review":
            payload = json.loads(self._extract_between(prompt, "AGENT_REPAIR_JSON_START", "AGENT_REPAIR_JSON_END"))
            current_by_id = {
                item["id"]: item["translation"]
                for item in payload["current_translations"]
            }
            return (
                {
                    "lines": [
                        {"id": item["id"], "translation": current_by_id[item["id"]]}
                        for item in payload["source_lines"]
                    ],
                    "review_notes": "agent repaired",
                },
                Usage(input_tokens=7, output_tokens=7, total_tokens=14),
            )
        raise RuntimeError(f"Unexpected task: {task}")

    def check_credentials(self) -> tuple[bool, str]:
        return True, "ok"

    def _extract_between(self, text: str, start_marker: str, end_marker: str) -> str:
        start_index = text.index(start_marker) + len(start_marker)
        end_index = text.index(end_marker, start_index)
        return text[start_index:end_index].strip()


class BackendRequestFailureBackend(LLMBackend):
    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        _ = messages
        raise BackendRequestError(
            "provider failed",
            metadata=BackendErrorMetadata(provider="test", retryable=False, status_code=401),
        )

    def check_credentials(self) -> tuple[bool, str]:
        return True, "ok"


class PipelineTestCase(unittest.TestCase):
    def test_dry_run_returns_batch_plan_without_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "sample.txt"
            input_path.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")

            options = PipelineOptions(
                input_path=input_path,
                batch_size=2,
                dry_run=True,
                work_dir=temp_path / "runtime",
            )
            pipeline = SubtitlePipeline(
                backend=None,
                options=options,
                dashboard=QuietDashboard(),
            )

            result = pipeline.run()

            self.assertTrue(result.dry_run)
            self.assertEqual(
                [(entry.index, entry.size, entry.first_id, entry.last_id) for entry in result.planned_batches],
                [(1, 2, "1", "2"), (2, 2, "3", "4"), (3, 1, "5", "5")],
            )
            self.assertIsNone(result.output_path)
            self.assertFalse((temp_path / "sample.translated.txt").exists())
            self.assertFalse(result.state_path.exists())

    def test_mock_translation_writes_output_and_runtime_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "episode.txt"
            input_path.write_text("Hello Alice.\nDamn it.\nMove.\n", encoding="utf-8")

            options = PipelineOptions(
                input_path=input_path,
                provider="mock",
                model="mock-zh",
                batch_size=2,
                final_review=True,
                work_dir=temp_path / "runtime",
            )
            pipeline = SubtitlePipeline(
                backend=build_backend("mock", "mock-zh"),
                options=options,
                dashboard=QuietDashboard(),
            )

            result = pipeline.run()

            self.assertEqual(result.batches_translated, 2)
            self.assertEqual(result.review_batches, 1)
            self.assertTrue(result.output_path.exists())
            self.assertTrue(result.state_path.exists())
            self.assertTrue(result.glossary_path.exists())
            self.assertGreater(result.usage.total_tokens, 0)

            output_text = result.output_path.read_text(encoding="utf-8")
            self.assertIn("[MOCK-ZH] Hello Alice.", output_text)
            self.assertIn("[MOCK-ZH] Damn it.", output_text)

            glossary = json.loads(result.glossary_path.read_text(encoding="utf-8"))
            self.assertEqual(glossary["Alice"], "Alice")

            state = json.loads(result.state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["translation_batches_completed"], 2)
            self.assertEqual(state["review_batches_completed"], 1)
            self.assertTrue(state["validation_completed"])
            self.assertNotIn("translated_segments", state)
            self.assertNotIn("reviewed_segments", state)

            translated_shards = sorted(result.state_path.parent.joinpath("translated_batches").glob("*.json"))
            reviewed_shards = sorted(result.state_path.parent.joinpath("reviewed_batches").glob("*.json"))
            self.assertEqual(len(translated_shards), 2)
            self.assertEqual(len(reviewed_shards), 1)

    def test_resume_restores_from_incremental_batch_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "resume.txt"
            input_path.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
            work_dir = temp_path / "runtime"

            first_pipeline = SubtitlePipeline(
                backend=ScriptedBackend(fail_on_call=2),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=2,
                    final_review=False,
                    retries=0,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            )

            with self.assertRaises(RuntimeError):
                first_pipeline.run()

            state_files = list(work_dir.glob("runs/*/run_state.json"))
            self.assertEqual(len(state_files), 1)
            state = json.loads(state_files[0].read_text(encoding="utf-8"))
            self.assertEqual(state["translation_batches_completed"], 1)
            self.assertFalse(state["validation_completed"])

            second_backend = ScriptedBackend()
            second_pipeline = SubtitlePipeline(
                backend=second_backend,
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=2,
                    final_review=False,
                    retries=0,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            )

            result = second_pipeline.run()

            self.assertEqual(second_backend.call_count, 2)
            self.assertEqual(result.batches_translated, 3)
            self.assertEqual(result.review_batches, 0)
            self.assertEqual(result.resumed_translation_batches, 1)
            self.assertEqual(result.resumed_review_batches, 0)
            self.assertEqual(
                result.output_path.read_text(encoding="utf-8"),
                "[SCRIPTED] one\n[SCRIPTED] two\n[SCRIPTED] three\n[SCRIPTED] four\n[SCRIPTED] five\n",
            )
            translated_shards = sorted(result.state_path.parent.joinpath("translated_batches").glob("*.json"))
            self.assertEqual(len(translated_shards), 3)

    def test_smart_batching_splits_large_dialogue_before_hard_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "long.txt"
            input_path.write_text(
                "\n".join([f"{'A' * 140}." for _ in range(15)]) + "\n",
                encoding="utf-8",
            )

            pipeline = SubtitlePipeline(
                backend=None,
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=50,
                    dry_run=True,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=QuietDashboard(),
            )

            result = pipeline.run()

            self.assertGreater(len(result.planned_batches), 1)
            self.assertLess(max(entry.size for entry in result.planned_batches), 15)

    def test_high_risk_fragment_batching_shrinks_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "fragments.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "i thought",
                        "we could still",
                        "make it out",
                        "before dawn",
                    ]
                    * 5
                )
                + "\n",
                encoding="utf-8",
            )

            pipeline = SubtitlePipeline(
                backend=None,
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=30,
                    dry_run=True,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=QuietDashboard(),
            )

            result = pipeline.run()

            self.assertGreater(len(result.planned_batches), 1)
            self.assertLessEqual(max(entry.size for entry in result.planned_batches), 9)

    def test_structural_translation_failures_trigger_recursive_split_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "split.txt"
            input_path.write_text("Alpha.\nBravo.\nCharlie.\nDelta.\n", encoding="utf-8")
            backend = StructuralFailureBackend()

            result = SubtitlePipeline(
                backend=backend,
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=8,
                    final_review=False,
                    retries=0,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=QuietDashboard(),
            ).run()

            self.assertEqual(
                result.output_path.read_text(encoding="utf-8"),
                "[SPLIT] Alpha.\n[SPLIT] Bravo.\n[SPLIT] Charlie.\n[SPLIT] Delta.\n",
            )
            self.assertEqual(backend.call_sizes, [4, 2, 1, 1, 2, 1, 1])

    def test_translation_failure_message_explains_missing_lines_and_smaller_batch_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "broken.txt"
            input_path.write_text("Alpha.\nBravo.\n", encoding="utf-8")

            pipeline = SubtitlePipeline(
                backend=AlwaysMissingLineBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=50,
                    final_review=False,
                    retries=0,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=QuietDashboard(),
            )

            with self.assertRaises(RuntimeError) as context:
                pipeline.run()

            message = str(context.exception)
            self.assertIn("Model output is missing subtitle entries or merged neighboring lines.", message)
            self.assertIn("Try rerunning with a smaller --batch-size", message)
            self.assertIn("--batch-size 25", message)
            self.assertIn("--batch-size 15", message)
            self.assertIn("\nFailure sample saved to:\n", message)

    def test_translation_failure_message_puts_failure_sample_on_new_line_for_generic_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "generic.txt"
            input_path.write_text("Alpha.\n", encoding="utf-8")

            pipeline = SubtitlePipeline(
                backend=AlwaysAttributeErrorBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=30,
                    final_review=False,
                    retries=0,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=QuietDashboard(),
            )

            with self.assertRaises(RuntimeError) as context:
                pipeline.run()

            message = str(context.exception)
            self.assertIn("Last error: 'str' object has no attribute 'get'.", message)
            self.assertIn("\nFailure sample saved to:\n", message)
            self.assertNotIn("'get' Failure sample saved to", message)

    def test_failure_sample_persists_attempt_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "failure.txt"
            input_path.write_text("Alpha.\n", encoding="utf-8")
            work_dir = temp_path / "runtime"

            pipeline = SubtitlePipeline(
                backend=AlwaysAttributeErrorBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=30,
                    final_review=False,
                    retries=1,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            )

            with self.assertRaises(RuntimeError):
                pipeline.run()

            runtime = build_runtime_paths(input_path=input_path, work_dir=work_dir, glossary_path=None)
            failure_path = runtime.failures_dir / "translate_batch_0001.json"
            failure = json.loads(failure_path.read_text(encoding="utf-8"))

            self.assertEqual(failure["stage"], "translate")
            self.assertEqual(failure["batch_index"], 1)
            self.assertEqual(len(failure["attempts"]), 2)
            self.assertEqual(failure["attempts"][0]["error"], "'str' object has no attribute 'get'")
            self.assertTrue(failure["attempts"][0]["messages"])

    def test_agent_repair_is_enabled_by_default_and_continues_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "agent.txt"
            input_path.write_text("Alpha.\n", encoding="utf-8")
            dashboard = RecordingDashboard()

            result = SubtitlePipeline(
                backend=AgentRepairTranslationBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=1,
                    final_review=False,
                    retries=0,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=dashboard,
            ).run()

            self.assertEqual(result.output_path.read_text(encoding="utf-8"), "[AGENT] Alpha.\n")
            self.assertEqual(len(result.agent_repairs), 1)
            self.assertTrue(result.agent_repairs[0].success)
            self.assertEqual(result.agent_repairs[0].stage, "translate")
            self.assertTrue(result.agent_repairs[0].log_path.exists())
            self.assertIn("running", [event["status"] for event in dashboard.agent_events])
            self.assertIn("repaired", [event["status"] for event in dashboard.agent_events])

            agent_log = json.loads(result.agent_repairs[0].log_path.read_text(encoding="utf-8"))
            self.assertTrue(agent_log["success"])
            self.assertEqual(agent_log["stage"], "translate")

    def test_no_agent_keeps_original_failure_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "agent-off.txt"
            input_path.write_text("Alpha.\n", encoding="utf-8")
            work_dir = temp_path / "runtime"

            pipeline = SubtitlePipeline(
                backend=AgentRepairTranslationBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=1,
                    final_review=False,
                    retries=0,
                    agent=False,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            )

            with self.assertRaises(RuntimeError) as context:
                pipeline.run()

            self.assertIn("Failure sample saved to:", str(context.exception))
            self.assertEqual(pipeline.agent_repairs, [])
            runtime = build_runtime_paths(input_path=input_path, work_dir=work_dir, glossary_path=None)
            self.assertFalse(runtime.agent_logs_dir.exists())

    def test_agent_failure_records_attempts_and_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "agent-fails.txt"
            input_path.write_text("Alpha.\n", encoding="utf-8")
            work_dir = temp_path / "runtime"

            with self.assertRaises(RuntimeError) as context:
                SubtitlePipeline(
                    backend=AgentRepairTranslationBackend(repair_succeeds=False),
                    options=PipelineOptions(
                        input_path=input_path,
                        batch_size=1,
                        final_review=False,
                        retries=0,
                        agent_repair_attempts=2,
                        work_dir=work_dir,
                    ),
                    dashboard=QuietDashboard(),
                ).run()

            message = str(context.exception)
            self.assertIn("Agent repair failed after 2 attempts.", message)
            self.assertIn("Agent log saved to:", message)

            runtime = build_runtime_paths(input_path=input_path, work_dir=work_dir, glossary_path=None)
            failure = json.loads((runtime.failures_dir / "translate_batch_0001.json").read_text(encoding="utf-8"))
            self.assertEqual(len(failure["agent_attempts"]), 2)

            agent_log = json.loads((runtime.agent_logs_dir / "translate_batch_0001.json").read_text(encoding="utf-8"))
            self.assertFalse(agent_log["success"])
            self.assertEqual(len(agent_log["attempts"]), 2)

    def test_backend_request_errors_do_not_trigger_agent_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "backend-error.txt"
            input_path.write_text("Alpha.\n", encoding="utf-8")
            work_dir = temp_path / "runtime"
            pipeline = SubtitlePipeline(
                backend=BackendRequestFailureBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=1,
                    final_review=False,
                    retries=0,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            )

            with self.assertRaises(RuntimeError):
                pipeline.run()

            self.assertEqual(pipeline.agent_repairs, [])
            runtime = build_runtime_paths(input_path=input_path, work_dir=work_dir, glossary_path=None)
            self.assertFalse(runtime.agent_logs_dir.exists())

    def test_agent_can_repair_review_output_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "review-agent.txt"
            input_path.write_text("Meet Alice now.\n", encoding="utf-8")

            result = SubtitlePipeline(
                backend=AgentRepairReviewBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=1,
                    final_review=True,
                    retries=0,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=QuietDashboard(),
            ).run()

            self.assertEqual(result.review_batches, 1)
            self.assertEqual(result.output_path.read_text(encoding="utf-8"), "[SCRIPTED] Meet Alice now.\n")
            self.assertEqual(len(result.agent_repairs), 1)
            self.assertEqual(result.agent_repairs[0].stage, "review")

    def test_cache_hit_reuses_review_response_without_backend_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "cache.txt"
            input_path.write_text("Hello Alice.\nMove.\n", encoding="utf-8")
            work_dir = temp_path / "runtime"

            first_result = SubtitlePipeline(
                backend=ScriptedBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=10,
                    final_review=False,
                    resume=False,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            ).run()

            second_pipeline = SubtitlePipeline(
                backend=FailingBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=10,
                    final_review=False,
                    resume=False,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            )
            second_pipeline._lookup_translation_memory = lambda batch_segments: {}
            second_result = second_pipeline.run()

            self.assertEqual(first_result.review_batches, 0)
            self.assertGreaterEqual(second_result.cache_hits, 1)
            self.assertEqual(second_result.translation_memory_hits, 0)
            self.assertEqual(
                second_result.output_path.read_text(encoding="utf-8"),
                "[SCRIPTED] Hello Alice.\n[SCRIPTED] Move.\n",
            )

    def test_bilingual_render_reuses_existing_translation_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "render.txt"
            input_path.write_text("hello\nworld\n", encoding="utf-8")
            work_dir = temp_path / "runtime"

            first_result = SubtitlePipeline(
                backend=ScriptedBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=2,
                    bilingual=False,
                    final_review=False,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            ).run()

            second_result = SubtitlePipeline(
                backend=FailingBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=2,
                    bilingual=True,
                    final_review=False,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            ).run()

            self.assertEqual(first_result.batches_translated, 1)
            self.assertEqual(second_result.batches_translated, 1)
            self.assertEqual(second_result.resumed_translation_batches, 1)
            self.assertEqual(
                second_result.output_path.read_text(encoding="utf-8"),
                "hello\n[SCRIPTED] hello\nworld\n[SCRIPTED] world\n",
            )

    def test_bilingual_srt_output_stacks_source_and_translation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "dialogue.srt"
            input_path.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello there.\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\nMove.\n",
                encoding="utf-8",
            )

            result = SubtitlePipeline(
                backend=ScriptedBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=2,
                    bilingual=True,
                    final_review=False,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=QuietDashboard(),
            ).run()

            self.assertEqual(
                result.output_path.read_text(encoding="utf-8"),
                "1\n00:00:01,000 --> 00:00:02,000\nHello there.\n[SCRIPTED] Hello there.\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\nMove.\n[SCRIPTED] Move.\n",
            )

    def test_bilingual_vtt_output_preserves_passthrough_blocks_and_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "dialogue.vtt"
            input_path.write_text(
                "WEBVTT\n\n"
                "NOTE opening note\n\n"
                "intro\n"
                "00:00:01.000 --> 00:00:03.000 line:90%\n"
                "Hello there.\n",
                encoding="utf-8",
            )

            result = SubtitlePipeline(
                backend=ScriptedBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=2,
                    bilingual=True,
                    final_review=False,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=QuietDashboard(),
            ).run()

            self.assertEqual(
                result.output_path.read_text(encoding="utf-8"),
                "WEBVTT\n\n"
                "NOTE opening note\n\n"
                "intro\n"
                "00:00:01.000 --> 00:00:03.000 line:90%\n"
                "Hello there.\n"
                "[SCRIPTED] Hello there.\n",
            )

    def test_translation_memory_reuses_lines_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            work_dir = temp_path / "runtime"
            first_input = temp_path / "episode1.txt"
            second_input = temp_path / "episode2.txt"
            first_input.write_text("Same line\nAnother line\n", encoding="utf-8")
            second_input.write_text("Same line\nAnother line\n", encoding="utf-8")

            first_backend = ScriptedBackend()
            first_result = SubtitlePipeline(
                backend=first_backend,
                options=PipelineOptions(
                    input_path=first_input,
                    batch_size=2,
                    final_review=False,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            ).run()

            second_backend = FailingBackend()
            second_result = SubtitlePipeline(
                backend=second_backend,
                options=PipelineOptions(
                    input_path=second_input,
                    batch_size=2,
                    final_review=False,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            ).run()

            self.assertEqual(first_backend.call_count, 1)
            self.assertEqual(second_result.translation_memory_hits, 2)
            self.assertEqual(
                second_result.output_path.read_text(encoding="utf-8"),
                "[SCRIPTED] Same line\n[SCRIPTED] Another line\n",
            )
            tm_path = build_runtime_paths(
                input_path=first_input,
                work_dir=work_dir,
                glossary_path=None,
            ).translation_memory_path
            tm_data = json.loads(tm_path.read_text(encoding="utf-8"))
            self.assertIn("same line", tm_data)
            self.assertIn("another line", tm_data)

    def test_fast_translation_memory_does_not_reuse_in_normal_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            work_dir = temp_path / "runtime"
            first_input = temp_path / "episode-fast.txt"
            second_input = temp_path / "episode-normal.txt"
            first_input.write_text("Same line\n", encoding="utf-8")
            second_input.write_text("Same line\n", encoding="utf-8")

            fast_result = SubtitlePipeline(
                backend=ScriptedBackend(),
                options=PipelineOptions(
                    input_path=first_input,
                    batch_size=2,
                    fast_mode=True,
                    final_review=False,
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            ).run()

            with self.assertRaises(RuntimeError):
                SubtitlePipeline(
                    backend=FailingBackend(),
                    options=PipelineOptions(
                        input_path=second_input,
                        batch_size=2,
                        fast_mode=False,
                        final_review=False,
                        work_dir=work_dir,
                    ),
                    dashboard=QuietDashboard(),
                ).run()

            fast_tm_path = build_runtime_paths(
                input_path=first_input,
                work_dir=work_dir,
                glossary_path=None,
                fast_mode=True,
            ).translation_memory_path
            normal_tm_path = build_runtime_paths(
                input_path=first_input,
                work_dir=work_dir,
                glossary_path=None,
                fast_mode=False,
            ).translation_memory_path

            self.assertTrue(fast_tm_path.exists())
            self.assertFalse(normal_tm_path.exists())
            self.assertIn("[SCRIPTED] Same line", fast_result.output_path.read_text(encoding="utf-8"))

    def test_fast_mode_prefers_best_effort_completion_and_skips_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "fast.txt"
            input_path.write_text("Alpha.\nBravo.\n", encoding="utf-8")
            dashboard = QuietDashboard()
            backend = FastModeBestEffortBackend()

            result = SubtitlePipeline(
                backend=backend,
                options=PipelineOptions(
                    input_path=input_path,
                    batch_size=8,
                    fast_mode=True,
                    final_review=True,
                    retries=0,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=dashboard,
            ).run()

            self.assertEqual(backend.call_count, 1)
            self.assertEqual(result.review_batches, 0)
            self.assertEqual(dashboard.total_steps, 6)
            self.assertEqual(dashboard.completed_steps, dashboard.total_steps)
            self.assertEqual(
                result.output_path.read_text(encoding="utf-8"),
                "[FAST] Alpha.\nBravo.\n",
            )

    def test_target_language_alias_avoids_cross_language_translation_memory_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            work_dir = temp_path / "runtime"
            first_input = temp_path / "episode-zh.txt"
            second_input = temp_path / "episode-en.txt"
            first_input.write_text("Same line\n", encoding="utf-8")
            second_input.write_text("Same line\n", encoding="utf-8")

            first_result = SubtitlePipeline(
                backend=build_backend("mock", "mock-zh"),
                options=PipelineOptions(
                    input_path=first_input,
                    batch_size=2,
                    final_review=False,
                    target_language="zh",
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            ).run()

            second_result = SubtitlePipeline(
                backend=build_backend("mock", "mock-zh"),
                options=PipelineOptions(
                    input_path=second_input,
                    batch_size=2,
                    final_review=False,
                    target_language="en",
                    work_dir=work_dir,
                ),
                dashboard=QuietDashboard(),
            ).run()

            self.assertIn("[MOCK-ZH] Same line", first_result.output_path.read_text(encoding="utf-8"))
            self.assertIn("[MOCK-EN] Same line", second_result.output_path.read_text(encoding="utf-8"))

    def test_output_format_can_convert_srt_to_txt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "episode.srt"
            input_path.write_text(
                "1\n"
                "00:00:01,000 --> 00:00:03,000\n"
                "Hello there.\n\n"
                "2\n"
                "00:00:04,000 --> 00:00:06,000\n"
                "General Kenobi.\n",
                encoding="utf-8",
            )

            result = SubtitlePipeline(
                backend=ScriptedBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    output_format="txt",
                    batch_size=2,
                    final_review=False,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=QuietDashboard(),
            ).run()

            self.assertEqual(result.output_path.name, "episode.translated.txt")
            self.assertEqual(
                result.output_path.read_text(encoding="utf-8"),
                "[SCRIPTED] Hello there.\n[SCRIPTED] General Kenobi.\n",
            )

    def test_output_path_suffix_can_infer_output_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "episode.srt"
            output_path = temp_path / "exports" / "episode.en.txt"
            input_path.write_text(
                "1\n"
                "00:00:01,000 --> 00:00:03,000\n"
                "Hello there.\n",
                encoding="utf-8",
            )

            result = SubtitlePipeline(
                backend=ScriptedBackend(),
                options=PipelineOptions(
                    input_path=input_path,
                    output_path=output_path,
                    batch_size=2,
                    final_review=False,
                    work_dir=temp_path / "runtime",
                ),
                dashboard=QuietDashboard(),
            ).run()

            self.assertEqual(result.output_path, output_path)
            self.assertEqual(
                result.output_path.read_text(encoding="utf-8"),
                "[SCRIPTED] Hello there.\n",
            )

    def test_txt_input_cannot_convert_to_timed_output_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "episode.txt"
            input_path.write_text("Hello there.\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no timing information"):
                SubtitlePipeline(
                    backend=FailingBackend(),
                    options=PipelineOptions(
                        input_path=input_path,
                        output_format="srt",
                        final_review=False,
                        work_dir=temp_path / "runtime",
                    ),
                    dashboard=QuietDashboard(),
                )
