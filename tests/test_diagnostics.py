from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from subbake.diagnostics import diagnose_path, diagnose_text


class DiagnosticsTestCase(unittest.TestCase):
    def test_failure_json_diagnoses_line_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "translate_batch_0001.json"
            path.write_text(
                json.dumps(
                    {
                        "stage": "translate",
                        "batch_index": 1,
                        "attempts": [
                            {
                                "attempt": 1,
                                "error": "Line count mismatch: expected 3, got 2",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = diagnose_path(path)

        self.assertIn("dropped, inserted, or merged", report.diagnosis)
        self.assertTrue(any("--batch-size" in item for item in report.suggestions))

    def test_pasted_log_diagnoses_missing_credentials(self) -> None:
        report = diagnose_text("Error: Missing API key for OpenAI provider.")

        self.assertIn("credentials", report.diagnosis)
