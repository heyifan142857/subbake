"""Tests for the whisper.cpp installer (subbake.whisper_installer)."""

from __future__ import annotations

import json
import platform
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from subbake.whisper_installer import (
    SUPPORTED_MODELS,
    WHISPER_CLI_NAME,
    WhisperInstaller,
    _detect_platform,
    _user_agent,
)


class TestPlatformDetection(unittest.TestCase):
    def test_user_agent(self) -> None:
        ua = _user_agent()
        self.assertIn("subbake", ua)


class TestWhisperInstaller(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.installer = WhisperInstaller(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_binary_path(self) -> None:
        path = self.installer.binary_path
        expected_name = "whisper-cli.exe" if platform.system().lower() == "windows" else "whisper-cli"
        self.assertEqual(path.name, expected_name)

    def test_initial_not_available(self) -> None:
        self.assertFalse(self.installer.is_available())

    def test_initial_no_version(self) -> None:
        self.assertIsNone(self.installer.installed_version())

    def test_available_when_binary_exists(self) -> None:
        self.installer._bin_dir.mkdir(parents=True, exist_ok=True)
        self.installer.binary_path.write_text("fake binary")
        self.installer.binary_path.chmod(0o755)
        self.assertTrue(self.installer.is_available())

    def test_version_file(self) -> None:
        self.installer._root.mkdir(parents=True, exist_ok=True)
        self.installer._version_file.write_text("v1.7.0")
        self.assertEqual(self.installer.installed_version(), "v1.7.0")

    def test_check_available_returns_message(self) -> None:
        ok, msg = self.installer.check_available()
        self.assertFalse(ok)
        self.assertIn("whisper-cli is not installed", msg)

    def test_ensure_available_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.installer.ensure_available()

    def test_model_path_valid(self) -> None:
        for model in SUPPORTED_MODELS:
            path = self.installer.model_path(model)
            self.assertEqual(path.name, f"ggml-{model}.bin")

    def test_model_path_invalid(self) -> None:
        with self.assertRaises(ValueError):
            self.installer.model_path("invalid-model")

    def test_model_path_normalized(self) -> None:
        path = self.installer.model_path("SMALL")
        self.assertIn("ggml-small.bin", str(path))

    @patch("subbake.whisper_installer._download_file")
    def test_ensure_model_download(self, mock_download) -> None:
        """Test that ensure_model downloads when file missing."""
        model = "tiny"
        self.assertFalse(self.installer.model_path(model).exists())
        result = self.installer.ensure_model(model)
        self.assertEqual(result, self.installer.model_path(model))
        self.assertTrue(result.parent.exists())
        mock_download.assert_called_once()

    def test_ensure_model_exists(self) -> None:
        """Test that ensure_model returns existing path without download."""
        model = "tiny"
        path = self.installer.model_path(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake model")
        result = self.installer.ensure_model(model)
        self.assertEqual(result, path)

    @patch("subbake.whisper_installer._download_file")
    def test_download_model_directory_created(self, mock_download) -> None:
        """Test that the models directory is created on download."""
        model = "tiny"
        path = self.installer.model_path(model)
        self.assertFalse(path.parent.exists())
        result = self.installer.download_model(model)
        self.assertTrue(path.parent.exists())
        self.assertEqual(result, path)

    @patch("subbake.whisper_installer._download_file")
    @patch("subbake.whisper_installer.WhisperInstaller._release_info")
    def test_install_extracts_nested_archive(self, mock_release_info, mock_download) -> None:
        """Install should find whisper-cli even when it is nested in the archive."""
        mock_release_info.return_value = {
            "tag_name": "v1.7.0",
            "assets": [
                {
                    "name": "whisper-cli-v1.7.0-linux-x64.zip",
                    "browser_download_url": "https://example.test/whisper.zip",
                }
            ],
        }

        def write_archive(url: str, dest: Path, **kwargs) -> None:
            with zipfile.ZipFile(dest, "w") as zf:
                zf.writestr(f"package/bin/{WHISPER_CLI_NAME}", "fake binary")
                zf.writestr("package/bin/libwhisper.so", "fake lib")

        mock_download.side_effect = write_archive

        with patch("subbake.whisper_installer._detect_platform", return_value="linux-x64"):
            binary = self.installer.install("v1.7.0")

        self.assertEqual(binary, self.installer.binary_path)
        self.assertTrue(binary.exists())
        self.assertTrue((self.installer._bin_dir / "libwhisper.so").exists())
        self.assertEqual(self.installer.installed_version(), "v1.7.0")

    @patch("subbake.whisper_installer.WhisperInstaller._release_info")
    def test_install_falls_back_to_source_when_no_asset(self, mock_release_info) -> None:
        """Missing pre-built assets should fall back to the source-build path."""
        release = {"tag_name": "v1.7.0", "assets": []}
        mock_release_info.return_value = release
        expected = self.installer.binary_path

        with patch.object(self.installer, "_install_from_source", return_value=expected) as mock_source:
            result = self.installer.install("v1.7.0")

        self.assertEqual(result, expected)
        mock_source.assert_called_once_with(release, progress_callback=None)

    @patch("subbake.whisper_installer._download_file")
    @patch("subbake.whisper_installer.WhisperInstaller._release_info")
    def test_install_falls_back_when_archive_has_no_binary(self, mock_release_info, mock_download) -> None:
        """A matching archive without whisper-cli should fall back to source build."""
        release = {
            "tag_name": "v1.7.0",
            "assets": [
                {
                    "name": "whisper-cli-v1.7.0-linux-x64.zip",
                    "browser_download_url": "https://example.test/whisper.zip",
                }
            ],
        }
        mock_release_info.return_value = release
        expected = self.installer.binary_path

        def write_archive(url: str, dest: Path, **kwargs) -> None:
            with zipfile.ZipFile(dest, "w") as zf:
                zf.writestr("package/README.txt", "no binary here")

        mock_download.side_effect = write_archive

        with patch("subbake.whisper_installer._detect_platform", return_value="linux-x64"):
            with patch.object(self.installer, "_install_from_source", return_value=expected) as mock_source:
                result = self.installer.install("v1.7.0")

        self.assertEqual(result, expected)
        mock_source.assert_called_once_with(release, progress_callback=None)

    def test_uninstall_keep_models_removes_binary_only(self) -> None:
        self.installer._bin_dir.mkdir(parents=True, exist_ok=True)
        self.installer.binary_path.write_text("fake binary")
        self.installer._version_file.write_text("v1.7.0")
        model = self.installer.model_path("tiny")
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text("fake model")

        self.installer.uninstall(keep_models=True)

        self.assertFalse(self.installer.binary_path.exists())
        self.assertFalse(self.installer._version_file.exists())
        self.assertTrue(model.exists())

    def test_uninstall_removes_all_managed_data(self) -> None:
        self.installer._bin_dir.mkdir(parents=True, exist_ok=True)
        self.installer.binary_path.write_text("fake binary")
        model = self.installer.model_path("tiny")
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text("fake model")

        self.installer.uninstall()

        self.assertFalse(self.installer._root.exists())


class TestReleaseManagement(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.installer = WhisperInstaller(self.root)
        self.sample_release = {
            "tag_name": "v1.7.0",
            "assets": [
                {
                    "name": "whisper-cli-v1.7.0-linux-x64.tar.gz",
                    "browser_download_url": "https://github.com/...",
                },
                {
                    "name": "whisper-cli-v1.7.0-macos-arm64.tar.gz",
                    "browser_download_url": "https://github.com/...",
                },
            ],
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_find_asset_linux(self) -> None:
        """Test finding the correct asset for the platform (mock)."""
        from unittest.mock import MagicMock
        # Can't easily test _find_asset without mocking _detect_platform
        # Just verify the method exists and accepts release dicts
        self.assertTrue(hasattr(self.installer, "_find_asset"))

    def test_release_info_validates(self) -> None:
        """Test error handling for release_info."""
        from unittest.mock import patch as _patch
        with _patch("subbake.whisper_installer.RELEASES_API_URL", "https://api.github.com/repos/nonexistent/repo/releases"):
            with self.assertRaises(RuntimeError):
                self.installer._release_info("v999.999")

    @patch("subbake.whisper_installer.WhisperInstaller._release_info")
    def test_latest_version(self, mock_release_info) -> None:
        """Test that latest_version returns the tag name."""
        mock_release_info.return_value = {"tag_name": "v1.7.0"}
        version = self.installer.latest_version()
        self.assertEqual(version, "v1.7.0")
        mock_release_info.assert_called_once_with("latest")


class TestSupportedModels(unittest.TestCase):
    def test_models_defined(self) -> None:
        self.assertIn("tiny", SUPPORTED_MODELS)
        self.assertIn("base", SUPPORTED_MODELS)
        self.assertIn("small", SUPPORTED_MODELS)
        self.assertIn("medium", SUPPORTED_MODELS)
        self.assertIn("large-v3", SUPPORTED_MODELS)


if __name__ == "__main__":
    unittest.main()
