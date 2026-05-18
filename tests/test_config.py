from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subbake.config import (
    TRANSLATE_CONFIG_KEYS,
    discover_config_path,
    discover_global_config_path,
    discover_project_config_path,
    global_config_candidates,
    load_app_config,
    resolve_command_config,
)


class ConfigDiscoveryTestCase(unittest.TestCase):
    def test_project_config_beats_global_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_root = temp_path / "project"
            nested_dir = project_root / "nested"
            nested_dir.mkdir(parents=True, exist_ok=True)
            project_config = project_root / "subbake.toml"
            project_config.write_text("[defaults]\nprovider = 'mock'\n", encoding="utf-8")

            global_config_root = temp_path / "xdg"
            global_config = global_config_root / "subbake" / "config.toml"
            global_config.parent.mkdir(parents=True, exist_ok=True)
            global_config.write_text("[defaults]\nprovider = 'openai'\n", encoding="utf-8")

            with patch.dict("os.environ", {"XDG_CONFIG_HOME": str(global_config_root)}, clear=False):
                discovered = discover_config_path(nested_dir)

            self.assertEqual(discovered, project_config)
            self.assertEqual(discover_project_config_path(nested_dir), project_config)

    def test_xdg_global_config_is_used_when_project_config_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            global_config_root = temp_path / "xdg"
            global_config = global_config_root / "subbake" / "config.toml"
            global_config.parent.mkdir(parents=True, exist_ok=True)
            global_config.write_text("[defaults]\nprovider = 'mock'\n", encoding="utf-8")

            with patch.dict("os.environ", {"XDG_CONFIG_HOME": str(global_config_root)}, clear=False):
                discovered = discover_config_path(temp_path / "no-project")

            self.assertEqual(discovered, global_config)

    def test_macos_global_config_path_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir) / "home"
            config_path = home_dir / "Library" / "Application Support" / "subbake" / "config.toml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("[defaults]\nprovider = 'mock'\n", encoding="utf-8")

            with (
                patch("pathlib.Path.home", return_value=home_dir),
                patch("subbake.config.sys.platform", "darwin"),
                patch.dict("os.environ", {}, clear=True),
            ):
                discovered = discover_global_config_path()
                candidates = global_config_candidates()

            self.assertEqual(discovered, config_path)
            self.assertEqual(candidates[0], config_path)

    def test_windows_appdata_global_config_path_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            appdata_dir = temp_path / "AppData" / "Roaming"
            config_path = appdata_dir / "subbake" / "config.toml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("[defaults]\nprovider = 'mock'\n", encoding="utf-8")

            with (
                patch("pathlib.Path.home", return_value=temp_path / "home"),
                patch.dict("os.environ", {"APPDATA": str(appdata_dir)}, clear=True),
            ):
                discovered = discover_global_config_path()
                candidates = global_config_candidates()

            self.assertEqual(discovered, config_path)
            self.assertEqual(candidates[0], config_path)

    def test_translate_config_accepts_agent_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "subbake.toml"
            config_path.write_text(
                "[defaults]\n"
                "agent = false\n"
                "agent_repair_attempts = 1\n",
                encoding="utf-8",
            )

            values, _ = resolve_command_config(
                load_app_config(config_path),
                profile=None,
                allowed_keys=TRANSLATE_CONFIG_KEYS,
            )

            self.assertFalse(values["agent"])
            self.assertEqual(values["agent_repair_attempts"], 1)
