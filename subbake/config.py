from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import sys
import tomllib
from typing import Any

CONFIG_FILENAMES = ("subbake.toml", ".subbake.toml")
COMMON_CONFIG_KEYS = {
    "provider",
    "model",
    "api_key",
    "api_key_env",
    "base_url",
    "timeout",
}
TRANSLATE_CONFIG_KEYS = COMMON_CONFIG_KEYS | {
    "output_format",
    "batch_size",
    "fast",
    "bilingual",
    "source_language",
    "target_language",
    "retries",
    "final_review",
    "dry_run",
    "resume",
    "cache",
    "agent",
    "agent_repair_attempts",
    "work_dir",
    "glossary_path",
}
CHECK_KEY_CONFIG_KEYS = set(COMMON_CONFIG_KEYS)
_PATH_KEYS = {"work_dir", "glossary_path"}
_BOOL_KEYS = {"fast", "bilingual", "final_review", "dry_run", "resume", "cache", "agent"}
_INT_KEYS = {"batch_size", "retries", "agent_repair_attempts"}
_FLOAT_KEYS = {"timeout"}
_STRING_KEYS = (
    TRANSLATE_CONFIG_KEYS
    | CHECK_KEY_CONFIG_KEYS
    | {"default_profile"}
) - _PATH_KEYS - _BOOL_KEYS - _INT_KEYS - _FLOAT_KEYS
_RESERVED_TOP_LEVEL_KEYS = {"default_profile", "defaults", "profiles"}


@dataclass(slots=True)
class ConfigSelection:
    path: Path
    profile: str | None = None


@dataclass(slots=True)
class AppConfig:
    path: Path
    defaults: dict[str, Any] = field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_profile: str | None = None


def discover_config_path(search_from: Path | None = None) -> Path | None:
    project_path = discover_project_config_path(search_from)
    if project_path is not None:
        return project_path
    return discover_global_config_path()


def discover_project_config_path(search_from: Path | None = None) -> Path | None:
    current = (search_from or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        for filename in CONFIG_FILENAMES:
            candidate = directory / filename
            if candidate.exists():
                return candidate
    return None


def discover_global_config_path() -> Path | None:
    for candidate in global_config_candidates():
        if candidate.exists():
            return candidate
    return None


def global_config_candidates() -> list[Path]:
    home_dir = Path.home()
    candidates: list[Path] = []

    xdg_config_home = os.getenv("XDG_CONFIG_HOME")
    if xdg_config_home:
        candidates.append(Path(xdg_config_home) / "subbake" / "config.toml")

    appdata_dir = os.getenv("APPDATA")
    if appdata_dir:
        candidates.append(Path(appdata_dir) / "subbake" / "config.toml")

    if sys.platform == "darwin":
        candidates.append(home_dir / "Library" / "Application Support" / "subbake" / "config.toml")

    candidates.append(home_dir / ".config" / "subbake" / "config.toml")
    candidates.append(home_dir / ".subbake.toml")
    return _dedupe_paths(candidates)


def load_app_config(path: Path) -> AppConfig:
    resolved_path = path.resolve()
    with resolved_path.open("rb") as handle:
        data = tomllib.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file: {resolved_path}")

    unknown_top_level = set(data) - _RESERVED_TOP_LEVEL_KEYS
    if unknown_top_level:
        unknown = ", ".join(sorted(unknown_top_level))
        raise ValueError(
            f"Unsupported top-level config keys in {resolved_path.name}: {unknown}. "
            "Use [defaults], [profiles.<name>], and optional default_profile."
        )

    defaults = _normalize_config_mapping(
        data.get("defaults", {}),
        allowed_keys=TRANSLATE_CONFIG_KEYS,
        base_dir=resolved_path.parent,
        label="[defaults]",
    )

    profiles_raw = data.get("profiles", {})
    if profiles_raw is None:
        profiles_raw = {}
    if not isinstance(profiles_raw, dict):
        raise ValueError(f"[profiles] must be a table in {resolved_path.name}.")

    profiles: dict[str, dict[str, Any]] = {}
    for profile_name, profile_value in profiles_raw.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError(f"Profile names must be non-empty strings in {resolved_path.name}.")
        profiles[profile_name] = _normalize_config_mapping(
            profile_value,
            allowed_keys=TRANSLATE_CONFIG_KEYS,
            base_dir=resolved_path.parent,
            label=f"[profiles.{profile_name}]",
        )

    default_profile = data.get("default_profile")
    if default_profile is not None and not isinstance(default_profile, str):
        raise ValueError(f"default_profile must be a string in {resolved_path.name}.")

    return AppConfig(
        path=resolved_path,
        defaults=defaults,
        profiles=profiles,
        default_profile=default_profile,
    )


def resolve_command_config(
    config: AppConfig | None,
    *,
    profile: str | None,
    allowed_keys: set[str],
) -> tuple[dict[str, Any], ConfigSelection | None]:
    if config is None:
        return {}, None

    selected_profile = _resolve_profile_name(config, explicit_profile=profile)
    merged = dict(config.defaults)
    if selected_profile is not None:
        merged.update(config.profiles[selected_profile])

    filtered = {
        key: value
        for key, value in merged.items()
        if key in allowed_keys
    }
    api_key_env = filtered.pop("api_key_env", None)
    if "api_key" not in filtered and isinstance(api_key_env, str):
        env_value = os.getenv(api_key_env)
        if env_value:
            filtered["api_key"] = env_value

    return filtered, ConfigSelection(path=config.path, profile=selected_profile)


def format_config_selection(selection: ConfigSelection | None) -> str | None:
    if selection is None:
        return None
    if selection.profile:
        return f"{selection.path} (profile {selection.profile})"
    return str(selection.path)


def _resolve_profile_name(config: AppConfig, *, explicit_profile: str | None) -> str | None:
    if explicit_profile is not None:
        if explicit_profile not in config.profiles:
            raise ValueError(
                f"Config profile '{explicit_profile}' was not found in {config.path.name}."
            )
        return explicit_profile

    if config.default_profile is not None:
        if config.default_profile not in config.profiles:
            raise ValueError(
                f"default_profile '{config.default_profile}' was not found in {config.path.name}."
            )
        return config.default_profile

    if len(config.profiles) == 1:
        return next(iter(config.profiles))

    if len(config.profiles) > 1:
        raise ValueError(
            f"Multiple config profiles are defined in {config.path.name}. "
            "Set default_profile in the config file or pass --profile."
        )
    return None


def _normalize_config_mapping(
    raw_mapping: Any,
    *,
    allowed_keys: set[str],
    base_dir: Path,
    label: str,
) -> dict[str, Any]:
    if raw_mapping is None:
        return {}
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"{label} must be a table.")

    unknown_keys = set(raw_mapping) - allowed_keys
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unsupported config keys in {label}: {unknown}.")

    normalized: dict[str, Any] = {}
    for key, value in raw_mapping.items():
        normalized[key] = _coerce_config_value(
            key,
            value,
            base_dir=base_dir,
            label=label,
        )
    return normalized


def _coerce_config_value(
    key: str,
    value: Any,
    *,
    base_dir: Path,
    label: str,
) -> Any:
    if key in _PATH_KEYS:
        if not isinstance(value, str):
            raise ValueError(f"{label}.{key} must be a string path.")
        path = Path(value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path

    if key in _BOOL_KEYS:
        if not isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be true or false.")
        return value

    if key in _INT_KEYS:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be an integer.")
        return value

    if key in _FLOAT_KEYS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}.{key} must be a number.")
        return float(value)

    if key in _STRING_KEYS:
        if not isinstance(value, str):
            raise ValueError(f"{label}.{key} must be a string.")
        return value

    raise ValueError(f"Unsupported config key in {label}: {key}.")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped
