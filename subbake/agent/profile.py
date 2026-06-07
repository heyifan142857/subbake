"""Profile and config management for SubBakeAgent.

Extracted from ``_core.py``. Functions take an agent instance as the
first parameter, following the same pattern as ``executor.py`` and
``intent.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._core import SubBakeAgent

from subbake.config import TRANSLATE_CONFIG_KEYS, load_app_config, resolve_command_config
from subbake import config as _config
from subbake.runtime_options import merge_translation_values

from .trace import (
    _default_api_key_env,
    _prepend_default_profile,
    _toml_key,
    _toml_string,
    _verify_write_text,
)
from .ui import console_choose, prompt_text, select_from_list

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEW_PROFILE_VALUE = "__subbake_new_profile__"
CONFIG_BOOTSTRAP_CREATE = "create"
CONFIG_BOOTSTRAP_SKIP = "skip"
PROFILE_PROVIDER_OPTIONS = ("mock", "openai", "anthropic", "gemini", "openai-compatible")
PROFILE_TARGET_LANGUAGE_OPTIONS = ("Chinese", "zh", "en", "ja", "ko", "fr", "es", "de")
PROFILE_API_KEY_ENV_OPTIONS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def values_for_profile(agent: SubBakeAgent, profile: str | None) -> dict[str, Any]:
    """Resolve translation values for a given profile (or None for defaults)."""
    if agent.config is None:
        return merge_translation_values()
    config_values, _ = resolve_command_config(
        agent.config,
        profile=profile,
        allowed_keys=TRANSLATE_CONFIG_KEYS,
    )
    return merge_translation_values(config_values)


def initial_profile(agent: SubBakeAgent) -> str | None:
    """Determine the initial profile to use when the agent starts."""
    if agent.config is None or not agent.config.profiles:
        return None
    latest = agent.store.latest()
    if latest is not None and latest.profile in agent.config.profiles:
        return latest.profile
    if agent.config.default_profile is not None and agent.config.default_profile in agent.config.profiles:
        return agent.config.default_profile
    if len(agent.config.profiles) == 1:
        return next(iter(agent.config.profiles))
    names = sorted(agent.config.profiles)
    return console_choose(
        agent.console,
        agent.interactive,
        "Choose profile",
        [(name, name) for name in names],
        default=names[0],
    )


# ---------------------------------------------------------------------------
# Profile switching / picker
# ---------------------------------------------------------------------------


def switch_profile(agent: SubBakeAgent, profile_name: str) -> None:
    """Switch to a named profile, updating agent state."""
    if agent.config is None or profile_name not in agent.config.profiles:
        raise ValueError(f"Config profile '{profile_name}' was not found.")
    agent.profile = profile_name
    agent.values = values_for_profile(agent, profile_name)
    agent.session.profile = profile_name
    agent.console.print(
        f"[bold green]Profile switched:[/bold green] {profile_name} "
        f"({agent.values['provider']} / {agent.values['model']})"
    )


def open_profile_picker(agent: SubBakeAgent) -> None:
    """Show an interactive picker for selecting or creating a profile."""
    options: list[tuple[str, str]] = []
    if agent.config is not None:
        for name in sorted(agent.config.profiles):
            values = values_for_profile(agent, name)
            marker = "* " if name == agent.profile else ""
            options.append((name, f"{marker}{name}: {values['provider']} / {values['model']}"))
    options.append((NEW_PROFILE_VALUE, "new"))
    selected = select_from_list(
        agent.console,
        agent.interactive,
        "Model profile",
        options,
        default=agent.profile if agent.profile is not None else NEW_PROFILE_VALUE,
    )
    if selected == NEW_PROFILE_VALUE:
        create_profile_interactively(agent)
    elif selected:
        switch_profile(agent, selected)


def handle_profile_command(agent: SubBakeAgent, rest: str) -> None:
    """Handle the /profile (or /model) command."""
    profile_name = rest.strip()
    if not profile_name:
        if agent.interactive:
            open_profile_picker(agent)
        else:
            print_profiles(agent, include_new=True)
        return
    if profile_name.casefold() == "new":
        create_profile_interactively(agent)
        return
    switch_profile(agent, profile_name)


# ---------------------------------------------------------------------------
# Profile creation
# ---------------------------------------------------------------------------


def create_profile_interactively(agent: SubBakeAgent) -> None:
    """Walk the user through creating a new model profile."""
    if not agent.interactive:
        agent.console.print("Profile creation is available from the interactive /profile picker.")
        return

    config_path = config_path_for_profile_write(agent)

    profile_name_prompt = prompt_text(agent.console, agent.interactive, "New profile", "Profile name", default="")
    if profile_name_prompt is None or not profile_name_prompt.strip():
        cancel_profile_creation(agent)
        return
    profile_name = profile_name_prompt.strip()
    if agent.config is not None and profile_name in agent.config.profiles:
        raise ValueError(f"Config profile '{profile_name}' already exists.")

    provider_default = str(agent.values.get("provider") or "mock")
    provider_prompt = prompt_text(
        agent.console,
        agent.interactive,
        "New profile",
        "Provider",
        default=provider_default,
        completions=PROFILE_PROVIDER_OPTIONS,
    )
    if provider_prompt is None:
        cancel_profile_creation(agent)
        return
    provider = provider_prompt.strip() or provider_default

    model_default = str(agent.values.get("model") or "mock-zh")
    model_prompt = prompt_text(agent.console, agent.interactive, "New profile", "Model", default=model_default)
    if model_prompt is None:
        cancel_profile_creation(agent)
        return
    model = model_prompt.strip() or model_default

    api_key_env_default = _default_api_key_env(provider)
    api_key_env_prompt = prompt_text(
        agent.console,
        agent.interactive,
        "New profile",
        "API key environment variable",
        default=api_key_env_default,
        completions=PROFILE_API_KEY_ENV_OPTIONS,
    )
    if api_key_env_prompt is None:
        cancel_profile_creation(agent)
        return
    api_key_env = api_key_env_prompt.strip()

    base_url_prompt = prompt_text(agent.console, agent.interactive, "New profile", "Base URL", default="")
    if base_url_prompt is None:
        cancel_profile_creation(agent)
        return
    base_url = base_url_prompt.strip()

    target_language_default = str(agent.values.get("target_language") or "Chinese")
    target_language_prompt = prompt_text(
        agent.console,
        agent.interactive,
        "New profile",
        "Target language",
        default=target_language_default,
        completions=PROFILE_TARGET_LANGUAGE_OPTIONS,
    )
    if target_language_prompt is None:
        cancel_profile_creation(agent)
        return
    target_language = target_language_prompt.strip() or target_language_default

    profile_values = {
        "provider": provider,
        "model": model,
        "api_key_env": api_key_env,
        "base_url": base_url,
        "target_language": target_language,
    }
    append_profile_to_config(agent, config_path, profile_name, profile_values)
    agent.config_path = config_path
    agent.config = agent._load_config(config_path)
    agent.session.config_path = str(config_path)
    switch_profile(agent, profile_name)
    agent.console.print(f"[bold green]Profile created:[/bold green] {profile_name} in {config_path}")


def cancel_profile_creation(agent: SubBakeAgent) -> None:
    """Cancel an in-progress profile creation."""
    agent.console.print("[bold yellow]Profile creation cancelled.[/bold yellow]")


def config_path_for_profile_write(agent: SubBakeAgent) -> Path:
    """Determine where to write a new profile (existing config or global default)."""
    if agent.config_path is not None:
        return agent.config_path
    candidates = _config.global_config_candidates()
    if not candidates:
        raise RuntimeError("No global config path is available on this platform.")
    return candidates[0]


def append_profile_to_config(
    agent: SubBakeAgent,
    path: Path,
    profile_name: str,
    values: dict[str, str],
) -> None:
    """Append a new profile section to a ``subbake.toml`` config file."""
    path = path.expanduser()
    if path.exists():
        existing_config = load_app_config(path)
        if profile_name in existing_config.profiles:
            raise ValueError(f"Config profile '{profile_name}' already exists.")
        should_set_default = existing_config.default_profile is None and not existing_config.profiles
        content = path.read_text(encoding="utf-8")
    else:
        should_set_default = True
        content = ""
    prefix = content
    if should_set_default:
        prefix = _prepend_default_profile(prefix, profile_name)
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    if prefix and not prefix.endswith("\n\n"):
        prefix += "\n"

    lines = [f"[profiles.{_toml_key(profile_name)}]"]
    for key in ("provider", "model", "api_key_env", "base_url", "target_language"):
        value = str(values.get(key) or "").strip()
        if value:
            lines.append(f"{key} = {_toml_string(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    new_content = prefix + "\n".join(lines) + "\n"
    path.write_text(new_content, encoding="utf-8")
    _verify_write_text(path, new_content)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_profiles(agent: SubBakeAgent, *, include_new: bool = False) -> None:
    """Print the list of available profiles."""
    if agent.config is None or not agent.config.profiles:
        agent.console.print("No configured profiles were found. Using built-in mock defaults.")
        if include_new:
            agent.console.print("Use /profile in an interactive terminal and choose new to create one.")
        return
    agent.console.print("[bold green]Profiles:[/bold green]")
    for name in sorted(agent.config.profiles):
        marker = "*" if name == agent.profile else " "
        values = values_for_profile(agent, name)
        agent.console.print(f" {marker} {name}: {values['provider']} / {values['model']}")
    if include_new:
        agent.console.print("   new: create a new model profile from the interactive picker")


# ---------------------------------------------------------------------------
# Config bootstrap
# ---------------------------------------------------------------------------


def offer_config_bootstrap(agent: SubBakeAgent) -> None:
    """Offer to create a model profile when no config is found on startup."""
    if not agent.interactive or agent.config is not None:
        return
    selected = select_from_list(
        agent.console,
        agent.interactive,
        "No SubBake config found",
        [
            (CONFIG_BOOTSTRAP_CREATE, "create a model profile"),
            (CONFIG_BOOTSTRAP_SKIP, "continue with mock defaults"),
        ],
        default=CONFIG_BOOTSTRAP_CREATE,
    )
    if selected == CONFIG_BOOTSTRAP_CREATE:
        create_profile_interactively(agent)
        agent._record_event("config_bootstrap", "create", {"config_path": agent.session.config_path})
        return
    agent.console.print("[bold yellow]No config created.[/bold yellow] Continuing with built-in mock defaults.")
    agent._record_event("config_bootstrap", "skip")
