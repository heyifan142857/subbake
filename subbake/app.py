from __future__ import annotations

import shutil
from pathlib import Path

import typer
from click.core import ParameterSource
from rich.console import Console

from subbake import __version__
from subbake.config import (
    CHECK_KEY_CONFIG_KEYS,
    TRANSLATE_CONFIG_KEYS,
    discover_config_path,
    format_config_selection,
    load_app_config,
    resolve_command_config,
)
from subbake.entities import DEFAULT_BATCH_SIZE, PipelineResult
from subbake.models import build_backend
from subbake.pipeline import SubtitlePipeline
from subbake.runtime_options import (
    build_pipeline_options,
    merge_translation_values,
)
from subbake.series import translate_series
from subbake.storage import build_runtime_paths
from subbake.ui import Dashboard

APP_HELP = """LLM subtitle translation CLI with Chinese as the default target language, or another target such as en / ja / fr.

Common commands:
  sbake
  sbake translate input.srt --provider openai
  sbake series ./Season01 --provider openai
  sbake resume
  sbake translate input.vtt --bilingual
  sbake translate input.srt --dry-run
  sbake check-key --provider openai
  sbake clean input.srt
  sbake clean . --all

Common options for `sbake translate`:
  --output         Set the output file path
  --output-format  Force the output format: srt / vtt / txt
  --provider       Choose the model provider, such as mock / openai / anthropic / gemini
  --model          Set the model name
  --base-url       Set the OpenAI-compatible API base URL
  --api-key        Pass the API key directly
  --batch-size     Batch size, default is 30
  --fast           Prioritize speed and successful completion over maximum quality
  --bilingual      Output bilingual subtitles
  --target-language  Target language, for example Chinese / en / ja / fr
  --config         Use a specific subbake.toml file
  --profile        Choose a named config profile
  --dry-run        Parse and plan batches without calling the model
  --resume         Resume from run_state.json when available
  --cache          Reuse cached responses and translation-memory matches
  --no-agent       Disable default runtime agent repair for malformed model output
  --work-dir       Directory for cache / run state / failures
  --glossary-path  Path to the persistent glossary JSON file

Common options for `sbake clean`:
  --runs           Remove run state and failure samples
  --cache          Remove cached responses
  --glossary       Remove the persistent glossary file
  --all            Remove all runtime artifacts

Common options for `sbake check-key`:
  --provider       Choose the provider to validate
  --api-key        Pass the API key directly
  --base-url       Set the OpenAI-compatible API base URL
  --timeout        Set the network timeout in seconds

See full command options:
  sbake translate --help
  sbake series --help
  sbake check-key --help
  sbake clean --help
"""

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
    help=APP_HELP,
)
console = Console()


def _version_callback(value: bool) -> None:
    if not value:
        return
    typer.echo(f"subbake {__version__}")
    raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """subbake CLI."""
    if ctx.invoked_subcommand is None:
        from subbake.agent import start_interactive_agent

        start_interactive_agent(console=console, resume=False)
        raise typer.Exit()


def _load_command_config(
    *,
    explicit_config_path: Path | None,
    profile: str | None,
    allowed_keys: set[str],
) -> tuple[dict[str, object], object | None]:
    config_path = explicit_config_path
    if config_path is None:
        config_path = discover_config_path()
    if config_path is None:
        if profile is not None:
            raise ValueError("No subbake.toml file was found for the requested --profile.")
        return {}, None

    config = load_app_config(config_path)
    return resolve_command_config(
        config,
        profile=profile,
        allowed_keys=allowed_keys,
    )


def _configured_value(
    ctx: typer.Context,
    parameter_name: str,
    current_value: object,
    config_values: dict[str, object],
) -> object:
    if _is_commandline_value(ctx, parameter_name, current_value):
        return current_value
    return config_values.get(parameter_name, current_value)


def _is_commandline_value(ctx: typer.Context, parameter_name: str, current_value: object) -> bool:
    if _parameter_source_is_commandline(ctx, parameter_name):
        return True

    dashed_parameter_name = parameter_name.replace("_", "-")
    if dashed_parameter_name != parameter_name and _parameter_source_is_commandline(ctx, dashed_parameter_name):
        return True

    option_default = _command_option_default(ctx, parameter_name)
    return option_default is not _MISSING and current_value != option_default


_MISSING = object()


def _command_option_default(ctx: typer.Context, parameter_name: str) -> object:
    for parameter in ctx.command.params:
        if getattr(parameter, "name", None) == parameter_name:
            return getattr(parameter, "default", _MISSING)
    return _MISSING


def _parameter_source_is_commandline(ctx: typer.Context, parameter_name: str) -> bool:
    return ctx.get_parameter_source(parameter_name) == ParameterSource.COMMANDLINE


def _resolve_translation_values(
    *,
    ctx: typer.Context,
    explicit_config_path: Path | None,
    profile: str | None,
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
    output_format: str | None,
    batch_size: int,
    fast: bool,
    bilingual: bool,
    source_language: str,
    target_language: str,
    retries: int,
    final_review: bool,
    timeout: float,
    dry_run: bool,
    resume: bool,
    cache: bool,
    agent: bool,
    agent_repair_attempts: int,
    work_dir: Path | None,
    glossary_path: Path | None,
) -> tuple[dict[str, object], object | None]:
    config_values, config_selection = _load_command_config(
        explicit_config_path=explicit_config_path,
        profile=profile,
        allowed_keys=TRANSLATE_CONFIG_KEYS,
    )
    resolved = {
        "provider": _configured_value(ctx, "provider", provider, config_values),
        "model": _configured_value(ctx, "model", model, config_values),
        "api_key": _configured_value(ctx, "api_key", api_key, config_values),
        "base_url": _configured_value(ctx, "base_url", base_url, config_values),
        "output_format": _configured_value(ctx, "output_format", output_format, config_values),
        "batch_size": _configured_value(ctx, "batch_size", batch_size, config_values),
        "fast": _configured_value(ctx, "fast", fast, config_values),
        "bilingual": _configured_value(ctx, "bilingual", bilingual, config_values),
        "source_language": _configured_value(ctx, "source_language", source_language, config_values),
        "target_language": _configured_value(ctx, "target_language", target_language, config_values),
        "retries": _configured_value(ctx, "retries", retries, config_values),
        "final_review": _configured_value(ctx, "final_review", final_review, config_values),
        "timeout": _configured_value(ctx, "timeout", timeout, config_values),
        "dry_run": _configured_value(ctx, "dry_run", dry_run, config_values),
        "resume": _configured_value(ctx, "resume", resume, config_values),
        "cache": _configured_value(ctx, "cache", cache, config_values),
        "agent": _configured_value(ctx, "agent", agent, config_values),
        "agent_repair_attempts": _configured_value(
            ctx,
            "agent_repair_attempts",
            agent_repair_attempts,
            config_values,
        ),
        "work_dir": _configured_value(ctx, "work_dir", work_dir, config_values),
        "glossary_path": _configured_value(ctx, "glossary_path", glossary_path, config_values),
    }
    return merge_translation_values(resolved), config_selection


def _build_backend_from_values(values: dict[str, object]):
    if bool(values["dry_run"]):
        return None
    return build_backend(
        provider=str(values["provider"]),
        model=str(values["model"]),
        api_key=str(values["api_key"]) if values.get("api_key") is not None else None,
        base_url=str(values["base_url"]) if values.get("base_url") is not None else None,
        timeout_seconds=float(values["timeout"]),
    )


def _format_reuse_summary(result: PipelineResult) -> str | None:
    parts: list[str] = []
    if result.resumed_translation_batches:
        parts.append(f"{result.resumed_translation_batches} translated batch(es) from resume")
    if result.resumed_review_batches:
        parts.append(f"{result.resumed_review_batches} review batch(es) from resume")
    if result.translation_memory_hits:
        parts.append(f"{result.translation_memory_hits} line(s) from translation memory")
    if result.cache_hits:
        parts.append(f"{result.cache_hits} cached request(s)")
    if not parts:
        return None
    return ", ".join(parts)


def _format_agent_summary(result: PipelineResult) -> str | None:
    if not result.agent_repairs:
        return None
    triggered = len(result.agent_repairs)
    repaired = sum(1 for item in result.agent_repairs if item.success)
    failed = triggered - repaired
    batches = ", ".join(
        f"{item.stage} batch {item.batch_index}"
        for item in result.agent_repairs
    )
    paths = ", ".join(str(item.log_path) for item in result.agent_repairs)
    summary = f"{triggered} triggered, {repaired} repaired"
    if failed:
        summary += f", {failed} failed"
    summary += f" ({batches}). Logs: {paths}"
    return summary


@app.command()
def translate(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Input .srt, .vtt, or .txt file."),
    output: Path | None = typer.Option(None, "--output", "-o", dir_okay=False, help="Output file path."),
    output_format: str | None = typer.Option(
        None,
        "--output-format",
        help="Output format override: srt, vtt, or txt. When omitted, sbake infers the format from --output if it uses a supported suffix, otherwise it keeps the input format.",
    ),
    provider: str = typer.Option("mock", "--provider", help="LLM provider: mock, openai, anthropic, gemini."),
    model: str = typer.Option("mock-zh", "--model", help="Model name for the selected provider."),
    api_key: str | None = typer.Option(None, "--api-key", help="API key override for the provider."),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL."),
    config: Path | None = typer.Option(
        None,
        "--config",
        dir_okay=False,
        exists=True,
        resolve_path=True,
        help="Path to subbake.toml. By default sbake checks project config upward from the current directory, then falls back to home/global config.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Named config profile to use when subbake.toml defines multiple profiles.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        min=1,
        help="Subtitle entries per translation batch.",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Prioritize speed and successful completion over best quality by using lighter prompts and best-effort recovery.",
    ),
    bilingual: bool = typer.Option(False, "--bilingual", help="Emit bilingual subtitles."),
    source_language: str = typer.Option(
        "Auto",
        "--source-language",
        help="Source language hint. Supports common aliases like auto, en, ja, and zh.",
    ),
    target_language: str = typer.Option(
        "Chinese",
        "--target-language",
        help="Target language. Supports common aliases like zh, en, ja, ko, fr, es, and de.",
    ),
    retries: int = typer.Option(2, "--retries", min=0, help="Retries for malformed model output."),
    final_review: bool = typer.Option(
        True,
        "--final-review/--no-final-review",
        help="Run targeted consistency review on high-risk batches.",
    ),
    timeout: float = typer.Option(120.0, "--timeout", min=1.0, help="Per-request timeout in seconds."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only parse and show batch planning without calling the model."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from saved run state when available."),
    cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Reuse cached responses and translation-memory matches.",
    ),
    agent: bool = typer.Option(
        True,
        "--agent/--no-agent",
        help="Enable the default runtime agent repair for malformed model output.",
    ),
    agent_repair_attempts: int = typer.Option(
        2,
        "--agent-repair-attempts",
        min=0,
        help="Maximum agent repair attempts per failed batch.",
    ),
    work_dir: Path | None = typer.Option(None, "--work-dir", file_okay=False, help="Directory for cache, run state, failures, and default glossary."),
    glossary_path: Path | None = typer.Option(None, "--glossary-path", dir_okay=False, help="Persistent glossary JSON path."),
) -> None:
    """Translate subtitles while preserving subtitle structure."""

    try:
        values, config_selection = _resolve_translation_values(
            ctx=ctx,
            explicit_config_path=config,
            profile=profile,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            output_format=output_format,
            batch_size=batch_size,
            fast=fast,
            bilingual=bilingual,
            source_language=source_language,
            target_language=target_language,
            retries=retries,
            final_review=final_review,
            timeout=timeout,
            dry_run=dry_run,
            resume=resume,
            cache=cache,
            agent=agent,
            agent_repair_attempts=agent_repair_attempts,
            work_dir=work_dir,
            glossary_path=glossary_path,
        )
        options = build_pipeline_options(
            input_path=input_path,
            output_path=output,
            values=values,
        )

        backend = _build_backend_from_values(values)
        dashboard = Dashboard(console=console)
        pipeline = SubtitlePipeline(backend=backend, options=options, dashboard=dashboard)
        result = pipeline.run()
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("")
    if result.dry_run:
        from rich.table import Table

        console.print("[bold yellow]Dry run:[/bold yellow] no model API calls were made.")
        console.print(f"[bold green]Planned batches:[/bold green] {len(result.planned_batches)}")
        if result.planned_batches:
            table = Table(title="Batch Plan")
            table.add_column("Batch", justify="right")
            table.add_column("Lines", justify="right")
            table.add_column("IDs")
            for batch in result.planned_batches:
                table.add_row(
                    str(batch.index),
                    str(batch.size),
                    f"{batch.first_id} -> {batch.last_id}",
                )
            console.print(table)
        if result.glossary_path is not None:
            console.print(f"[bold green]Glossary:[/bold green] {result.glossary_path}")
        if result.state_path is not None:
            console.print(f"[bold green]Run state:[/bold green] {result.state_path}")
        config_description = format_config_selection(config_selection)
        if config_description is not None:
            console.print(f"[bold green]Config:[/bold green] {config_description}")
        return

    console.print(f"[bold green]Output:[/bold green] {result.output_path}")
    console.print(
        "[bold green]Usage:[/bold green] "
        f"{result.usage.input_tokens:,} in / {result.usage.output_tokens:,} out / {result.usage.total_tokens:,} total"
    )
    console.print(
        "[bold green]Batches:[/bold green] "
        f"{result.batches_translated} translated, {result.review_batches} reviewed"
    )
    reuse_summary = _format_reuse_summary(result)
    if reuse_summary is not None:
        console.print(f"[bold green]Reused:[/bold green] {reuse_summary}")
    agent_summary = _format_agent_summary(result)
    if agent_summary is not None:
        console.print(f"[bold green]Agent:[/bold green] {agent_summary}")
    config_description = format_config_selection(config_selection)
    if config_description is not None:
        console.print(f"[bold green]Config:[/bold green] {config_description}")
    if bool(values["fast"]):
        console.print("[bold green]Mode:[/bold green] fast")
    console.print(f"[bold green]Target language:[/bold green] {options.target_language}")
    if result.glossary_path is not None:
        console.print(f"[bold green]Glossary:[/bold green] {result.glossary_path}")
    if result.state_path is not None:
        console.print(f"[bold green]Run state:[/bold green] {result.state_path}")


@app.command()
def series(
    ctx: typer.Context,
    folder: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder containing .srt, .vtt, or .txt subtitle files."),
    output_format: str | None = typer.Option(
        None,
        "--output-format",
        help="Output format override for every file: srt, vtt, or txt.",
    ),
    provider: str = typer.Option("mock", "--provider", help="LLM provider: mock, openai, anthropic, gemini."),
    model: str = typer.Option("mock-zh", "--model", help="Model name for the selected provider."),
    api_key: str | None = typer.Option(None, "--api-key", help="API key override for the provider."),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL."),
    config: Path | None = typer.Option(
        None,
        "--config",
        dir_okay=False,
        exists=True,
        resolve_path=True,
        help="Path to subbake.toml.",
    ),
    profile: str | None = typer.Option(None, "--profile", help="Named config profile to use."),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, "--batch-size", min=1, help="Subtitle entries per translation batch."),
    fast: bool = typer.Option(False, "--fast", help="Prioritize speed and successful completion over best quality."),
    bilingual: bool = typer.Option(False, "--bilingual", help="Emit bilingual subtitles."),
    source_language: str = typer.Option("Auto", "--source-language", help="Source language hint."),
    target_language: str = typer.Option("Chinese", "--target-language", help="Target language."),
    retries: int = typer.Option(2, "--retries", min=0, help="Retries for malformed model output."),
    final_review: bool = typer.Option(True, "--final-review/--no-final-review", help="Run targeted consistency review."),
    timeout: float = typer.Option(120.0, "--timeout", min=1.0, help="Per-request timeout in seconds."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan work without calling the model."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from saved run state when available."),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Reuse cached responses and translation-memory matches."),
    agent: bool = typer.Option(True, "--agent/--no-agent", help="Enable runtime agent repair for malformed model output."),
    agent_repair_attempts: int = typer.Option(2, "--agent-repair-attempts", min=0, help="Maximum agent repair attempts per failed batch."),
    work_dir: Path | None = typer.Option(None, "--work-dir", file_okay=False, help="Shared directory for series cache, state, failures, and glossary."),
    glossary_path: Path | None = typer.Option(None, "--glossary-path", dir_okay=False, help="Persistent glossary JSON path."),
    recursive: bool = typer.Option(False, "--recursive", help="Translate subtitle files in subdirectories too."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing generated subtitle outputs."),
) -> None:
    """Translate a folder of episode subtitles with shared glossary and translation memory."""

    try:
        values, config_selection = _resolve_translation_values(
            ctx=ctx,
            explicit_config_path=config,
            profile=profile,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            output_format=output_format,
            batch_size=batch_size,
            fast=fast,
            bilingual=bilingual,
            source_language=source_language,
            target_language=target_language,
            retries=retries,
            final_review=final_review,
            timeout=timeout,
            dry_run=dry_run,
            resume=resume,
            cache=cache,
            agent=agent,
            agent_repair_attempts=agent_repair_attempts,
            work_dir=work_dir,
            glossary_path=glossary_path,
        )
        result = translate_series(
            root=folder,
            values=values,
            backend_factory=lambda: _build_backend_from_values(values),
            console=console,
            recursive=recursive,
            overwrite=overwrite,
        )
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("")
    console.print(f"[bold green]Series root:[/bold green] {result.root}")
    console.print(
        "[bold green]Series result:[/bold green] "
        f"{result.processed_count} processed, {result.skipped_count} skipped, {result.failure_count} failed"
    )
    config_description = format_config_selection(config_selection)
    if config_description is not None:
        console.print(f"[bold green]Config:[/bold green] {config_description}")
    if result.skipped:
        console.print("[bold yellow]Skipped:[/bold yellow]")
        for item in result.skipped[:10]:
            console.print(f"  - {item.input_path} ({item.reason})")
    if result.failures:
        console.print("[bold red]Failed:[/bold red]")
        for item in result.failures[:10]:
            console.print(f"  - {item.input_path} ({item.reason})")
        raise typer.Exit(code=1)


@app.command()
def resume(
    session_id: str | None = typer.Argument(
        None,
        help="Optional session ID to resume. If omitted, shows an interactive picker or resumes the latest session.",
    ),
) -> None:
    """Resume an interactive agent session."""

    from subbake.agent import start_interactive_agent

    start_interactive_agent(console=console, resume=session_id is None, session_id=session_id)


@app.command("check-key")
def check_key(
    ctx: typer.Context,
    provider: str = typer.Option("openai", "--provider", help="LLM provider: mock, openai, anthropic, gemini."),
    model: str = typer.Option(
        "check-only",
        "--model",
        help="Optional model name for backend initialization. Not required for most providers.",
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="API key override for the provider."),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL."),
    config: Path | None = typer.Option(
        None,
        "--config",
        dir_okay=False,
        exists=True,
        resolve_path=True,
        help="Path to subbake.toml. By default sbake checks project config upward from the current directory, then falls back to home/global config.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Named config profile to use when subbake.toml defines multiple profiles.",
    ),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0, help="Credential check timeout in seconds."),
) -> None:
    """Check whether the configured provider credentials are accepted."""

    try:
        config_values, config_selection = _load_command_config(
            explicit_config_path=config,
            profile=profile,
            allowed_keys=CHECK_KEY_CONFIG_KEYS,
        )
        provider = _configured_value(ctx, "provider", provider, config_values)
        model = _configured_value(ctx, "model", model, config_values)
        api_key = _configured_value(ctx, "api_key", api_key, config_values)
        base_url = _configured_value(ctx, "base_url", base_url, config_values)
        timeout = _configured_value(ctx, "timeout", timeout, config_values)

        backend = build_backend(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout,
        )
        valid, message = backend.check_credentials()
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if valid:
        console.print("[bold green]Credential check passed.[/bold green]")
        config_description = format_config_selection(config_selection)
        if config_description is not None:
            console.print(f"[bold green]Config:[/bold green] {config_description}")
        console.print(message)
        return

    console.print("[bold red]Credential check failed.[/bold red]")
    config_description = format_config_selection(config_selection)
    if config_description is not None:
        console.print(f"[bold green]Config:[/bold green] {config_description}")
    console.print(message)
    raise typer.Exit(code=1)


@app.command()
def clean(
    target: Path = typer.Argument(
        Path("."),
        exists=True,
        file_okay=True,
        dir_okay=True,
        resolve_path=True,
        help="Subtitle file or directory used to locate runtime artifacts.",
    ),
    work_dir: Path | None = typer.Option(
        None,
        "--work-dir",
        file_okay=False,
        resolve_path=True,
        help="Explicit runtime directory to clean instead of deriving .subbake from the target.",
    ),
    runs: bool = typer.Option(False, "--runs", help="Remove run state and failure samples."),
    cache: bool = typer.Option(False, "--cache", help="Remove cached model responses."),
    glossary: bool = typer.Option(False, "--glossary", help="Remove the persistent glossary file."),
    all: bool = typer.Option(
        False,
        "--all",
        help="Remove all runtime artifacts. This is the default for directory targets.",
    ),
) -> None:
    """Remove cached runtime files, run state, failure samples, and glossary data."""

    runtime_root, run_dir, glossary_file = _resolve_clean_paths(
        target=target,
        work_dir=work_dir,
    )
    remove_runs, remove_cache, remove_glossary = _resolve_clean_selection(
        target=target,
        runs=runs,
        cache=cache,
        glossary=glossary,
        all=all,
    )

    removed: list[str] = []
    missing: list[str] = []

    if remove_runs:
        removed_path = run_dir if target.is_file() else runtime_root / "runs"
        _remove_path(removed_path, removed, missing, "runs")
    if remove_cache:
        _remove_path(runtime_root / "cache", removed, missing, "cache")
    if remove_glossary:
        _remove_globbed_files(
            runtime_root=runtime_root,
            pattern="glossary*.json",
            fallback_path=glossary_file,
            removed=removed,
            missing=missing,
            label="glossary",
        )

    if target.is_dir() or work_dir is not None or all:
        _prune_empty_runtime_root(runtime_root)
    elif run_dir.parent.exists() and not any(run_dir.parent.iterdir()):
        run_dir.parent.rmdir()
        _prune_empty_runtime_root(runtime_root)

    if removed:
        console.print("[bold green]Removed:[/bold green]")
        for item in removed:
            console.print(f"  - {item}")
    else:
        console.print("[bold yellow]Nothing removed.[/bold yellow]")

    if missing:
        console.print("[bold yellow]Not found:[/bold yellow]")
        for item in missing:
            console.print(f"  - {item}")


def _resolve_clean_paths(
    target: Path,
    work_dir: Path | None,
) -> tuple[Path, Path, Path]:
    if target.is_file():
        runtime_paths = build_runtime_paths(
            input_path=target,
            work_dir=work_dir,
            glossary_path=None,
        )
        return runtime_paths.root_dir, runtime_paths.run_dir, runtime_paths.glossary_path

    runtime_root = work_dir or target / ".subbake"
    return runtime_root, runtime_root / "runs", runtime_root / "glossary.json"


def _resolve_clean_selection(
    *,
    target: Path,
    runs: bool,
    cache: bool,
    glossary: bool,
    all: bool,
) -> tuple[bool, bool, bool]:
    if all:
        return True, True, True
    if runs or cache or glossary:
        return runs, cache, glossary
    if target.is_file():
        return True, False, False
    return True, True, True


def _remove_path(
    path: Path,
    removed: list[str],
    missing: list[str],
    label: str,
) -> None:
    if not path.exists():
        missing.append(f"{label}: {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    removed.append(f"{label}: {path}")


def _remove_globbed_files(
    *,
    runtime_root: Path,
    pattern: str,
    fallback_path: Path,
    removed: list[str],
    missing: list[str],
    label: str,
) -> None:
    matches = sorted(
        path
        for path in runtime_root.glob(pattern)
        if path.is_file()
    )
    if not matches:
        missing.append(f"{label}: {fallback_path}")
        return
    for path in matches:
        _remove_path(path, removed, missing, label)


def _prune_empty_runtime_root(runtime_root: Path) -> None:
    if not runtime_root.exists():
        return
    for child in runtime_root.iterdir():
        if child.is_dir():
            return
        if child.is_file():
            return
    runtime_root.rmdir()


if __name__ == "__main__":
    app()
