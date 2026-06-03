# Changelog

This file tracks notable changes for each release.

## [Unreleased]

## [0.4.0] - 2026-06-03

### Added

- `sbake` without a subcommand now opens a conversational agent interface while keeping `sbake translate` and other classic commands unchanged.
- The agent can decide from natural language whether to translate `@file`, translate an `@folder` as a series, diagnose SubBake failure logs, edit generated translated subtitles, or perform simple project-local text file work such as create, append, replace text, rename, and delete.
- Agent sessions are persisted locally under `.subbake/agent/sessions/*.json`, with `/session` and `sbake resume` support for returning to previous conversations.
- Slash command completion is available in the agent: type `/`, keep typing to filter, use `Tab` for unique completions, and use Up/Down plus Enter to choose from command or picker menus.
- `/model`, `/profile`, and `/session` now open compact inline pickers in interactive terminals instead of full-screen dialogs; profile pickers include a `new` option for creating a profile.
- New profile creation now uses a compact inline wizard with provider, API-key environment, and target-language completion instead of full-screen input dialogs.
- The profile creation wizard now writes `default_profile` when it creates the first profile in a config file.
- When the interactive agent starts without any config file, it offers to create the first model profile automatically instead of requiring manual `subbake.toml` setup.
- Plan mode is available for mutating agent actions through `Shift+Tab` or `/plan`, with `/approve` and `/reject` for proposed tool calls.
- Series translation can process a subtitle folder with shared glossary and translation memory context across the run.

### Changed

- README and PyPI README now present the interactive agent as a primary workflow and keep detailed setup guidance in the project Wiki.
- Agent inline pickers now use terminal-theme-friendly styling instead of a hard-coded dark background.
- `pyproject.toml` now advertises `agent` as a package keyword.

### Fixed

- Command-line options such as `--target-language` now reliably override config profile values across Click/Typer versions.
- `click` is now declared explicitly as a package dependency.

### Safety

- Agent file operations are internal tools rather than slash commands; users describe intent naturally and the agent selects the tool.
- Project-local file mutations are guarded, block protected paths such as `.git`, `.venv`, and `.subbake`, and create backups for destructive edits.

## [0.3.2] - 2026-05-18

### Added

- Runtime agent repair is now enabled by default for malformed model output, with `--no-agent` and `agent = false` to disable it.
- Agent repair attempts now have a Rich dashboard log panel, persisted `agent_logs`, failure-sample attempt details, and final CLI summaries when triggered.

## [0.3.1] - 2026-05-18

### Fixed

- Final review prompts now use one complete authoritative line list with source text for every entry, preventing short entries such as numeric exclamations from being dropped during review.

## [0.3.0] - 2026-04-21
### Added

- A `--fast` mode that uses lighter prompts, skips final review, and prefers best-effort completion when model structure is unstable.
- Target language aliases such as `en`, `ja`, `ko`, `fr`, `es`, and `de`, plus matching source-language alias support.
- `subbake.toml` configuration support with auto-discovery, `--config`, `--profile`, `[defaults]`, and named profiles for model/provider presets.
- An `--output-format` option that can convert subtitle output between `srt`, `vtt`, and `txt`, plus output-path suffix inference such as `.srt -> .txt`.
- A `gemini` provider that uses Google's official Gemini OpenAI-compatible endpoint with `GEMINI_API_KEY`.

### Changed

- Persistent glossary and translation-memory files are now isolated per language pair to avoid cross-language reuse.
- Translation memory is now also isolated by fast vs standard mode, so normal runs do not reuse lower-quality fast-mode translations.
- Mock translations now reflect the requested target language, which keeps local testing aligned with the new alias-aware target-language behavior.
- Command-line options now override config values, while config defaults and profiles fill in omitted parameters.
- When multiple config profiles exist, `sbake` now requires `default_profile` or an explicit `--profile` instead of silently picking the first one.
- Config discovery now follows `command line > project config > home/global config > built-in defaults`, with global config support for Linux, macOS, and Windows locations.
- README examples now use a `chatgpt` profile instead of `deepseek`, and the output section documents explicit output paths and cross-format rendering.
- The config-file docs now point to a repository example file and more clearly recommend home/global config for personal default usage, with project config as an override layer.
- README is now trimmed down into a shorter landing page, with more detailed usage notes reorganized into wiki-style local pages.
- README now uses a shorter documentation section that points directly to the project wiki instead of listing multiple inline doc links.

## [0.2.0] - 2026-04-21

### Added

- Incremental checkpoint storage with lightweight `run_state.json` plus per-batch shards under `translated_batches/` and `reviewed_batches/`.
- Cross-file translation memory persisted to `.subbake/translation_memory.json`.
- Split `translation_fingerprint` and `render_fingerprint` so bilingual rendering changes can reuse finished translations.
- Provider-side retry handling for OpenAI-compatible and Anthropic requests with exponential backoff, `Retry-After`, request ids, and structured failure metadata.
- Dashboard ETA estimation that updates during translation and review batches.
- Regression tests for incremental resume, render reuse, translation memory reuse, provider parsing behavior, prompt shaping, dashboard ETA, adaptive batching, and structural split retries.
- Regression tests for cache hits, failure sample persistence, `clean`, malformed JSON responses, transport exceptions, and bilingual SRT/VTT rendering.

### Changed

- Default `--batch-size` is now `30`, which is a better quality-throughput balance for subtitle translation than the previous default of `50`.
- Translation prompts now use compact JSON payloads, omit timestamps, and more strongly forbid merging subtitle entries even when one spoken sentence spans multiple subtitle lines.
- Final review is now targeted at high-risk batches instead of replaying every batch.
- Translation batching is now adaptive: it considers character load, estimated tokens, semantic boundaries, split-sentence risk, speaker changes, and formatting risk instead of only a fixed entry count.
- Structural validation failures during translation now trigger automatic sub-batch retries before the batch is marked failed.
- Translation failure messages now explain likely causes such as missing or merged lines, empty translations, rate limits, and transport failures, and suggest retry guidance such as lowering `--batch-size`.
- SRT parsing is now more forgiving: cue indices are optional, cue timing settings are preserved, and wilder real-world timing lines are normalized on render.
- CLI help and README wording now describe targeted review, intelligent batching, and incremental runtime artifacts more accurately.

### Fixed

- OpenAI-compatible responses that use `text` or `target` instead of `translation` are now accepted when parsing translation lines.
- Glossary updates are accepted both as a list of entries and as a plain source-to-target mapping.
- Existing translations can now be reused when only the render mode changes, such as switching to bilingual output.
- Project metadata now advertises Python `3.14` support in package classifiers.

### Docs

- README now documents incremental batch shard outputs and clarifies that final review only runs on high-risk batches.
