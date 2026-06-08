# Changelog

This file tracks notable changes for each release.

## [Unreleased]

## [0.4.3] - 2026-06-08

### Added

- **Chat handler**: the interactive agent now responds to conversational queries with LLM-powered replies, using up to 16 recent session events for context, instead of falling back to a hardcoded message. Falls back gracefully when no LLM backend or MockBackend is active.
- **Language inference**: when a translation request omits the target language, the agent detects the user's own query language (Japanese kana → Japanese, Korean hangul → Korean, Chinese CJK → Chinese, Latin → English) and uses it as the translation target. An explicit target language in the message still takes priority.
- **REPL history navigation**: up/down arrow keys recall previously typed commands, built from session events (capped at 100 entries).
- **Loading spinner**: a Rich `console.status` spinner appears while waiting for chat LLM responses, matching the existing pattern in the agent loop.
- **Esc/Ctrl+C interrupt**: pressing Esc or Ctrl+C at the prompt immediately cancels the current input and returns to a fresh prompt, giving a quick way to discard a half-typed command.
- **In-flight operation cancellation**: in-progress translation, series, and editing operations can be cancelled with Esc. Pipeline checkpoints check for cancellation between all stages (load, parse, translate, validate, review, write). Introduces `subbake/cancellation.py` with `cancellation_scope`, `run_interruptibly`, and `OperationCancelledError`.
- **`/resume` session picker**: typing `/resume` shows a selectable list of recent sessions instead of silently resuming the latest one. After switching, the last 3 user/assistant exchanges are replayed for context.
- **`/history` command**: shows full conversation history with an optional numeric limit (e.g., `/history 5`).
- **`sbake resume <id>`**: resume a specific session by ID instead of the most recent one.
- **Esc/Ctrl+C distinction**: Esc cancels current input (returns to fresh prompt), Ctrl+C exits the agent entirely (session is auto-saved). Applied to the REPL, inline pickers, and inline-text prompts.
- **Conversation summary on resume**: a summary of prior conversation is injected into the LLM context on session resume, preventing the agent from re-greeting the user.
- **Cancellation test suite**: tests for cancellation scope creation, nested scopes, `run_interruptibly` during LLM calls, and cancellation checkpoints across all pipeline stages (`test_cancellation.py`).

### Fixed

- **Session history persistence**: assistant messages and cancellation events are now persisted in session history, so `/resume` and `/history` show complete conversations instead of only user messages. Tool execution results (translate, series, edit) are accompanied by a human-readable "assistant" event.
- **Diagnose/edit/translation summaries**: events now carry human-readable summary fields for accurate conversation replay.
- **Inline-picker Ctrl+C propagation**: inline pickers no longer catch `KeyboardInterrupt` with blanket try/except, so Ctrl+C exits the agent as intended.

## [0.4.2] - 2026-06-07

### Added

- **Intent classification gate**: the agent pre-classifies user intents before entering the full agent loop. High-confidence intents (translation, series, editing, diagnosis, file operations, browsing, profile switching) go directly to tool execution, and the agent loop only loads tools relevant to the detected category, improving speed and focus.
- **Intent-confidence gating**: low-confidence requests are routed to clarification prompts, and medium-confidence requests prompt the user for confirmation before proceeding, preventing premature or incorrect tool calls.
- **`/undo` command**: reverts the last file operation (create, modify, append, rename, delete) by restoring from its backup. Series translation outputs are grouped so a single `/undo` removes all files produced by a batch translation run.
- **Content write verification**: every file write operation now reads back and compares the written content to guard against silent data corruption from disk-full conditions, permission changes, or filesystem bugs. Covers session saves, config writes, subtitle edits, pipeline output, and storage operations.
- **Plan preview**: when a plan is pending, the agent now shows a preview of the queued tool calls so users can see what will be executed before approving.
- **Observation summaries**: agent observations carry a `context_summary` field with per-tool-type summarization (`list_files` counts by kind, `search_files` shows top candidates, `candidate_subtitles` lists matches, `recent_translations` reports the latest record, `read_file_preview` includes char count).
- **Structured tool specs**: tools use JSON Schema definitions organized into categorized groups (translate_file, translate_series, edit_subtitle, diagnose, file_operation, browse, profile, chat), enabling intent-based filtering of available tools.
- **Intent hint propagation**: the agent loop state now carries an `intent_hint` with category, confidence, and reasoning, so the agent loop model trusts the pre-classified intent instead of re-deriving it from scratch.
- **Mock intent classifier**: the mock backend supports keyword-based classification, enabling offline testing of the intent gate.
- Tests for the intent classification gate (`test_intent_gate.py`) covering all categories, fallback classification, confidence gating, and required-arg validation.
- Tests for observation summarization (`test_observation_summary.py`) covering all summarizer types.
- CLI integration tests for `/undo` covering single-file removal, overwrite restoration, and grouped series-output undo.

### Changed

- `.gitignore` now excludes `CLAUDE.md` from version control.
- The `read_file` tool moved from the `file_operation` category to `browse`, since it is a non-mutating operation.
- Agent loop context uses compact `to_context_dict()` representations with summaries instead of full observation dumps, reducing prompt bloat.
- **Second-wave module extraction**: `_core.py` reduced from ~2340 to ~746 lines by extracting 10 new modules into the `agent/` package — `deterministic.py` (deterministic tool execution), `discovery.py` (file discovery), `executor.py` (tool dispatch), `intent.py` (intent classification), `plan.py` (plan mode), `profile.py` (profile management), `session_ops.py` (session persistence), `target.py` (target resolution), `text_helpers.py` (text utilities), and `undo.py` (undo support). Remaining `_core.py` focuses solely on agent session lifecycle and the main loop entrypoint.
- UI helpers (`select_from_list`, `prompt_text`, `console_choose`) extracted from `_core.py` into `ui.py` as standalone functions, eliminating tight coupling to the agent instance.
- Re-exports in `agent/__init__.py` reorganized to reflect the new module layout — constants and helper functions re-exported from their owning modules (`intent`, `profile`) instead of `_core`.
- Tests updated to use new import paths: `profile` tests call `create_profile_interactively`/`offer_config_bootstrap` via `subbake.agent.profile`, and intent-gate tests call standalone functions (`_mock_classify_intent`, `_fallback_intent_classification`, `apply_confidence_gate`, `intent_to_decision`) from `subbake.agent.intent` instead of agent instance methods.

### Fixed

- Removed redundant right-arrow keybinding for slash-command completion; Tab is the canonical completion key.
- The original `_run_agent_loop` signature is preserved with an optional `state` parameter, keeping backward compatibility for existing callers.

### Removed

- Dead code: `_decide_next_action` and `_build_agent_decision_messages` methods removed from the agent (replaced by the intent gate + agent loop flow).

## [0.4.1] - 2026-06-03

### Added

- The interactive agent now gives a concrete pre-flight message before starting translation work, including file count, input suffixes, render mode, target language, and output format when known.
- The agent now prints a completion message after file and series translation runs.
- Natural-language series requests can now target the current directory, filter input files by explicit subtitle suffixes such as `.srt`, and pass explicit translation options including bilingual output, target/source language, output format, recursive mode, overwrite, dry run, fast mode, and final-review control.
- The agent can retarget recent or referenced generated subtitle outputs, such as changing a mistaken Chinese-only translation into Chinese-English bilingual subtitles without requiring the user to repeat the source path.
- The agent can now retarget a previous translation by title text, such as `The Matrix Revolutions`, when the matching subtitle file is in the current directory.
- Title matching and file search now expand known cross-language movie title aliases, such as `黑客帝国` to `The Matrix`.

### Fixed

- Translation tool-call messages are now shown before non-translation tools execute, while translation tools provide their own richer progress wording before the Rich dashboard appears.
- Requests such as `生成 txt 格式` no longer cause the current-directory series detector to treat `.txt` as an input-file filter.
- Referencing a generated subtitle such as `episode.translated.srt` in a retargeting request now resolves back to the source subtitle before rerendering.
- Agent file search now matches file names as well as file contents, and common search requests are parsed locally instead of relying on model-provided search patterns.
- The interactive agent startup banner now includes the package version.

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
