# subbake

[![PyPI version](https://img.shields.io/pypi/v/subbake)](https://pypi.org/project/subbake/)
[![Python versions](https://img.shields.io/pypi/pyversions/subbake)](https://pypi.org/project/subbake/)
[![CI](https://github.com/heyifan142857/SubBake/actions/workflows/ci.yml/badge.svg)](https://github.com/heyifan142857/SubBake/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/heyifan142857/SubBake/blob/main/LICENSE)

`subbake` is an agent-first subtitle translation CLI.

It translates `.srt`, `.vtt`, and line-based `.txt` subtitles with LLMs, while preserving subtitle structure and keeping long-running translation work recoverable through cache, translation memory, glossary, and resume state.

The main entrypoint is the interactive agent:

```bash
sbake
```

Inside the agent, you can point at files or folders, switch model profiles, resume sessions, inspect SubBake failure logs, make constrained edits to generated translated subtitles, and perform simple project-local file operations.
Typing `/` opens the command picker; keep typing to filter, then use `Tab` to complete or Up/Down and Enter to choose. `Shift+Tab` toggles plan mode, where mutating actions are proposed for approval before execution. `/session`, `/model`, and `/profile` open compact inline pickers for previous sessions and model profiles, with `new` available when creating a profile interactively.
If no config file is found, the interactive agent can create the first model profile for you.

```text
@episode01.srt
@Season01
/model
/session
分析 @.subbake/runs/.../failures/translate_batch_0001.json
把 @episode01.translated.srt 里的角色名统一一下
创建 @notes.txt 记录后续需要检查的术语
把 @notes.txt 改名成 @translation-notes.txt
```

For detailed setup, provider configuration, examples, and workflow guidance, use the [project Wiki](https://github.com/heyifan142857/SubBake/wiki).

## Why SubBake

- Agent workflow: run `sbake`, reference `@file` or `@folder`, switch profiles with `/model`, switch sessions with `/session`, and toggle plan mode with `Shift+Tab`.
- Subtitle-safe translation: preserves ids, order, timing, cue settings, and line counts.
- Series support: translate a whole season folder with shared glossary and translation memory.
- Runtime repair: malformed model output can be logged, diagnosed, and automatically repaired during translation.
- Review pass: high-risk batches can receive targeted consistency review.
- Practical persistence: cache, run state, failure samples, glossary, and translation memory live under `.subbake`.
- Config profiles: `subbake.toml` supports multiple provider/model setups for quick switching.

## Agent Boundaries

The agent can translate subtitle files, translate folders as a series, switch configured profiles, diagnose SubBake failure logs, edit generated translated subtitles, and perform simple project-local file work such as creating, appending, replacing text, renaming, and deleting files.

It does not expose low-level file tools as user commands; describe the task naturally and the agent decides which internal tool to use. In plan mode, mutating tool calls are held until `/approve` and can be discarded with `/reject`. It also does not operate outside the current project root. File edits and deletes are limited to project-local text files; protected runtime or repository paths such as `.git`, `.venv`, and `.subbake` are blocked. Mutating file operations create backups under `.subbake/agent/backups`.

## Install

```bash
pip install subbake
```

Then configure a provider profile. See the [configuration example](https://github.com/heyifan142857/SubBake/blob/main/examples/subbake.toml) and the [Wiki](https://github.com/heyifan142857/SubBake/wiki).

## CLI Modes

Interactive agent:

```bash
sbake
sbake resume
```

Classic single-file command:

```bash
sbake translate input.srt --profile chatgpt
```

Series command:

```bash
sbake series ./Season01 --profile chatgpt
```

Credential and cleanup utilities:

```bash
sbake check-key --profile chatgpt
sbake clean .
```

## Documentation

The Wiki is the primary user guide:

- [Project Wiki](https://github.com/heyifan142857/SubBake/wiki)
- [Configuration example](https://github.com/heyifan142857/SubBake/blob/main/examples/subbake.toml)
- [GitHub repository](https://github.com/heyifan142857/SubBake)

Command help is also available from the CLI:

```bash
sbake --help
sbake translate --help
sbake series --help
sbake check-key --help
sbake clean --help
```
