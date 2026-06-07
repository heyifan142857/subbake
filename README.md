# subbake

[![PyPI version](https://img.shields.io/pypi/v/subbake)](https://pypi.org/project/subbake/)
[![Python versions](https://img.shields.io/pypi/pyversions/subbake)](https://pypi.org/project/subbake/)
[![CI](https://github.com/heyifan142857/SubBake/actions/workflows/ci.yml/badge.svg)](https://github.com/heyifan142857/SubBake/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`subbake` is an agent-first subtitle translation CLI.

It translates `.srt`, `.vtt`, and line-based `.txt` subtitles with LLMs, while preserving subtitle structure and keeping long-running translation work recoverable through cache, translation memory, glossary, and resume state.

The main entrypoint is the interactive agent — reference files with `@` and use `/` commands:

```bash
sbake
```

```
翻译 @episode01.srt         translate a subtitle file
翻译 @Season01              translate a folder as a series
/model chatgpt              switch model profiles
/session                    switch between agent sessions
```

No config file? The agent creates your first profile on startup.

For detailed setup and examples, use the [project Wiki](https://github.com/heyifan142857/SubBake/wiki).

![subbake CLI demo](assets/subbake-demo.gif)

## Why SubBake

- **Agent workflow**: run `sbake`, reference `@file`/`@folder`, switch profiles with `/model` or sessions with `/session`, toggle plan mode with `Shift+Tab`.
- **Subtitle-safe**: preserves ids, timing, cue settings, and line counts.
- **Series support**: translate a whole season with shared glossary and translation memory.
- **Runtime repair**: malformed model output is logged and repaired automatically.
- **Review pass**: high-risk batches receive targeted consistency review.
- **Persistence**: cache, run state, glossary, and translation memory live under `.subbake`.
- **Config profiles**: `subbake.toml` supports multiple provider presets for quick switching.

## Agent Boundaries

The agent handles translation, series, profile switching, failure diagnosis, subtitle editing, and project-local file operations (create, append, replace, rename, delete).

It does not expose low-level file tools as direct commands — describe the task naturally and the agent selects the tool. In plan mode (`Shift+Tab`), mutating actions wait for `/approve` before execution. The agent is sandboxed to the project root; `.git`, `.venv`, `.subbake`, and `__pycache__` are blocked. Destructive operations create backups under `.subbake/agent/backups`.

## Install

```bash
pip install subbake
```

Then configure a provider profile. See [examples/subbake.toml](examples/subbake.toml) and the [Wiki](https://github.com/heyifan142857/SubBake/wiki).

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
- [Configuration example](examples/subbake.toml)
- [PyPI package](https://pypi.org/project/subbake/)

Command help is also available from the CLI:

```bash
sbake --help
sbake translate --help
sbake series --help
sbake check-key --help
sbake clean --help
```
