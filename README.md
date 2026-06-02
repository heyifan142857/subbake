# subbake

[![PyPI version](https://img.shields.io/pypi/v/subbake)](https://pypi.org/project/subbake/)
[![Python versions](https://img.shields.io/pypi/pyversions/subbake)](https://pypi.org/project/subbake/)
[![CI](https://github.com/heyifan142857/SubBake/actions/workflows/ci.yml/badge.svg)](https://github.com/heyifan142857/SubBake/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`subbake` is an agent-first subtitle translation CLI.

It translates `.srt`, `.vtt`, and line-based `.txt` subtitles with LLMs, while preserving subtitle structure and keeping long-running translation work recoverable through cache, translation memory, glossary, and resume state.

The main entrypoint is the interactive agent:

```bash
sbake
```

Inside the agent, you can point at files or folders, switch model profiles, resume sessions, inspect SubBake failure logs, and ask it to make constrained edits to generated translated subtitles.

```text
@episode01.srt
@Season01
/model fast_zh
分析 @.subbake/runs/.../failures/translate_batch_0001.json
/edit @episode01.translated.srt 统一角色名译法
```

For detailed setup, provider configuration, examples, and workflow guidance, use the [project Wiki](https://github.com/heyifan142857/SubBake/wiki).

![subbake CLI demo](assets/subbake-demo.gif)

## Why SubBake

- Agent workflow: run `sbake`, reference `@file` or `@folder`, switch profiles with `/model`, and resume with `sbake resume`.
- Subtitle-safe translation: preserves ids, order, timing, cue settings, and line counts.
- Series support: translate a whole season folder with shared glossary and translation memory.
- Runtime repair: malformed model output can be logged, diagnosed, and automatically repaired during translation.
- Review pass: high-risk batches can receive targeted consistency review.
- Practical persistence: cache, run state, failure samples, glossary, and translation memory live under `.subbake`.
- Config profiles: `subbake.toml` supports multiple provider/model setups for quick switching.

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
