# subbake

[![PyPI version](https://img.shields.io/pypi/v/subbake)](https://pypi.org/project/subbake/)
[![Python versions](https://img.shields.io/pypi/pyversions/subbake)](https://pypi.org/project/subbake/)
[![CI](https://github.com/heyifan142857/SubBake/actions/workflows/ci.yml/badge.svg)](https://github.com/heyifan142857/SubBake/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/heyifan142857/SubBake/blob/main/LICENSE)

`subbake` 是一个字幕翻译 CLI，支持 `.srt`、`.vtt` 和按行处理的 `.txt`。

它默认把字幕翻译为中文，也支持 `en`、`ja`、`ko`、`fr`、`es`、`de` 等常用目标语言，并提供智能批次切分、上下文记忆、缓存、断点续跑、高风险批次复审和默认开启的运行时 agent 自修复。

## 安装

```bash
pip install subbake
```

## 快速开始

```bash
sbake translate input.srt --provider openai --model your-model
```

OpenAI 兼容接口：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://your-provider.example.com/v1"
```

Gemini：

```bash
export GEMINI_API_KEY="your_api_key"
sbake translate input.srt --provider gemini --model gemini-2.5-flash
```

Anthropic：

```bash
export ANTHROPIC_API_KEY="your_api_key"
sbake translate input.srt --provider anthropic --model your-model
```

本地联调：

```bash
sbake translate input.srt --provider mock
```

## 文档

- [项目首页](https://github.com/heyifan142857/SubBake)
- [文档与使用说明](https://github.com/heyifan142857/SubBake/wiki)
- [配置文件示例](https://github.com/heyifan142857/SubBake/blob/main/examples/subbake.toml)

## 常用命令

```bash
sbake translate --help
sbake check-key --help
sbake clean --help
```
