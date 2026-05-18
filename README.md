# subbake

[![PyPI version](https://img.shields.io/pypi/v/subbake)](https://pypi.org/project/subbake/)
[![Python versions](https://img.shields.io/pypi/pyversions/subbake)](https://pypi.org/project/subbake/)
[![CI](https://github.com/heyifan142857/SubBake/actions/workflows/ci.yml/badge.svg)](https://github.com/heyifan142857/SubBake/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`subbake` 是一个简单的字幕翻译 CLI，默认将字幕翻译为中文，也可以通过 `--target-language en` 这类参数切到其他常用语言。

它的目标是用尽量直接的命令行工作流处理字幕翻译，同时保留对批量翻译、断点续跑、缓存和复审这些实用能力的支持。

![subbake CLI demo](assets/subbake-demo.gif)

## 核心能力

- 支持 `.srt`、`.vtt` 和按行处理的 `.txt`
- 支持常用目标语言缩写，如 `zh`、`en`、`ja`
- 智能批量翻译、上下文记忆和 `--fast` 快速模式
- glossary、cache、translation memory、断点续跑
- 默认开启运行时 agent 自修复：模型输出结构错误时读取失败日志、自动修正并继续跑，可用 `--no-agent` 关闭
- 高风险 batch 定向复审与失败样本落盘
- `subbake.toml` 配置文件和多 profile 模型配置
- 基于 `rich` 的命令行可视化，包括进度、时间线和 Token 用量

## 快速开始

安装并运行：

```bash
pip install subbake
sbake translate input.srt --provider openai --model your-model
```

使用 OpenAI 兼容接口时：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://your-provider.example.com/v1"
```

Gemini 使用：

```bash
export GEMINI_API_KEY="your_api_key"
sbake translate input.srt --provider gemini --model gemini-2.5-flash
```

Anthropic 使用：

```bash
export ANTHROPIC_API_KEY="your_api_key"
sbake translate input.srt --provider anthropic --model your-model
```

内置 `mock` 后端可用于本地联调：

```bash
sbake translate input.srt --provider mock
```

翻译到其他目标语言：

```bash
sbake translate input.srt --provider openai --model your-model --target-language en
```

配置文件示例见 [examples/subbake.toml](examples/subbake.toml)。

## 文档

文档与使用说明见 [项目 Wiki](https://github.com/heyifan142857/SubBake/wiki)。

完整命令说明仍可直接查看：

```bash
sbake translate --help
sbake check-key --help
sbake clean --help
```
