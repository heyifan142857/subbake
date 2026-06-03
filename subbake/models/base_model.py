from __future__ import annotations

import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.message import Message
from http.client import HTTPMessage
from typing import Any

from subbake.entities import GlossaryEntry, TranslationLine, Usage
from subbake.languages import language_short_code, normalize_language_name
from subbake.title_matching import normalize_title_text, title_tokens_from_text


@dataclass(slots=True)
class BackendErrorMetadata:
    provider: str
    retryable: bool
    status_code: int | None = None
    request_id: str | None = None
    response_body: str | None = None
    reason: str | None = None
    retry_after_seconds: float | None = None
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "retryable": self.retryable,
            "status_code": self.status_code,
            "request_id": self.request_id,
            "reason": self.reason,
            "retry_after_seconds": self.retry_after_seconds,
            "url": self.url,
        }


class BackendRequestError(RuntimeError):
    def __init__(self, message: str, *, metadata: BackendErrorMetadata) -> None:
        super().__init__(message)
        self.metadata = metadata


class LLMBackend(ABC):
    @abstractmethod
    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        raise NotImplementedError

    @abstractmethod
    def check_credentials(self) -> tuple[bool, str]:
        raise NotImplementedError


class MockBackend(LLMBackend):
    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        prompt = "\n".join(message["content"] for message in messages)
        task = _extract_between(prompt, "TASK_START", "TASK_END").strip()
        usage = Usage(
            input_tokens=_estimate_tokens(prompt),
            output_tokens=0,
            total_tokens=0,
        )

        if task == "translate_subtitles":
            context = json.loads(_extract_between(prompt, "CONTEXT_JSON_START", "CONTEXT_JSON_END"))
            payload = json.loads(_extract_between(prompt, "BATCH_JSON_START", "BATCH_JSON_END"))
            target_language = normalize_language_name(str(context.get("tgt", "Chinese")))
            tag = language_short_code(target_language)
            lines = []
            glossary_updates = []
            for item in payload["lines"]:
                source_text = item["text"]
                translated = "" if not source_text.strip() else f"[MOCK-{tag}] {source_text}"
                lines.append({"id": item["id"], "translation": translated})
                names = re.findall(r"\b[A-Z][a-zA-Z]+\b", source_text)
                for name in names:
                    glossary_updates.append({"source": name, "target": name})
            result = {
                "lines": lines,
                "summary": "Mock summary of the latest subtitle batch.",
                "glossary_updates": glossary_updates,
            }
        elif task == "review_translations":
            payload = json.loads(_extract_between(prompt, "REVIEW_JSON_START", "REVIEW_JSON_END"))
            result = {
                "lines": [
                    {"id": item["id"], "translation": item["translation"]}
                    for item in payload["lines"]
                ],
                "review_notes": "Mock review kept translations unchanged.",
            }
        elif task == "agent_repair_translation":
            context = json.loads(_extract_between(prompt, "AGENT_REPAIR_JSON_START", "AGENT_REPAIR_JSON_END"))
            target_language = normalize_language_name(str(context.get("target_language", "Chinese")))
            tag = language_short_code(target_language)
            result = {
                "lines": [
                    {
                        "id": item["id"],
                        "translation": "" if not item["text"].strip() else f"[MOCK-{tag}] {item['text']}",
                    }
                    for item in context["source_lines"]
                ],
                "summary": "Mock agent repaired the subtitle batch.",
                "glossary_updates": [],
            }
        elif task == "agent_repair_review":
            context = json.loads(_extract_between(prompt, "AGENT_REPAIR_JSON_START", "AGENT_REPAIR_JSON_END"))
            current_by_id = {
                item["id"]: item.get("translation", "")
                for item in context.get("current_translations", [])
            }
            result = {
                "lines": [
                    {
                        "id": item["id"],
                        "translation": current_by_id.get(item["id"]) or item["text"],
                    }
                    for item in context["source_lines"]
                ],
                "review_notes": "Mock agent repaired the review batch.",
            }
        elif task == "agent_edit_subtitle":
            payload = json.loads(_extract_between(prompt, "EDIT_JSON_START", "EDIT_JSON_END"))
            result = {
                "lines": [
                    {"id": item["id"], "translation": item["translation"]}
                    for item in payload["lines"]
                ],
                "edit_notes": "Mock edit kept subtitles unchanged.",
            }
        elif task == "agent_decide":
            context = json.loads(_extract_between(prompt, "AGENT_CONTEXT_JSON_START", "AGENT_CONTEXT_JSON_END"))
            result = _mock_agent_decision(context)
        elif task == "agent_loop_decide":
            context = json.loads(_extract_between(prompt, "AGENT_LOOP_CONTEXT_JSON_START", "AGENT_LOOP_CONTEXT_JSON_END"))
            result = _mock_agent_loop_decision(context)
        else:
            raise ValueError(f"Unsupported mock task: {task}")

        rendered = json.dumps(result, ensure_ascii=False)
        usage.output_tokens = _estimate_tokens(rendered)
        usage.total_tokens = usage.input_tokens + usage.output_tokens
        return result, usage

    def check_credentials(self) -> tuple[bool, str]:
        return True, "Mock provider does not require an API key."


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        provider_label: str = "OpenAI-compatible",
        api_key_env_var: str = "OPENAI_API_KEY",
        base_url_env_var: str = "OPENAI_BASE_URL",
        default_base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.model = model
        self.provider_label = provider_label
        self.api_key_env_var = api_key_env_var
        self.base_url_env_var = base_url_env_var
        self.api_key = api_key or os.getenv(self.api_key_env_var)
        self.base_url = (base_url or os.getenv(self.base_url_env_var) or default_base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retry_attempts = 3
        if not self.api_key:
            raise ValueError(
                f"Missing API key for {self.provider_label} provider. "
                f"Set {self.api_key_env_var} or use --api-key."
            )

    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        try:
            return self._request_with_retries(payload)
        except BackendRequestError as exc:
            body = (exc.metadata.response_body or "").lower()
            if exc.metadata.status_code == 400 and "response_format" in body:
                fallback = {"model": self.model, "messages": messages}
                return self._request_with_retries(fallback)
            raise

    def check_credentials(self) -> tuple[bool, str]:
        request = urllib.request.Request(
            url=f"{self.base_url}/models",
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return False, _format_http_error(self.provider_label, exc.code, body)
        except urllib.error.URLError as exc:
            return False, f"{self.provider_label} credential check failed: {exc.reason}"

        model_count = len(data.get("data", [])) if isinstance(data, dict) else 0
        if model_count:
            return True, f"Credentials look valid. {model_count} model(s) visible from {self.provider_label}."
        return True, f"Credentials look valid. Successfully reached {self.provider_label}."

    def _request_with_retries(self, payload: dict) -> tuple[dict, Usage]:
        last_error: BackendRequestError | None = None
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                return self._request_once(payload)
            except BackendRequestError as exc:
                last_error = exc
                if attempt >= self.max_retry_attempts or not exc.metadata.retryable:
                    raise
                delay_seconds = self._retry_delay_seconds(exc.metadata, attempt)
                time.sleep(delay_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenAI request retry loop ended unexpectedly.")

    def _request_once(self, payload: dict) -> tuple[dict, Usage]:
        url = f"{self.base_url}/chat/completions"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                headers = response.headers
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise self._build_http_error(self.provider_label, exc, url) from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            raise self._build_transport_error(self.provider_label, exc, url)

        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", _estimate_tokens(json.dumps(payload))),
            output_tokens=usage_data.get("completion_tokens", _estimate_tokens(content)),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        if usage.total_tokens == 0:
            usage.total_tokens = usage.input_tokens + usage.output_tokens
        return parsed, usage

    def _retry_delay_seconds(self, metadata: BackendErrorMetadata, attempt: int) -> float:
        if metadata.retry_after_seconds is not None:
            return metadata.retry_after_seconds
        base_delay = min(8.0, 0.75 * (2 ** (attempt - 1)))
        jitter = random.uniform(0.0, 0.25)
        return base_delay + jitter

    def _build_http_error(self, provider_label: str, exc: urllib.error.HTTPError, url: str) -> BackendRequestError:
        body = exc.read().decode("utf-8", errors="replace")
        status_code = getattr(exc, "code", None)
        request_id = _extract_request_id(exc.headers)
        metadata = BackendErrorMetadata(
            provider=provider_label,
            retryable=_is_retryable_http_status(status_code),
            status_code=status_code,
            request_id=request_id,
            response_body=body,
            retry_after_seconds=_extract_retry_after_seconds(exc.headers),
            url=url,
        )
        return BackendRequestError(
            _format_backend_error_message(metadata),
            metadata=metadata,
        )

    def _build_transport_error(self, provider_label: str, exc: BaseException, url: str) -> BackendRequestError:
        metadata = BackendErrorMetadata(
            provider=provider_label,
            retryable=True,
            reason=_stringify_transport_reason(exc),
            url=url,
        )
        return BackendRequestError(
            _format_backend_error_message(metadata),
            metadata=metadata,
        )


class GeminiBackend(OpenAIBackend):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            provider_label="Gemini",
            api_key_env_var="GEMINI_API_KEY",
            base_url_env_var="GEMINI_BASE_URL",
            default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )


class AnthropicBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.max_retry_attempts = 3
        if not self.api_key:
            raise ValueError("Missing API key for Anthropic provider. Set ANTHROPIC_API_KEY or use --api-key.")

    def generate_json(self, messages: list[dict[str, str]]) -> tuple[dict, Usage]:
        system_parts = [message["content"] for message in messages if message["role"] == "system"]
        body_messages = [
            {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}
            for message in messages
            if message["role"] != "system"
        ]
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": "\n\n".join(system_parts),
            "messages": body_messages,
        }
        data, _ = self._request_with_retries(payload)

        chunks = [
            item.get("text", "")
            for item in data.get("content", [])
            if item.get("type") == "text"
        ]
        text = "\n".join(chunks)
        parsed = _extract_json_object(text)
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", _estimate_tokens(json.dumps(payload))),
            output_tokens=usage_data.get("output_tokens", _estimate_tokens(text)),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )
        if usage.total_tokens == 0:
            usage.total_tokens = usage.input_tokens + usage.output_tokens
        return parsed, usage

    def _request_with_retries(self, payload: dict) -> tuple[dict, Message | HTTPMessage | None]:
        last_error: BackendRequestError | None = None
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                return self._request_once(payload)
            except BackendRequestError as exc:
                last_error = exc
                if attempt >= self.max_retry_attempts or not exc.metadata.retryable:
                    raise
                delay_seconds = self._retry_delay_seconds(exc.metadata, attempt)
                time.sleep(delay_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Anthropic request retry loop ended unexpectedly.")

    def _request_once(self, payload: dict) -> tuple[dict, Message | HTTPMessage | None]:
        url = "https://api.anthropic.com/v1/messages"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8")), response.headers
        except urllib.error.HTTPError as exc:
            raise self._build_http_error("Anthropic", exc, url) from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            raise self._build_transport_error("Anthropic", exc, url)

    def _retry_delay_seconds(self, metadata: BackendErrorMetadata, attempt: int) -> float:
        if metadata.retry_after_seconds is not None:
            return metadata.retry_after_seconds
        base_delay = min(8.0, 0.75 * (2 ** (attempt - 1)))
        jitter = random.uniform(0.0, 0.25)
        return base_delay + jitter

    def _build_http_error(self, provider_label: str, exc: urllib.error.HTTPError, url: str) -> BackendRequestError:
        body = exc.read().decode("utf-8", errors="replace")
        status_code = getattr(exc, "code", None)
        request_id = _extract_request_id(exc.headers)
        metadata = BackendErrorMetadata(
            provider=provider_label,
            retryable=_is_retryable_http_status(status_code),
            status_code=status_code,
            request_id=request_id,
            response_body=body,
            retry_after_seconds=_extract_retry_after_seconds(exc.headers),
            url=url,
        )
        return BackendRequestError(
            _format_backend_error_message(metadata),
            metadata=metadata,
        )

    def _build_transport_error(self, provider_label: str, exc: BaseException, url: str) -> BackendRequestError:
        metadata = BackendErrorMetadata(
            provider=provider_label,
            retryable=True,
            reason=_stringify_transport_reason(exc),
            url=url,
        )
        return BackendRequestError(
            _format_backend_error_message(metadata),
            metadata=metadata,
        )

    def check_credentials(self) -> tuple[bool, str]:
        request = urllib.request.Request(
            url="https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return False, _format_http_error("Anthropic", exc.code, body)
        except urllib.error.URLError as exc:
            return False, f"Anthropic credential check failed: {exc.reason}"

        model_count = len(data.get("data", [])) if isinstance(data, dict) else 0
        if model_count:
            return True, f"Credentials look valid. {model_count} model(s) visible from Anthropic."
        return True, "Credentials look valid. Successfully reached Anthropic."


def build_backend(
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 120.0,
) -> LLMBackend:
    normalized = provider.lower()
    if normalized == "mock":
        return MockBackend()
    if normalized in {"openai", "openai-compatible", "compatible"}:
        return OpenAIBackend(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
    if normalized == "gemini":
        return GeminiBackend(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
    if normalized == "anthropic":
        return AnthropicBackend(
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def parse_translation_lines(items: list[dict]) -> list[TranslationLine]:
    lines: list[TranslationLine] = []
    for item in items:
        translation = item.get("translation")
        if translation is None:
            translation = item.get("text")
        if translation is None:
            translation = item.get("target")
        if translation is None:
            raise KeyError("translation")
        lines.append(TranslationLine(id=str(item["id"]), translation=str(translation)))
    return lines


def parse_glossary_entries(items: list[dict | str] | dict[str, str]) -> list[GlossaryEntry]:
    if isinstance(items, dict):
        iterable = [{"source": key, "target": value} for key, value in items.items()]
    else:
        iterable = items
    entries: list[GlossaryEntry] = []
    for item in iterable:
        entry = _coerce_glossary_entry(item)
        if entry is not None:
            entries.append(entry)
    return entries


def _coerce_glossary_entry(item: dict | str) -> GlossaryEntry | None:
    if isinstance(item, dict):
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if source and target:
            return GlossaryEntry(source=source, target=target)
        return None
    if isinstance(item, str):
        return _parse_glossary_string(item)
    return None


def _parse_glossary_string(text: str) -> GlossaryEntry | None:
    stripped = text.strip()
    if not stripped:
        return None

    parenthetical = re.match(
        r"^(?P<target>[^()]+?)\s*\((?P<source>[^()]+)\)\s*(?:[:：-].*)?$",
        stripped,
    )
    if parenthetical is not None:
        source = parenthetical.group("source").strip()
        target = parenthetical.group("target").strip(" -–—:：")
        if source and target:
            return GlossaryEntry(source=source, target=target)

    for delimiter in (" - ", " – ", " — ", ": ", "："):
        if delimiter not in stripped:
            continue
        left, right = stripped.split(delimiter, 1)
        source, target = _order_glossary_pair(left.strip(), right.strip())
        if source and target:
            return GlossaryEntry(source=source, target=target)
    return None


def _order_glossary_pair(left: str, right: str) -> tuple[str, str]:
    if not left or not right:
        return "", ""

    left_has_latin = _contains_latin(left)
    right_has_latin = _contains_latin(right)
    left_has_cjk = _contains_cjk(left)
    right_has_cjk = _contains_cjk(right)
    if left_has_latin and right_has_cjk and not right_has_latin:
        return left, right
    if right_has_latin and left_has_cjk and not left_has_latin:
        return right, left
    return left, right


def _contains_latin(text: str) -> bool:
    return re.search(r"[A-Za-z]", text) is not None


def _contains_cjk(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def _mock_agent_decision(context: dict[str, Any]) -> dict[str, Any]:
    message = str(context.get("user_message") or "")
    lowered = message.casefold()
    references = [
        item for item in context.get("references", [])
        if isinstance(item, dict)
    ]
    mode = str(context.get("mode") or "chat")

    def decision(tool_name: str, arguments: dict[str, Any], text: str = "Running tool.") -> dict[str, Any]:
        if mode == "plan":
            return {
                "action": "plan",
                "message": f"Plan:\n- {text}",
                "tool_calls": [{"tool_name": tool_name, "arguments": arguments}],
            }
        return {
            "action": "tool_call",
            "message": text,
            "tool_name": tool_name,
            "arguments": arguments,
        }

    if any(word in lowered for word in ("profile", "model", "模型", "配置")) and "切" in message:
        profile = _last_word(message)
        return decision("switch_profile", {"profile": profile}, f"Switching profile to {profile}.")

    if references and any(word in lowered for word in ("分析", "诊断", "错误", "失败", "log", "error", "diagnose")):
        return decision("diagnose_path", {"path": references[0]["path"]}, "Diagnosing failure log.")

    if references and _looks_like_edit_request(message, references[0]):
        instruction = _remove_mock_refs(message).strip() or message
        return decision(
            "edit_subtitle",
            {"path": references[0]["path"], "instruction": instruction},
            "Editing generated subtitle.",
        )

    if any(word in lowered for word in ("删除", "delete", "remove")) and references:
        return decision("delete_file", {"path": references[0]["path"]}, "Deleting file.")

    if any(word in lowered for word in ("改名", "rename", "重命名")):
        if len(references) >= 2:
            return decision(
                "rename_path",
                {"old_path": references[0]["path"], "new_path": references[1]["path"]},
                "Renaming file.",
            )
        return {"action": "ask_user", "message": "Which source and destination paths should I use?"}

    if any(word in lowered for word in ("追加", "append")) and references:
        return decision(
            "append_file",
            {"path": references[0]["path"], "content": _content_after_refs(message)},
            "Appending file.",
        )

    if any(word in lowered for word in ("替换", "replace")) and references and "=>" in message:
        old, _, new = _content_after_refs(message).partition("=>")
        return decision(
            "replace_in_file",
            {"path": references[0]["path"], "old": old.strip(), "new": new.strip()},
            "Replacing text.",
        )

    if any(word in lowered for word in ("创建", "create", "新建")) and references:
        return decision(
            "create_file",
            {"path": references[0]["path"], "content": _content_after_refs(message)},
            "Creating file.",
        )

    if references and any(word in lowered for word in ("读取", "查看", "read", "show")):
        return decision("read_file", {"path": references[0]["path"]}, "Reading file.")

    if references and any(word in lowered for word in ("搜索", "search", "查找")):
        return decision("search_files", {"path": references[0]["path"], "pattern": _last_word(message)}, "Searching files.")

    if references:
        first = references[0]
        if first.get("is_dir"):
            return decision("translate_series", {"path": first["path"]}, "Translating subtitle folder.")
        suffix = str(first.get("suffix") or "").lower()
        if suffix in {".srt", ".vtt", ".txt"}:
            return decision("translate_file", {"path": first["path"]}, "Translating subtitle file.")
        return decision("diagnose_path", {"path": first["path"]}, "Diagnosing referenced file.")

    return {
        "action": "respond",
        "message": "I can help with subtitle translation, diagnostics, profiles, and project-local file changes. Reference files with @path.",
    }


def _mock_agent_loop_decision(context: dict[str, Any]) -> dict[str, Any]:
    message = str(context.get("user_message") or context.get("original_user_message") or "")
    lowered = message.casefold()
    references = [
        item for item in context.get("references", [])
        if isinstance(item, dict)
    ]
    observations = [
        item for item in context.get("observations", [])
        if isinstance(item, dict)
    ]

    def discover(
        tool_name: str,
        arguments: dict[str, Any],
        text: str,
        *,
        confidence: float = 0.75,
    ) -> dict[str, Any]:
        return {
            "action": "tool_call",
            "message": text,
            "tool_name": tool_name,
            "arguments": arguments,
            "reason": text,
            "confidence": confidence,
        }

    def final(
        tool_name: str,
        arguments: dict[str, Any],
        text: str,
        *,
        confidence: float = 0.85,
    ) -> dict[str, Any]:
        return {
            "action": "final_tool_call",
            "message": text,
            "tool_name": tool_name,
            "arguments": arguments,
            "reason": text,
            "confidence": confidence,
        }

    latest_observation = observations[-1] if observations else None
    if latest_observation is not None:
        tool_name = str(latest_observation.get("tool_name") or "")
        data = latest_observation.get("data")
        if not isinstance(data, dict):
            data = {}

        if tool_name in {"candidate_subtitles", "search_files"}:
            candidates = [
                item for item in data.get("candidates", [])
                if isinstance(item, dict)
            ]
            if _mock_search_requested(message):
                return {
                    "action": "respond",
                    "message": _mock_candidate_response(candidates, data.get("matches")),
                    "reason": "Search results were observed.",
                    "confidence": 0.85,
                }
            strong = _mock_strong_subtitle_candidates(candidates)
            if len(strong) == 1:
                path = _mock_executable_candidate_path(strong[0])
                return final(
                    "translate_file",
                    {"path": path},
                    f"Translating selected subtitle {path}.",
                    confidence=0.9,
                )
            if len(strong) > 1:
                choices = ", ".join(_mock_executable_candidate_path(candidate) for candidate in strong[:5])
                return {
                    "action": "ask_user",
                    "message": f"I found multiple subtitle candidates. Please choose one: {choices}",
                    "reason": "Multiple strong subtitle candidates remain.",
                    "confidence": 0.9,
                }
            if not _mock_has_observation(observations, "recent_translations"):
                return discover(
                    "recent_translations",
                    {},
                    "Checking recent translations because no subtitle candidate was selected.",
                    confidence=0.65,
                )
            return {
                "action": "ask_user",
                "message": "I could not find a subtitle candidate. Please reference the source subtitle with @path.",
                "reason": "No candidate subtitle found.",
                "confidence": 0.8,
            }

        if tool_name == "recent_translations":
            records = [
                item for item in data.get("translations", [])
                if isinstance(item, dict)
            ]
            if records:
                first = records[0]
                selected_tool = str(first.get("tool_name") or "translate_file")
                path = str(first.get("path") or "")
                arguments: dict[str, Any] = {"path": path}
                if selected_tool == "translate_series":
                    if first.get("suffixes"):
                        arguments["suffixes"] = first["suffixes"]
                    if first.get("recursive") is not None:
                        arguments["recursive"] = first["recursive"]
                return final(
                    selected_tool,
                    arguments,
                    f"Retargeting recent translation {path}.",
                    confidence=0.82,
                )
            return {
                "action": "ask_user",
                "message": "I do not have a recent translation to retarget. Please reference the subtitle with @path.",
                "reason": "No recent translation was observed.",
                "confidence": 0.82,
            }

        if tool_name == "list_files":
            files = [
                str(item.get("path"))
                for item in data.get("files", [])
                if isinstance(item, dict) and item.get("path")
            ]
            return {
                "action": "respond",
                "message": "\n".join(files) if files else "No files found.",
                "reason": "Listed project files.",
                "confidence": 0.9,
            }

        if tool_name == "read_file_preview":
            return {
                "action": "respond",
                "message": str(data.get("text") or ""),
                "reason": "Read file preview.",
                "confidence": 0.9,
            }

    if any(word in lowered for word in ("profile", "model", "模型", "配置")) and "切" in message:
        profile = _last_word(message)
        return final("switch_profile", {"profile": profile}, f"Switching profile to {profile}.")

    if references and any(word in lowered for word in ("分析", "诊断", "错误", "失败", "log", "error", "diagnose")):
        return final("diagnose_path", {"path": references[0]["path"]}, "Diagnosing failure log.")

    if references and _looks_like_edit_request(message, references[0]):
        instruction = _remove_mock_refs(message).strip() or message
        return final(
            "edit_subtitle",
            {"path": references[0]["path"], "instruction": instruction},
            "Editing generated subtitle.",
        )

    if _mock_search_requested(message):
        path = references[0]["path"] if references else "."
        return discover(
            "search_files",
            {"path": path, "pattern": _mock_search_pattern(message)},
            "Searching project files.",
        )

    if references and any(word in lowered for word in ("读取", "查看", "read", "show")):
        return discover(
            "read_file_preview",
            {"path": references[0]["path"], "limit": 4000},
            "Reading file preview.",
        )

    if references:
        first = references[0]
        if first.get("is_dir"):
            return final("translate_series", {"path": first["path"]}, "Translating subtitle folder.")
        suffix = str(first.get("suffix") or "").lower()
        if suffix in {".srt", ".vtt", ".txt"}:
            return final("translate_file", {"path": first["path"]}, "Translating subtitle file.")
        return final("diagnose_path", {"path": first["path"]}, "Diagnosing referenced file.")

    if _mock_list_request(message):
        return discover("list_files", {"path": ".", "recursive": False}, "Listing project files.")

    if _mock_retarget_translation_requested(message):
        if _mock_has_title_hint(message):
            return discover(
                "candidate_subtitles",
                {"path": ".", "query": message},
                "Finding candidate subtitles before translation.",
                confidence=0.82,
            )
        return discover(
            "recent_translations",
            {},
            "Checking recent translations before retargeting.",
            confidence=0.72,
        )

    return {
        "action": "respond",
        "message": "I can help with subtitle translation, diagnostics, profiles, and project-local file changes. Reference files with @path.",
        "reason": "No concrete action was detected.",
        "confidence": 0.6,
    }


def _looks_like_edit_request(message: str, reference: dict[str, Any]) -> bool:
    lowered = message.casefold()
    return bool(reference.get("generated_subtitle")) and any(
        word in lowered
        for word in ("修改", "修正", "统一", "edit", "fix", "change")
    )


def _remove_mock_refs(message: str) -> str:
    return re.sub(r"@(?:\"[^\"]+\"|'[^']+'|\S+)", "", message)


def _content_after_refs(message: str) -> str:
    cleaned = _remove_mock_refs(message).strip()
    cleaned = re.sub(r"^(创建|新建|追加|替换)\s*", "", cleaned).strip()
    cleaned = re.sub(r"^(create|append|replace)\b", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _mock_retarget_translation_requested(message: str) -> bool:
    lowered = message.casefold()
    if not any(
        word in lowered
        for word in (
            "变成",
            "变为",
            "改成",
            "改为",
            "换成",
            "做成",
            "重新生成",
            "再生成",
            "rerender",
            "re-render",
            "make",
            "translate",
            "翻译",
        )
    ):
        return False
    return any(word in lowered for word in ("字幕", "subtitle", "翻译", "translation", "双语", "bilingual", "中英"))


def _mock_search_requested(message: str) -> bool:
    lowered = message.casefold()
    return any(word in lowered for word in ("搜索", "查找", "search", "find"))


def _mock_list_request(message: str) -> bool:
    lowered = message.casefold()
    return any(word in lowered for word in ("当前目录", "目录下", "current directory", "cwd")) and any(
        word in lowered
        for word in ("有什么", "列", "查看", "读取", "list", "show", "read")
    )


def _mock_has_title_hint(message: str) -> bool:
    tokens = [
        token
        for token in title_tokens_from_text(message)
        if token not in {"the", "a", "an", "to", "of", "and", "or", "can", "could"}
    ]
    if tokens:
        return True
    spans = re.findall(r"[A-Za-z0-9][A-Za-z0-9 ._'()-]{2,}", message)
    return any(len(normalize_title_text(span).split()) >= 1 for span in spans)


def _mock_search_pattern(message: str) -> str:
    cleaned = _remove_mock_refs(message).strip()
    for marker in ("搜索", "查找", "search", "find"):
        match = re.search(rf"\b{re.escape(marker)}\b|{re.escape(marker)}", cleaned, flags=re.IGNORECASE)
        if match is not None:
            return cleaned[match.end():].strip(" ：:=,，。")
    return _last_word(cleaned)


def _mock_strong_subtitle_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subtitle_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("kind") or "") in {"source", "translated", "bilingual"}
    ]
    if not subtitle_candidates:
        return []
    top_score = float(subtitle_candidates[0].get("score") or 0)
    return [
        candidate
        for candidate in subtitle_candidates
        if float(candidate.get("score") or 0) >= top_score - 8.0
    ]


def _mock_executable_candidate_path(candidate: dict[str, Any]) -> str:
    inferred_source_path = candidate.get("inferred_source_path")
    if inferred_source_path:
        return str(inferred_source_path)
    return str(candidate.get("path") or "")


def _mock_candidate_response(candidates: list[dict[str, Any]], matches: object) -> str:
    lines: list[str] = []
    for candidate in candidates[:10]:
        path = candidate.get("path")
        if not path:
            continue
        line = f"{path} kind={candidate.get('kind')} score={candidate.get('score')}"
        inferred_source_path = candidate.get("inferred_source_path")
        if inferred_source_path:
            line = f"{line} inferred_source_path={inferred_source_path}"
        lines.append(line)
    if not lines and isinstance(matches, list):
        lines = [str(match) for match in matches[:10]]
    return "\n".join(lines) if lines else "No matches."


def _mock_has_observation(observations: list[dict[str, Any]], tool_name: str) -> bool:
    return any(str(observation.get("tool_name") or "") == tool_name for observation in observations)


def _last_word(message: str) -> str:
    parts = [part for part in re.split(r"\s+", message.strip()) if part]
    return parts[-1] if parts else ""


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    start_index = text.index(start_marker) + len(start_marker)
    end_index = text.index(end_marker, start_index)
    return text[start_index:end_index].strip()


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, end_index = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
        if end_index:
            break
    raise ValueError(f"Failed to parse JSON object from model output: {text}")


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


def _format_http_error(provider_label: str, status_code: int, body: str) -> str:
    normalized = body.strip().replace("\n", " ")
    if status_code in {401, 403}:
        return f"{provider_label} rejected the credentials ({status_code}): {normalized}"
    return f"{provider_label} credential check failed ({status_code}): {normalized}"


def _extract_request_id(headers: Message | HTTPMessage | None) -> str | None:
    if headers is None:
        return None
    for header_name in ("x-request-id", "request-id", "anthropic-request-id", "x-goog-request-id"):
        value = headers.get(header_name)
        if value:
            return str(value)
    return None


def _extract_retry_after_seconds(headers: Message | HTTPMessage | None) -> float | None:
    if headers is None:
        return None
    retry_after = headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


def _is_retryable_http_status(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code in {408, 409, 429} or 500 <= status_code < 600


def _stringify_transport_reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return str(exc)


def _format_backend_error_message(metadata: BackendErrorMetadata) -> str:
    parts = [f"{metadata.provider} request failed"]
    if metadata.status_code is not None:
        parts.append(f"status={metadata.status_code}")
    if metadata.request_id:
        parts.append(f"request_id={metadata.request_id}")
    if metadata.reason:
        parts.append(f"reason={metadata.reason}")
    if metadata.retryable:
        parts.append("retryable=yes")
    else:
        parts.append("retryable=no")
    body = (metadata.response_body or "").strip().replace("\n", " ")
    if body:
        parts.append(f"body={body[:400]}")
    return "; ".join(parts)
