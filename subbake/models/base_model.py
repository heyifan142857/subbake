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
