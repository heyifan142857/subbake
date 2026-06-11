"""Transcription backends for audio/video to subtitle conversion.

Follows the same ABC + factory pattern as ``subbake.models.base_model``.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from subbake.entities import SubtitleDocument, SubtitleSegment
from subbake.parsers import load_document
from subbake.parsers.srt_parser import parse_srt_document
from subbake.parsers.vtt_parser import parse_vtt_document
from subbake.whisper_installer import WhisperInstaller


SUPPORTED_FORMATS = frozenset({"srt", "vtt", "txt"})
SUPPORTED_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"})
SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv"})


def is_media_file(path: Path) -> bool:
    """Return True if *path* has a supported audio or video extension."""
    suffix = path.suffix.lower()
    return suffix in SUPPORTED_AUDIO_EXTENSIONS or suffix in SUPPORTED_VIDEO_EXTENSIONS


def _ensure_audio(input_path: Path) -> Path:
    """Extract audio from video if needed. Return a WAV path suitable for transcription."""
    suffix = input_path.suffix.lower()
    if suffix in SUPPORTED_AUDIO_EXTENSIONS:
        return input_path
    if suffix in SUPPORTED_VIDEO_EXTENSIONS:
        return _extract_audio(input_path)
    raise ValueError(
        "Unsupported file type: {}. Supported: audio (wav/mp3/m4a/...) or video (mp4/mkv/...)".format(
            input_path.suffix
        )
    )


def _extract_audio(video_path: Path) -> Path:
    """Use ffmpeg to extract 16 kHz mono WAV audio from a video file."""
    if not _ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required to extract audio from video.\n"
            "Install it: sudo apt install ffmpeg  (Linux)\n"
            "            brew install ffmpeg       (macOS)\n"
            "            choco install ffmpeg       (Windows)"
        )

    wav_path = video_path.with_suffix(".wav").with_name(
        "{}_audio.wav".format(video_path.stem)
    )
    # Use a fixed temp path next to the input for simplicity
    tmp_wav = video_path.parent / ".subbake" / "tmp" / "{}_audio.wav".format(video_path.stem)
    tmp_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",             # overwrite existing
        "-i", str(video_path),
        "-vn",            # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",   # 16kHz sample rate
        "-ac", "1",       # mono
        str(tmp_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg failed: {}".format(result.stderr.strip()))
    return tmp_wav


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _parse_subtitle_text(text: str, fmt: str, path: Path) -> SubtitleDocument:
    """Parse subtitle text (SRT/VTT) into a SubtitleDocument."""
    if fmt == "srt":
        return _parse_srt_text(text, path)
    if fmt == "vtt":
        return _parse_vtt_text(text, path)
    # For txt, just wrap each line as a segment
    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    segments = [
        SubtitleSegment(id=str(i + 1), text=line, start="", end="")
        for i, line in enumerate(lines)
    ]
    return SubtitleDocument(path=path, format="txt", segments=segments)


def _parse_srt_text(text: str, path: Path) -> SubtitleDocument:
    """Parse SRT text (from memory) into a SubtitleDocument."""
    import tempfile as _tf
    with _tf.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp_path = Path(f.name)
    try:
        return parse_srt_document(tmp_path)
    finally:
        tmp_path.unlink()


def _parse_vtt_text(text: str, path: Path) -> SubtitleDocument:
    import tempfile as _tf
    with _tf.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp_path = Path(f.name)
    try:
        return parse_vtt_document(tmp_path)
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# TranscriberBackend ABC
# ---------------------------------------------------------------------------


class TranscriberBackend(ABC):
    """Abstract base for transcription backends (whisper.cpp, Whisper API, …)."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        output_format: str = "srt",
        prompt: str | None = None,
    ) -> SubtitleDocument:
        """Transcribe an audio file and return parsed subtitles."""
        raise NotImplementedError

    @abstractmethod
    def check_available(self) -> tuple[bool, str]:
        """Return (ok, message) describing whether this backend is usable."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Whisper API backend  (OpenAI-compatible /v1/audio/transcriptions)
# ---------------------------------------------------------------------------


class WhisperAPIBackend(TranscriberBackend):
    """Uses the OpenAI-compatible Whisper API (OpenAI, Groq, Together AI, …)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "whisper-1",
        timeout_seconds: float = 300.0,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    def check_available(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, (
                "Whisper API requires an API key.\n"
                "Set OPENAI_API_KEY environment variable, or pass --api-key, "
                "or configure api_key in subbake.toml."
            )
        return True, "Whisper API configured (model={})".format(self._model)

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        output_format: str = "srt",
        prompt: str | None = None,
    ) -> SubtitleDocument:
        """Transcribe using the OpenAI Whisper API."""
        if not self._api_key:
            raise RuntimeError("Whisper API key not configured.")

        url = "{}/audio/transcriptions".format(self._base_url)
        fmt = output_format.lower().lstrip(".")
        if fmt not in ("json", "text", "srt", "vtt", "verbose_json"):
            fmt = "srt"

        # Build multipart form data
        boundary = "----SubBakeBoundary{}".format(id(self))
        body = io.BytesIO()

        def _add_form_field(name: str, value: str) -> None:
            body.write(b"--" + boundary.encode() + b"\r\n")
            body.write('Content-Disposition: form-data; name="{}"\r\n\r\n'.format(name).encode())
            body.write(value.encode() + b"\r\n")

        def _add_form_file(field: str, filename: str, data: bytes) -> None:
            body.write(b"--" + boundary.encode() + b"\r\n")
            body.write(
                'Content-Disposition: form-data; name="{}"; filename="{}"\r\n'.format(field, filename).encode()
            )
            body.write(b"Content-Type: audio/wav\r\n\r\n")
            body.write(data)
            body.write(b"\r\n")

        _add_form_field("model", self._model)
        _add_form_field("response_format", fmt)
        if language:
            _add_form_field("language", language)
        if prompt:
            _add_form_field("prompt", prompt)

        with open(audio_path, "rb") as f:
            _add_form_file("file", audio_path.name, f.read())

        body.write(b"--" + boundary.encode() + b"--\r\n")

        # Send request
        req = urllib.request.Request(
            url,
            data=body.getvalue(),
            headers={
                "Content-Type": "multipart/form-data; boundary={}".format(boundary),
                "Authorization": "Bearer {}".format(self._api_key),
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                response_data = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "Whisper API error ({}): {}".format(exc.code, error_body)
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Whisper API request failed: {}".format(exc.reason)) from exc

        # Parse response
        if fmt in ("srt", "vtt"):
            # API returns raw SRT/VTT text
            return _parse_subtitle_text(response_data, fmt, audio_path)
        elif fmt == "json":
            data = json.loads(response_data)
            text = data.get("text", "")
        elif fmt == "verbose_json":
            data = json.loads(response_data)
            segments_data = data.get("segments", [])
            segments = [
                SubtitleSegment(
                    id=str(seg.get("id", i + 1)),
                    text=seg.get("text", "").strip(),
                    start=_format_seconds(seg.get("start", 0)),
                    end=_format_seconds(seg.get("end", 0)),
                )
                for i, seg in enumerate(segments_data)
            ]
            return SubtitleDocument(path=audio_path, format="txt", segments=segments)
        else:
            text = response_data

        # Wrap raw text as a single-segment txt document
        lines = [l for l in text.strip().splitlines() if l.strip()]
        segments = [
            SubtitleSegment(id=str(i + 1), text=line, start="", end="")
            for i, line in enumerate(lines)
        ]
        return SubtitleDocument(path=audio_path, format="txt", segments=segments)


def _format_seconds(seconds: float) -> str:
    """Format float seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, secs, millis)


# ---------------------------------------------------------------------------
# whisper.cpp backend  (local subprocess)
# ---------------------------------------------------------------------------


class WhisperCPPBackend(TranscriberBackend):
    """Uses the local whisper-cli binary (whisper.cpp)."""

    def __init__(
        self,
        *,
        project_root: Path,
        model: str = "small",
        binary_path: Path | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self._installer = WhisperInstaller(project_root)
        self._model_name = model
        self._custom_binary = binary_path
        self._extra_args = extra_args or []

    def check_available(self) -> tuple[bool, str]:
        return self._installer.check_available()

    def _resolve_binary(self) -> Path:
        if self._custom_binary and self._custom_binary.exists():
            return self._custom_binary
        return self._installer.ensure_available()

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        output_format: str = "srt",
        prompt: str | None = None,
    ) -> SubtitleDocument:
        """Transcribe using local whisper-cli."""
        binary = self._resolve_binary()
        model_path = self._installer.ensure_model(self._model_name)
        fmt = output_format.lower().lstrip(".")

        # whisper-cli writes output to a file next to the input
        out_dir = audio_path.parent
        base_name = audio_path.stem

        cmd = [
            str(binary),
            "-m", str(model_path),
            "-f", str(audio_path),
            "-os",  # output SRT (or -ovtt for VTT)
            "--output-file", str(out_dir / base_name),
        ]

        if fmt == "vtt":
            cmd.append("--output-vtt")
        else:
            cmd.append("--output-srt")

        if language:
            cmd.extend(["-l", language])
        if prompt:
            cmd.extend(["--prompt", prompt])
        cmd.extend(self._extra_args)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError("whisper-cli failed: {}".format(error_msg))

        # Load the generated subtitle file
        output_suffix = ".vtt" if fmt == "vtt" else ".srt"
        output_path = out_dir / "{}{}".format(base_name, output_suffix)

        if not output_path.exists():
            # Try with .wav.srt (whisper-cli may append to input name)
            output_path = out_dir / "{}{}".format(audio_path.name, output_suffix)
        if not output_path.exists():
            raise RuntimeError(
                "whisper-cli did not produce output file (checked {} and {}). "
                "Output: {}".format(
                    out_dir / "{}{}".format(base_name, output_suffix),
                    out_dir / "{}{}".format(audio_path.name, output_suffix),
                    result.stdout.strip(),
                )
            )

        try:
            doc = load_document(output_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to parse whisper.cpp output: {}".format(exc)
            ) from exc

        return doc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_transcriber(
    *,
    provider: str = "whisper_api",
    project_root: Path | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    whisper_model: str = "small",
    timeout_seconds: float = 300.0,
) -> TranscriberBackend:
    """Factory: return a TranscriberBackend based on *provider*.

    Supported providers:
    - ``whisper_api``  — OpenAI-compatible Whisper API
    - ``whisper_cpp``  — local whisper.cpp binary
    """
    normalized = provider.lower()
    if normalized == "whisper_api":
        kwargs: dict[str, Any] = {
            "timeout_seconds": timeout_seconds,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if model:
            kwargs["model"] = model
        return WhisperAPIBackend(**kwargs)
    if normalized == "whisper_cpp":
        if project_root is None:
            raise ValueError("project_root is required for whisper_cpp provider")
        return WhisperCPPBackend(
            project_root=project_root,
            model=whisper_model,
        )
    raise ValueError(
        "Unsupported transcriber provider: {}. Supported: whisper_api, whisper_cpp".format(provider)
    )


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------


def transcribe_to_srt(
    input_path: Path,
    *,
    project_root: Path,
    provider: str = "whisper_api",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    whisper_model: str = "small",
    language: str | None = None,
    output_format: str = "srt",
    prompt: str | None = None,
    output_path: Path | None = None,
    dry_run: bool = False,
) -> Path | SubtitleDocument:
    """Transcribe an audio/video file and save/return the subtitle document.

    Returns the output Path when *output_path* or a default path is used,
    or the SubtitleDocument when ``dry_run=True``.
    """
    if dry_run:
        # Just validate inputs without transcribing
        if not input_path.exists():
            raise FileNotFoundError("File not found: {}".format(input_path))
        if not is_media_file(input_path):
            raise ValueError("Not a supported media file: {}".format(input_path))
        # Return an empty document as a placeholder
        return SubtitleDocument(path=input_path, format=output_format, segments=[])

    # Extract audio if needed
    audio_path = _ensure_audio(input_path)

    # Build backend
    backend = build_transcriber(
        provider=provider,
        project_root=project_root,
        api_key=api_key,
        base_url=base_url,
        model=model,
        whisper_model=whisper_model,
    )

    # Transcribe
    doc = backend.transcribe(
        audio_path,
        language=language,
        output_format=output_format,
        prompt=prompt,
    )

    # Determine output path
    if output_path is not None:
        out = output_path
    else:
        fmt_ext = ".{}".format(output_format.lower().lstrip("."))
        out = input_path.with_suffix(fmt_ext)

    # Clean up extracted audio if it was temporary
    if audio_path != input_path and audio_path.exists():
        try:
            audio_path.unlink()
        except OSError:
            pass

    # Render and save
    from subbake.parsers import render_document
    content = render_document(doc, doc.segments, bilingual=False, output_format=output_format)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")

    return out
