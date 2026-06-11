"""Tests for the transcription module (subbake.transcriber)."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subbake.transcriber import (
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    TranscriberBackend,
    WhisperAPIBackend,
    WhisperCPPBackend,
    build_transcriber,
    is_media_file,
    _ensure_audio,
    _parse_subtitle_text,
)


class FakeResponse:
    def __init__(self, data: bytes, status: int = 200) -> None:
        self._data = data
        self.status = status

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass


class TestIsMediaFile(unittest.TestCase):
    def test_audio_extensions(self) -> None:
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            self.assertTrue(is_media_file(Path(f"test{ext}")), f"expected {ext} to be media")

    def test_video_extensions(self) -> None:
        for ext in SUPPORTED_VIDEO_EXTENSIONS:
            self.assertTrue(is_media_file(Path(f"video{ext}")), f"expected {ext} to be media")

    def test_subtitle_not_media(self) -> None:
        self.assertFalse(is_media_file(Path("test.srt")))
        self.assertFalse(is_media_file(Path("test.vtt")))
        self.assertFalse(is_media_file(Path("test.txt")))

    def test_unknown_not_media(self) -> None:
        self.assertFalse(is_media_file(Path("test.pdf")))


class TestEnsureAudio(unittest.TestCase):
    def test_audio_passthrough(self) -> None:
        wav = Path("/tmp/test.wav")
        result = _ensure_audio(wav)
        self.assertEqual(result, wav)

    def test_video_needs_ffmpeg(self) -> None:
        mp4 = Path("/tmp/test.mp4")
        with self.assertRaises(RuntimeError) as ctx:
            _ensure_audio(mp4)
        self.assertIn("ffmpeg", str(ctx.exception))

    def test_unsupported_format(self) -> None:
        pdf = Path("/tmp/test.pdf")
        with self.assertRaises(ValueError):
            _ensure_audio(pdf)


class TestTranscriberBackend(unittest.TestCase):
    def test_abc_cannot_instantiate(self) -> None:
        with self.assertRaises(TypeError):
            TranscriberBackend()  # type: ignore


class TestWhisperAPIBackend(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = WhisperAPIBackend(api_key="test-key")

    def test_check_available_with_key(self) -> None:
        ok, msg = self.backend.check_available()
        self.assertTrue(ok)
        self.assertIn("whisper-1", msg)

    def test_check_available_without_key(self) -> None:
        backend = WhisperAPIBackend(api_key="")
        ok, msg = backend.check_available()
        self.assertFalse(ok)
        self.assertIn("API key", msg)

    @patch("urllib.request.urlopen")
    def test_transcribe_srt_response(self, mock_urlopen) -> None:
        """Test parsing an SRT response from the API."""
        srt_response = (
            "1\n"
            "00:00:01,000 --> 00:00:04,000\n"
            "Hello world.\n\n"
            "2\n"
            "00:00:05,000 --> 00:00:08,000\n"
            "How are you?\n"
        )

        mock_urlopen.return_value = FakeResponse(srt_response.encode("utf-8"))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = Path(f.name)

        try:
            doc = self.backend.transcribe(audio_path, output_format="srt")
            self.assertEqual(doc.format, "srt")
            self.assertEqual(len(doc.segments), 2)
            self.assertEqual(doc.segments[0].text, "Hello world.")
            self.assertEqual(doc.segments[1].text, "How are you?")
        finally:
            audio_path.unlink(missing_ok=True)

    @patch("urllib.request.urlopen")
    def test_transcribe_json_response(self, mock_urlopen) -> None:
        json_response = json.dumps({"text": "Hello world. How are you?"})

        mock_urlopen.return_value = FakeResponse(json_response.encode("utf-8"))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = Path(f.name)

        try:
            doc = self.backend.transcribe(audio_path, output_format="json")
            self.assertEqual(doc.format, "txt")
            self.assertEqual(len(doc.segments), 1)
            self.assertEqual(doc.segments[0].text, "Hello world. How are you?")
        finally:
            audio_path.unlink(missing_ok=True)

    @patch("urllib.request.urlopen")
    def test_transcribe_with_language_and_prompt(self, mock_urlopen) -> None:
        """Test that language and prompt are passed in the form data."""
        srt_response = "1\n00:00:01,000 --> 00:00:02,000\nTest\n"
        mock_urlopen.return_value = FakeResponse(srt_response.encode("utf-8"))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake")
            audio_path = Path(f.name)

        try:
            doc = self.backend.transcribe(audio_path, language="en", prompt="Technical terms")
            self.assertEqual(len(doc.segments), 1)
            # Verify the request body contained language and prompt
            request_body = mock_urlopen.call_args[0][0].data.decode()
            self.assertIn('name="language"', request_body)
            self.assertIn("en", request_body)
            self.assertIn('name="prompt"', request_body)
            self.assertIn("Technical terms", request_body)
        finally:
            audio_path.unlink(missing_ok=True)

    @patch("urllib.request.urlopen")
    def test_transcribe_api_error(self, mock_urlopen) -> None:
        """Test handling of API errors."""
        from urllib.error import HTTPError
        from email.message import Message

        msg = Message()
        error_response = b'{"error": "invalid file"}'

        mock_urlopen.side_effect = HTTPError(
            "https://api.openai.com/v1/audio/transcriptions",
            400,
            "Bad Request",
            msg,
            io.BytesIO(error_response),
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake")
            audio_path = Path(f.name)

        try:
            with self.assertRaises(RuntimeError) as ctx:
                self.backend.transcribe(audio_path)
            self.assertIn("400", str(ctx.exception))
            self.assertIn("invalid file", str(ctx.exception))
        finally:
            audio_path.unlink(missing_ok=True)


class TestBuildTranscriber(unittest.TestCase):
    def test_whisper_api_default(self) -> None:
        backend = build_transcriber(provider="whisper_api")
        self.assertIsInstance(backend, WhisperAPIBackend)

    def test_whisper_cpp_needs_project_root(self) -> None:
        with self.assertRaises(ValueError):
            build_transcriber(provider="whisper_cpp")

    def test_whisper_cpp_with_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = build_transcriber(provider="whisper_cpp", project_root=Path(tmp))
            self.assertIsInstance(backend, WhisperCPPBackend)

    def test_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            build_transcriber(provider="invalid")


class TestParseSubtitleText(unittest.TestCase):
    def test_parse_srt_text(self) -> None:
        srt = (
            "1\n"
            "00:00:01,000 --> 00:00:04,000\n"
            "Hello world.\n"
        )
        path = Path("/fake/test.wav")
        doc = _parse_subtitle_text(srt, "srt", path)
        self.assertEqual(doc.format, "srt")
        self.assertEqual(len(doc.segments), 1)
        self.assertEqual(doc.segments[0].text, "Hello world.")

    def test_parse_vtt_text(self) -> None:
        vtt = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:04.000\n"
            "Hello VTT\n"
        )
        path = Path("/fake/test.wav")
        doc = _parse_subtitle_text(vtt, "vtt", path)
        self.assertEqual(doc.format, "vtt")
        self.assertEqual(len(doc.segments), 1)

    def test_parse_txt_text(self) -> None:
        text = "Line one\nLine two\nLine three"
        path = Path("/fake/test.wav")
        doc = _parse_subtitle_text(text, "txt", path)
        self.assertEqual(doc.format, "txt")
        self.assertEqual(len(doc.segments), 3)


if __name__ == "__main__":
    unittest.main()
