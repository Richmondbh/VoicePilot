"""
Speech-to-text with faster-whisper (a fast re-implementation of OpenAI Whisper).
Runs locally on the CPU - free, no API key.

Usage pattern taken from the faster-whisper README:
https://github.com/SYSTRAN/faster-whisper#usage
faster-whisper decodes audio with PyAV, so the browser's .webm recordings work
without installing ffmpeg separately.
"""
import tempfile
from pathlib import Path

from . import config

_model = None


def get_model():
    """Load the Whisper model once (first call downloads it, ~150 MB for 'base')."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe_file(path: str | Path) -> str:
    segments, _info = get_model().transcribe(
        str(path), language=config.WHISPER_LANGUAGE, beam_size=5, vad_filter=True
    )
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe_bytes(data: bytes, suffix: str = ".webm") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return transcribe_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
