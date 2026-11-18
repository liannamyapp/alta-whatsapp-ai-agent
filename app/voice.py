"""Speech-to-text and text-to-speech for WhatsApp voice notes."""
from __future__ import annotations

import io

from .config import settings
from .llm import client


def transcribe(audio: bytes, filename: str = "voice.ogg") -> str:
    buf = io.BytesIO(audio)
    buf.name = filename  # the API infers the format from the name
    resp = client().audio.transcriptions.create(model=settings.stt_model, file=buf)
    return resp.text.strip()


def synthesize(text: str) -> bytes:
    """Return an OGG/Opus clip, the format WhatsApp uses for voice notes."""
    resp = client().audio.speech.create(model=settings.tts_model, voice=settings.tts_voice,
                                        input=text, response_format="opus")
    return resp.content
