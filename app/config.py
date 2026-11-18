"""Runtime configuration, read from environment variables."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "gpt-4o-mini"))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "gpt-4o"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "text-embedding-3-small"))
    stt_model: str = field(default_factory=lambda: _env("STT_MODEL", "whisper-1"))
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "tts-1"))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "nova"))

    greenapi_url: str = field(default_factory=lambda: _env("GREENAPI_URL", "https://api.green-api.com"))
    greenapi_media_url: str = field(default_factory=lambda: _env("GREENAPI_MEDIA_URL", "https://media.green-api.com"))
    greenapi_instance: str = field(default_factory=lambda: _env("GREENAPI_INSTANCE_ID"))
    greenapi_token: str = field(default_factory=lambda: _env("GREENAPI_TOKEN"))
    # WhatsApp chat id of the staff member who receives handoffs, e.g. 37477000000@c.us
    staff_chat_id: str = field(default_factory=lambda: _env("STAFF_CHAT_ID"))
    webhook_secret: str = field(default_factory=lambda: _env("WEBHOOK_SECRET"))

    knowledge_dir: Path = ROOT / "knowledge"
    # Vercel functions can only write to /tmp
    data_dir: Path = Path("/tmp/alta") if os.getenv("VERCEL") else ROOT / "data"
    # On serverless hosts work must finish before the response is returned
    sync_webhook: bool = field(default_factory=lambda: bool(os.getenv("VERCEL")) or _env("SYNC_WEBHOOK") == "1")
    top_k: int = 4
    min_score: float = 0.25
    history_turns: int = 8


settings = Settings()
