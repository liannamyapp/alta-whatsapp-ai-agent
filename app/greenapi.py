"""GREEN-API (WhatsApp) client and webhook parsing."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import settings


@dataclass
class Incoming:
    message_id: str
    chat_id: str
    sender_name: str
    kind: str            # "text" | "voice"
    text: str = ""
    audio_url: str = ""


def parse_webhook(payload: dict) -> Incoming | None:
    """Return a normalized incoming message, or None for webhooks we ignore."""
    if payload.get("typeWebhook") != "incomingMessageReceived":
        return None
    sender = payload.get("senderData", {})
    chat_id = sender.get("chatId", "")
    if not chat_id or chat_id.endswith("@g.us"):   # ignore group chats
        return None
    data = payload.get("messageData", {})
    mtype = data.get("typeMessage")
    base = dict(message_id=payload.get("idMessage", ""), chat_id=chat_id,
                sender_name=sender.get("senderName", ""))
    if mtype == "textMessage":
        return Incoming(kind="text", text=data.get("textMessageData", {}).get("textMessage", ""), **base)
    if mtype in ("extendedTextMessage", "quotedMessage"):
        return Incoming(kind="text", text=data.get("extendedTextMessageData", {}).get("text", ""), **base)
    if mtype == "audioMessage":
        return Incoming(kind="voice", audio_url=data.get("fileMessageData", {}).get("downloadUrl", ""), **base)
    return None


class GreenAPI:
    def __init__(self, http: httpx.Client | None = None):
        self.http = http or httpx.Client(timeout=30)

    def _url(self, method: str, media: bool = False) -> str:
        host = settings.greenapi_media_url if media else settings.greenapi_url
        return f"{host}/waInstance{settings.greenapi_instance}/{method}/{settings.greenapi_token}"

    def send_text(self, chat_id: str, text: str) -> None:
        self.http.post(self._url("sendMessage"), json={"chatId": chat_id, "message": text}).raise_for_status()

    def send_audio(self, chat_id: str, audio: bytes, filename: str = "reply.ogg") -> None:
        files = {"file": (filename, audio, "audio/ogg")}
        self.http.post(self._url("sendFileByUpload", media=True),
                       data={"chatId": chat_id}, files=files).raise_for_status()

    def download(self, url: str) -> bytes:
        r = self.http.get(url)
        r.raise_for_status()
        return r.content
