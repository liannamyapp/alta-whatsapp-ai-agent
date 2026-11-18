"""FastAPI service: GREEN-API webhook -> (STT) -> agent -> (TTS) -> WhatsApp reply."""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Callable

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from .agent import Agent
from .config import settings
from .greenapi import GreenAPI, Incoming, parse_webhook
from .store import Store

log = logging.getLogger("alta")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Pipeline:
    """Everything needed to answer one WhatsApp message. Dependencies are injectable for tests."""

    def __init__(self, agent: Agent, wa: GreenAPI, store: Store,
                 transcribe: Callable[[bytes], str], synthesize: Callable[[str], bytes]):
        self.agent, self.wa, self.store = agent, wa, store
        self.transcribe, self.synthesize = transcribe, synthesize
        self._seen: OrderedDict[str, None] = OrderedDict()

    def is_duplicate(self, message_id: str) -> bool:
        if not message_id:
            return False
        if message_id in self._seen:
            return True
        self._seen[message_id] = None
        if len(self._seen) > 2000:
            self._seen.popitem(last=False)
        return False

    def handle(self, msg: Incoming) -> str:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        text = msg.text
        if msg.kind == "voice":
            s = time.perf_counter()
            text = self.transcribe(self.wa.download(msg.audio_url))
            timings["stt_ms"] = (time.perf_counter() - s) * 1000
            log.info("STT %s: %r", msg.chat_id, text)
        if not text.strip():
            return ""
        result = self.agent.respond(msg.chat_id, text)
        timings["llm_ms"] = result.llm_ms
        if msg.kind == "voice":
            # Voice in -> voice out (plus the text, so the client can re-read it).
            s = time.perf_counter()
            try:
                self.wa.send_audio(msg.chat_id, self.synthesize(result.reply))
            except Exception:  # never lose the answer because TTS failed
                log.exception("TTS/audio send failed, falling back to text")
            timings["tts_ms"] = (time.perf_counter() - s) * 1000
        self.wa.send_text(msg.chat_id, result.reply)
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        self.store.add_turn(msg.chat_id, msg.kind, timings, result.tools_called, result.retrieved_ids)
        log.info("reply %s tools=%s %.0fms", msg.chat_id, result.tools_called, timings["total_ms"])
        return result.reply


def build_pipeline() -> Pipeline:
    from .llm import embed
    from .rag import KnowledgeIndex
    from .voice import synthesize, transcribe

    store = Store(settings.data_dir / "alta.db")
    wa = GreenAPI()
    index = KnowledgeIndex.build(settings.knowledge_dir, embed, cache_dir=settings.data_dir)

    def notify_staff(chat_id: str, reason: str, text: str):
        if settings.staff_chat_id:
            phone = chat_id.split("@")[0]
            wa.send_text(settings.staff_chat_id,
                         f"ALTA bot handoff\nClient: +{phone}\nReason: {reason}\nLast message: {text}")

    agent = Agent(index, store, on_handoff=notify_staff)
    return Pipeline(agent, wa, store, transcribe, synthesize)


app = FastAPI(title="ALTA Academy WhatsApp voice agent")
_pipeline: Pipeline | None = None


def pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def _check_secret(authorization: str | None):
    if settings.webhook_secret and authorization != f"Bearer {settings.webhook_secret}":
        raise HTTPException(status_code=401, detail="bad token")


def _safe_handle(msg: Incoming):
    try:
        pipeline().handle(msg)
    except Exception:
        log.exception("failed to handle message %s", msg.message_id)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/webhook")
async def webhook(request: Request, background: BackgroundTasks, authorization: str | None = Header(None)):
    # GREEN-API sends `Authorization: Bearer <webhookUrlToken>` when the token is set in the console.
    _check_secret(authorization)
    msg = parse_webhook(await request.json())
    if msg and not pipeline().is_duplicate(msg.message_id):
        if settings.sync_webhook:
            _safe_handle(msg)                        # serverless: finish before returning
        else:
            background.add_task(_safe_handle, msg)   # long-running server: answer GREEN-API immediately
    return {"ok": True}


class ChatIn(BaseModel):
    chat_id: str = "local-test"
    text: str


@app.post("/chat")
def chat(body: ChatIn, authorization: str | None = Header(None)):
    """Test the agent over HTTP without WhatsApp."""
    _check_secret(authorization)
    r = pipeline().agent.respond(body.chat_id, body.text)
    return {"reply": r.reply, "tools": r.tools_called, "retrieved": r.retrieved_ids, "handoff": r.handoff_reason}


@app.get("/stats")
def stats(authorization: str | None = Header(None)):
    _check_secret(authorization)
    return pipeline().store.stats()
