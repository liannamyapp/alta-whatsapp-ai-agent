"""Offline tests: no network, no API key. LLM and embeddings are replaced with fakes."""
import hashlib
import json
import re
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app.agent import Agent
from app.config import settings
from app.greenapi import Incoming, parse_webhook
from app.main import Pipeline
from app.rag import KnowledgeIndex, load_chunks
from app.store import Store
from evals.run_evals import normalize_digits, script_of, write_report


def fake_embed(texts):
    """Hashed bag-of-words vectors: good enough to test retrieval plumbing."""
    out = []
    for t in texts:
        v = np.zeros(256)
        for w in re.findall(r"\w+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        out.append(v.tolist())
    return out


def tool_call(name, args, cid="c1"):
    return NS(id=cid, function=NS(name=name, arguments=json.dumps(args)))


def reply(content=None, calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))])


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, tools):
        self.calls.append(messages)
        return self.responses.pop(0)


@pytest.fixture
def index():
    return KnowledgeIndex.build(settings.knowledge_dir, fake_embed)


def test_chunks_split_by_heading():
    titles = [c.title for c in load_chunks(settings.knowledge_dir)]
    assert "AI courses" in titles and "Payment" in titles
    assert len(titles) == len(set(titles))


def test_search_finds_payment(index):
    hits = index.search("payment idram bank card", min_score=0.0)
    assert hits[0][0].title == "Payment"


def test_agent_searches_then_answers(index):
    store = Store(":memory:")
    llm = ScriptedLLM(reply(calls=[tool_call("search_knowledge_base", {"query": "AI courses price"})]),
                      reply("AI for developers costs 70,000 AMD per month."))
    r = Agent(index, store, chat_fn=llm).respond("u1", "How much is AI for developers?")
    assert r.tools_called == ["search_knowledge_base"]
    assert r.retrieved_ids
    tool_msg = llm.calls[1][-1]
    assert tool_msg["role"] == "tool" and "70,000" in tool_msg["content"]
    assert store.history("u1", 4)[-1]["content"].startswith("AI for developers")


def test_handoff_notifies_staff(index):
    store, notified = Store(":memory:"), []
    llm = ScriptedLLM(reply(calls=[tool_call("handoff_to_human", {"reason": "start date unknown"})]),
                      reply("A staff member will reply soon."))
    agent = Agent(index, store, chat_fn=llm, on_handoff=lambda *a: notified.append(a))
    r = agent.respond("u2", "When does the next group start?")
    assert r.handoff_reason == "start date unknown"
    assert notified[0][0] == "u2"
    assert store.stats()["handoffs"] == 1


def test_lead_is_saved(index):
    store = Store(":memory:")
    llm = ScriptedLLM(reply(calls=[tool_call("create_lead", {"name": "Anna", "course": "AI for everyone",
                                                             "format": "group"})]),
                      reply("Thanks Anna, we registered you."))
    Agent(index, store, chat_fn=llm).respond("u3", "I'm Anna, sign me up")
    assert store.stats()["leads"] == 1


def test_step_limit_falls_back_to_handoff(index):
    store = Store(":memory:")
    loop = [reply(calls=[tool_call("search_knowledge_base", {"query": "x"})]) for _ in range(4)]
    r = Agent(index, store, chat_fn=ScriptedLLM(*loop)).respond("u4", "???")
    assert r.handoff_reason == "agent_step_limit"
    assert r.reply


def test_parse_webhook_text_voice_and_group():
    base = {"typeWebhook": "incomingMessageReceived", "idMessage": "m1",
            "senderData": {"chatId": "37400000000@c.us", "senderName": "Ani"}}
    text = parse_webhook({**base, "messageData": {"typeMessage": "textMessage",
                                                  "textMessageData": {"textMessage": "Barev"}}})
    assert text.kind == "text" and text.text == "Barev"
    voice = parse_webhook({**base, "messageData": {"typeMessage": "audioMessage",
                                                   "fileMessageData": {"downloadUrl": "https://x/a.ogg"}}})
    assert voice.kind == "voice" and voice.audio_url.endswith(".ogg")
    group = {**base, "senderData": {"chatId": "123@g.us"},
             "messageData": {"typeMessage": "textMessage", "textMessageData": {"textMessage": "hi"}}}
    assert parse_webhook(group) is None
    assert parse_webhook({"typeWebhook": "stateInstanceChanged"}) is None


class FakeWA:
    def __init__(self):
        self.texts, self.audios = [], []

    def send_text(self, chat_id, text): self.texts.append((chat_id, text))
    def send_audio(self, chat_id, audio): self.audios.append((chat_id, audio))
    def download(self, url): return b"OGG"


def test_voice_pipeline_stt_agent_tts(index):
    store, wa = Store(":memory:"), FakeWA()
    llm = ScriptedLLM(reply("We accept Idram and bank cards."))
    p = Pipeline(Agent(index, store, chat_fn=llm), wa, store,
                 transcribe=lambda audio: "How can I pay?", synthesize=lambda text: b"VOICE")
    p.handle(Incoming("m1", "37400000000@c.us", "Ani", "voice", audio_url="https://x"))
    assert wa.audios == [("37400000000@c.us", b"VOICE")]
    assert wa.texts[0][1] == "We accept Idram and bank cards."
    s = store.stats()
    assert s["voice_turns"] == 1 and s["latency_p50_ms"] is not None


def test_tts_failure_still_sends_text(index):
    store, wa = Store(":memory:"), FakeWA()
    p = Pipeline(Agent(index, store, chat_fn=ScriptedLLM(reply("ok"))), wa, store,
                 transcribe=lambda a: "hi", synthesize=lambda t: (_ for _ in ()).throw(RuntimeError("tts down")))
    p.handle(Incoming("m2", "1@c.us", "", "voice", audio_url="u"))
    assert wa.texts == [("1@c.us", "ok")]


def test_duplicate_webhooks_ignored(index):
    p = Pipeline(None, FakeWA(), Store(":memory:"), None, None)
    assert not p.is_duplicate("abc") and p.is_duplicate("abc")


def test_eval_helpers(tmp_path, monkeypatch):
    assert normalize_digits("50,000 AMD and 125 000") == "50000 AMD and 125000"
    assert script_of("Դասընթացը արժե 50000 դրամ") == "hy"
    assert script_of("Курс стоит 50000 драм") == "ru"
    assert script_of("The course costs 50000 AMD") == "en"
    import evals.run_evals as ev
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    rows = [{"id": "a", "reply": "x", "handoff": True, "expect_handoff": True, "latency_ms": 900,
             "retrieval_hit": True, "facts_ok": False, "lang_ok": True, "tool_ok": None}]
    write_report(rows, use_judge=False)
    report = (tmp_path / "report.md").read_text()
    assert "handoff_recall | 100%" in report and "facts_ok" in report


def test_webhook_endpoint_auth_and_dispatch(index, monkeypatch):
    from dataclasses import replace
    from fastapi.testclient import TestClient
    import app.main as m
    store, wa = Store(":memory:"), FakeWA()
    m._pipeline = Pipeline(Agent(index, store, chat_fn=ScriptedLLM(reply("Barev!"))), wa, store, None, None)
    monkeypatch.setattr(m, "settings", replace(settings, webhook_secret="s3cret"))
    c = TestClient(m.app)
    body = {"typeWebhook": "incomingMessageReceived", "idMessage": "w1",
            "senderData": {"chatId": "1@c.us"},
            "messageData": {"typeMessage": "textMessage", "textMessageData": {"textMessage": "Hi"}}}
    assert c.post("/webhook", json=body).status_code == 401
    assert c.post("/webhook", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert wa.texts == [("1@c.us", "Barev!")]
    m._pipeline = None


def test_allowed_chats_and_forwarding(index, monkeypatch):
    from dataclasses import replace
    from fastapi.testclient import TestClient
    import app.main as m
    forwarded = []
    monkeypatch.setattr(m.httpx, "post", lambda url, json, headers, timeout: forwarded.append((url, headers)))
    store, wa = Store(":memory:"), FakeWA()
    m._pipeline = Pipeline(Agent(index, store, chat_fn=ScriptedLLM(reply("Hi!"))), wa, store, None, None)
    monkeypatch.setattr(m, "settings", replace(settings, webhook_secret="tok", allowed_chats="+37491111111",
                                               forward_webhook_url="https://crm/hook"))
    c = TestClient(m.app)
    msg = lambda cid, mid: {"typeWebhook": "incomingMessageReceived", "idMessage": mid,
                            "senderData": {"chatId": cid},
                            "messageData": {"typeMessage": "textMessage", "textMessageData": {"textMessage": "x"}}}
    c.post("/webhook", json=msg("37499999999@c.us", "a"), headers={"Authorization": "tok"})
    assert wa.texts == []                                  # not allowed -> bot silent
    c.post("/webhook", json=msg("37491111111@c.us", "b"), headers={"Authorization": "tok"})
    assert wa.texts == [("37491111111@c.us", "Hi!")]
    assert forwarded == [("https://crm/hook", {"Authorization": "tok"})] * 2   # CRM still gets everything
    monkeypatch.setattr(m, "settings", replace(settings, allowed_chats="none"))
    assert not m.is_allowed("1@c.us")
    m._pipeline = None
