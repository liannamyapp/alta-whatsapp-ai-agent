# ALTA Academy — WhatsApp voice & text AI agent

An LLM-based customer assistant for **ALTA Academy**, an online education center in Armenia.
Clients write or send **voice notes** on WhatsApp; the agent answers in Armenian, Russian or English,
using only the academy's official information, registers enrollment leads, and hands the
conversation to a staff member whenever it does not know the answer.

## Architecture

```
WhatsApp ──► GREEN-API ──webhook──► FastAPI /webhook ──(background task)──┐
                                                                         ▼
                      voice note? ── download ─► STT (Whisper) ─► text
                                                                         ▼
                      Agent (gpt-4o-mini, tool calling, max 4 steps)
                        ├─ search_knowledge_base ─► RAG index (embeddings + cosine)
                        ├─ create_lead           ─► SQLite
                        └─ handoff_to_human      ─► SQLite + WhatsApp message to staff
                                                                         ▼
                      voice in? ─► TTS (Opus/OGG) ─► voice reply   +   text reply
                                                                         ▼
                      per-turn metrics (latency, STT/LLM/TTS ms, tools, retrieved chunks) ─► SQLite
```

| File | Purpose |
|---|---|
| `app/rag.py` | Splits Markdown knowledge into topic chunks, embeds them, caches vectors, cosine search with a score threshold |
| `app/agent.py` | System prompt, tool schemas, tool-calling loop, step-limit fallback to a human |
| `app/main.py` | Webhook, auth (`Bearer` token), dedup of repeated webhooks, voice/text pipeline, `/chat`, `/stats` |
| `app/greenapi.py` | GREEN-API webhook parsing (text, extended text, audio; groups ignored) and sending |
| `app/voice.py` | Speech-to-text and text-to-speech |
| `app/store.py` | Conversation history, leads, handoffs, turn metrics, aggregate stats |
| `evals/` | Evaluation dataset (24 cases, 3 languages) and runner |
| `tests/` | 12 offline tests with fake LLM/embeddings (no API key needed) |

## Design decisions

- **Grounding over fluency.** Facts may only come from `search_knowledge_base`. Unknown topics
  (start dates, duration, discounts) are deliberately absent from the knowledge base, and the
  eval set contains "trap" questions about them: the correct behavior is a handoff, not a guess.
- **Topic-sized chunks.** The knowledge base is small and structured, so chunks follow headings
  (one topic = one chunk). This keeps prices and their course names in the same chunk.
- **Voice in, voice out.** A voice note gets a voice reply *and* a text copy; if TTS fails the
  text is still sent. Replies are kept to 1–3 sentences without lists or emojis so they sound natural.
- **Fast webhook ack.** The webhook returns immediately and the work runs in a background task,
  so GREEN-API never times out or retries; repeated message ids are ignored.

## Measuring effectiveness

### Offline evals — `python -m evals.run_evals`
Each case runs through the real agent and is scored on:

| Metric | How it is measured |
|---|---|
| retrieval_recall | the expected knowledge topic is among retrieved chunks |
| fact_accuracy | required facts (prices, email) appear in the reply — deterministic, digit-normalized |
| handoff precision / recall | `handoff_to_human` called exactly when expected (unknown info, human request, complaint) |
| tool_accuracy | `create_lead` called for enrollment intents |
| language_match | reply script matches the question (Armenian / Cyrillic / Latin) |
| judge_correct | LLM-as-judge (gpt-4o) score ≥ 4/5, with the knowledge base as the only truth |
| hallucination_rate | judge flags any ALTA fact not present in the knowledge base |
| latency p50 / p95 | end-to-end agent time per question |

Results go to `evals/report.md` (summary + every failure) and `evals/results.jsonl` (per case).
Workflow: change a prompt, a tool description or the knowledge base → rerun evals → compare reports.

### Online metrics — `GET /stats`
Live traffic: conversations, turns, voice share, leads, handoff rate, latency p50/p95.
Handoff rate and lead count are the business KPIs; latency is tracked per stage (STT / LLM / TTS)
in the `turns` table to see where time goes.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
cp .env.example .env            # fill in keys, then: export $(cat .env | xargs)
pytest -q                       # offline tests
python -m evals.run_evals       # evals (uses the OpenAI API)
uvicorn app.main:app --reload
curl -X POST localhost:8000/chat -H "Authorization: Bearer $WEBHOOK_SECRET" \
     -H "Content-Type: application/json" -d '{"text":"Ինչ արժե ֆրոնտենդի դասընթացը?"}'
```

## Deploy

Vercel (Python runtime, `index.py` entrypoint; the webhook is processed synchronously and data lives in `/tmp`), Render via `render.yaml` (persistent process, background tasks), or any Docker host via `Dockerfile`. Then in the GREEN-API
console set the webhook URL to `https://<your-app>/webhook`, set `webhookUrlToken` to the same
value as `WEBHOOK_SECRET`, and enable incoming message webhooks.

## Next steps

Real-time phone calls (Twilio/Telnyx Media Streams + streaming STT/TTS, VAD and barge-in),
weekly review of handoff transcripts to grow the knowledge base, and adding failed real
conversations to the eval set.
