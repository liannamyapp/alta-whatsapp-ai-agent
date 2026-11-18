"""SQLite storage: conversation history, leads, handoffs and per-turn metrics."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, chat_id TEXT, role TEXT, content TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY, chat_id TEXT, name TEXT, course TEXT, format TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS handoffs (
    id INTEGER PRIMARY KEY, chat_id TEXT, reason TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY, chat_id TEXT, modality TEXT, latency_ms REAL,
    stt_ms REAL, llm_ms REAL, tts_ms REAL, tools TEXT, retrieved TEXT, ts REAL);
"""


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _exec(self, sql: str, args: tuple = ()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def add_message(self, chat_id: str, role: str, content: str):
        self._exec("INSERT INTO messages(chat_id, role, content, ts) VALUES (?,?,?,?)",
                   (chat_id, role, content, time.time()))

    def history(self, chat_id: str, turns: int) -> list[dict]:
        rows = self._exec(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?",
            (chat_id, turns * 2)).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def add_lead(self, chat_id: str, name: str, course: str, fmt: str):
        self._exec("INSERT INTO leads(chat_id, name, course, format, ts) VALUES (?,?,?,?,?)",
                   (chat_id, name, course, fmt, time.time()))

    def add_handoff(self, chat_id: str, reason: str):
        self._exec("INSERT INTO handoffs(chat_id, reason, ts) VALUES (?,?,?)", (chat_id, reason, time.time()))

    def add_turn(self, chat_id: str, modality: str, timings: dict, tools: list[str], retrieved: list[str]):
        self._exec(
            "INSERT INTO turns(chat_id, modality, latency_ms, stt_ms, llm_ms, tts_ms, tools, retrieved, ts)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (chat_id, modality, timings.get("total_ms"), timings.get("stt_ms"), timings.get("llm_ms"),
             timings.get("tts_ms"), json.dumps(tools), json.dumps(retrieved), time.time()))

    def stats(self) -> dict:
        q = lambda sql: self._exec(sql).fetchone()[0]  # noqa: E731
        lat = [r[0] for r in self._exec("SELECT latency_ms FROM turns ORDER BY latency_ms").fetchall()]
        pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None  # noqa: E731
        turns = q("SELECT COUNT(*) FROM turns")
        handoffs = q("SELECT COUNT(*) FROM handoffs")
        return {
            "turns": turns,
            "conversations": q("SELECT COUNT(DISTINCT chat_id) FROM turns"),
            "leads": q("SELECT COUNT(*) FROM leads"),
            "handoffs": handoffs,
            "handoff_rate": round(handoffs / turns, 3) if turns else None,
            "voice_turns": q("SELECT COUNT(*) FROM turns WHERE modality='voice'"),
            "latency_p50_ms": pct(0.5),
            "latency_p95_ms": pct(0.95),
        }
