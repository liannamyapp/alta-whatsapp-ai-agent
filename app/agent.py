"""The conversational agent: system prompt, tool definitions and the tool-calling loop."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import settings
from .rag import KnowledgeIndex
from .store import Store

SYSTEM_PROMPT = """You are the WhatsApp assistant of ALTA Academy, an online education center in Armenia.

Rules:
1. Always reply in the language the client wrote in (Armenian, Russian or English).
2. Facts about ALTA Academy (courses, prices, payment, certificates, contacts) must come ONLY
   from the search_knowledge_base tool. Call it before answering any factual question.
   Never invent prices, dates, schedules, durations or discounts.
3. If the knowledge base does not contain the answer, or the client asks for a human, is upset,
   or wants to negotiate, call handoff_to_human and tell the client a staff member will reply soon.
4. When a client clearly wants to enroll, collect their name and the course, then call create_lead.
5. Keep answers short and friendly: 1-3 sentences, plain text, no markdown. Answers may be read
   aloud as voice messages, so avoid lists, emojis and URLs unless the client asks for a link.
"""

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": "Search ALTA Academy's official information (courses, prices, payment, "
                       "certificates, account, contacts). Use for every factual question.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query, preferably in English."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "handoff_to_human",
        "description": "Transfer the conversation to a staff member when the answer is not in the "
                       "knowledge base, the client asks for a person, complains, or negotiates.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "Short reason for the handoff."}},
            "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "create_lead",
        "description": "Register a client who wants to enroll.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "course": {"type": "string"},
            "format": {"type": "string", "enum": ["group", "individual", "unknown"]}},
            "required": ["name", "course"]}}},
]


@dataclass
class AgentResult:
    reply: str
    tools_called: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    handoff_reason: str | None = None
    llm_ms: float = 0.0


class Agent:
    def __init__(self, index: KnowledgeIndex, store: Store, chat_fn: Callable[..., Any] | None = None,
                 on_handoff: Callable[[str, str, str], None] | None = None, max_steps: int = 4):
        self.index = index
        self.store = store
        self.chat_fn = chat_fn or self._openai_chat
        self.on_handoff = on_handoff
        self.max_steps = max_steps

    @staticmethod
    def _openai_chat(messages: list[dict], tools: list[dict]):
        from .llm import client
        return client().chat.completions.create(
            model=settings.chat_model, messages=messages, tools=tools, temperature=0.2)

    def _run_tool(self, name: str, args: dict, chat_id: str, user_text: str, result: AgentResult) -> str:
        if name == "search_knowledge_base":
            hits = self.index.search(args.get("query") or user_text)
            result.retrieved_ids.extend(c.id for c, _ in hits)
            if not hits:
                return "NO_RESULTS: the knowledge base has nothing on this. Hand off to a human."
            return "\n\n".join(f"[{c.id} score={s:.2f}]\n{c.text}" for c, s in hits)
        if name == "handoff_to_human":
            reason = args.get("reason", "unspecified")
            result.handoff_reason = reason
            self.store.add_handoff(chat_id, reason)
            if self.on_handoff:
                self.on_handoff(chat_id, reason, user_text)
            return "Handoff registered. A staff member will contact the client."
        if name == "create_lead":
            self.store.add_lead(chat_id, args.get("name", ""), args.get("course", ""), args.get("format", "unknown"))
            return "Lead saved."
        return f"Unknown tool {name}"

    def respond(self, chat_id: str, user_text: str) -> AgentResult:
        result = AgentResult(reply="")
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    *self.store.history(chat_id, settings.history_turns),
                    {"role": "user", "content": user_text}]
        t0 = time.perf_counter()
        for _ in range(self.max_steps):
            resp = self.chat_fn(messages=messages, tools=TOOLS)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                result.reply = (msg.content or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result.tools_called.append(tc.function.name)
                output = self._run_tool(tc.function.name, args, chat_id, user_text, result)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
        if not result.reply:
            result.reply = "Thank you! A staff member will reply to you shortly."
            if not result.handoff_reason:
                self._run_tool("handoff_to_human", {"reason": "agent_step_limit"}, chat_id, user_text, result)
                result.tools_called.append("handoff_to_human")
        result.llm_ms = (time.perf_counter() - t0) * 1000
        self.store.add_message(chat_id, "user", user_text)
        self.store.add_message(chat_id, "assistant", result.reply)
        return result
