"""Offline evaluation of the ALTA agent.

Usage:  python -m evals.run_evals            (needs OPENAI_API_KEY)
Writes: evals/results.jsonl and evals/report.md

Metrics
- retrieval_recall   expected knowledge topic is among retrieved chunks (fact questions)
- fact_accuracy      required facts (prices, email...) appear in the reply (deterministic)
- handoff_precision / handoff_recall   handoff_to_human called exactly when it should be
- tool_accuracy      expected tool (e.g. create_lead) was called
- language_match     reply is in the same script as the question (hy / ru / en)
- judge_correct      LLM judge: reply is correct and helpful (score >= 4 of 5)
- hallucination_rate LLM judge: reply contains claims not supported by the knowledge base
- latency p50 / p95  end-to-end agent latency per question
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import Agent  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag import KnowledgeIndex  # noqa: E402
from app.store import Store  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent

JUDGE_PROMPT = """You grade replies of a customer-support assistant for ALTA Academy.
The ONLY source of truth is this knowledge base:
<kb>
{kb}
</kb>
Client question: {question}
Assistant reply: {reply}
The assistant {handoff} transfer the client to a human.

Return JSON: {{"score": 1-5, "hallucination": true|false, "reason": "..."}}
- score 5 = correct, complete, concise; 1 = wrong or useless. If the KB lacks the answer,
  a reply that says so and transfers to a human deserves 5.
- hallucination = true if the reply states ANY fact about ALTA (price, date, duration,
  discount, schedule, course) that is not in the KB."""


def script_of(text: str) -> str:
    hy = len(re.findall(r"[Ա-֏]", text))
    ru = len(re.findall(r"[Ѐ-ӿ]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if hy >= max(ru, 1) and hy >= en * 0.3:
        return "hy"
    if ru >= max(hy, 1) and ru >= en * 0.3:
        return "ru"
    return "en"


def normalize_digits(text: str) -> str:
    # "50,000" / "50 000" / "50.000" -> "50000"
    return re.sub(r"(?<=\d)[\s,.  ](?=\d{3}\b)", "", text)


def judge(question: str, reply: str, handoff: bool, kb: str) -> dict:
    from app.llm import client
    resp = client().chat.completions.create(
        model=settings.judge_model, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            kb=kb, question=question, reply=reply, handoff="DID" if handoff else "did NOT")}])
    return json.loads(resp.choices[0].message.content)


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))] if s else 0.0


def main(use_judge: bool = True):
    from app.llm import embed
    cases = [json.loads(l) for l in (EVAL_DIR / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    index = KnowledgeIndex.build(settings.knowledge_dir, embed, cache_dir=settings.data_dir)
    kb = "\n\n".join(c.text for c in index.chunks)
    title_of = {c.id: c.title for c in index.chunks}

    rows = []
    for case in cases:
        store = Store(":memory:")
        agent = Agent(index, store)
        t0 = time.perf_counter()
        r = agent.respond(f"eval-{case['id']}", case["question"])
        latency = (time.perf_counter() - t0) * 1000
        handed_off = "handoff_to_human" in r.tools_called
        topics = {title_of.get(i) for i in r.retrieved_ids}
        reply_norm = normalize_digits(r.reply)
        row = {
            "id": case["id"], "category": case["category"], "lang": case["lang"],
            "question": case["question"], "reply": r.reply, "tools": r.tools_called,
            "retrieved_topics": sorted(t for t in topics if t), "latency_ms": round(latency),
            "handoff": handed_off, "expect_handoff": case["expect_handoff"],
            "retrieval_hit": all(t in topics for t in case["expected_topics"]) if case["expected_topics"] else None,
            "facts_ok": all(f.lower() in reply_norm.lower() for f in case["must_include"]) if case["must_include"] else None,
            "lang_ok": script_of(r.reply) == case["lang"],
            "tool_ok": (case["expect_tool"] in r.tools_called) if case.get("expect_tool") else None,
        }
        if use_judge:
            j = judge(case["question"], r.reply, handed_off, kb)
            row.update(judge_score=j.get("score"), hallucination=bool(j.get("hallucination")),
                       judge_reason=j.get("reason", ""))
        rows.append(row)
        print(f"{case['id']:<24} handoff={handed_off!s:<5} {latency:6.0f}ms  {r.reply[:70]!r}")

    (EVAL_DIR / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    write_report(rows, use_judge)


def rate(rows, key) -> tuple[float, int]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (sum(vals) / len(vals) if vals else float("nan")), len(vals)


def write_report(rows: list[dict], use_judge: bool):
    tp = sum(r["handoff"] and r["expect_handoff"] for r in rows)
    fp = sum(r["handoff"] and not r["expect_handoff"] for r in rows)
    fn = sum(not r["handoff"] and r["expect_handoff"] for r in rows)
    lat = [r["latency_ms"] for r in rows]
    metrics = {
        "retrieval_recall": rate(rows, "retrieval_hit"),
        "fact_accuracy": rate(rows, "facts_ok"),
        "tool_accuracy": rate(rows, "tool_ok"),
        "language_match": rate(rows, "lang_ok"),
        "handoff_precision": (tp / (tp + fp) if tp + fp else float("nan"), tp + fp),
        "handoff_recall": (tp / (tp + fn) if tp + fn else float("nan"), tp + fn),
    }
    if use_judge:
        for r in rows:
            r["judge_ok"] = (r.get("judge_score") or 0) >= 4
        metrics["judge_correct"] = rate(rows, "judge_ok")
        metrics["hallucination_rate"] = rate(rows, "hallucination")

    lines = ["# ALTA agent - evaluation report", "",
             f"Model: `{settings.chat_model}`, judge: `{settings.judge_model}`, cases: {len(rows)}", "",
             "| Metric | Value | n |", "|---|---|---|"]
    for name, (v, n) in metrics.items():
        lines.append(f"| {name} | {'n/a' if v != v else f'{v:.0%}'} | {n} |")
    lines += [f"| latency p50 | {statistics.median(lat):.0f} ms | {len(lat)} |",
              f"| latency p95 | {pct(lat, 0.95):.0f} ms | {len(lat)} |", "",
              "## Failures", ""]
    for r in rows:
        problems = [k for k in ("retrieval_hit", "facts_ok", "lang_ok", "tool_ok") if r.get(k) is False]
        if r["handoff"] != r["expect_handoff"]:
            problems.append("handoff")
        if r.get("hallucination"):
            problems.append("hallucination")
        if use_judge and not r.get("judge_ok"):
            problems.append(f"judge={r.get('judge_score')}")
        if problems:
            lines.append(f"- **{r['id']}** ({', '.join(problems)}): {r['reply'][:160]}")
    (EVAL_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main(use_judge="--no-judge" not in sys.argv)
