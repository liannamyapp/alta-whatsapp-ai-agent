"""Minimal retrieval-augmented generation index.

Knowledge files (Markdown) are split on '# ' headings so each chunk is one
self-contained topic (prices, payment, contacts...). Chunks are embedded once
and cached on disk; the cache is rebuilt automatically when a file changes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import settings

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    text: str


def split_markdown(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    title, lines = "", []

    def flush():
        body = "\n".join(lines).strip()
        if title and body:
            cid = f"{path.stem}#{len(chunks)}"
            chunks.append(Chunk(cid, path.name, title, f"{title}\n{body}"))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            flush()
            title, lines = line[2:].strip(), []
        else:
            lines.append(line)
    flush()
    return chunks


def load_chunks(knowledge_dir: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for p in sorted(knowledge_dir.glob("*.md")):
        out.extend(split_markdown(p))
    return out


class KnowledgeIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embed_fn: EmbedFn):
        self.chunks = chunks
        self.vectors = vectors
        self.embed_fn = embed_fn

    @classmethod
    def build(cls, knowledge_dir: Path, embed_fn: EmbedFn, cache_dir: Path | None = None) -> "KnowledgeIndex":
        chunks = load_chunks(knowledge_dir)
        key = settings.embed_model + "".join(c.text for c in chunks)
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        cache = (cache_dir / f"index_{digest}.json") if cache_dir else None
        if cache and cache.exists():
            vectors = np.array(json.loads(cache.read_text()), dtype=np.float32)
        else:
            vectors = np.array(embed_fn([c.text for c in chunks]), dtype=np.float32)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(vectors.tolist()))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        return cls(chunks, vectors, embed_fn)

    def search(self, query: str, k: int | None = None, min_score: float | None = None) -> list[tuple[Chunk, float]]:
        k = k or settings.top_k
        min_score = settings.min_score if min_score is None else min_score
        q = np.array(self.embed_fn([query])[0], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9
        scores = self.vectors @ q
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]
