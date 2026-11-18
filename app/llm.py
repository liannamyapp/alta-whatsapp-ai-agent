"""Thin wrapper around the OpenAI client so the rest of the code (and tests) can swap it."""
from functools import lru_cache

from openai import OpenAI

from .config import settings


@lru_cache(maxsize=1)
def client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=settings.openai_api_key)


def embed(texts: list[str]) -> list[list[float]]:
    resp = client().embeddings.create(model=settings.embed_model, input=texts)
    return [d.embedding for d in resp.data]
