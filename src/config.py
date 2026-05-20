"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_chat_model: str
    openai_embedding_model: str
    google_api_key: str | None
    gemini_embedding_model: str
    gemini_reasoning_model: str
    default_chunk_size: int = 900
    default_chunk_overlap: int = 140
    max_context_chunks: int = 6


def get_settings() -> Settings:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    return Settings(
        openai_api_key=openai_api_key if openai_api_key and openai_api_key != "your_openai_api_key_here" else None,
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        google_api_key=google_api_key if google_api_key and google_api_key != "your_google_ai_studio_key_here" else None,
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        gemini_reasoning_model=os.getenv("GEMINI_REASONING_MODEL", "gemini-2.5-flash-lite"),
    )
