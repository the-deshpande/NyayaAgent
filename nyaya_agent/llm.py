from __future__ import annotations

import os
from functools import lru_cache

from langchain_openrouter import ChatOpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI

from nyaya_agent.settings import CHAT_MODEL_ID

@lru_cache(maxsize=1)
def get_chat_model():
    if os.getenv("OPENROUTER_API_KEY"):
        return ChatOpenRouter(model=CHAT_MODEL_ID, temperature=0)
    elif os.getenv("GEMINI_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
    else:
        raise ValueError("Neither OPENROUTER_API_KEY nor GEMINI_API_KEY is set in environment.")