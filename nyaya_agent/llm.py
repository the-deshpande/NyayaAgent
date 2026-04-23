from __future__ import annotations

from functools import lru_cache

from langchain_openrouter import ChatOpenRouter

from nyaya_agent.settings import CHAT_MODEL_ID


@lru_cache(maxsize=1)
def get_chat_model():
    # Use model from openrouter with 0 temperature
    model = ChatOpenRouter(model=CHAT_MODEL_ID, temperature=0)
    return model
    