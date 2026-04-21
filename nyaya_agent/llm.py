from __future__ import annotations

from functools import lru_cache

from langchain_openrouter import ChatOpenRouter

from nyaya_agent.settings import CHAT_MODEL_ID


@lru_cache(maxsize=1)
def get_chat_model():
    # # Use Google AI Studio (API key), not Vertex AI, when model id looks like Gemini.
    # mid = (CHAT_MODEL_ID or "").lower()
    # if "gemini" in mid:
    #     return init_chat_model(
    #         CHAT_MODEL_ID,
    #         model_provider="google_genai",
    #         temperature=0,
    #     )
    # return init_chat_model(CHAT_MODEL_ID, temperature=0)
    model = ChatOpenRouter(model=CHAT_MODEL_ID, temperature=0)
    return model
    