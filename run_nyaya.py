from __future__ import annotations

import json
import logging
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from nyaya_agent.graph import build_graph
from nyaya_agent.llm import get_chat_model
from nyaya_agent.memory import ChatMemoryStore
from nyaya_agent.settings import SQLITE_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CLI_SESSION_ID = "cli"


def chat_loop() -> int:
    load_dotenv()

    model = get_chat_model()
    messages: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(
            content=(
                "You are Nyaya Agent (chat mode). Be helpful and concise. "
                "If legal sources are not provided, clearly state uncertainty and suggest what to retrieve."
            )
        )
    ]

    print("\nChat mode enabled. Type `quit()` to exit.\n")
    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            return 0

        if user_text == "quit()":
            return 0
        if not user_text:
            continue

        logger.info(f"User input: {user_text}")
        messages.append(HumanMessage(content=user_text))
        ai = model.invoke(messages)
        messages.append(ai)
        logger.info("AI response received")
        print(f"nyaya> {ai.content}\n")


def main() -> int:
    load_dotenv()

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: python run_nyaya.py \"<your legal/compliance question>\"")
        return 2

    store = ChatMemoryStore(SQLITE_PATH)
    ctx = store.get_context(CLI_SESSION_ID)
    logger.info(f"Loaded context with {len(ctx.messages)} messages for session {CLI_SESSION_ID}")

    app = build_graph()
    logger.info("Invoking Nyaya Agent graph")
    out = app.invoke(
        {
            "query": query,
            "conversation_summary": ctx.summary,
            "recent_messages": ctx.messages,
        }
    )

    memo = out.get("memo") or {}
    print(json.dumps(memo, indent=2, ensure_ascii=False))

    reply = (out.get("assistant_message") or out.get("chat_reply") or "").strip()
    if reply:
        logger.info("Appending exchange to memory store")
        store.append_exchange(CLI_SESSION_ID, query, reply)

    choice = input("\nDo you wish to just chat? (y/n): ").strip().lower()
    if choice in {"y", "yes"}:
        return chat_loop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
