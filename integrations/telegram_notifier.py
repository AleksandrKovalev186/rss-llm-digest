import asyncio

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.types import LinkPreviewOptions
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

from integrations.message_state import State
from memory.vector_store import retrieve_rules
from settings.config import settings

FORMATTER_PROMPT = """Format the news summaries below into Telegram HTML using these rules:
{rules}
Output formatted text only, no explanations.\n\n{summaries}"""


def split_messages(text: str, max_length: int = 4096) -> list[str]:
    """Разбивает текст на части по логическим границам между новостями."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    separator = "━━━━━━━━━━━━━━━━━━━━"
    parts = text.split(separator)
    current_chunk = ""

    for part in parts:
        candidate = current_chunk + (separator if current_chunk else "") + part
        if len(candidate) <= max_length:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = part

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text[:max_length]]


async def telegram_node(state: State) -> State:
    rules = retrieve_rules(
        query="Telegram HTML formatting tags escaping line breaks limits",
        k=4
    )

    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    response = await llm.ainvoke([
        HumanMessage(content=FORMATTER_PROMPT.format(
            rules=rules,
            summaries=state.summaries
        ))
    ])

    chunks = split_messages(response.content)

    async with Bot(token=settings.telegram_bot_token) as bot:
        try:
            for i, chunk in enumerate(chunks, start=1):
                await bot.send_message(
                    chat_id=settings.telegram_chat_id,
                    text=chunk,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                    link_preview_options=LinkPreviewOptions(is_disabled=True)
                )

                if len(chunks) > 1 and i < len(chunks):
                    await asyncio.sleep(0.5)

        except Exception as e:
            print(f" {e}")

    return state
