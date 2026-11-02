import asyncio
import html
import re

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.types import LinkPreviewOptions
from loguru import logger

from graph.state import State
from settings.config import settings

SEPARATOR = "━━━━━━━━━━━━━━━━━━━━"
FIELD_PATTERN = re.compile(
    r"^(TITLE|SUMMARY|WHY_IT_MATTERS|SOURCE|TARGET_AUDIENCE|RELATED):\s*(.*)$"
)
FIELD_LABELS = {
    "SUMMARY": "Summary",
    "WHY_IT_MATTERS": "Why it matters",
    "TARGET_AUDIENCE": "Audience",
    "RELATED": "Related",
}


def _format_line(line: str) -> str:
    match = FIELD_PATTERN.match(line.strip())
    if not match:
        return html.escape(line, quote=False)

    field, value = match.groups()
    if field == "TITLE":
        return f"<b>{html.escape(value, quote=False)}</b>"
    if field == "SOURCE":
        return f'<a href="{html.escape(value)}">Source</a>' if value else ""
    label = f"<b>{FIELD_LABELS[field]}:</b>"
    return f"{label} {html.escape(value, quote=False)}" if value else label


def format_digest(text: str) -> str:
    """Turn the plain-text digest into Telegram HTML; the LLM never writes tags itself."""
    items = re.split(r"^\s*---\s*$", text.strip(), flags=re.MULTILINE)
    formatted = [
        "\n".join(_format_line(line) for line in item.strip().splitlines()) for item in items
    ]
    return f"\n{SEPARATOR}\n".join(item for item in formatted if item.strip())


def _line_split(text: str, max_length: int) -> list[str]:
    """Split on line boundaries: format_digest never lets an HTML tag span lines.

    shortcut: a single line longer than max_length is still sent whole and Telegram rejects it;
    summary lines are far below 4096 chars, so this is not handled.
    """
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        if current and len(current) + len(line) > max_length:
            chunks.append(current.strip())
            current = ""
        current += line
    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_messages(text: str, max_length: int = 4096) -> list[str]:
    """Splits the text into segments along logical boundaries between news items."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    parts = text.split(SEPARATOR)
    current_chunk = ""

    for part in parts:
        candidate = current_chunk + (SEPARATOR if current_chunk else "") + part
        if len(candidate) <= max_length:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(part) > max_length:
                line_parts = _line_split(part.strip(), max_length)
                chunks.extend(line_parts[:-1])
                current_chunk = line_parts[-1]
            else:
                current_chunk = part

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks or _line_split(text, max_length)


async def telegram_node(state: State) -> State:
    chunks = split_messages(format_digest(state.summaries))

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
            logger.error(e)

    return state
