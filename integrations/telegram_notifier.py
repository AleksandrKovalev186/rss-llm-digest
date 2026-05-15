import asyncio
import re

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.types import LinkPreviewOptions
from langchain_core.messages import HumanMessage
from loguru import logger

from settings.llm import get_chat_llm

from integrations.message_state import State
from settings.config import settings

FORMATTER_PROMPT = """You are a Telegram message formatter.
Convert the news summaries below to Telegram HTML.
Output ONLY the formatted text — no code blocks, no explanations, no markdown.

ALLOWED TAGS (only these, nothing else):
<b>bold</b>  <i>italic</i>  <u>underline</u>  <s>strikethrough</s>
<a href="URL">link</a>  <code>inline code</code>  <pre>block</pre>

RULES:
- Special characters must be escaped: & → &amp;  < → &lt;  > → &gt;
- Do NOT use <div>, <p>, <br>, <h1>-<h6>, or any other tags not listed above
- Do NOT wrap output in ```html, ```, or any code block markers
- Separate news items with: ━━━━━━━━━━━━━━━━━━━━

{summaries}"""

_TELEGRAM_ALLOWED = frozenset({
    'b', 'strong', 'i', 'em', 'u', 'ins', 's', 'strike', 'del',
    'a', 'code', 'pre', 'blockquote',
})
_BLOCK_TAGS = frozenset({
    'div', 'p', 'section', 'article', 'header', 'footer', 'li', 'ul', 'ol', 'br',
})


def _sanitize_html(text: str) -> str:
    """Strip HTML tags unsupported by Telegram; replace block elements with newlines."""
    text = re.sub(r'^\s*```[\w]*\s*\n?', '', text)
    text = re.sub(r'\n?\s*```\s*$', '', text)

    def _replace(m: re.Match) -> str:
        inner = m.group(1).lstrip('/').split()[0].lower()
        if inner in _TELEGRAM_ALLOWED:
            return m.group(0)
        return '\n' if inner in _BLOCK_TAGS else ''

    result = re.sub(r'<(/?[\w][^>]*)>', _replace, text)
    return re.sub(r'\n{3,}', '\n\n', result).strip()


def _char_split(text: str, max_length: int) -> list[str]:
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def split_messages(text: str, max_length: int = 4096) -> list[str]:
    """Splits the text into segments along logical boundaries between news items."""
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
            if len(part) > max_length:
                char_parts = _char_split(part.strip(), max_length)
                chunks.extend(char_parts[:-1])
                current_chunk = char_parts[-1]
            else:
                current_chunk = part

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks or _char_split(text, max_length)


async def telegram_node(state: State) -> State:
    llm = get_chat_llm()
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: llm.invoke([
            HumanMessage(content=FORMATTER_PROMPT.format(
                summaries=state.summaries
            ))
        ])
    )

    chunks = split_messages(_sanitize_html(response.content))

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
