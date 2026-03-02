import asyncio
import csv
from pathlib import Path
from typing import List, Literal

import feedparser
import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from integrations.email_notifier import email_node
from integrations.message_state import State
from integrations.telegram_notifier import telegram_node
from memory.vector_store import init_vectorstore, store_news, retrieve_news
from settings.config import settings


def read_urls(filename: str) -> List[str]:
    with open(filename, "r", encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=',')
        rows = list(csv_reader)
    return rows


urls = read_urls('urls.csv')


@tool
def rss_feed(feed_urls: List[str], count: int = 2) -> str:
    """
      Fetch latest entries from given RSS feed URLs.

      Parameters:
          feed_urls (List[str]) — list of RSS/Atom feed URLs.
          count (int) — number of newest items to return per feed.

      Returns:
          str — formatted text with titles and links.
      """
    result = []
    parsed_entries = []
    for url in feed_urls:
        feed = feedparser.parse(url)

        if not feed.entries:
            result.append(f"RSS feed {url} has no entries or could not be parsed.\n")
            continue

        for i, entry in enumerate(feed.entries[:count], start=1):
            title = entry.get("title", "No title")
            link = entry.get("link", "No link")
            content = entry.get("summary", "No content")[:250]

            result.append(f"{i}. Title: {title}\n   Link: {link}\n   Content: {content}\n")

            parsed_entries.append({
                "title": title,
                "link": link,
                "content": content,
                "source_url": url
            })

    store_news(parsed_entries)
    return "\n".join(result)


@tool
def search_rss_history(query: str, k: int = 3) -> str:
    """
    Search previously fetched RSS articles by semantic similarity.
    Use this to find related articles and their links from past runs.

    Parameters:
        query — topic or keywords to search for
        k — number of results to return

    Returns:
        str — relevant articles with titles and links
    """
    return retrieve_news(query, k)


def integration_router(state: State) -> Literal['telegram_node', 'email_node']:
    if settings.channel_to_send == 'TELEGRAM':
        return "telegram_node"
    elif settings.channel_to_send == 'EMAIL':
        return "email_node"
    raise ValueError(
        "No output channel selected. "
    )


data = yaml.safe_load(Path("system_prompt.yaml").read_text())

messages_data = {msg["role"]: msg["content"] for msg in data["messages"]}

prompt = ChatPromptTemplate.from_messages([
    ("system", messages_data["system"]),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatDeepSeek(model="deepseek-chat", max_tokens=2000)

tools = [rss_feed, search_rss_history]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)


async def summarizer_node(state: State) -> State:
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    result = await agent_executor.ainvoke({
        "input": f"Use the rss_feed tool to fetch items from {urls}, "
                 f"then summarize each item according to the system instructions."
    })

    state.summaries = result["output"]
    return state


def build_graph():
    builder = StateGraph(State)

    builder.add_node('summarizer', summarizer_node)
    builder.add_node('telegram_node', telegram_node)
    builder.add_node('email_node', email_node)

    builder.add_edge(START, 'summarizer')
    builder.add_conditional_edges('summarizer', integration_router)
    builder.add_edge('telegram_node', END)
    builder.add_edge('email_node', END)

    return builder.compile()


async def run_agent():
    init_vectorstore()
    graph = build_graph()

    await graph.ainvoke({})


async def run_pipeline():
    init_vectorstore()

    hour, minute = settings.schedule_time.split(":")

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_agent,
        trigger=CronTrigger(hour=int(hour), minute=int(minute)),
    )
    scheduler.start()
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(run_pipeline())
