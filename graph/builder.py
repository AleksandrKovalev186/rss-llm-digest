import asyncio
from typing import Literal

from langchain.agents import create_agent
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from graph.state import State
from integrations.email_notifier import email_node
from integrations.telegram_notifier import telegram_node
from settings.config import settings
from settings.llm import get_chat_llm
from settings.prompts import load_system_prompt
from tools.rss import read_urls, rss_feed, search_rss_history


def integration_router(state: State) -> Literal['telegram_node', 'email_node']:
    if settings.channel_to_send == 'TELEGRAM':
        return "telegram_node"
    elif settings.channel_to_send == 'EMAIL':
        return "email_node"
    raise ValueError(
        "No output channel selected. "
    )


async def summarizer_node(state: State) -> State:
    agent = create_agent(
        model=get_chat_llm(),
        tools=[rss_feed, search_rss_history],
        system_prompt=load_system_prompt(),
    )
    urls = read_urls('urls.csv')

    # ChatHuggingFace has no async support for HuggingFacePipeline, so run sync invoke in a thread.
    result = await asyncio.to_thread(agent.invoke, {
        "messages": [{
            "role": "user",
            "content": f"Use the rss_feed tool to fetch items from {urls}, "
                       f"then summarize each item according to the system instructions.",
        }]
    })

    state.summaries = result["messages"][-1].content
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
