import feedparser
from pathlib import Path

import yaml
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek

urls = "https://aws.amazon.com/blogs/training-and-certification/feed"

@tool
def rss_feed(url: str) -> str:
    """
    Fetches an RSS feed and returns the latest 5 items
    with title and link in a readable text format.
    """
    feed = feedparser.parse(url)

    if not feed.entries:
        return "RSS feed has no entries or could not be parsed."

    result = []

    for i, entry in enumerate(feed.entries[:5], start=1):
        title = entry.get("title", "No title")
        link = entry.get("link", "No link")
        content = entry.get("summary", "No content")[:500]  # Берем первые 500 символов

        result.append(f"{i}. Title: {title}\n   Link: {link}\n   Content: {content}\n")

    return "\n".join(result)


data = yaml.safe_load(Path("system_prompt.yaml").read_text())

messages_data = {msg["role"]: msg["content"] for msg in data["messages"]}

prompt = ChatPromptTemplate.from_messages([
    ("system", messages_data["system"]),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


llm = ChatDeepSeek(model="deepseek-chat")

tools = [rss_feed]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

if __name__ == '__main__':
    result = agent_executor.invoke(
        {
            "input": f"Use the rss_feed tool to fetch items from {urls}, then summarize each item according to the system instructions."
        }
    )

    print(result["output"])