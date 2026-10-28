import csv
from typing import List

import feedparser
from langchain_core.tools import tool

from memory.vector_store import store_news, retrieve_news


def read_urls(filename: str) -> List[str]:
    with open(filename, "r", encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=',')
        rows = list(csv_reader)
    return rows


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
