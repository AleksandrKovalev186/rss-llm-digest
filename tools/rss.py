import csv
from typing import List

import feedparser
from langchain_core.tools import tool

from memory.vector_store import store_news, retrieve_news

MAX_SEARCH_CALLS = 3

# shortcut: assumes one pipeline run at a time; move into the graph state if runs overlap.
_returned_links: set[str] = set()
_search_calls = 0


def reset_run_state() -> None:
    global _search_calls
    _returned_links.clear()
    _search_calls = 0


def read_urls(filename: str) -> List[str]:
    with open(filename, "r", encoding='utf-8') as file:
        return [row[0] for row in csv.reader(file, delimiter=',') if row]


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
    skipped = 0
    for url in feed_urls:
        feed = feedparser.parse(url)

        if not feed.entries:
            result.append(f"RSS feed {url} has no entries or could not be parsed.\n")
            continue

        for i, entry in enumerate(feed.entries[:count], start=1):
            title = entry.get("title", "No title")
            link = entry.get("link", "No link")
            content = entry.get("summary", "No content")[:250]

            if link in _returned_links:
                skipped += 1
                continue
            _returned_links.add(link)

            result.append(f"{i}. Title: {title}\n   Link: {link}\n   Content: {content}\n")

            parsed_entries.append({
                "title": title,
                "link": link,
                "content": content,
                "source_url": url
            })

    # Chroma rejects an empty upsert, e.g. when every feed is unreachable.
    if parsed_entries:
        store_news(parsed_entries)
    if skipped:
        result.append(
            f"{skipped} item(s) were already returned earlier in this run and are omitted. "
            f"Do not call rss_feed again; use the earlier results."
        )
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
    global _search_calls
    if _search_calls >= MAX_SEARCH_CALLS:
        return "Search limit reached for this run. Do not call search_rss_history again; " \
               "write the summaries now."
    _search_calls += 1
    return retrieve_news(query, k, exclude_links=_returned_links) or "No related articles found."
