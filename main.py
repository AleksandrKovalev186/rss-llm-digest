import feedparser
from pathlib import Path
from langchain_core.prompts.loading import load_prompt
urls = "https://aws.amazon.com/blogs/training-and-certification/feed"


prompt_config = Path("system_prompt.yaml")
prompt = load_prompt(prompt_config)


def rss_feed(url: str):
    feed = feedparser.parse(url)
    print(feed['entries'][:10])


if __name__ == '__main__':
    rss_feed(urls)
