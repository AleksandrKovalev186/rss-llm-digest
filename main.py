import feedparser

urls = "https://aws.amazon.com/blogs/training-and-certification/feed"


def rss_feed(url: str):
    feed = feedparser.parse(url)
    print(feed['entries'][:5])


if __name__ == '__main__':
    rss_feed(urls)
