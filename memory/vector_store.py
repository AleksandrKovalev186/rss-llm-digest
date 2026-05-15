from datetime import datetime, UTC, timedelta

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from settings.config import settings

NEWS_COLLECTION = "news_collection"

_rss_store: Chroma | None = None
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def init_vectorstore() -> None:
    global _rss_store

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    client.get_or_create_collection(NEWS_COLLECTION)

    _rss_store = Chroma(
        client=client,
        collection_name=NEWS_COLLECTION,
        embedding_function=_embeddings,
    )


def store_news(entries: list[dict[str, str]]) -> None:
    if _rss_store is None:
        raise RuntimeError("Rss store is not initialized.")

    texts = [
        f"{e['title']}\n{e['content']}" for e in entries
    ]
    ids = [e["link"] for e in entries]
    metadata = [
        {
            "title": e["title"],
            "link": e["link"],
            "source_url": e["source_url"],
            "fetched_at": datetime.now(UTC).timestamp(),
        }
        for e in entries
    ]

    _rss_store.add_texts(texts=texts, ids=ids, metadatas=metadata)


def retrieve_news(query: str, k: int = 3) -> str:
    threshold = (datetime.now() - timedelta(minutes=20)).timestamp()
    if _rss_store is None:
        raise RuntimeError("Rss store is not initialized.")

    docs = _rss_store.similarity_search(
        query,
        k=k,
        filter={"fetched_at": {"$lt": threshold}}  # type: ignore[arg-type]
    )
    return "\n\n---\n\n".join(
        f"Title: {doc.metadata['title']}\nLink: {doc.metadata['link']}\n{doc.page_content}"
        for doc in docs
    )
