from datetime import datetime, UTC

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings.config import settings

TELEGRAM_RULE_COLLECTION = "telegram_rules"
NEWS_COLLECTION = "news_collection"

_vectorstore: Chroma | None = None
_rss_store: Chroma | None = None
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def init_vectorstore() -> None:
    global _vectorstore
    global _rss_store

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    tg_collection = client.get_or_create_collection(TELEGRAM_RULE_COLLECTION)
    client.get_or_create_collection(NEWS_COLLECTION)

    _rss_store = Chroma(
        client=client,
        collection_name=NEWS_COLLECTION,
        embedding_function=_embeddings,
    )

    if tg_collection.count() == 0:
        loader = WebBaseLoader("https://core.telegram.org/bots/api#html-style")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        _vectorstore = Chroma.from_documents(
            chunks,
            _embeddings,
            client=client,
            collection_name=TELEGRAM_RULE_COLLECTION,
        )
    else:
        _vectorstore = Chroma(
            client=client,
            collection_name=TELEGRAM_RULE_COLLECTION,
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
            "fetched_at": datetime.now(UTC).isoformat(),
        }
        for e in entries
    ]

    _rss_store.add_texts(texts=texts, ids=ids, metadatas=metadata)


def retrieve_rules(query: str, k: int = 4) -> str:
    if _vectorstore is None:
        raise RuntimeError(
            "Vector store is not initialized. "
            "Call init_vectorstore() before using retrieve_rules()."
        )

    docs = _vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def retrieve_news(query: str, k: int = 3) -> str:
    if _rss_store is None:
        raise RuntimeError(
            "Rss store is not initialized. "
            "Call init_vectorstore() before using retrieve_rules()."
        )

    docs = _rss_store.similarity_search(query, k=k)
    return "\n\n---\n\n".join(
        f"Title: {doc.metadata['title']}\nLink: {doc.metadata['link']}\n{doc.page_content}"
        for doc in docs
    )
