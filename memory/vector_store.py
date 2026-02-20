import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings.config import settings

COLLECTION_NAME = "telegram_rules"

_vectorstore: Chroma | None = None


def init_vectorstore() -> None:
    global _vectorstore

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        loader = WebBaseLoader("https://core.telegram.org/bots/api#html-style")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        _vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            client=client,
            collection_name=COLLECTION_NAME,
        )
    else:
        _vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )


def retrieve_rules(query: str, k: int = 4) -> str:
    if _vectorstore is None:
        raise RuntimeError(
            "Vector store is not initialized. "
            "Call init_vectorstore() before using retrieve_rules()."
        )

    docs = _vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
