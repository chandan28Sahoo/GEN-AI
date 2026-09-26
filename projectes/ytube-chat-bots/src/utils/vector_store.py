import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def build_vector_store(
    text: str,
    collection_name: str = "ytube_chat",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    persist_dir: str = "./data/vector_store",
) -> Chroma:
    """Split text and persist into a ChromaDB vector store.

    Args:
        text: Raw transcript text.
        collection_name: ChromaDB collection name.
        chunk_size: Token size per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        persist_dir: Local directory to persist the vector store.

    Returns:
        A Chroma vector store instance ready for retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = splitter.create_documents([text])

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    return vectorstore


def video_id_exists(
    video_id: str,
    persist_dir: str = "./data/vector_store",
) -> bool:
    """Check whether a ChromaDB collection for this video_id already exists on disk."""
    if not os.path.isdir(persist_dir):
        return False
    client = chromadb.PersistentClient(path=persist_dir)
    existing = {col.name for col in client.list_collections()}
    return video_id in existing


def get_or_build_vector_store(
    video_id: str,
    transcript: str,
    persist_dir: str = "./data/vector_store",
) -> Chroma:
    """Load the vector store if the video was already indexed, otherwise build it.

    Args:
        video_id: YouTube video ID used as the ChromaDB collection name.
        transcript: Raw transcript text (used only when building).
        persist_dir: Local directory to persist the vector store.

    Returns:
        A Chroma vector store instance ready for retrieval.
    """
    if video_id_exists(video_id, persist_dir):
        print(f"⚡ Cache hit — loading existing vector store for '{video_id}'.")
        return load_vector_store(collection_name=video_id, persist_dir=persist_dir)

    print(f"🔨 Building vector store for '{video_id}'...")
    return build_vector_store(transcript, collection_name=video_id, persist_dir=persist_dir)


def load_vector_store(
    collection_name: str = "ytube_chat",
    persist_dir: str = "./data/vector_store",
) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
