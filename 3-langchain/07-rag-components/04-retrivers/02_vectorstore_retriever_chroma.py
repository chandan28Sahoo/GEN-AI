# -*- coding: utf-8 -*-
"""
================================================================================
RETRIEVER 2: Vector Store Retriever (Chroma)
================================================================================

📌 WHAT IS IT?
--------------
A Vector Store Retriever is the most fundamental retriever in RAG pipelines.
It converts your documents into numerical vector representations (embeddings),
stores them in a vector database (Chroma in this case), and retrieves the most
semantically similar documents for a given query.

Chroma is an open-source, in-memory vector database that is easy to use for
local development and prototyping.

⚙️ HOW DOES IT WORK?
---------------------
1. INDEXING (done once):
   - Each document is passed through an embedding model.
   - The embedding model converts the text into a dense float vector
     (e.g., 1024 dimensions with NVIDIA's model).
   - These vectors are stored in Chroma's collection.

2. RETRIEVAL (done at query time):
   - The user query is also embedded into a vector.
   - Chroma performs a cosine similarity (or dot product) search against all
     stored document vectors.
   - The top-k most similar documents are returned.

3. Two ways to retrieve:
   a) `vectorstore.as_retriever()` — Returns a standard LangChain Retriever object.
      Plugs directly into chains like RetrievalQA.
   b) `vectorstore.similarity_search()` — Direct method call, returns documents.
      Useful for debugging or one-off queries.

🛠️ WHAT PROBLEM DOES IT SOLVE?
---------------------------------
- Traditional keyword search (like SQL LIKE or grep) fails to understand meaning.
  e.g., "car" vs "automobile" — keyword search misses the connection.
- Vector search solves this by comparing meaning (semantics) rather than
  exact words.
- Allows LLMs to "look up" relevant information from large document collections
  before generating an answer.

✅ WHEN TO USE?
---------------
- The foundation of almost every RAG pipeline — use it as your default choice.
- When you have a fixed, relatively small set of documents (< 1 million chunks).
- Local development or prototyping (Chroma runs in memory — no setup needed).
- When you need simple semantic search without diversity or query expansion.

⚠️ LIMITATIONS:
----------------
- Returns top-k results purely by similarity score — can return redundant/
  near-duplicate results if documents overlap a lot.
- Chroma in-memory mode does not persist data between sessions (use
  `persist_directory` param for persistence).
- Embedding quality directly impacts retrieval quality.

================================================================================
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.documents import Document

# ==============================================================================
# Setup
# ==============================================================================

os.environ["NVIDIA_API_KEY"] = "your-nvidia-api-key-here"

# ==============================================================================
# Sample Documents
# ==============================================================================

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="NVIDIA provides powerful embedding models."),
]

# ==============================================================================
# Initialize Embedding Model
# ==============================================================================

# NVIDIAEmbeddings uses NVIDIA's hosted embedding API to convert text -> vectors.
# Each text chunk becomes a dense float vector (e.g., 1024 dimensions).
embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")

# ==============================================================================
# Create Chroma Vector Store
# ==============================================================================

# from_documents() does two things:
#   1. Embeds all documents using the embedding model.
#   2. Stores those embeddings in Chroma (in-memory by default).
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# ==============================================================================
# METHOD A: Use as_retriever() — LangChain Retriever Interface
# ==============================================================================

# as_retriever() wraps the vector store in a standard LangChain Retriever.
# This lets you plug it directly into chains (e.g., RetrievalQA).
# k=2: return top 2 most similar documents.
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

print("=== METHOD A: as_retriever() Results ===\n")
for i, doc in enumerate(results):
    print(f"--- Result {i + 1} ---")
    print(doc.page_content)
    print()

# ==============================================================================
# METHOD B: Direct similarity_search() — One-off Query
# ==============================================================================

results = vectorstore.similarity_search(query, k=2)

print("=== METHOD B: similarity_search() Results ===\n")
for i, doc in enumerate(results):
    print(f"--- Result {i + 1} ---")
    print(doc.page_content)
    print()
