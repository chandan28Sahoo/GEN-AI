# -*- coding: utf-8 -*-
"""
================================================================================
RETRIEVER 3: MMR (Maximal Marginal Relevance) Retriever — FAISS
================================================================================

📌 WHAT IS IT?
--------------
MMR (Maximal Marginal Relevance) is an advanced retrieval strategy that balances
two competing goals:
  1. RELEVANCE   — Results must be similar to the query.
  2. DIVERSITY   — Results must NOT be too similar to each other.

It is used as a `search_type` option on any LangChain vector store retriever.
This example uses FAISS (Facebook AI Similarity Search), a highly efficient
vector similarity library from Meta.

FAISS is faster than Chroma for large datasets and supports GPU acceleration.

⚙️ HOW DOES IT WORK?
---------------------
Standard similarity search picks the top-k documents purely by similarity to
the query. This can return multiple near-duplicate results.

MMR fixes this with an iterative algorithm:
  1. Find a large candidate pool (fetch_k documents most similar to the query).
  2. Pick the first document that is most similar to the query.
  3. For each remaining candidate, compute a score:
       MMR score = λ * similarity(doc, query) - (1 - λ) * max_similarity(doc, selected_docs)
     Where λ (lambda_mult) controls the trade-off:
       - λ = 1.0  → Pure similarity (same as standard search).
       - λ = 0.0  → Pure diversity (maximally different from already-selected docs).
       - λ = 0.5  → Balanced trade-off (recommended default).
  4. Select the document with the highest MMR score.
  5. Repeat steps 3-4 until k documents are selected.

🛠️ WHAT PROBLEM DOES IT SOLVE?
---------------------------------
Imagine your vector store has 10 documents about "LangChain", all slightly
reworded. A standard similarity search for "What is LangChain?" would return
multiple near-duplicates — wasting the LLM's context window.

MMR solves this by ensuring each returned document adds NEW information,
making the retrieved context more comprehensive and less redundant.

✅ WHEN TO USE?
---------------
- When your document collection has many similar/duplicate chunks.
- When you want to provide the LLM with a broader, more diverse context.
- When users ask broad questions that can be answered from multiple angles.
- When you have large knowledge bases (wikis, books, large codebases) where
  the same concept is discussed in many places.

⚠️ LIMITATIONS:
----------------
- Slightly slower than standard similarity search due to the iterative process.
- The lambda_mult parameter needs tuning — too low means highly off-topic results.
- fetch_k should be larger than k for MMR to work effectively.

================================================================================
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.documents import Document

# ==============================================================================
# Setup
# ==============================================================================

os.environ["NVIDIA_API_KEY"] = "your-nvidia-api-key-here"

embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")

# ==============================================================================
# Sample Documents (intentionally repetitive to show MMR's power)
# ==============================================================================

# Notice docs 1 and 2 are very similar ("LangChain" + "LLM").
# Standard search would likely return both; MMR will prefer diversity.
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# ==============================================================================
# Create FAISS Vector Store
# ==============================================================================

# FAISS builds an index in memory for fast similarity search.
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# ==============================================================================
# Create MMR Retriever
# ==============================================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",          # Enable MMR instead of default cosine similarity
    search_kwargs={
        "k": 3,                 # Number of documents to return
        "lambda_mult": 0.5,     # Trade-off: 0=max diversity, 1=max similarity
        # "fetch_k": 20,        # Optional: candidate pool size (default 20)
    }
)

# ==============================================================================
# Query
# ==============================================================================

query = "What is langchain?"
results = retriever.invoke(query)

print("=== MMR Retriever Results (diverse, non-redundant) ===\n")
for i, doc in enumerate(results):
    print(f"--- Result {i + 1} ---")
    print(doc.page_content)
    print()
