# 03 - Vector Stores & ChromaDB

---

## 🌟 1. What is a Vector Store in LangChain?

A **Vector Store** (or Vector Database) is a database specifically optimized for storing, indexing, and querying high-dimensional vector embeddings alongside document text and metadata.

### The Vector Search Workflow
1. Documents are converted into embeddings and stored in the database.
2. When a user asks a question, their query is converted into an embedding.
3. The database performs an **Approximate Nearest Neighbor (ANN)** search to retrieve the most mathematically similar documents in milliseconds.

```
                           ┌──────────────────────────────────────────────────┐
                           │               ChromaDB Vector Store              │
                           ├──────────────────────────────────────────────────┤
                           │ Doc ID 1: [0.12, -0.44, ...] | Meta: {"team":"RCB"}
User Query:                │ Doc ID 2: [0.89,  0.11, ...] | Meta: {"team":"MI"}
"Who is a fast bowler?" ──►│ Doc ID 3: [0.03, -0.85, ...] | Meta: {"team":"CSK"}
                           └────────────────────────┬─────────────────────────┘
                                                    │
                                                    ▼
                                     Top-K Most Relevant Documents
                                     (e.g., Jasprit Bumrah doc)
```

---

## ❓ 2. Why Do We Need Vector Stores?

### The Problems They Solve:
1. **Sub-second Retrieval at Scale**: While calculating cosine similarity in NumPy works for 10 documents, it cannot scale to 100,000 documents. Vector stores use hierarchical clustering indexes (HNSW, IVF) to search massive datasets in milliseconds.
2. **Metadata Filtering (Hybrid Search)**: Filter search results by author, date, user ID, or category *before* or *during* semantic search.
3. **Persistence & CRUD**: Easily add new documents, update existing knowledge, and delete outdated entries without recomputing the entire database.

---

## 🚀 3. How Do We Use ChromaDB in LangChain?

### 📂 File in this Submodule

- [1_langchain_chroma.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/03-vector-stores/1_langchain_chroma.py): Comprehensive runnable script demonstrating document creation, persistence, similarity search, scores, metadata filtering, updates, and deletion.

---

### Step-by-Step Code Walkthrough: Full CRUD & Search in ChromaDB

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Step 1: Prepare documents with metadata
docs = [
    Document(
        page_content="Virat Kohli is an aggressive batsman who led Royal Challengers Bangalore.",
        metadata={"team": "RCB", "role": "batsman"}
    ),
    Document(
        page_content="Jasprit Bumrah is a lethal fast bowler for Mumbai Indians known for yorkers.",
        metadata={"team": "MI", "role": "bowler"}
    ),
    Document(
        page_content="MS Dhoni is a legendary captain and wicketkeeper for Chennai Super Kings.",
        metadata={"team": "CSK", "role": "wicketkeeper"}
    )
]

# Step 2: Initialize ChromaDB (persisted locally on disk)
persist_path = str(Path(__file__).parent / 'my_chroma_db')

vector_store = Chroma(
    collection_name="ipl_players",
    embedding_function=OpenAIEmbeddings(),
    persist_directory=persist_path
)

# Step 3: Add documents to vector database
doc_ids = vector_store.add_documents(docs)
print(f"Added {len(doc_ids)} documents to ChromaDB.")

# Step 4: Semantic Similarity Search (Top-2 matches)
print("\n--- Similarity Search ---")
results = vector_store.similarity_search(query="Who is known for bowling yorkers?", k=2)
for r in results:
    print(f"Match: {r.page_content} (Team: {r.metadata['team']})")

# Step 5: Similarity Search with Distance Scores
print("\n--- Search with Distance Scores ---")
scored_results = vector_store.similarity_search_with_score(query="wicketkeeper captain", k=1)
doc, distance = scored_results[0]
print(f"Closest match (Distance: {distance:.4f}): {doc.page_content}")

# Step 6: Metadata Filtering (Search ONLY within CSK)
print("\n--- Metadata Filtered Search ---")
filtered_results = vector_store.similarity_search(
    query="cricket player",
    filter={"team": "CSK"}
)
for r in filtered_results:
    print("CSK Player:", r.page_content)

# Step 7: Update a Document
updated_kohli = Document(
    page_content="Virat Kohli holds the all-time record for most runs scored in IPL history.",
    metadata={"team": "RCB", "role": "batsman", "status": "legend"}
)
vector_store.update_document(document_id=doc_ids[0], document=updated_kohli)

# Step 8: Delete a Document
vector_store.delete(ids=[doc_ids[1]])
print("Deleted document successfully.")
```

---

## ⚖️ Vector Database Landscape

| Database | Type | Strengths | Best Use Case |
| :--- | :--- | :--- | :--- |
| **ChromaDB** | Local / Embedded | Zero setup, open-source, fast local prototyping | Development, local RAG, demos |
| **FAISS** (Meta) | In-Memory Library | Extreme search speed on CPU & GPU | Fast clustering & similarity research |
| **Pinecone** / **Qdrant** | Managed Cloud | Scalable to billions of vectors, high availability | Production enterprise cloud applications |
| **pgvector** | PostgreSQL Extension | Combine vector search and relational SQL queries | Existing Postgres infrastructure |
