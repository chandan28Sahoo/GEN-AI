# -*- coding: utf-8 -*-
"""
================================================================================
RETRIEVER 4: MultiQuery Retriever — FAISS
================================================================================

📌 WHAT IS IT?
--------------
MultiQueryRetriever is an advanced LangChain retriever that uses an LLM to
automatically generate MULTIPLE variations of a user's query, runs all of them
against the vector store, and returns the UNION of all results (de-duplicated).

It acts as a "query expansion" layer before the actual vector search.

⚙️ HOW DOES IT WORK?
---------------------
Standard retrieval: 1 query → 1 embedding → top-k results.

MultiQuery retrieval:
  1. LLM receives the original user query.
  2. LLM generates N alternative phrasings of the same question from different
     angles (e.g., different wording, sub-questions, hypothetical framings).
  3. Each alternative query is embedded and used to search the vector store.
  4. All results from all queries are merged and de-duplicated.
  5. The final combined set of documents is returned.

Example:
  Original query: "How to improve energy levels and maintain balance?"
  LLM-generated alternatives might be:
    - "What are natural ways to boost daily energy?"
    - "How does sleep and diet affect energy balance?"
    - "Techniques for maintaining physical and mental energy throughout the day"

  Each of these may hit different documents in the vector store, giving a
  richer, more complete context to the LLM.

🛠️ WHAT PROBLEM DOES IT SOLVE?
---------------------------------
Single-query retrieval suffers from "vocabulary mismatch" — the exact words in
your query may not match the words in the relevant documents.

Example:
  Query: "Ways to feel more energetic"
  Document: "Drinking sufficient water throughout the day helps maintain metabolism"
  → These may not have a high cosine similarity despite being semantically related.

MultiQueryRetriever generates diverse queries that increase the probability of
matching relevant documents that a single query might miss.

This example also compares MultiQuery vs simple Similarity retrieval side-by-side
to show the difference.

✅ WHEN TO USE?
---------------
- When queries are complex, ambiguous, or can be interpreted in multiple ways.
- When documents use different vocabulary than the user query.
- When you want better recall (retrieve more relevant documents) at the cost
  of a small latency overhead (LLM call to generate sub-queries).
- When the question covers multiple aspects (e.g., "health, diet, and sleep").

⚠️ LIMITATIONS:
----------------
- Adds LLM API latency and cost (one extra LLM call per retrieval).
- May retrieve more documents than needed — combine with a compressor to trim.
- Not ideal for real-time, latency-sensitive applications.
- The quality of generated sub-queries depends on the LLM.

================================================================================
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever

# ==============================================================================
# Setup
# ==============================================================================

os.environ["NVIDIA_API_KEY"] = "your-nvidia-api-key-here"

embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

# ==============================================================================
# Sample Documents (mixed topics to highlight retrieval differences)
# ==============================================================================

all_docs = [
    # Health-related documents
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.",     metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.",            metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),

    # Unrelated documents (noise)
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.",      metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.",       metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.",             metadata={"source": "I5"}),
]

# ==============================================================================
# Create FAISS Vector Store
# ==============================================================================

vectorstore = FAISS.from_documents(documents=all_docs, embedding=embedding_model)

# ==============================================================================
# Retriever A: Standard Similarity Retriever (baseline for comparison)
# ==============================================================================

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ==============================================================================
# Retriever B: MultiQuery Retriever
# ==============================================================================

# from_llm() creates a MultiQueryRetriever that:
#   1. Uses `llm` to generate multiple alternative queries.
#   2. Uses the inner `retriever` to search for each alternative query.
#   3. Merges and de-duplicates all results.
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

# ==============================================================================
# Query Both Retrievers and Compare
# ==============================================================================

query = "How to improve energy levels and maintain balance?"

similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

# --- Standard Similarity Results ---
print("=== Standard Similarity Retriever Results ===")
print("(These may miss relevant docs due to vocabulary mismatch)\n")
for i, doc in enumerate(similarity_results):
    print(f"--- Result {i + 1} [source: {doc.metadata.get('source')}] ---")
    print(doc.page_content)
    print()

print("=" * 80)

# --- MultiQuery Results ---
print("\n=== MultiQuery Retriever Results ===")
print("(More diverse — LLM-generated sub-queries improve recall)\n")
for i, doc in enumerate(multiquery_results):
    print(f"--- Result {i + 1} [source: {doc.metadata.get('source')}] ---")
    print(doc.page_content)
    print()
