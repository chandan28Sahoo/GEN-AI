# -*- coding: utf-8 -*-
"""
================================================================================
RETRIEVER 1: Wikipedia Retriever
================================================================================

📌 WHAT IS IT?
--------------
WikipediaRetriever is a built-in LangChain retriever that fetches documents
directly from Wikipedia using the Wikipedia API. It does NOT require a vector
store or embedding model — it retrieves documents in real-time from the web.

⚙️ HOW DOES IT WORK?
---------------------
1. You pass a natural language query to the retriever.
2. Internally, it calls the Wikipedia API with that query as a search term.
3. Wikipedia returns the top matching article(s) as full-text documents.
4. Each article is wrapped into a LangChain `Document` object with:
   - `page_content`: The full article text.
   - `metadata`: Includes source URL, title, etc.
5. You get back a list of `Document` objects ready to use in your RAG pipeline.

🛠️ WHAT PROBLEM DOES IT SOLVE?
---------------------------------
- Eliminates the need to manually scrape or download Wikipedia articles.
- Provides on-the-fly access to a massive, constantly updated knowledge base.
- Useful when you don't want to build and maintain a custom vector store for
  general knowledge questions.

✅ WHEN TO USE?
---------------
- When the user asks open-ended factual or encyclopedic questions.
- When you want to augment LLM responses with up-to-date, real-world knowledge.
- In educational apps, research assistants, or general Q&A bots where the
  domain is broad and hard to pre-index.
- Quick prototyping of RAG pipelines without needing a vector store.

⚠️ LIMITATIONS:
----------------
- Requires internet access at query time.
- Results depend on Wikipedia's search ranking, which may not always match
  what the user truly needs.
- Not suitable for private/internal documents or domain-specific knowledge.

================================================================================
"""

import os
from langchain_community.retrievers import WikipediaRetriever

# ==============================================================================
# Setup
# ==============================================================================

# No API key needed for Wikipedia — it's a public API.
# However, if you use this as part of a larger LangChain pipeline with NVIDIA
# models, set your NVIDIA key like this:
# os.environ["NVIDIA_API_KEY"] = "your-key-here"

# ==============================================================================
# Initialize the Wikipedia Retriever
# ==============================================================================

# top_k_results: how many Wikipedia articles to fetch per query
# lang: the language of Wikipedia to search (e.g., "en", "hi", "de")
retriever = WikipediaRetriever(top_k_results=2, lang="en")

# ==============================================================================
# Query the Retriever
# ==============================================================================

query = "the geopolitical history of india and pakistan from the perspective of a chinese"
docs = retriever.invoke(query)

# ==============================================================================
# Display Results
# ==============================================================================

print("=== Wikipedia Retriever Results ===\n")

for i, doc in enumerate(docs):
    print(f"--- Result {i + 1} ---")
    print(f"Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Title:  {doc.metadata.get('title', 'N/A')}")
    print(f"Content Preview:\n{doc.page_content[:500]}...")  # truncate for display
    print()
