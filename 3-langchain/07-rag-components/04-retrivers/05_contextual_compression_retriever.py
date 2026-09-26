# -*- coding: utf-8 -*-
"""
================================================================================
RETRIEVER 5: Contextual Compression Retriever — FAISS
================================================================================

📌 WHAT IS IT?
--------------
ContextualCompressionRetriever is a two-stage retriever that:
  Stage 1 (Base Retriever): Fetches full documents from the vector store.
  Stage 2 (Compressor):     Passes those documents through an LLM to extract
                             only the sentences/passages that are RELEVANT to
                             the user's query, discarding the rest.

The compressor used here is `LLMChainExtractor`, which uses a chain-of-thought
LLM prompt to identify and return only the relevant portions of each document.

⚙️ HOW DOES IT WORK?
---------------------
Without compression:
  Query: "What is photosynthesis?"
  Retrieved doc: "The Grand Canyon is one of the most visited natural wonders...
                  Photosynthesis is the process by which green plants convert
                  sunlight into energy... Millions of tourists visit every year..."
  → The LLM receives the ENTIRE document including irrelevant tourist info.

With Contextual Compression:
  Stage 1: Base retriever fetches full documents (same as before).
  Stage 2: LLMChainExtractor sends each document + query to the LLM with:
           "Given this document and question, extract only the relevant part."
  → The LLM extracts: "Photosynthesis is the process by which green plants
                        convert sunlight into energy."
  → Only this compressed excerpt is passed to the final LLM for answer generation.

Key components:
  - `base_retriever`: Any standard vector store retriever.
  - `base_compressor`: The component that filters/extracts relevant content.
                       Options include:
                         • LLMChainExtractor – uses LLM to extract relevant text.
                         • LLMChainFilter – uses LLM to filter out irrelevant docs.
                         • EmbeddingsFilter – uses embedding similarity to filter.
                         • DocumentCompressorPipeline – combines multiple compressors.

🛠️ WHAT PROBLEM DOES IT SOLVE?
---------------------------------
Real-world documents are long and contain mixed content — a paragraph about
photosynthesis buried in an article about the Grand Canyon.

Problems with full-document retrieval:
  1. Context window waste: Irrelevant text uses up LLM token budget.
  2. Distraction: LLMs can be misled or confused by irrelevant content.
  3. Noise in RAG: Lower answer quality when context contains off-topic info.

Contextual Compression extracts only the needle from the haystack, giving the
LLM clean, focused, high-quality context.

✅ WHEN TO USE?
---------------
- When your document chunks are large or contain multiple unrelated topics.
- When LLM context windows are limited and every token counts.
- When you want to improve answer precision without aggressive chunking.
- In production RAG systems where answer quality is critical.
- When documents naturally mix topics (e.g., web pages, long-form articles).

⚠️ LIMITATIONS:
----------------
- Adds significant latency: One LLM call per retrieved document.
- Higher cost: Each document compression requires an LLM API call.
- May over-compress: Sometimes removes useful context if the LLM is too
  aggressive. Tune with better prompts or use EmbeddingsFilter as a
  cheaper alternative.
- Not suitable for latency-sensitive real-time applications without caching.

================================================================================
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# ==============================================================================
# Setup
# ==============================================================================

os.environ["NVIDIA_API_KEY"] = "your-nvidia-api-key-here"

embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")

# ==============================================================================
# Sample Documents (intentionally contain MIXED content)
# ==============================================================================
# These documents contain photosynthesis info buried inside unrelated text.
# This mimics real-world messy documents (web pages, reports, books).

docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# ==============================================================================
# Stage 1: Base Vector Store Retriever (FAISS)
# ==============================================================================

# This fetches the top-k full documents — before compression.
vectorstore = FAISS.from_documents(docs, embedding_model)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ==============================================================================
# Stage 2: Compressor — LLMChainExtractor
# ==============================================================================

# LLMChainExtractor uses the LLM to read each retrieved document and extract
# only the sentences directly relevant to the user's query.
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
compressor = LLMChainExtractor.from_llm(llm)

# ==============================================================================
# Create the Contextual Compression Retriever
# ==============================================================================

# Combines Stage 1 (base_retriever) + Stage 2 (compressor):
# base_retriever fetches docs → compressor trims them → clean excerpts returned.
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# ==============================================================================
# Query
# ==============================================================================

query = "What is photosynthesis?"

# Each result is a compressed excerpt — only the relevant part of each document.
compressed_results = compression_retriever.invoke(query)

print("=== Contextual Compression Retriever Results ===")
print("(Only relevant excerpts extracted from each document)\n")

for i, doc in enumerate(compressed_results):
    print(f"--- Result {i + 1} [source: {doc.metadata.get('source')}] ---")
    print(doc.page_content)
    print()
