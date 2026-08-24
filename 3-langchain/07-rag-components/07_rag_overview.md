# Module 07: Retrieval-Augmented Generation (RAG) Architecture Overview

---

## 🌟 1. What is RAG (Retrieval-Augmented Generation)?

**Retrieval-Augmented Generation (RAG)** is the industry-standard architecture for grounding AI models on your private or custom data.

Instead of retraining or fine-tuning expensive models, RAG works by **dynamically finding relevant context** in your documents and injecting it into the prompt sent to the LLM.

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │                  END-TO-END RAG FLOW                   │
                                  └────────────────────────────────────────────────────────┘

 1. INGESTION PHASE (Offline Indexing)
 ═════════════════════════════════════
  PDF / CSV / TXT ──► [ Document Loaders ] ──► [ Text Splitters ] ──► [ Embedding Model ] ──► [ Vector Store (Chroma) ]
                          (01-loaders)            (02-splitters)          (01-models/03)           (03-vector-stores)

 2. RETRIEVAL & GENERATION PHASE (Online Querying)
 ═════════════════════════════════════════════════
  User Query ─────────► [ Embedding Model ] ──► Vector Similarity Search ──► Top-K Relevant Document Chunks
                                                                                       │
                                                                                       ▼
  User Query + Retrieved Chunks ──────────────► [ Chat Model ] ────────────► Grounded, Fact-Based Answer
```

---

## ❓ 2. Why Do We Need RAG?

### The Core Limitations of Raw LLMs:
1. **Knowledge Cutoff**: Models do not know about events, news, or changes after their training date.
2. **Private Data Isolation**: Models know nothing about your company's proprietary codebase, internal wikis, or user records.
3. **Hallucinations**: Without grounding facts in the prompt, LLMs can fabricate plausibly sounding falsehoods.

### How RAG Solves This:
- **Zero Hallucination Grounding**: The LLM is instructed: *"Answer the question based ONLY on the provided context."*
- **Source Attributions**: Every response can cite the exact document and page number.
- **Cost Effective**: Updating knowledge requires only indexing a new document into ChromaDB (fractions of a cent), rather than retraining a model ($10,000+).

---

## 🚀 3. How the 3 RAG Components Connect in LangChain

```
07-rag-components/
├── 01-document-loaders/   ──► 1. Ingestion: Load raw PDFs, CSVs, TXT, directories into LangChain Document objects.
├── 02-text-splitters/     ──► 2. Chunking: Split documents into semantically coherent overlapping passages.
└── 03-vector-stores/      ──► 3. Indexing & Search: Store vector embeddings in ChromaDB and retrieve top-K matches.
```

---

## 📚 Direct Submodule Guide Links

1. **[01-document-loaders Guide](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/01-document-loaders/01_document_loaders.md)**
   - Extract text from `.txt`, `.pdf`, `.csv`, and multi-file directories with metadata.
2. **[02-text-splitters Guide](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/02-text-splitters/02_text_splitters.md)**
   - Chunk text using Recursive, Markdown, Python code, and Semantic chunking.
3. **[03-vector-stores Guide](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/03-vector-stores/03_vector_stores.md)**
   - ChromaDB initialization, document indexing, similarity search, scores, and metadata filtering.
