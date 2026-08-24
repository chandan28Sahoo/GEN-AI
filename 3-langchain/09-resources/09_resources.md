# 09 - Reference Books, Roadmaps & Curated Resources

---

## 🌟 1. What is this Module?

This module contains reference textbooks, course curriculums, official documentation links, and learning roadmaps to guide your progression from basic prompt engineering to production AI engineering.

---

## 📚 Included Reference Books & Documents

1. **[Building Machine Learning Systems with Python - Second Edition.pdf](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/09-resources/Building%20Machine%20Learning%20Systems%20with%20Python%20-%20Second%20Edition.pdf)**
   - **What it covers**: Feature engineering, supervised learning algorithms (Linear Models, Decision Trees, SVMs), unsupervised clustering, recommendation systems, and evaluation metrics using Python and Scikit-Learn.
   - **Why read it**: Provides the mathematical intuition behind vector spaces, distance metrics (Cosine vs Euclidean), and classification algorithms.

2. **[dl-curriculum.pdf](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/09-resources/dl-curriculum.pdf)**
   - **What it covers**: Deep learning fundamentals, Artificial Neural Networks (ANNs), Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Attention Mechanisms, and Transformer architecture.
   - **Why read it**: Essential foundational curriculum for understanding how modern LLMs are structured.

---

## 🗺️ Recommended Generative AI Study Roadmap

```
Phase 1: Foundations
┌────────────────────────────────────────────────────────┐
│  • Python (OOP, Pydantic, Type Annotations)            │
│  • NumPy & Vector Math (Dot products, Cosine Distance) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
Phase 2: Core LangChain & LLM APIs
┌────────────────────────────────────────────────────────┐
│  • Models (LLMs, ChatModels, Embeddings)               │
│  • Prompt Engineering (Templates, Messages, Memory)    │
│  • Output Parsers & Structured Output (.with_struct)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
Phase 3: Pipelines & Composition (LCEL)
┌────────────────────────────────────────────────────────┐
│  • LCEL Primitives (Sequence, Parallel, Branch, Lambda)│
│  • Streaming & Asynchronous execution                  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
Phase 4: Retrieval-Augmented Generation (RAG)
┌────────────────────────────────────────────────────────┐
│  • Document Ingestion (PDF, CSV, TXT)                  │
│  • Chunking Strategies (Recursive, Code, Semantic)     │
│  • Vector Databases (ChromaDB, Hybrid Search, Filter)  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
Phase 5: Production & Autonomous Agents
┌────────────────────────────────────────────────────────┐
│  • Tool Calling & Function Execution                   │
│  • LangGraph (Stateful, Multi-Agent Cyclic Graphs)     │
│  • Observability & Tracing (LangSmith)                 │
└────────────────────────────────────────────────────────┘
```

---

## 🌐 Official Documentation & External Links

| Resource | Official URL | Description |
| :--- | :--- | :--- |
| **LangChain Python Documentation** | [python.langchain.com](https://python.langchain.com/) | Official tutorials, guides, and component references. |
| **LangChain Expression Language (LCEL)** | [LCEL Docs](https://python.langchain.com/docs/concepts/lcel/) | Official guide on composition and streaming. |
| **ChromaDB Documentation** | [docs.trychroma.com](https://docs.trychroma.com/) | Chroma vector store installation and query filtering. |
| **Hugging Face Hub** | [huggingface.co/models](https://huggingface.co/models) | Open-source LLMs and SentenceTransformers embeddings. |
| **Pydantic Documentation** | [docs.pydantic.dev](https://docs.pydantic.dev/) | Python data validation and JSON schema generation. |
