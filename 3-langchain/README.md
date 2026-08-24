# 🦜️🔗 LangChain Learning Curriculum: From Zero to Hero

A comprehensive, hands-on, step-by-step learning repository for mastering **Generative AI and LLM Application Engineering with LangChain**.

Designed for **beginner to intermediate developers**, this repository contains runnable code examples, intuitive architectural diagrams, and comprehensive topic-specific markdown guides answering **What?**, **Why?**, and **How to use?** for every topic.

---

## 🗺️ Master Curriculum Roadmap

```
 01-models               02-prompts              03-output-parsers
┌────────────────┐     ┌────────────────┐      ┌────────────────┐
│ • LLMs         │ ──► │ • Templates    │  ──► │ • StrParser    │
│ • Chat Models  │     │ • Roles        │      │ • JsonParser   │
│ • Embeddings   │     │ • Chat History │      │ • Pydantic     │
└────────────────┘     └────────────────┘      └────────────────┘
                                                       │
                                                       ▼
 06-runnables-lcel       05-chains               04-structured-output
┌────────────────┐     ┌────────────────┐      ┌────────────────┐
│ • Sequence     │ ◄── │ • Simple       │  ◄── │ • with_struct. │
│ • Parallel     │     │ • Sequential   │      │ • Pydantic     │
│ • Passthrough  │     │ • Parallel     │      │ • TypedDict    │
│ • Branch       │     │ • Conditional  │      │ • JSON Schema  │
└────────────────┘     └────────────────┘      └────────────────┘
        │
        ▼
 07-rag-components       08-practice             09-resources
┌────────────────┐     ┌────────────────┐      ┌────────────────┐
│ • Loaders      │ ──► │ • Hands-on     │  ──► │ • PDF Books    │
│ • Text Splits  │     │ • Mini RAG Bot │      │ • Roadmaps     │
│ • ChromaDB     │     │ • Challenges   │      │ • Cheat Sheets │
└────────────────┘     └────────────────┘      └────────────────┘
```

---

## 📚 Detailed Topic Guides (What? Why? How?)

| Module | Topic Guide | Key Concepts Covered |
| :--- | :--- | :--- |
| **01-models** | [**01_models_overview.md**](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/01_models_overview.md)<br>• [01_llms.md](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/01-llms/01_llms.md)<br>• [02_chat_models.md](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/02-chat-models/02_chat_models.md)<br>• [03_embedding_models.md](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/03-embedding-models/03_embedding_models.md) | Completion LLMs, Chat Models (Groq, Gemini, OpenAI, Hugging Face), Vector Embeddings, Cosine Similarity math |
| **02-prompts** | [**02_prompts.md**](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/02-prompts/02_prompts.md) | `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`, Streamlit UI integration, Multi-turn Chatbot |
| **03-output-parsers** | [**03_output_parsers.md**](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/03-output-parsers/03_output_parsers.md) | Transforming raw text into `str`, `json`, `ResponseSchema`, and strictly validated `Pydantic` models |
| **04-structured-output** | [**04_structured_output.md**](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/04-structured-output/04_structured_output.md) | Native Tool Calling with `.with_structured_output()`, Pydantic vs TypedDict vs JSON schema |
| **05-chains** | [**05_chains.md**](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/05-chains/05_chains.md) | Composing pipelines with `\|`: Simple Linear, Sequential Multi-Step, Parallel, and Conditional Routing chains |
| **06-runnables-lcel** | [**06_runnables_lcel.md**](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/06-runnables-lcel/06_runnables_lcel.md) | Core LCEL Primitives: `RunnableSequence`, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, `RunnableBranch` |
| **07-rag-components** | [**07_rag_overview.md**](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/07_rag_overview.md)<br>• [01_document_loaders.md](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/01-document-loaders/01_document_loaders.md)<br>• [02_text_splitters.md](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/02-text-splitters/02_text_splitters.md)<br>• [03_vector_stores.md](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/07-rag-components/03-vector-stores/03_vector_stores.md) | End-to-end RAG architecture: Ingestion (PDF, CSV, TXT, Dir), Chunking (Recursive, Code, Semantic), & ChromaDB Indexing |
| **08-practice** | [**08_practice.md**](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/08-practice/08_practice.md) | Practical exercises & 4 real-world project blueprints (including a local PDF Mini-RAG assistant) |
| **09-resources** | [**09_resources.md**](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/09-resources/09_resources.md) | Deep Learning curriculum, Machine Learning textbooks, and full Gen AI career study path |

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone and enter directory
git clone <repo-url>
cd GEN-AI

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```ini
OPENAI_API_KEY="your-openai-api-key"
GROQ_API_KEY="your-groq-api-key"
GOOGLE_AI_API_KEY="your-google-gemini-key"
HUGGINGFACE_API_TOKEN="your-huggingface-token"
```

### 3. Run Any Script
```bash
# Run a chat model example:
python 3-langchain/01-models/02-chat-models/1_chatmodel_groq.py

# Run parallel chain:
python 3-langchain/05-chains/3_parallel_chain.py

# Run ChromaDB Vector Store:
python 3-langchain/07-rag-components/03-vector-stores/1_langchain_chroma.py

# Run Streamlit Prompt UI:
streamlit run 3-langchain/02-prompts/1_prompt_ui.py
```
