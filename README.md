# 🧠 Generative AI & LangChain Learning Repository

A comprehensive, hands-on learning workspace for building Generative AI applications using **LangChain**, modern Large Language Models (LLMs), prompt engineering, chains, and Retrieval-Augmented Generation (RAG).

---

## 🗺️ Learning Path & Curriculum Structure

All learning materials are organized sequentially inside the [**`3-langchain/`**](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/README.md) directory with dedicated, in-depth **`<topic_name>.md`** guides for every topic answering **What?**, **Why?**, and **How to use?**:

```
3-langchain/
├── 01-models/
│   ├── 01_models_overview.md                        ──► Master guide to Models & Embeddings
│   ├── 01-llms/ (01_llms.md)                        ──► Text-completion models (Legacy)
│   ├── 02-chat-models/ (02_chat_models.md)          ──► OpenAI, Google Gemini, Groq, Hugging Face
│   └── 03-embedding-models/ (03_embedding_models.md)──► Text embeddings & cosine similarity math
├── 02-prompts/ (02_prompts.md)                      ──► Prompt Templates, Messages, History & Chatbots
├── 03-output-parsers/ (03_output_parsers.md)        ──► Str, JSON, Structured, and Pydantic Output Parsers
├── 04-structured-output/ (04_structured_output.md)  ──► Native Tool Calling (.with_structured_output), Pydantic & TypedDict
├── 05-chains/ (05_chains.md)                        ──► Simple, Sequential, Parallel, and Conditional Chains
├── 06-runnables-lcel/ (06_runnables_lcel.md)        ──► LangChain Expression Language (LCEL) & Core Runnables
├── 07-rag-components/
│   ├── 07_rag_overview.md                           ──► End-to-end RAG architecture overview
│   ├── 01-document-loaders/ (01_document_loaders.md)──► Ingest TXT, PDF, CSV, and Directories
│   ├── 02-text-splitters/ (02_text_splitters.md)    ──► Recursive, Markdown, Python, and Semantic chunking
│   └── 03-vector-stores/ (03_vector_stores.md)      ──► ChromaDB Vector Store CRUD & Metadata Filtering
├── 08-practice/ (08_practice.md)                    ──► Hands-on coding exercises & real-world mini-projects
└── 09-resources/ (09_resources.md)                  ──► Textbooks, Deep Learning curriculum, and study paths
```

---

## 🛠️ Prerequisites & Setup

### 1. Clone & Setup Virtual Environment
```bash
# Clone repository
git clone <repo-url>
cd GEN-AI

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```ini
OPENAI_API_KEY="your-openai-api-key"
GROQ_API_KEY="your-groq-api-key"
GOOGLE_AI_API_KEY="your-google-gemini-key"
HUGGINGFACE_API_TOKEN="your-huggingface-token"
```

### 3. Verify Installation
Confirm your LangChain environment is configured properly:
```bash
python verify_langchain.py
```

---

## 🚀 Running Examples

You can run any script from the project root:

```bash
# Chat Model with Groq
python 3-langchain/01-models/02-chat-models/1_chatmodel_groq.py

# Parallel Chain (Notes + Quiz Generator)
python 3-langchain/05-chains/3_parallel_chain.py

# ChromaDB Vector Store
python 3-langchain/07-rag-components/03-vector-stores/1_langchain_chroma.py

# Streamlit Prompt UI
streamlit run 3-langchain/02-prompts/1_prompt_ui.py
```

---

## 💡 Troubleshooting & Tips

- **VS Code "Import could not be resolved"**:
  1. Press `Cmd + Shift + P` (or `Ctrl + Shift + P`)
  2. Select `Python: Select Interpreter`
  3. Choose the `.venv` Python interpreter: `./.venv/bin/python`

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](file:///Users/chandansahoo/Documents/GEN-AI/LICENSE) file for details.
