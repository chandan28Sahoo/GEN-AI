# 🎬 YouTube Chatbot

A RAG-powered chatbot that lets you chat with any YouTube video using LangChain, ChromaDB, and an LLM of your choice.

## Features

- 🔗 Extract transcripts from any YouTube URL
- ✂️ Smart text chunking & embedding
- 🧠 Vector store retrieval (ChromaDB)
- 💬 Conversational Q&A with memory
- 🖥️ Streamlit UI

## Project Structure

```
ytube-chat-bots/
├── src/
│   ├── chains/         # LangChain chains (RAG, conversational)
│   ├── tools/          # YouTube transcript fetching tools
│   ├── prompts/        # Prompt templates
│   └── utils/          # Helper utilities
├── data/
│   ├── transcripts/    # Raw & processed transcripts (auto-generated)
│   └── vector_store/   # ChromaDB persistent store (auto-generated)
├── notebooks/          # Experimentation notebooks
├── tests/              # Unit & integration tests
├── config/             # Config files (models, chunking, etc.)
├── app.py              # Streamlit app entry point
├── main.py             # CLI entry point
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run the app
streamlit run app.py
```

## Usage

```python
# CLI
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Then chat!
```
