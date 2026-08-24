# Module 01: Models & Embeddings - Master Overview

---

## 🌟 1. What is the Models Layer in LangChain?

The **Models Layer** is the fundamental interface in LangChain that enables your Python code to interact with AI models across different providers (OpenAI, Google Gemini, Groq, Hugging Face, Anthropic).

LangChain divides models into three distinct categories:
1. **[01-llms](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/01-llms/01_llms.md)**: Pure text-completion models (`String In` $\rightarrow$ `String Out`).
2. **[02-chat-models](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/02-chat-models/02_chat_models.md)**: Conversational models with role awareness (`Messages In` $\rightarrow$ `AIMessage Out`).
3. **[03-embedding-models](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/03-embedding-models/03_embedding_models.md)**: Semantic vector representation models (`Text In` $\rightarrow$ `Numbers Out`).

```
                              ┌───────────────────────────────────────────────┐
                              │            LANGCHAIN MODELS LAYER             │
                              └───────────────────────┬───────────────────────┘
                                                      │
         ┌────────────────────────────────────────────┼──────────────────────────────────────────┐
         ▼                                            ▼                                          ▼
┌──────────────────┐                         ┌──────────────────┐                       ┌──────────────────┐
│     01-llms      │                         │  02-chat-models  │                       │ 03-embeddings    │
├──────────────────┤                         ├──────────────────┤                       ├──────────────────┤
│ Text Completion  │                         │ Structured Roles │                       │ Semantic Vectors │
│ str -> str       │                         │ [Msg] -> AIMsg   │                       │ str -> [float]   │
│ e.g. OpenAI()    │                         │ e.g. ChatOpenAI()│                       │ e.g. Embeddings()│
└──────────────────┘                         └──────────────────┘                       └──────────────────┘
```

---

## ❓ 2. Why Do We Need LangChain Models Rather Than Raw SDKs?

Without LangChain, switching from OpenAI to Groq or Google Gemini requires rewriting your entire prompt formatting, error handling, parameter naming, and parsing logic.

### Key Benefits of LangChain's Unified Model Interface:
1. **Standardized Method Signatures**: Every model implements `.invoke()`, `.stream()`, `.batch()`, and async equivalents (`.ainvoke()`).
2. **Seamless Switching**: Swap `ChatOpenAI(model="gpt-4o")` for `ChatGroq(model="llama-3.3-70b")` with zero changes to downstream chain logic.
3. **LCEL Compatibility**: Models can be piped directly into prompt templates and output parsers (`prompt | model | parser`).

---

## 🚀 3. How to Navigate This Module

Start with the following topics in order:
1. **[01-llms Guide](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/01-llms/01_llms.md)** - Understand legacy text completion.
2. **[02-chat-models Guide](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/02-chat-models/02_chat_models.md)** - Master modern conversational models across 4 providers.
3. **[03-embedding-models Guide](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/03-embedding-models/03_embedding_models.md)** - Learn vector representations and document similarity math.
