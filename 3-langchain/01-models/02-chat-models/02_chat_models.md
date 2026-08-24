# 02 - Chat Models & Multi-Provider Integration

---

## 🌟 1. What is a Chat Model in LangChain?

A **Chat Model** is an AI model fine-tuned using conversational dialogue datasets and Reinforcement Learning from Human Feedback (RLHF).

Instead of treating your prompt as raw text, Chat Models understand **structured message roles**:
- **`SystemMessage`**: The "director". Configures the model's identity, tone, persona, constraints, and instructions.
- **`HumanMessage`**: The "user". Represents the query or instructions coming from a human user.
- **`AIMessage`**: The "assistant". The AI's generated response (which can be fed back into future turns for chat history).

```
   ┌──────────────────────────────────────────────────────────┐
   │                     Input Messages                       │
   │  [SystemMessage("You are a helpful Python mentor.")]      │
   │  [HumanMessage("What is list comprehension in Python?")]  │
   └────────────────────────────┬─────────────────────────────┘
                                │
                        [ Chat Model Engine ]
                                │
   ┌────────────────────────────▼─────────────────────────────┐
   │                     Output Message                       │
   │  AIMessage(                                              │
   │      content="List comprehension is a concise way...",   │
   │      response_metadata={'token_usage': 42, ...}          │
   │  )                                                       │
   └──────────────────────────────────────────────────────────┘
```

---

## ❓ 2. Why Do We Need It & When Is It Used?

### The Problem It Solves
1. **Multi-Turn Conversations**: Chat models maintain context across multiple turns of interaction by consuming an ordered list of previous `HumanMessage` and `AIMessage` items.
2. **Behavioral Guardrails**: `SystemMessage` prevents prompt injection, sets safety boundaries, and enforces persona rules.
3. **Unified API Across Providers**: Different AI vendors (OpenAI, Google, Groq, Anthropic, Hugging Face) have different underlying REST APIs. LangChain provides a single, uniform interface (`.invoke()`, `.stream()`, `.batch()`) across all providers.

---

## 🚀 3. How Do We Use It Across Different Providers?

### 📂 Provider Examples in this Folder

| File | Provider | Package | Default Model |
| :--- | :--- | :--- | :--- |
| [1_chatmodel_groq.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/02-chat-models/1_chatmodel_groq.py) | **Groq** (Ultra-fast LPU inference) | `langchain-groq` | `llama-3.3-70b-versatile` |
| [2_chatmodel_gemini.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/02-chat-models/2_chatmodel_gemini.py) | **Google Gemini** | `langchain-google-genai` | `gemini-1.5-flash` |
| [3_chatmodel_openai.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/02-chat-models/3_chatmodel_openai.py) | **OpenAI** | `langchain-openai` | `gpt-4o` / `gpt-4` |
| [4_chatmodel_hf.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/02-chat-models/4_chatmodel_hf.py) | **Hugging Face Hub** | `langchain-huggingface` | Open-source serverless models |

---

### Step-by-Step Implementation Guide

#### A. Groq (`1_chatmodel_groq.py`)
Groq delivers near-instantaneous inference speeds using specialized LPU chips:
```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq Chat Model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=1000,
    temperature=0.7
)

# Structure conversational roles
messages = [
    SystemMessage(content="You are an expert Python tutor."),
    HumanMessage(content="Explain decorators with a simple real-world example.")
]

# Invoke model
response = llm.invoke(messages)
print(response.content)
```

#### B. Google Gemini (`2_chatmodel_gemini.py`)
```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_AI_API_KEY")
)

messages = [
    SystemMessage(content="You are a helpful coding assistant."),
    HumanMessage(content="Explain Python generators in 3 bullet points.")
]

response = llm.invoke(messages)
print(response.content)
```

#### C. OpenAI (`3_chatmodel_openai.py`)
```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4o', temperature=0.7)
result = model.invoke("Write a 5-line poem about coding in Python")

print(result.content)
```

#### D. Hugging Face Serverless (`4_chatmodel_hf.py`)
```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Connect to Hugging Face Inference Endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
)

# 2. Wrap as a LangChain Chat Model
chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is the difference between supervised and unsupervised learning?")
print(response.content)
```

---

## ⚙️ Key Configuration Parameters

| Parameter | Type | Purpose |
| :--- | :--- | :--- |
| `temperature` | `float` (0.0 – 2.0) | `0.0` for deterministic extraction/code; `0.7`+ for storytelling/brainstorming. |
| `max_tokens` | `int` | Upper cap on generated output tokens. |
| `model` / `model_name` | `str` | Exact model identifier (e.g. `gpt-4o-mini`, `gemini-1.5-flash`, `llama-3.3-70b-versatile`). |
| `streaming` | `bool` | Set to `True` to stream tokens as they are generated. |

---

## ⚠️ Common Pitfalls & Best Practices

1. **Extracting text**: The return value of `.invoke()` is an `AIMessage` object, not a raw string. Access the generated text using **`response.content`**.
2. **Rate Limits & API Costs**: Groq has a generous free tier for developers; OpenAI and Google Gemini charge per million input/output tokens. Always manage `max_tokens` in development.
3. **Environment Security**: Never hardcode API keys directly in source files. Always use `dotenv.load_dotenv()` and keep `.env` in your `.gitignore`.
