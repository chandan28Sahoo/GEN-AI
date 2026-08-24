# 01 - Large Language Models (LLMs)

---

## 🌟 1. What is an LLM in LangChain?

In LangChain, **LLM** refers specifically to **pure text completion models** (such as OpenAI's `gpt-3.5-turbo-instruct` or older GPT-3 models).

Unlike chat models that understand back-and-forth conversations, a traditional LLM operates as a **raw text completer**:
- **Input**: A single, raw string of text (`str`).
- **Output**: A single, raw string continuation (`str`).

```
┌────────────────────────────────┐
│       Input Text String        │
│  "The capital of France is"    │
└───────────────┬────────────────┘
                │
         [ LLM Engine ] (Predicts the most probable next tokens)
                │
┌───────────────▼────────────────┐
│      Output Text String        │
│          " Paris."             │
└────────────────────────────────┘
```

---

## ❓ 2. Why Do We Need It & When Is It Used?

### The Problem It Solves
Early generative AI systems were designed around single-shot prompt completion. If you need simple, non-conversational text continuation (such as autocompleting an email sentence, filling in a template, or classifying a single sentence), pure LLMs provide a simple string-to-string API without the overhead of role formatting (`SystemMessage`, `HumanMessage`, `AIMessage`).

### LLMs vs. Chat Models: The Core Difference

| Feature | LLMs (Text Completion) | Chat Models (Conversational) |
| :--- | :--- | :--- |
| **Input Type** | Single string `str` | List of structured message objects `[BaseMessage]` |
| **Output Type** | Single string `str` | Structured `AIMessage` with metadata |
| **Role Awareness** | None (Treats everything as raw text) | Understands `System`, `Human`, `AI`, and `Tool` roles |
| **Primary Use Cases** | Legacy codebases, raw completion tasks | Chatbots, agents, multi-step reasoning, tool calling |
| **LangChain Package** | `from langchain_openai import OpenAI` | `from langchain_openai import ChatOpenAI` |

> [!NOTE]
> Most modern frontier models (GPT-4o, Claude 3.5, Gemini 1.5, Llama 3) are trained specifically as **Chat Models**. Text completion LLMs (`gpt-3.5-turbo-instruct`) are maintained primarily for backward compatibility.

---

## 🚀 3. How Do We Use It?

### Step-by-Step Code Walkthrough

Refer to: [1_llm_demo.py](file:///Users/chandansahoo/Documents/GEN-AI/3-langchain/01-models/01-llms/1_llm_demo.py)

```python
from langchain_openai import OpenAI
from dotenv import load_dotenv

# Step 1: Load environment variables (reads OPENAI_API_KEY from .env)
load_dotenv()

# Step 2: Initialize the LLM instance
# gpt-3.5-turbo-instruct is OpenAI's recommended completion model
llm = OpenAI(
    model='gpt-3.5-turbo-instruct',
    temperature=0.7,   # Controls randomness (0.0 = deterministic, 1.0 = creative)
    max_tokens=100     # Upper limit on generated response length
)

# Step 3: Invoke the model with a raw text string
prompt = "What is the capital of India?"
result = llm.invoke(prompt)

# Step 4: Print the plain string result
print(result)
```

---

## ⚙️ Key Configuration Parameters

| Parameter | Type | Default | What it does |
| :--- | :--- | :--- | :--- |
| `model` | `str` | `gpt-3.5-turbo-instruct` | Identifies the model checkpoint on the provider's server. |
| `temperature` | `float` | `0.7` | Controls sampling entropy. Use `0.0` for factual/math tasks, `0.7`-`1.0` for creative writing. |
| `max_tokens` | `int` | `256` | Maximum number of tokens the model is allowed to generate in the completion. |
| `top_p` | `float` | `1.0` | Nucleus sampling probability threshold. |
| `stop` | `list[str]` | `None` | Optional stop sequences where the model terminates generation immediately. |

---

## ⚠️ Common Pitfalls & Pro Tips

1. **Passing Messages to an LLM**: If you pass a list like `[HumanMessage(...)]` to an `OpenAI()` completion object, it will convert it to a string representation instead of understanding roles. Always use `ChatOpenAI()` for message lists.
2. **Missing API Keys**: Always ensure `OPENAI_API_KEY` is present in your environment or passed explicitly via `api_key="..."`.
3. **Deprecation Notice**: Avoid using older models like `text-davinci-003` or `text-curie-001` as OpenAI has shut down those endpoints.
